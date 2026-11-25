"""tqstudio web server: the tQuant engine behind a JSON API, and the web pages that call it.

Run from the repository root (``python -m tqstudio``); colleagues on the same network open
``http://<this pc>:8010``.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import tensorquant as tq

from .views.calibration import (CalibrationRuns, save_flags, set_folder, set_view, sets_view, slv_check, slv_monitor,
                                test_view, underlying_view, universe_view)
from .views.market import catalog, equity_tickers, fit_view, fx_view, market_view, rates_view, snapshots
from .views.tradepad import ModelCache, describe_trade, market_inputs, price_trade
from .views.tradepad import catalog as tradepad_catalog
from .paths import MARKETDATA
from .views.universe import (DownloadRuns, add_ticker, allowed_client, app_settings, remove_ticker,
                             server_authorized, set_active)
from .views.universe import universe_view as market_universe_view

ROOT = Path(__file__).resolve().parent
DATA_SOURCE = os.environ.get("TQSTUDIO_DATA", "live")   # live: Bloomberg snapshots; synthetic: generated ones
MARKET_DATA = MARKETDATA / ("live" if DATA_SOURCE == "live" else "")     # the shared market data folder
CALIBRATIONS = MARKETDATA / "calibrations" / DATA_SOURCE                    # calibration sets (git-ignored)
DOCS_BUILD = ROOT.parent / "study" / "docs" / "build"      # LaTeX builds (latexmk in study/docs; not in git)
DOCUMENTS = {"models": ("Models and Pricing", "models_and_pricing.pdf"),
             "technical": ("Architecture", "architecture.pdf")}


class Engine:
    """Market snapshots loaded on demand and cached by ``(date, set)`` (the Tradepad prices on the latest one), the
    computed views, and a lock: one computation at a time (TensorFlow, global evaluation date set per call)."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.lock = threading.Lock()
        self.markets: dict = {}
        self.views: dict = {}
        try:
            self.market = self._load(None, None)          # latest snapshot (Tradepad)
        except FileNotFoundError:
            self.market = None                            # no snapshot yet: the app starts empty, waiting for a download
        self.models = ModelCache(CALIBRATIONS)
        self.jobs: dict = {}                              # pending second phases of the prices (greeks)
        self.projections: dict = {}                       # projections of the last prices, computed on request
        self._job_ids = itertools.count(1)

    def _load(self, day: Optional[str], calibration_set: Optional[str]) -> tq.MarketEnvironment:
        mkt = tq.MarketEnvironment.from_folder(self.folder, date.fromisoformat(day) if day else None,
                                               calibration_set or None)
        self.markets[(mkt.evaluation_date.isoformat(), mkt.calibration_set)] = mkt
        return mkt

    def snapshot(self, day: Optional[str], calibration_set: Optional[str]) -> tq.MarketEnvironment:
        """The market of a snapshot (``None``: the latest date / set), evaluation date set (call under the lock)."""
        mkt = self.markets.get((day or None, calibration_set or None))
        if mkt is None:
            mkt = self._load(day, calibration_set)
            self.markets[(day or None, calibration_set or None)] = mkt      # also under the requested key
        tq.Settings.evaluation_date = mkt.evaluation_date
        return mkt

    def view(self, kind: str, key: str, day: Optional[str], calibration_set: Optional[str], build) -> dict:
        with self.lock:
            mkt = self.snapshot(day, calibration_set)
            cache_key = (kind, key, mkt.evaluation_date.isoformat(), mkt.calibration_set)
            if cache_key not in self.views:
                self.views[cache_key] = build(mkt)
            return self.views[cache_key]

    def price(self, request: dict) -> dict:
        """Tradepad pricing on the snapshot of ``request["date"]`` (latest set of that date; latest when absent).
        ``request["phase"] == "price"``: the price at once and a ``job`` id for :meth:`greeks` (the rest of the
        computation is kept here: the last few only)."""
        with self.lock:
            mkt = self.snapshot(request.get("date"), None)
            out = price_trade(mkt, self.models, request, n_paths=int(request.get("n_paths", 50_000)),
                              phase=request.get("phase", "all"))
            pending = out.pop("pending", None)
            projections = out.pop("projections", None)
            if pending is not None:
                job = str(next(self._job_ids))
                self.jobs[job] = (mkt, pending)
                for old in list(self.jobs)[:-4]:          # keep the last few
                    self.jobs.pop(old, None)
                if projections is not None:
                    self.projections[job] = (mkt, projections)
                    for old in list(self.projections)[:-4]:
                        self.projections.pop(old, None)
                out["job"] = job
                out["has_projections"] = projections is not None
            return out

    def greeks(self, job: str) -> dict:
        """Second phase of a :meth:`price` with ``phase="price"``: the greeks (and the whole result)."""
        with self.lock:
            if job not in self.jobs:
                raise KeyError(f"unknown or expired pricing job {job}")
            mkt, finish = self.jobs.pop(job)
            tq.Settings.evaluation_date = mkt.evaluation_date
            return finish()

    def project(self, job: str) -> dict:
        """Projections of the price *job* (expected flows, autocall probabilities, forwards), from its paths; the
        result is kept."""
        with self.lock:
            if job not in self.projections:
                raise KeyError(f"no projections for pricing job {job} (price again)")
            mkt, build = self.projections[job]
            if callable(build):
                tq.Settings.evaluation_date = mkt.evaluation_date
                self.projections[job] = (mkt, build())
            return self.projections[job][1]

    def refresh_latest(self) -> None:
        """A new snapshot was written: the latest one (``None`` keys, Tradepad market) is looked up again."""
        with self.lock:
            self.markets.pop((None, None), None)
            self.market = self._load(None, None)

    def tradepad(self, build, day: Optional[str] = None, request: Optional[dict] = None):
        with self.lock:
            mkt = self.snapshot(day, None)
            return build(mkt) if request is None else build(mkt, request)


engine = Engine(MARKET_DATA)
app = FastAPI(title="tqstudio")


@app.middleware("http")
async def no_cache_pages(request, call_next):
    """Pages and static files are revalidated at every load (ETag, 304 when unchanged): no stale JavaScript after
    an update."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/api/tickers")
def tickers() -> dict:
    if engine.market is None:
        return {"snapshot": None, "tickers": []}
    return {"snapshot": engine.market.evaluation_date.isoformat(), "tickers": equity_tickers(engine.market)}


@app.get("/api/snapshots")
def list_snapshots() -> dict:
    return {"source": DATA_SOURCE, "snapshots": snapshots(MARKET_DATA)}


def _view(kind: str, key: str, day: Optional[str], calibration_set: Optional[str], build) -> dict:
    try:
        return engine.view(kind, key, day, calibration_set, build)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"not in this snapshot: {exc}")


@app.get("/api/catalog")
def market_catalog(date: Optional[str] = None, set: Optional[str] = None) -> dict:
    return _view("catalog", "", date, set, catalog)


@app.get("/api/market/equity/{ticker}")
def market_equity(ticker: str, date: Optional[str] = None, set: Optional[str] = None) -> dict:
    def build(mkt):
        folder = MARKET_DATA / mkt.evaluation_date.strftime("%Y%m%d") / str(mkt.calibration_set)
        return market_view(mkt, ticker, folder)

    return _view("equity", ticker, date, set, build)


@app.get("/api/market/equity/{ticker}/fit")
def market_equity_fit(ticker: str, date: Optional[str] = None, set: Optional[str] = None) -> dict:
    """The listed options of the vol surface repriced by Black-Scholes (surface vol and smile vol) vs the market."""
    def build(mkt):
        folder = MARKET_DATA / mkt.evaluation_date.strftime("%Y%m%d") / str(mkt.calibration_set)
        return fit_view(mkt, ticker, folder)

    return _view("fit", ticker, date, set, build)


@app.get("/api/market/rates/{ccy}/{curve}")
def market_rates(ccy: str, curve: str, date: Optional[str] = None, set: Optional[str] = None) -> dict:
    def build(mkt):
        folder = MARKET_DATA / mkt.evaluation_date.strftime("%Y%m%d") / str(mkt.calibration_set)
        return rates_view(mkt, ccy, curve, folder)

    return _view("rates", f"{ccy}:{curve}", date, set, build)


@app.get("/api/market/fx/{pair}")
def market_fx(pair: str, date: Optional[str] = None, set: Optional[str] = None) -> dict:
    return _view("fx", pair, date, set, lambda mkt: fx_view(mkt, pair))


downloads = DownloadRuns()
_refreshed = {"result": None}                         # the last download whose snapshot the engine has picked up


def is_local(request: Request) -> bool:
    """The request comes from an authorized pc (``authorized_pcs`` of ``tqstudio/config.yaml``, read at each request):
    it may change the universe."""
    return allowed_client(request.client.host if request.client else None, app_settings()["authorized_pcs"])


def can_download(request: Request) -> bool:
    """A download may start: live data, this machine authorized (it has the Bloomberg terminal the download runs on)
    and the request from an authorized pc."""
    return DATA_SOURCE == "live" and server_authorized(app_settings()["authorized_pcs"]) and is_local(request)


@app.get("/api/market/universe")
def market_universe(request: Request, date: Optional[str] = None, set: Optional[str] = None) -> dict:
    """The market data universe with the status of each instrument in the snapshot *date* / *set* (latest when
    absent), and whether this client may start a download."""
    found = snapshots(MARKET_DATA)
    day = next((s for s in found if s["date"] == date), found[0] if found else None)
    folder = None
    if day is not None:
        cset = set if set in day["sets"] else day["sets"][0]
        folder = MARKET_DATA / day["date"].replace("-", "") / cset
    return {**market_universe_view(folder), "source": DATA_SOURCE,
            "can_download": can_download(request), "can_edit": is_local(request),
            "server_authorized": server_authorized(app_settings()["authorized_pcs"]),
            "download": downloads.status()}


@app.post("/api/market/universe")
def market_universe_add(request: Request, body: dict = Body(...)) -> dict:
    """Add a Bloomberg ticker to the universe (``{"asset_class", "ticker"}``): downloaded at the next download."""
    if not is_local(request):
        raise HTTPException(status_code=403, detail="the universe can be changed only from the pcs of authorized_pcs (tqstudio/config.yaml)")
    try:
        return add_ticker(body.get("asset_class", ""), body.get("ticker", ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/market/universe/remove")
def market_universe_remove(request: Request, body: dict = Body(...)) -> dict:
    """Remove a ticker from the universe (``{"asset_class", "ticker"}``): commented out, not downloaded any more."""
    if not is_local(request):
        raise HTTPException(status_code=403, detail="the universe can be changed only from the pcs of authorized_pcs (tqstudio/config.yaml)")
    try:
        return remove_ticker(body.get("asset_class", ""), body.get("ticker", ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/market/universe/active")
def market_universe_active(request: Request, body: dict = Body(...)) -> dict:
    """Switch the ``active`` flag of an instrument or a curve (``{"name", "active"}``): active ones are downloaded by
    a full download."""
    if not is_local(request):
        raise HTTPException(status_code=403, detail="the universe can be changed only from the pcs of authorized_pcs (tqstudio/config.yaml)")
    try:
        return set_active(body.get("name", ""), bool(body.get("active")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/market/download")
def market_download(request: Request, body: Optional[dict] = Body(None)) -> dict:
    """Start a download: every active instrument, or ``{"select": [Bloomberg tickers]}`` (those equities with the
    active rates, FX and curves); ``"calibrate": true`` also calibrates the models of the new snapshot."""
    if DATA_SOURCE != "live":
        raise HTTPException(status_code=400, detail="downloads write live snapshots: start tqstudio with --data live")
    if not server_authorized(app_settings()["authorized_pcs"]):
        raise HTTPException(status_code=403, detail="Not authorized: this tqstudio runs on a pc without a Bloomberg "
                                                    "terminal (not in authorized_pcs of tqstudio/config.yaml). Start "
                                                    "the download from the tqstudio of an authorized pc: its snapshot "
                                                    "appears here.")
    if not is_local(request):
        who = request.client.host if request.client else "unknown"
        raise HTTPException(status_code=403, detail=f"Not authorized: your pc ({who}) is not in authorized_pcs of "
                                                    "tqstudio/config.yaml, the pcs allowed to start downloads.")
    try:
        body = body or {}

        def calibrate(folder):                         # the models of the new snapshot, their flags
            runs.start({"date": f"{folder.parent.name[:4]}-{folder.parent.name[4:6]}-{folder.parent.name[6:]}",
                        "set": folder.name})
            return f"calibration started on {folder.parent.name}/{folder.name}"

        return downloads.start(body.get("select") or None, then=calibrate if body.get("calibrate") else None)
    except ValueError as exc:                          # already running, or no Python with blpapi
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/market/download")
def market_download_status() -> dict:
    status = downloads.status()
    if not status["running"] and status.get("exit_code") == 0 and status.get("result") != _refreshed["result"]:
        engine.refresh_latest()                       # the new snapshot becomes the latest (Tradepad, calibration)
        _refreshed["result"] = status.get("result")
    return status


@app.get("/api/tradepad/catalog")
def tradepad_products(date: Optional[str] = None) -> dict:
    try:
        return engine.tradepad(tradepad_catalog, date)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/tradepad/trade")
def tradepad_trade(request: dict = Body(...)) -> dict:
    try:
        return engine.tradepad(describe_trade, request.get("date"), request)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/tradepad/market")
def tradepad_market(request: dict = Body(...)) -> dict:
    try:
        return engine.tradepad(market_inputs, request.get("date"), request)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/price")
def price(request: dict = Body(...)) -> dict:
    try:
        return engine.price(request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/price/projections")
def price_projections(request: dict = Body(...)) -> dict:
    try:
        return engine.project(str(request.get("job")))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/price/greeks")
def price_greeks(request: dict = Body(...)) -> dict:
    try:
        return engine.greeks(str(request.get("job")))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


runs = CalibrationRuns()


def _source(folder):
    """The snapshot a calibration set was calibrated on (evaluation date set; call under the lock)."""
    info = json.loads((folder / "calibration_info.json").read_text(encoding="utf-8"))["snapshot"]
    return engine.snapshot(info["date"], info["set"])


@app.get("/api/calibration/sets")
def calibration_sets() -> dict:
    return {**sets_view(CALIBRATIONS), "run": runs.status()}


@app.get("/api/calibration/set/{day}/{cset}")
def calibration_set(day: str, cset: str) -> dict:
    try:
        return set_view(set_folder(CALIBRATIONS, day, cset))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/calibration/underlying/{day}/{cset}/{ticker}")
def calibration_underlying(day: str, cset: str, ticker: str) -> dict:
    try:
        folder = set_folder(CALIBRATIONS, day, cset)
        with engine.lock:
            latest = engine.snapshot(None, None)
            source = _source(folder)
            same = (latest.evaluation_date, latest.calibration_set) == (source.evaluation_date, source.calibration_set)
            return underlying_view(folder, ticker, source, None if same else latest)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/calibration/test")
def calibration_test(request: dict = Body(...)) -> dict:
    try:
        folder = set_folder(CALIBRATIONS, request["date"], request["set"])
        with engine.lock:
            return test_view(folder, request["ticker"], _source(folder), rerun=bool(request.get("rerun")),
                             n_paths=int(request.get("n_paths", 20_000)))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/calibration/universe")
def calibration_universe() -> dict:
    with engine.lock:
        return universe_view(engine.snapshot(None, None), CALIBRATIONS)


@app.post("/api/calibration/universe")
def calibration_universe_save(request: dict = Body(...)) -> dict:
    try:
        return save_flags(request["asset_class"], request["name"], request.get("flags") or {})
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/calibration/slv")
def calibration_slv() -> dict:
    with engine.lock:
        latest = engine.snapshot(None, None)
        return slv_monitor(CALIBRATIONS, latest, engine.snapshot)


@app.post("/api/calibration/slv/check")
def calibration_slv_check(request: dict = Body(...)) -> dict:
    try:
        with engine.lock:
            return slv_check(CALIBRATIONS, engine.snapshot(None, None), request["ticker"])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/calibration/run")
def calibration_run(request: dict = Body(...)) -> dict:
    try:
        return runs.start(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/calibration/run")
def calibration_run_status() -> dict:
    return runs.status()


@app.get("/api/documents")
def documents() -> dict:
    snapshot = engine.market.evaluation_date.isoformat() if engine.market is not None else None
    return {"snapshot": snapshot, "source": DATA_SOURCE,
            "documents": [{"key": k, "title": title, "available": (DOCS_BUILD / name).is_file()}
                          for k, (title, name) in DOCUMENTS.items()]}


@app.get("/documents/{key}")
def document(key: str) -> FileResponse:
    if key not in DOCUMENTS or not (DOCS_BUILD / DOCUMENTS[key][1]).is_file():
        raise HTTPException(status_code=404, detail=f"document {key} not built (run latexmk in study/docs)")
    return FileResponse(DOCS_BUILD / DOCUMENTS[key][1], media_type="application/pdf",
                        content_disposition_type="inline")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "static" / "home.html")


@app.get("/market")
def market_page() -> FileResponse:
    return FileResponse(ROOT / "static" / "market.html")


@app.get("/tradepad")
def tradepad() -> FileResponse:
    return FileResponse(ROOT / "static" / "tradepad.html")


@app.get("/calibration")
def calibration_page() -> FileResponse:
    return FileResponse(ROOT / "static" / "calibration.html")
