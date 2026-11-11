"""tqstudio web server: the tQuant engine behind a JSON API, and the web pages that call it.

Run from the repository root (``python -m tqstudio``); colleagues on the same network open
``http://<this pc>:8010``.
"""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import tensorquant as tq

from .views.market import catalog, equity_tickers, fx_view, market_view, rates_view, snapshots
from .views.tradepad import ModelCache, price_trade
from .views.tradepad import catalog as tradepad_catalog

ROOT = Path(__file__).resolve().parent
MARKET_DATA = ROOT.parent / "marketdata"
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
        self.market = self._load(None, None)              # latest snapshot (Tradepad)
        self.models = ModelCache()

    def _load(self, day: Optional[str], calibration_set: Optional[str]) -> tq.MarketEnvironment:
        mkt = tq.MarketEnvironment.from_folder(self.folder, date.fromisoformat(day) if day else None,
                                               calibration_set or None)
        self.markets[(mkt.evaluation_date.isoformat(), mkt.calibration_set)] = mkt
        return mkt

    def snapshot(self, day: Optional[str], calibration_set: Optional[str]) -> tq.MarketEnvironment:
        """The market of a snapshot (``None``: the latest date / set), evaluation date set (call under the lock)."""
        mkt = self.markets.get((day, calibration_set)) if day and calibration_set else None
        if mkt is None:
            mkt = self._load(day, calibration_set)
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
        with self.lock:
            tq.Settings.evaluation_date = self.market.evaluation_date
            return price_trade(self.market, self.models, request, n_paths=int(request.get("n_paths", 50_000)))


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
    return {"snapshot": engine.market.evaluation_date.isoformat(), "tickers": equity_tickers(engine.market)}


@app.get("/api/snapshots")
def list_snapshots() -> dict:
    return {"snapshots": snapshots(MARKET_DATA)}


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
    return _view("equity", ticker, date, set, lambda mkt: market_view(mkt, ticker))


@app.get("/api/market/rates/{ccy}/{curve}")
def market_rates(ccy: str, curve: str, date: Optional[str] = None, set: Optional[str] = None) -> dict:
    return _view("rates", f"{ccy}:{curve}", date, set, lambda mkt: rates_view(mkt, ccy, curve))


@app.get("/api/market/fx/{pair}")
def market_fx(pair: str, date: Optional[str] = None, set: Optional[str] = None) -> dict:
    return _view("fx", pair, date, set, lambda mkt: fx_view(mkt, pair))


@app.get("/api/tradepad/catalog")
def tradepad_products() -> dict:
    with engine.lock:
        tq.Settings.evaluation_date = engine.market.evaluation_date
        return tradepad_catalog(engine.market)


@app.post("/api/price")
def price(request: dict = Body(...)) -> dict:
    try:
        return engine.price(request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/documents")
def documents() -> dict:
    return {"snapshot": engine.market.evaluation_date.isoformat(),
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
