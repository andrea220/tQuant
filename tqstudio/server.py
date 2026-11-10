"""tqstudio web server: the tQuant engine behind a JSON API, and the web pages that call it.

Run from the repository root (``python -m tqstudio``); colleagues on the same network open
``http://<this pc>:8000``.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import tensorquant as tq

from .views.market import equity_tickers, market_view
from .views.tradepad import BARRIER_TYPES, MODELS, PRODUCTS, ModelCache, price_trade

ROOT = Path(__file__).resolve().parent
MARKET_DATA = ROOT.parent / "marketdata"


class Engine:
    """The loaded market (latest snapshot) and a lock: one computation at a time (TensorFlow, global evaluation
    date)."""

    def __init__(self, folder: Path):
        self.lock = threading.Lock()
        self.market = tq.MarketEnvironment.from_folder(folder)
        tq.Settings.evaluation_date = self.market.evaluation_date
        self.cache: dict = {}
        self.models = ModelCache()

    def market_view(self, ticker: str) -> dict:
        with self.lock:
            if ticker not in self.cache:
                self.cache[ticker] = market_view(self.market, ticker)
            return self.cache[ticker]

    def price(self, request: dict) -> dict:
        with self.lock:
            return price_trade(self.market, self.models, request["trade"], request["model"],
                               n_paths=int(request.get("n_paths", 50_000)))


engine = Engine(MARKET_DATA)
app = FastAPI(title="tqstudio")


@app.get("/api/tickers")
def tickers() -> dict:
    return {"snapshot": engine.market.evaluation_date.isoformat(), "tickers": equity_tickers(engine.market)}


@app.get("/api/market/{ticker}")
def market(ticker: str) -> dict:
    try:
        return engine.market_view(ticker)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown underlying {ticker}")


@app.get("/api/tradepad/options")
def tradepad_options() -> dict:
    return {"models": MODELS, "products": PRODUCTS, "barrier_types": BARRIER_TYPES,
            "snapshot": engine.market.evaluation_date.isoformat()}


@app.post("/api/price")
def price(request: dict = Body(...)) -> dict:
    try:
        return engine.price(request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/tradepad")
def tradepad() -> FileResponse:
    return FileResponse(ROOT / "static" / "tradepad.html")
