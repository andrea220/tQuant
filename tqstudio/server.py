"""tqstudio web server: the tQuant engine behind a JSON API, and the web pages that call it.

Run from the repository root (``python -m tqstudio``); colleagues on the same network open
``http://<this pc>:8000``.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import tensorquant as tq

from .views.market import equity_tickers, market_view

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

    def market_view(self, ticker: str) -> dict:
        with self.lock:
            if ticker not in self.cache:
                self.cache[ticker] = market_view(self.market, ticker)
            return self.cache[ticker]


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


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")
