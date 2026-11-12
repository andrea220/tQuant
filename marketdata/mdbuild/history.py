"""Daily price history of the universe in one Parquet file (long format ``date, ticker, value``), updated
incrementally: new tickers get the whole window, the others only the days after their last stored date."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["date", "ticker", "value"]


def read_history(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "ticker": pd.Series(dtype=str),
                             "value": pd.Series(dtype=float)})
    return pd.read_parquet(path)[COLUMNS]


def update_history(path: Path, source, tickers: list[str], field: str, years: int, today: date) -> pd.DataFrame:
    """Download what is missing up to *today* and rewrite the file.

    Returns:
        pd.DataFrame: The whole history (sorted by ticker and date, one value per pair).
    """
    stored = read_history(path)
    start_window = today - timedelta(days=int(round(365.25 * years)))
    last = stored.groupby("ticker")["date"].max() if len(stored) else pd.Series(dtype="datetime64[ns]")
    new = [t for t in tickers if t not in last.index]
    pieces = [stored]
    if new:
        pieces.append(source.history(new, field, start_window, today))
    by_start: dict = {}
    for t in tickers:
        if t in last.index:
            start = (pd.Timestamp(last[t]) + pd.Timedelta(days=1)).date()
            if start <= today:
                by_start.setdefault(start, []).append(t)
    for start, group in by_start.items():
        pieces.append(source.history(group, field, start, today))
    full = pd.concat([p for p in pieces if len(p)], ignore_index=True) if any(len(p) for p in pieces) else stored
    full["date"] = pd.to_datetime(full["date"])
    full = (full.dropna(subset=["value"]).drop_duplicates(["date", "ticker"], keep="last")
            .sort_values(["ticker", "date"]).reset_index(drop=True))
    full = full[full["date"] >= pd.Timestamp(start_window)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    full[COLUMNS].to_parquet(path, index=False)
    return full


def correlations(history: pd.DataFrame, names: dict[str, str], window: int) -> pd.DataFrame:
    """Correlation of daily log returns over the last *window* dates common to all the tickers of *names*.

    Args:
        history: Long table ``date, ticker, value``.
        names: Bloomberg ticker -> factor name of the correlation matrix (``"EQ:EUR:ENI_IM"``, ``"FX:USD:EURUSD"``).
        window: Number of returns.

    Returns:
        pd.DataFrame: ``factor_1, factor_2, corr`` (upper triangle), the layout of ``correlations.csv``.
    """
    wide = history[history["ticker"].isin(names)].pivot(index="date", columns="ticker", values="value")
    wide = wide[[t for t in names if t in wide.columns]].dropna().sort_index()
    returns = np.log(wide).diff().dropna().tail(int(window))
    if len(returns) < 2:
        raise ValueError("not enough common history for the correlations")
    corr = returns.corr()
    cols = list(corr.columns)
    rows = [{"factor_1": names[a], "factor_2": names[b], "corr": float(corr.loc[a, b])}
            for i, a in enumerate(cols) for b in cols[i + 1:]]
    return pd.DataFrame(rows, columns=["factor_1", "factor_2", "corr"])
