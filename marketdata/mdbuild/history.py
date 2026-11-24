"""Daily price history of the universe in one Parquet file (long format ``date, ticker, value``), updated
incrementally: new tickers get the whole window, the others only the days after their last stored date."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["date", "ticker", "value"]
EIGEN_FLOOR = 1e-4          # smallest eigenvalue of a repaired correlation matrix


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


def nearest_correlation(matrix: np.ndarray, floor: float = 0.0) -> tuple[np.ndarray, float]:
    """A valid correlation matrix close to *matrix* (symmetric, unit diagonal, positive semi-definite): eigenvalues
    below *floor* raised to it, then the diagonal rescaled to 1. Returns the matrix and its smallest eigenvalue before
    the repair (unchanged when it is not negative)."""
    a = 0.5 * (matrix + matrix.T)
    w, v = np.linalg.eigh(a)
    if w.min() >= floor:
        return a, float(w.min())
    b = (v * np.maximum(w, floor)) @ v.T
    d = np.sqrt(np.diag(b))
    out = b / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return out, float(w.min())


def rolling_correlations(history: pd.DataFrame, names: dict[str, str], frequency: str = "W-FRI",
                         window: int = 13, percentile: float = 90.0, fx_percentile: float = 50.0) -> pd.DataFrame:
    """Correlations from their rolling values: log returns of the closes at *frequency* (the last close of each week),
    correlation over every window of *window* returns along the whole history (one step a period), and per pair a
    percentile of those window correlations: *percentile* between equities (a stressed, high correlation: the prudent
    one for the worst-of sold to investors), *fx_percentile* for the pairs with an FX rate (the median: a central
    estimate keeping its sign, the one the quanto drift needs). Each pair on its own common periods (a short history
    does not shorten the others); the matrix repaired to a valid correlation (:func:`nearest_correlation`).

    Args:
        history: Long table ``date, ticker, value``.
        names: Bloomberg ticker -> factor name of the correlation matrix (``"EQ:EUR:ENI_IM"``, ``"FX:USD:EURUSD"``).
        frequency: Pandas period of the returns (``"W-FRI"``: weekly, Friday).
        window: Returns per window (13 weeks = 3 months).
        percentile: Percentile of the window correlations between equities (90).
        fx_percentile: Percentile of the window correlations of the pairs with an FX rate (``FX:`` factors; 50).

    Returns:
        pd.DataFrame: ``factor_1, factor_2, corr`` (upper triangle), the layout of ``correlations.csv``; ``attrs``:
        the settings, the number of windows per pair (minimum), the smallest eigenvalue before the repair and the
        largest change of a pair made by the repair.

    Raises:
        ValueError: If no pair has a full window of common returns.
    """
    wide = history[history["ticker"].isin(names)].pivot(index="date", columns="ticker", values="value").sort_index()
    tickers = [t for t in names if t in wide.columns]
    closes = wide[tickers].resample(frequency).last()
    returns = np.log(closes).diff()
    n = len(tickers)
    raw = np.eye(n)
    counts = []
    for i in range(n):
        for j in range(i + 1, n):
            pair = returns[[tickers[i], tickers[j]]].dropna()
            rolling = pair[tickers[i]].rolling(window).corr(pair[tickers[j]]).dropna()
            rolling = rolling[np.isfinite(rolling)]
            if len(rolling):
                fx = names[tickers[i]].startswith("FX:") or names[tickers[j]].startswith("FX:")
                raw[i, j] = raw[j, i] = float(np.percentile(rolling.to_numpy(), fx_percentile if fx else percentile))
                counts.append(len(rolling))
            else:
                raw[i, j] = raw[j, i] = np.nan
    if not counts:
        raise ValueError("not enough common history for the correlations")
    raw = np.nan_to_num(raw, nan=0.0)                       # a pair without a window: uncorrelated
    fixed, min_eigen = nearest_correlation(raw, EIGEN_FLOOR)       # positive definite: Cholesky for the simulation
    rows = [{"factor_1": names[tickers[i]], "factor_2": names[tickers[j]], "corr": float(fixed[i, j])}
            for i in range(n) for j in range(i + 1, n)]
    out = pd.DataFrame(rows, columns=["factor_1", "factor_2", "corr"])
    out.attrs.update(method="rolling percentile", frequency=frequency, window=int(window), percentile=float(percentile),
                     fx_percentile=float(fx_percentile),
                     min_windows=int(min(counts)), min_eigenvalue=min_eigen,
                     max_repair=float(np.max(np.abs(fixed - raw))) if n > 1 else 0.0)
    return out


FX_VOL_TENORS = (1, 3, 6, 12, 24, 36, 60, 84, 120)          # months: the expiries of the flat FX vol surfaces
FX_VOL_MONEYNESS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)           # of the spot: the strikes


def historical_fx_vols(history: pd.DataFrame, pairs: dict[str, tuple], today, frequency: str = "W-FRI",
                       periods: int = 52) -> pd.DataFrame:
    """Historical vol of each FX pair, as a flat surface (``fx_vol.csv``): the standard deviation of the log returns
    of the closes at *frequency* over the last *periods* returns, annualised (52 weekly returns a year). For the
    quanto drift until implied FX vols are downloaded.

    Args:
        history: Long table ``date, ticker, value``.
        pairs: Bloomberg ticker -> ``(base, quote, spot)`` of each FX pair.
        today: Snapshot date (the expiries of the grid from it).
        frequency: Pandas period of the returns (``"W-FRI"``: weekly, Friday).
        periods: Returns used (52: one year of weeks).

    Returns:
        pd.DataFrame: ``base, quote, expiry, strike, vol`` on ``FX_VOL_TENORS`` x ``FX_VOL_MONEYNESS`` of the spot;
        ``attrs``: the settings and the vol per pair.
    """
    per_year = {"W": 52.0, "D": 252.0, "B": 252.0, "M": 12.0}[frequency[0]]
    rows, vols = [], {}
    for bbg, (base, quote, spot) in pairs.items():
        closes = history[history["ticker"] == bbg].set_index("date")["value"].sort_index()
        closes = closes[closes.index <= pd.Timestamp(today)].resample(frequency).last().dropna()
        returns = np.log(closes).diff().dropna().iloc[-periods:]
        if len(returns) < 2:
            continue
        vol = float(returns.std(ddof=1) * np.sqrt(per_year))
        vols[f"{base}{quote}"] = vol
        for months in FX_VOL_TENORS:
            expiry = (pd.Timestamp(today) + pd.DateOffset(months=months)).date().isoformat()
            rows += [{"base": base, "quote": quote, "expiry": expiry, "strike": float(spot) * m, "vol": vol}
                     for m in FX_VOL_MONEYNESS]
    out = pd.DataFrame(rows, columns=["base", "quote", "expiry", "strike", "vol"])
    out.attrs.update(method="historical", frequency=frequency, periods=int(periods), vols=vols)
    return out


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
