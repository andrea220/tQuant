"""Market data view of one underlying: plain JSON-ready data (numbers, strings, lists) for the web app and the
``testviews`` notebooks."""

from __future__ import annotations

from datetime import timedelta

import numpy as np

import tensorquant as tq

TENORS = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0, "7Y": 7.0, "10Y": 10.0}
N_GRID = 120


def equity_tickers(mkt: tq.MarketEnvironment) -> list[str]:
    """Equity tickers of the market, in numeric order (TICKER2 before TICKER10)."""
    names = {k.split(":")[2] for k in mkt.risk_factors() if k.startswith("EQ:") and k.endswith(":SPOT")}
    return sorted(names, key=lambda s: (s.rstrip("0123456789"), int(s[len(s.rstrip("0123456789")):] or 0)))


def market_view(mkt: tq.MarketEnvironment, ticker: str) -> dict:
    """Spot data, forwards and dividends, discount curve, implied vol surface and correlations of *ticker*.

    Args:
        mkt (MarketEnvironment): The market (its evaluation date must be ``tq.Settings.evaluation_date``).
        ticker (str): Equity ticker.

    Returns:
        dict: JSON-ready sections ``summary``, ``forwards``, ``forward_curve``, ``dividends``, ``curve``, ``vol``,
        ``correlations``.

    Raises:
        KeyError: If *ticker* is not an equity of the market.
    """
    if ticker not in equity_tickers(mkt):
        raise KeyError(ticker)
    eq = mkt.get_equity(ticker)
    s0 = float(eq.spot)
    curve = eq.discount_curve
    summary = {"ticker": ticker, "currency": eq.currency.name, "snapshot": mkt.evaluation_date.isoformat(),
               "spot": s0, "repo": float(eq.repo), "dividend_model": eq.dividend_model,
               "div_yield": None if eq.div_yield is None else float(eq.div_yield), "discount_curve": curve.name}

    forwards = [{"tenor": k, "T": t, "forward": float(eq.forward(t)), "forward_over_spot": float(eq.forward(t)) / s0,
                 "pv_dividends": float(eq.pv_dividends(t)), "implied_div_yield": float(eq.implied_div_yield(t)),
                 "discount_factor": float(curve.discount(t))} for k, t in TENORS.items()]
    grid = np.linspace(1 / 365, max(TENORS.values()), N_GRID)
    forward_curve = {"T": grid.tolist(), "forward": [float(eq.forward(t)) for t in grid],
                     "implied_div_yield": [float(eq.implied_div_yield(t)) for t in grid]}

    dividends = []
    if eq.dividends is not None:
        pay = eq.dividends.payment_dates or [None] * len(eq.dividends)
        dividends = [{"ex_date": d.isoformat(), "pay_date": p.isoformat() if p else None, "amount": float(a),
                      "pct_spot": 100 * float(a) / s0}
                     for d, p, a in zip(eq.dividends.ex_dates, pay, eq.dividends.amounts)]

    pillars = [float(t) for t in curve.pillars]
    zero_grid = np.linspace(1 / 365, max(pillars), N_GRID)
    curve_view = {"name": curve.name,
                  "pillars": [{"date": d.isoformat(), "T": t, "zero_rate": float(r)}
                              for d, t, r in zip(curve.dates, pillars, np.asarray(curve.rates, float))],
                  "grid": {"T": zero_grid.tolist(), "zero_rate": [float(curve.zero_rate(t)) for t in zero_grid]}}

    vol = None
    vs = eq.vol_surface
    if vs is not None:
        maturities = [float(t) for t in vs.maturity]
        vol = {"expiries": [(mkt.evaluation_date + timedelta(days=int(round(365 * t)))).isoformat()
                            for t in maturities],
               "T": maturities,
               "strikes": [float(k) for k in vs.strike],
               "moneyness": [float(k) / s0 for k in vs.strike],
               "matrix": vs.volatility_matrix.numpy().tolist(),
               "atm_forward": [float(vs.volatility(strike=float(eq.forward(t)), tenor=t)) for t in maturities]}

    correlations = []
    corr = mkt.get_correlation_matrix()
    name = f"EQ:{eq.currency.name}:{ticker}"
    if corr is not None and name in corr.names:
        correlations = sorted(({"factor": other, "correlation": float(corr.correlation(name, other))}
                               for other in corr.names if other != name), key=lambda r: -r["correlation"])

    return {"summary": summary, "forwards": forwards, "forward_curve": forward_curve, "dividends": dividends,
            "curve": curve_view, "vol": vol, "correlations": correlations}
