"""Market data views (equity underlying, rate curve, FX pair) and the snapshot catalog: plain JSON-ready data
(numbers, strings, lists) for the web app and the ``testviews`` notebooks."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import tensorquant as tq

TENORS = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0, "7Y": 7.0, "10Y": 10.0}
N_GRID = 120
RATE_TENORS = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0, "7Y": 7.0,
               "10Y": 10.0, "15Y": 15.0, "20Y": 20.0, "30Y": 30.0}


def snapshots(root: Path) -> list[dict]:
    """Market data snapshots in *root* (``<yyyymmdd>/<set>/``), latest first: ``[{date, sets}]`` (sets latest first)."""
    out = []
    for day in sorted((d for d in Path(root).iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8),
                      key=lambda d: d.name, reverse=True):
        sets = sorted((s.name for s in day.iterdir() if s.is_dir() and not s.name.startswith("_")), reverse=True)
        if sets:
            out.append({"date": datetime.strptime(day.name, "%Y%m%d").date().isoformat(), "sets": sets})
    return out


def catalog(mkt: tq.MarketEnvironment) -> dict:
    """What the market holds, by asset class: equity tickers, rate curves (with their vols), FX pairs."""
    keys = set(mkt.risk_factors())
    curves = sorted({tuple(k.split(":")[1:3]) for k in keys if k.startswith("IR:") and k.endswith(":SPOT")})
    pairs = sorted({k.split(":")[2] for k in keys if k.startswith("FX:") and k.endswith(":SPOT")})
    return {"snapshot": {"date": mkt.evaluation_date.isoformat(), "set": getattr(mkt, "calibration_set", None)},
            "equity": equity_tickers(mkt),
            "rates": [{"id": f"{ccy}:{name}", "ccy": ccy, "name": name,
                       "caplet_vol": f"IR:{ccy}:{name}:VOL" in keys, "swaption_vol": f"IR:{ccy}:{name}:SWVOL" in keys}
                      for ccy, name in curves],
            "fx": [{"id": pair, "base": pair[:3], "quote": pair[3:],
                    "vol": any(k.startswith("FX:") and k.endswith(f":{pair}:VOL") for k in keys)} for pair in pairs]}


def _expiry(mkt: tq.MarketEnvironment, t: float) -> str:
    return (mkt.evaluation_date + timedelta(days=int(round(365 * t)))).isoformat()


def rates_view(mkt: tq.MarketEnvironment, ccy: str, name: str) -> dict:
    """Curve *ccy* / *name*: zero rates and discount factors at the pillars and standard tenors, zero and
    instantaneous forward curves; caplet vols and the swaption cube of the index when the market has them.

    Raises:
        KeyError: If the curve is not in the market.
    """
    keys = mkt.risk_factors()
    if f"IR:{ccy}:{name}:SPOT" not in keys:
        raise KeyError(f"{ccy} {name}")
    currency = tq.Currency[ccy]
    curve = mkt.get_ir_curve(currency, name)
    pillars = [float(t) for t in curve.pillars]
    last = max(pillars)
    tenors = [{"tenor": k, "T": t, "zero_rate": float(curve.zero_rate(t)), "discount_factor": float(curve.discount(t)),
               "forward_3m": float(curve.forward_rate(t, t + 0.25))} for k, t in RATE_TENORS.items() if t <= last]
    grid = np.linspace(1 / 365, last, N_GRID)
    out = {"summary": {"curve": name, "currency": ccy, "key": curve.name, "snapshot": mkt.evaluation_date.isoformat(),
                       "pillars": len(pillars), "last_pillar": curve.dates[-1].isoformat(),
                       "daycount": str(curve.daycounter_convention.name)},
           "pillars": [{"date": d.isoformat(), "T": t, "zero_rate": float(r), "discount_factor": float(curve.discount(t))}
                       for d, t, r in zip(curve.dates, pillars, np.asarray(curve.rates, float))],
           "tenors": tenors,
           "grid": {"T": grid.tolist(), "zero_rate": [float(curve.zero_rate(t)) for t in grid],
                    "inst_forward": [float(curve.inst_fwd(t)) for t in grid]},
           "caplet_vol": None, "swaption_vol": None}
    if f"IR:{ccy}:{name}:VOL" in keys:
        vs = mkt.get_ir_vol_surface(currency, name)
        out["caplet_vol"] = {"vol_type": vs.vol_type, "T": [float(t) for t in vs.maturity],
                             "expiries": [_expiry(mkt, float(t)) for t in vs.maturity],
                             "strikes": [float(k) for k in vs.strike], "matrix": vs.volatility_matrix.numpy().tolist()}
    if f"IR:{ccy}:{name}:SWVOL" in keys:
        cube = mkt.get_swaption_vol_cube(currency, name)
        out["swaption_vol"] = {"vol_type": cube.vol_type, "expiries": [float(t) for t in cube.expiries],
                               "tenors": [float(t) for t in cube.tenors],
                               "strike_offsets": [float(k) for k in cube.strike_offsets],
                               "cube": cube.volatilities.numpy().tolist()}
    return out


def fx_view(mkt: tq.MarketEnvironment, pair: str) -> dict:
    """FX pair (e.g. ``EURUSD``, 1 base = x quote): spot, forwards by covered interest parity on the two default
    curves, implied vol surface (smiles, ATM-forward term structure).

    Raises:
        KeyError: If the pair is not stored in the market.
    """
    base, quote = pair[:3], pair[3:]
    keys = mkt.risk_factors()
    if not any(k.startswith("FX:") and k.endswith(f":{pair}:SPOT") for k in keys):
        raise KeyError(pair)
    spot = float(mkt.get_fx_spot(base, quote))
    def default_curve(ccy):                      # forwards need both currencies' default curves
        try:
            return mkt.get_ir_curve(tq.Currency[ccy])
        except (ValueError, KeyError):
            return None

    base_curve, quote_curve = default_curve(base), default_curve(quote)
    forwards, forward_curve = [], None
    if base_curve is not None and quote_curve is not None:
        for k, t in TENORS.items():
            fwd = float(mkt.get_fx_forward(base, quote, t))
            forwards.append({"tenor": k, "T": t, "forward": fwd, "points": 1e4 * (fwd - spot),
                             "df_base": float(base_curve.discount(t)), "df_quote": float(quote_curve.discount(t))})
        grid = np.linspace(1 / 365, max(TENORS.values()), N_GRID)
        forward_curve = {"T": grid.tolist(), "forward": [float(mkt.get_fx_forward(base, quote, t)) for t in grid]}
    out = {"summary": {"pair": pair, "base": base, "quote": quote, "snapshot": mkt.evaluation_date.isoformat(),
                       "spot": spot, "base_curve": base_curve.name if base_curve is not None else "missing",
                       "quote_curve": quote_curve.name if quote_curve is not None else "missing"},
           "forwards": forwards,
           "forward_curve": forward_curve,
           "vol": None}
    if any(k.startswith("FX:") and k.endswith(f":{pair}:VOL") for k in keys):
        vs = mkt.get_fx_vol_surface(base, quote)
        maturities = [float(t) for t in vs.maturity]
        out["vol"] = {"T": maturities, "expiries": [_expiry(mkt, t) for t in maturities],
                      "strikes": [float(k) for k in vs.strike], "moneyness": [float(k) / spot for k in vs.strike],
                      "matrix": vs.volatility_matrix.numpy().tolist(),
                      "atm_forward": [float(vs.volatility(strike=float(mkt.get_fx_forward(base, quote, t)), tenor=t))
                                      for t in maturities]}
    return out


def equity_tickers(mkt: tq.MarketEnvironment) -> list[str]:
    """Equity tickers of the market, in numeric order (TICKER2 before TICKER10)."""
    names = {k.split(":")[2] for k in mkt.risk_factors() if k.startswith("EQ:") and k.endswith(":SPOT")}
    return sorted(names, key=lambda s: (s.rstrip("0123456789"), int(s[len(s.rstrip("0123456789")):] or 0)))


def market_view(mkt: tq.MarketEnvironment, ticker: str) -> dict:
    """Spot data, forwards and dividends, implied vol surface and correlations of *ticker*.

    Args:
        mkt (MarketEnvironment): The market (its evaluation date must be ``tq.Settings.evaluation_date``).
        ticker (str): Equity ticker.

    Returns:
        dict: JSON-ready sections ``summary``, ``forwards``, ``forward_curve``, ``dividends``, ``vol``,
        ``correlations``.

    Raises:
        KeyError: If *ticker* is not an equity of the market.
    """
    if ticker not in equity_tickers(mkt):
        raise KeyError(ticker)
    with mkt.get_equity(ticker).discount_curve.pinned():       # thousands of discount factors, one market
        return _equity_view(mkt, ticker)


def _equity_view(mkt: tq.MarketEnvironment, ticker: str) -> dict:
    eq = mkt.get_equity(ticker)
    s0 = float(eq.spot)
    curve = eq.discount_curve
    summary = {"ticker": ticker, "currency": eq.currency.name, "snapshot": mkt.evaluation_date.isoformat(),
               "spot": s0, "repo": float(eq.repo), "dividend_model": eq.dividend_model,
               "div_yield": None if eq.div_yield is None else float(eq.div_yield)}

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
            "vol": vol, "correlations": correlations}
