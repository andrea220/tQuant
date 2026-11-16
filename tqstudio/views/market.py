"""Market data views (equity underlying, rate curve, FX pair) and the snapshot catalog: plain JSON-ready data
(numbers, strings, lists) for the web app and the ``testviews`` notebooks."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import json

import numpy as np
import pandas as pd

import tensorquant as tq
from tensorquant.calibration.equityvol import LognormalMixture

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


def smile_quotes(folder: Path, ticker: str, spot: float, surface=None, target: float = 0.75):
    """Calibrated smiles of *ticker* against its option quotes, from the snapshot files ``eq_vol_calibration.csv``
    (mixture parameters per expiry) and ``eq_vol_quotes.csv`` (bid / mid / ask vols): per expiry the model smile
    (redrawn from the parameters), the stored surface at that maturity and the quotes. ``None`` without the files.

    Returns:
        dict | None: ``expiries`` (list) and ``default`` (index of the expiry nearest *target* years).
    """
    cal_path, quotes_path = Path(folder) / "eq_vol_calibration.csv", Path(folder) / "eq_vol_quotes.csv"
    if not cal_path.is_file() or not quotes_path.is_file():
        return None
    cal, quotes = pd.read_csv(cal_path), pd.read_csv(quotes_path)
    cal, quotes = cal[cal["ticker"] == ticker], quotes[quotes["ticker"] == ticker]
    if "weights" not in cal or cal["weights"].isna().all():
        return None
    out = []
    for row in cal[cal["weights"].notna()].itertuples(index=False):
        mixture = LognormalMixture(json.loads(row.weights), json.loads(row.means), json.loads(row.stdevs),
                                   float(row.default))
        q = quotes[quotes["expiry"] == row.expiry].sort_values("strike")
        lo = min(0.5 * row.forward, q["strike"].min() if len(q) else row.forward)
        hi = max(1.5 * row.forward, q["strike"].max() if len(q) else row.forward)
        line = np.linspace(lo, hi, 121)
        used = q[q["used"].astype(bool) & ~q["settlement"].astype(bool)]    # market quotes (settlement: bid = ask)
        bid = used["iv_bid"].fillna(used["iv_mid"])
        inside = ((used["model_vol"] >= bid - 1e-12) & (used["model_vol"] <= used["iv_ask"] + 1e-12)).mean() \
            if len(used) else float("nan")
        nan = lambda x: None if x is None or not np.isfinite(x) else float(x)
        out.append({
            "expiry": row.expiry, "t": float(row.t), "forward": float(row.forward), "repo": float(row.repo),
            "quotes": int(row.quotes), "source": row.source, "fit_rms": float(row.fit_rms),
            "inside": nan(inside), "atm_vol": float(row.atm_vol),
            "line": {"moneyness": (line / spot).tolist(),
                     "model": [nan(v) for v in mixture.implied_vols(row.forward, line, row.t)],
                     "surface": None if surface is None else
                     [float(surface.volatility(strike=float(k), tenor=float(row.t))) for k in line]},
            "points": [{"moneyness": r.strike / spot, "strike": r.strike, "type": r.option_type, "bid": nan(r.iv_bid),
                        "mid": nan(r.iv_mid), "ask": nan(r.iv_ask), "settlement": bool(r.settlement),
                        "used": bool(r.used)} for r in q.itertuples(index=False)]})
    if not out:
        return None
    default = int(np.argmin([abs(e["t"] - target) for e in out]))
    return {"expiries": out, "default": default}


def fit_view(mkt: tq.MarketEnvironment, ticker: str, folder: Path) -> dict:
    """The listed options used to build the vol surface of *ticker*, repriced by :class:`BlackScholesPricer` on the
    snapshot market (spot, dividends, repo, discount curve, vol read from the stored surface: what a pricer sees),
    and with the vol of the calibrated smile at the strike instead (same inputs: the fit alone, without the
    interpolation between the surface nodes).

    Args:
        mkt (MarketEnvironment): The snapshot market (evaluation date set by the caller).
        ticker (str): Equity ticker.
        folder (Path): The snapshot folder (``eq_vol_quotes.csv``, ``eq_option_quotes.csv``).

    Returns:
        dict: ``spot``, ``quotes`` (one per used option: expiry, strike, type, exercise, settlement, bid / ask / mid
        prices, Black vols bid / mid / ask, surface and smile vols and prices) and ``summary`` per model (share of the
        two-sided quotes priced within bid / ask, rms and max difference in % of the spot and in vol points).

    Raises:
        FileNotFoundError: Without the option quote files (a snapshot built before the equity vols).
    """
    from tensorquant.pricers.black import BlackScholesPricer, bjerksund_stensland_calc, blackscholes_calc

    vq_path, oq_path = Path(folder) / "eq_vol_quotes.csv", Path(folder) / "eq_option_quotes.csv"
    if not vq_path.is_file() or not oq_path.is_file():
        raise FileNotFoundError(f"no option quotes in the snapshot for {ticker}")
    vq = pd.read_csv(vq_path)
    vq = vq[(vq["ticker"] == ticker) & vq["used"].astype(bool)]
    oq = pd.read_csv(oq_path, dtype={"last_update_time": str}, keep_default_na=True)
    if "last_update_time" not in oq:                               # snapshots before the time was kept
        oq["last_update_time"] = ""
    oq = oq[oq["ticker"] == ticker][["expiry", "strike", "option_type", "exercise", "bid", "ask", "last",
                                     "last_update", "last_update_time", "quote_date"]]
    rows = vq.merge(oq, on=["expiry", "strike", "option_type"], how="left")
    eq = mkt.get_equity(ticker)
    spot = float(eq.spot)
    ccy = eq.ccy if hasattr(eq, "ccy") else mkt.get_currency(ticker)
    pricer = BlackScholesPricer()                                     # defaults: as the vols were implied
    out = []
    for r in rows.itertuples(index=False):
        expiry = datetime.strptime(str(r.expiry), "%Y-%m-%d").date()
        if expiry <= mkt.evaluation_date:
            continue
        option_type = tq.OptionType.Call if str(r.option_type).upper().startswith("C") else tq.OptionType.Put
        american = str(getattr(r, "exercise", "")).lower().startswith("american")
        option = tq.VanillaOption(ccy, mkt.evaluation_date, expiry, option_type, float(r.strike), underlying=ticker,
                                  exercise_type=tq.ExerciseType.American if american else tq.ExerciseType.European)
        s_net, k, rate, q, sigma, t = pricer._build_inputs(option, mkt)
        calc = bjerksund_stensland_calc if american else blackscholes_calc

        def price(vol):
            return float(calc(s_net, k, rate, tf_constant(vol), t, q, option_type))

        settlement = bool(r.settlement)
        bid = float(r.last) if settlement else float(r.bid)
        ask = float(r.last) if settlement else float(r.ask)
        surface_vol, smile_vol = float(sigma), float(r.model_vol)
        time = r.last_update_time if isinstance(r.last_update_time, str) else ""
        out.append({"expiry": r.expiry, "strike": float(r.strike), "option_type": "C" if option_type == tq.OptionType.Call else "P",
                    "exercise": "American" if american else "European", "settlement": settlement,
                    "last_update": str(r.last_update) if isinstance(r.last_update, str) else None,
                    "last_update_time": time or None,
                    "previous_session": isinstance(r.last_update, str) and isinstance(r.quote_date, str)
                    and r.last_update < r.quote_date,
                    "bid": bid, "ask": ask, "mid": 0.5 * (bid + ask),
                    "vol_bid": _num(r.iv_bid), "vol_mid": _num(r.iv_mid), "vol_ask": _num(r.iv_ask),
                    "surface_vol": surface_vol, "surface_price": price(surface_vol),
                    "smile_vol": smile_vol, "smile_price": price(smile_vol)})
    summary = {}
    for model in ("surface", "smile"):
        two = [o for o in out if not o["settlement"]]
        inside = [o["bid"] - 1e-12 <= o[f"{model}_price"] <= o["ask"] + 1e-12 for o in two]
        diff = np.array([100.0 * (o[f"{model}_price"] - o["mid"]) / spot for o in out])
        dvol = np.array([100.0 * (o[f"{model}_vol"] - o["vol_mid"]) for o in out if o["vol_mid"] is not None])
        summary[model] = {"within_bid_ask": float(np.mean(inside)) if inside else None, "two_sided": len(two),
                          "rms_pct_spot": float(np.sqrt(np.mean(diff ** 2))) if diff.size else None,
                          "max_pct_spot": float(np.max(np.abs(diff))) if diff.size else None,
                          "rms_volpts": float(np.sqrt(np.mean(dvol ** 2))) if dvol.size else None,
                          "max_volpts": float(np.max(np.abs(dvol))) if dvol.size else None}
    quote_date = rows["quote_date"].dropna().astype(str)
    return {"ticker": ticker, "spot": spot, "quotes": out, "summary": summary, "count": len(out),
            "quote_date": quote_date.iloc[0] if len(quote_date) else None,
            "has_times": any(o["last_update_time"] for o in out),
            "previous_session": sum(1 for o in out if o["previous_session"])}


def tf_constant(x: float):
    import tensorflow as tf
    return tf.constant(float(x), dtype=tf.float64)


def _num(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def market_view(mkt: tq.MarketEnvironment, ticker: str, folder: Path = None) -> dict:
    """Spot data, forwards and dividends, implied vol surface and correlations of *ticker*.

    Args:
        mkt (MarketEnvironment): The market (its evaluation date must be ``tq.Settings.evaluation_date``).
        ticker (str): Equity ticker.

    Returns:
        dict: JSON-ready sections ``summary``, ``forwards``, ``forward_curve``, ``dividends``, ``vol``,
        ``correlations`` (``names``, ``labels``, ``matrix`` of the snapshot, or ``None``).

    Raises:
        KeyError: If *ticker* is not an equity of the market.
    """
    if ticker not in equity_tickers(mkt):
        raise KeyError(ticker)
    with mkt.get_equity(ticker).discount_curve.pinned():       # thousands of discount factors, one market
        view = _equity_view(mkt, ticker)
    eq = mkt.get_equity(ticker)
    view["smile_quotes"] = None if folder is None else smile_quotes(folder, ticker, float(eq.spot), eq.vol_surface)
    return view


def _equity_view(mkt: tq.MarketEnvironment, ticker: str) -> dict:
    eq = mkt.get_equity(ticker)
    s0 = float(eq.spot)
    curve = eq.discount_curve
    summary = {"ticker": ticker, "currency": eq.currency.name, "snapshot": mkt.evaluation_date.isoformat(),
               "spot": s0, "repo": float(eq.repo_rate(1.0)), "dividend_model": eq.dividend_model,
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

    # the whole correlation matrix of the snapshot (the page shows the sub-matrix of the factors picked)
    correlations = None
    corr = mkt.get_correlation_matrix()
    if corr is not None:
        correlations = {"names": list(corr.names), "labels": [n.split(":")[-1] for n in corr.names],
                        "matrix": corr.matrix.numpy().tolist()}

    # repo term structure (implied from the options): pillars and zero repo rate on a grid to 6 years
    repo = None
    if isinstance(eq.repo, tq.RateCurve):
        pillars = [float(t) for t in eq.repo.pillars]
        repo_grid = np.linspace(1 / 365, max(6.0, max(pillars)), N_GRID)
        repo = {"pillars": [{"date": d.isoformat(), "T": t, "rate": float(r)}
                            for d, t, r in zip(eq.repo.dates, pillars, np.asarray(eq.repo.rates, float))],
                "grid": {"T": repo_grid.tolist(), "rate": [float(eq.repo_rate(t)) for t in repo_grid]}}

    return {"summary": summary, "forwards": forwards, "forward_curve": forward_curve, "dividends": dividends,
            "repo": repo, "vol": vol, "correlations": correlations}
