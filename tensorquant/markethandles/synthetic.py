"""Synthetic market data: curves, equities, FX and correlations in the ``marketdata/`` table layout.

``generate_market_data`` returns the tables of :meth:`MarketEnvironment.from_frames` (write them
with ``MarketEnvironment.to_folder`` or ``DataFrame.to_csv``); ``generate_market`` builds the
:class:`MarketEnvironment` directly. Fixed seed, independent random stream per block (curves,
equities, FX, correlations), all dates relative to the evaluation date.

- Curves: Nelson-Siegel zero rates per currency (smooth forwards); EUR 6M Euribor = ESTR + a Nelson-Siegel spread.
- Rate vols: EUR 6M caplet normal (Bachelier) vols from a normal SABR (beta = 0, Hagan) smile per expiry.
- Equities: spot, repo, discrete dividends, SSVI vol surface in forward moneyness
  (``Equity.forward``, dividends included).
- FX: spots and SSVI vol surfaces on the FX forward (covered interest parity).
- Correlations: one-factor model (every pair in a range, PSD by construction).

SSVI (Gatheral-Jacquier): ``w(k, T) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))``,
``theta(T) = atm(T)^2 T`` increasing, ``phi(theta) = eta / sqrt(theta (1 + theta))`` with
``eta (1 + |rho|) <= 2``: no static arbitrage by construction.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

import numpy as np
import pandas as pd

# One OIS curve per currency: name, Nelson-Siegel long level beta0 and short level beta0 + beta1
CURVES = {"EUR": ("ESTR", 0.028, 0.020), "GBP": ("SONIA", 0.042, 0.040), "USD": ("SOFR", 0.039, 0.042)}
CURVE_TENORS_M = [1, 3, 6, 12, 24, 36, 60, 84, 120, 180, 240, 360]
FX_SPOTS = [("EUR", "USD", 1.0850), ("GBP", "USD", 1.2700)]   # 1 base = spot quote
EUR6M_SPREAD = ((0.0020, 0.0030), (0.0030, 0.0045))          # short, long spread ranges over ESTR
CAPLET_EXPIRIES_M = [6, 12, 18, 24, 36, 48, 60, 84, 120]
CAPLET_STRIKES = [round(k, 4) for k in np.arange(-0.005, 0.0601, 0.005)]
VOL_TENORS_M = [1, 3, 6, 12, 18, 24, 36, 60]
EQ_MONEYNESS = np.round(np.arange(0.6, 1.401, 0.1), 2)
FX_MONEYNESS = np.round(np.arange(0.8, 1.201, 0.05), 2)
DIV_FREQUENCY = {"EUR": 1, "GBP": 2, "USD": 4}               # payments per year
DIV_HORIZON_Y = 5
CORR_RANGE = (0.30, 0.80)


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    return date(d.year + y, m + 1, min(d.day, 28))


def _streams(seed: int) -> dict:
    names = ("curves", "equities", "fx", "correlations", "rate_vols")   # child i does not depend on the count
    return dict(zip(names, (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(names)))))


def _ssvi_vol(t: float, k: np.ndarray, atm: float, ts: float, rho: float, eta: float) -> np.ndarray:
    """SSVI implied vol at maturity t and forward log-moneyness k (ATM vol term structure atm + ts (sqrt t - 1))."""
    theta = (atm + ts * (np.sqrt(t) - 1.0)) ** 2 * t
    phi = eta / np.sqrt(theta * (1.0 + theta))
    w = 0.5 * theta * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2))
    return np.sqrt(w / t)


def _ssvi_params(rng, atm_range, ts_range, rho_range) -> dict:
    atm, ts = rng.uniform(*atm_range), rng.uniform(*ts_range)
    rho, u = rng.uniform(*rho_range), rng.uniform(0.50, 0.95)
    return dict(atm=atm, ts=ts, rho=rho, eta=u * 2.0 / (1.0 + abs(rho)))


def curve_table(evaluation_date: date, rng) -> pd.DataFrame:
    """Nelson-Siegel zero rates ``b0 + b1 L1(T) + b2 L2(T)``, ``L1 = (1 - e^{-T/tau}) / (T/tau)``,
    ``L2 = L1 - e^{-T/tau}``; hump ``b2`` and decay ``tau`` random per currency."""
    rows = []
    for ccy, (name, long, short) in CURVES.items():
        b2, tau = rng.uniform(-0.010, 0.010), rng.uniform(1.0, 3.0)
        for m in CURVE_TENORS_M:
            t = m / 12.0
            l1 = (1.0 - np.exp(-t / tau)) / (t / tau)
            rate = long + (short - long) * l1 + b2 * (l1 - np.exp(-t / tau))
            rows.append({"ccy": ccy, "curve": name, "pillar": add_months(evaluation_date, m),
                         "value": round(float(rate), 8), "quote": "zero", "daycount": "Actual365",
                         "interp": "LINEAR"})
    # EUR 6M Euribor: ESTR + spread(T), spread Nelson-Siegel from a short to a long level (drawn after the OIS curves)
    s_short, s_long = rng.uniform(*EUR6M_SPREAD[0]), rng.uniform(*EUR6M_SPREAD[1])
    estr = {r["pillar"]: r["value"] for r in rows if r["curve"] == "ESTR"}
    for m in CURVE_TENORS_M:
        t, d = m / 12.0, add_months(evaluation_date, m)
        l1 = (1.0 - np.exp(-t / 2.0)) / (t / 2.0)
        rows.append({"ccy": "EUR", "curve": "6M", "pillar": d, "value": round(estr[d] + s_long + (s_short - s_long) * l1, 8),
                     "quote": "zero", "daycount": "Actual365", "interp": "LINEAR"})
    return pd.DataFrame(rows)


def sabr_normal_vol(F: float, K: np.ndarray, T: float, alpha: float, rho: float, nu: float) -> np.ndarray:
    """Normal (Bachelier) implied vol of the normal SABR model (beta = 0), Hagan et al. (2002)."""
    K = np.asarray(K, dtype=float)
    zeta = nu / alpha * (F - K)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log((np.sqrt(1.0 - 2.0 * rho * zeta + zeta ** 2) + zeta - rho) / (1.0 - rho))
        ratio = np.where(np.abs(zeta) < 1e-10, 1.0, zeta / x)
    return alpha * ratio * (1.0 + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2 * T)


def caplet_vol_table(evaluation_date: date, curves: pd.DataFrame, rng) -> pd.DataFrame:
    """EUR 6M caplet normal vols on (fixing expiry x absolute strike), normal SABR around the 6M forward."""
    from .ircurve import RateCurve
    from ..timehandles.daycounter import DayCounter, DayCounterConvention

    grp = curves[(curves.ccy == "EUR") & (curves.curve == "6M")]
    curve = RateCurve(evaluation_date, list(grp["pillar"]), list(grp["value"]), "LINEAR", DayCounterConvention.Actual365)
    a360 = DayCounter(DayCounterConvention.Actual360)
    a_short, a_long = rng.uniform(0.0070, 0.0095), rng.uniform(0.0055, 0.0075)   # ATM normal vol, short / long
    rho, nu = rng.uniform(-0.20, 0.20), rng.uniform(0.25, 0.45)
    rows = []
    for m in CAPLET_EXPIRIES_M:
        expiry, end = add_months(evaluation_date, m), add_months(evaluation_date, m + 6)
        t = (expiry - evaluation_date).days / 365.0
        F = (float(curve.discount(expiry)) / float(curve.discount(end)) - 1.0) / a360.year_fraction(expiry, end)
        alpha = a_long + (a_short - a_long) * np.exp(-t / 3.0)
        vols = sabr_normal_vol(F, CAPLET_STRIKES, t, alpha, rho, nu)
        rows += [{"ccy": "EUR", "curve": "6M", "expiry": expiry, "strike": k, "vol": round(float(v), 8),
                  "vol_type": "normal"} for k, v in zip(CAPLET_STRIKES, vols)]
    return pd.DataFrame(rows)


def generate_market_data(
    evaluation_date: date,
    seed: int = 42,
    tickers_per_currency: Sequence[tuple] = (("EUR", 30), ("GBP", 10), ("USD", 10)),
) -> dict:
    """Synthetic market tables (``{file name: DataFrame}``, layout of ``MarketEnvironment.from_frames``).

    Args:
        evaluation_date (date): Snapshot date (all dates relative to it).
        seed (int): Seed (one independent stream per block).
        tickers_per_currency: ``(currency, count)``; tickers are numbered ``TICKER1, TICKER2, ...`` in order.

    Returns:
        dict: ``ir_curves.csv``, ``ir_vol.csv``, ``fx_spot.csv``, ``fx_vol.csv``, ``eq_spot.csv``, ``eq_vol.csv``,
            ``eq_dividends.csv``, ``correlations.csv``.
    """
    from .marketenvironment import MarketEnvironment
    from ..timehandles.utils import Settings

    rng = _streams(seed)
    currencies = [ccy for ccy, n in tickers_per_currency for _ in range(n)]
    tickers = [f"TICKER{i}" for i in range(1, len(currencies) + 1)]
    curves = curve_table(evaluation_date, rng["curves"])
    ir_vol = caplet_vol_table(evaluation_date, curves, rng["rate_vols"])
    fx_spot = pd.DataFrame(FX_SPOTS, columns=["base", "quote", "spot"])

    # equities: spots, repos, dividends, SSVI parameters
    spot_rows, div_rows, ssvi = [], [], {}
    r = rng["equities"]
    for ticker, ccy in zip(tickers, currencies):
        spot = round(float(r.uniform(10, 500)), 2)
        spot_rows.append({"ticker": ticker, "ccy": ccy, "spot": spot,
                          "repo": round(float(r.uniform(0.0, 0.01)), 5), "div_yield": None})
        ssvi[ticker] = _ssvi_params(r, (0.15, 0.40), (-0.04, 0.02), (-0.80, -0.30))
        freq = DIV_FREQUENCY[ccy]
        amount = spot * r.uniform(0.0, 0.06) / freq
        ex_date = evaluation_date + timedelta(days=int(r.integers(5, 365 // freq)))
        while ex_date <= add_months(evaluation_date, 12 * DIV_HORIZON_Y):
            div_rows.append({"ticker": ticker, "ccy": ccy, "ex_date": ex_date,
                             "pay_date": ex_date + timedelta(days=14), "amount": round(amount, 4)})
            ex_date = add_months(ex_date, 12 // freq)
            amount *= 1.03 ** (1 / freq)
    eq_spot, eq_div = pd.DataFrame(spot_rows), pd.DataFrame(div_rows)

    # forwards from the library (dividends, repo, curves) for the SSVI moneyness
    previous = Settings.evaluation_date
    Settings.evaluation_date = evaluation_date
    try:
        base = MarketEnvironment.from_frames({"ir_curves.csv": curves, "fx_spot.csv": fx_spot,
                                              "eq_spot.csv": eq_spot, "eq_dividends.csv": eq_div}, evaluation_date)
        expiries = [add_months(evaluation_date, m) for m in VOL_TENORS_M]
        times = [(d - evaluation_date).days / 365.0 for d in expiries]
        vol_rows = []
        for row in spot_rows:
            eq = base.get_equity(row["ticker"])
            strikes = np.round(row["spot"] * EQ_MONEYNESS, 2)
            for d, t in zip(expiries, times):
                vols = _ssvi_vol(t, np.log(strikes / float(eq.forward(d))), **ssvi[row["ticker"]])
                vol_rows += [{"ticker": row["ticker"], "ccy": row["ccy"], "expiry": d, "strike": float(k),
                              "vol": round(float(v), 6)} for k, v in zip(strikes, vols)]
        fx_rows = []
        for b, q, s in FX_SPOTS:
            params = _ssvi_params(rng["fx"], (0.06, 0.10), (-0.01, 0.01), (-0.30, 0.30))
            strikes = np.round(s * FX_MONEYNESS, 4)
            for d, t in zip(expiries, times):
                vols = _ssvi_vol(t, np.log(strikes / float(base.get_fx_forward(b, q, d))), **params)
                fx_rows += [{"base": b, "quote": q, "expiry": d, "strike": float(k), "vol": round(float(v), 6)}
                            for k, v in zip(strikes, vols)]
    finally:
        Settings.evaluation_date = previous

    # correlations: one-factor rho_ij = b_i b_j (every pair in CORR_RANGE, PSD)
    lo, hi = CORR_RANGE
    b = rng["correlations"].uniform(np.sqrt(lo), np.sqrt(hi), len(tickers))
    corr = np.round(np.outer(b, b), 4)
    np.fill_diagonal(corr, 1.0)
    names = [f"EQ:{ccy}:{t}" for t, ccy in zip(tickers, currencies)]
    corr_rows = [{"factor_1": names[i], "factor_2": names[j], "corr": corr[i, j]}
                 for i in range(len(names)) for j in range(i + 1, len(names))]

    tables = {
        "ir_curves.csv": curves, "ir_vol.csv": ir_vol, "fx_spot.csv": fx_spot, "fx_vol.csv": pd.DataFrame(fx_rows),
        "eq_spot.csv": eq_spot, "eq_vol.csv": pd.DataFrame(vol_rows), "eq_dividends.csv": eq_div,
        "correlations.csv": pd.DataFrame(corr_rows),
    }
    assert tables["eq_vol.csv"]["vol"].between(0.03, 1.5).all() and tables["fx_vol.csv"]["vol"].between(0.01, 1.0).all()
    return tables


def generate_market(evaluation_date: date, seed: int = 42, **kwargs):
    """Synthetic :class:`MarketEnvironment` (``generate_market_data`` + ``from_frames``)."""
    from .marketenvironment import MarketEnvironment

    return MarketEnvironment.from_frames(generate_market_data(evaluation_date, seed, **kwargs), evaluation_date)
