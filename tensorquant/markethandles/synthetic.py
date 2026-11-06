"""Synthetic market data: curves, equities, FX and correlations in the ``marketdata/`` table layout.

``generate_market_data`` returns the tables of :meth:`MarketEnvironment.from_frames` (write them
with ``MarketEnvironment.to_folder`` or ``DataFrame.to_csv``); ``generate_market`` builds the
:class:`MarketEnvironment` directly. Fixed seed, independent random stream per block (curves,
equities, FX, correlations), all dates relative to the evaluation date.

- Curves: Nelson-Siegel zero rates per currency (smooth forwards); EUR 6M Euribor = ESTR + a Nelson-Siegel spread.
- Rate vols: EUR 6M caplet normal (Bachelier) vols from a normal SABR (beta = 0, Hagan) smile per expiry;
  EUR 6M swaption normal vol cube (expiry x tenor x strike offset), normal SABR per (expiry, tenor).
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
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd

# One OIS curve per currency: name, Nelson-Siegel long level beta0 and short level beta0 + beta1
CURVES = {"EUR": ("ESTR", 0.028, 0.020), "GBP": ("SONIA", 0.042, 0.040), "USD": ("SOFR", 0.039, 0.042)}
CURVE_TENORS_M = [1, 3, 6, 12, 24, 36, 60, 84, 120, 180, 240, 360]
FX_SPOTS = [("EUR", "USD", 1.0850), ("GBP", "USD", 1.2700)]   # 1 base = spot quote
EUR6M_SPREAD = ((0.0020, 0.0030), (0.0030, 0.0045))          # short, long spread ranges over ESTR
CAPLET_EXPIRIES_M = list(range(6, 121, 6))                       # every 6M: on the fixings of the caplets
CAPLET_STRIKES = [round(k, 4) for k in np.arange(-0.005, 0.0601, 0.005)]
SWAPTION_EXPIRIES_M = [6, 12, 24, 36, 60, 84, 120]
SWAPTION_TENORS = ["1Y", "2Y", "5Y", "10Y", "20Y"]
SWAPTION_OFFSETS = [-0.02, -0.01, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.01, 0.02]
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
    names = ("curves", "equities", "fx", "correlations", "rate_vols", "swaption_vols")   # child i: count-independent
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


def _rate_model_parameters(rng) -> dict:
    """G2++ parameters behind the synthetic EUR 6M caplet vols and swaption cube (first draws of the stream)."""
    return dict(a=rng.uniform(0.02, 0.06), b=rng.uniform(0.40, 0.90), sigma=rng.uniform(0.008, 0.011),
                eta=rng.uniform(0.007, 0.010), rho=rng.uniform(-0.85, -0.60))


def _rate_model(evaluation_date: date, curves: pd.DataFrame, params: dict):
    """The G2++ on the ESTR curve and the discount functions (date -> float) of ESTR and EUR 6M."""
    from .ircurve import RateCurve
    from ..models.gaussianshortrate import GaussianShortRateModel
    from ..timehandles.daycounter import DayCounterConvention

    def curve(name):
        grp = curves[(curves.ccy == "EUR") & (curves.curve == name)]
        return RateCurve(evaluation_date, list(grp["pillar"]), list(grp["value"]), "LINEAR", DayCounterConvention.Actual365)

    disc, fwd = curve("ESTR"), curve("6M")
    # discount factors as numbers, cached per date (the node schedules share most of their dates)
    p_disc = lru_cache(maxsize=None)(lambda d: float(disc.discount(d)))
    p_fwd = lru_cache(maxsize=None)(lambda d: float(fwd.discount(d)))
    return GaussianShortRateModel.g2pp(disc, **params), p_disc, p_fwd


def caplet_vol_table(evaluation_date: date, curves: pd.DataFrame, params: dict, rng) -> pd.DataFrame:
    """EUR 6M caplet normal vols on (fixing expiry x absolute strike).

    ATM vols from the G2++ *params* on the ESTR curve (closed-form caplets on ``[expiry, expiry + 6M]`` with the 6M basis,
    the same model as :func:`swaption_vol_table`), smile from a normal SABR around the 6M forward:
    ``vol(K) = atm * sabr(F, K) / sabr(F, F)``.
    """
    from ..timehandles.daycounter import DayCounter, DayCounterConvention

    model, disc, fwd = _rate_model(evaluation_date, curves, params)
    a360, a365 = DayCounter(DayCounterConvention.Actual360), DayCounter(DayCounterConvention.Actual365)
    rho, nu = rng.uniform(-0.20, 0.20), rng.uniform(0.25, 0.45)
    nodes = []
    for m in CAPLET_EXPIRIES_M:
        expiry, end = add_months(evaluation_date, m), add_months(evaluation_date, m + 6)
        tau = a360.year_fraction(expiry, end)
        F = (fwd(expiry) / fwd(end) - 1.0) / tau
        basis = (fwd(expiry) / fwd(end)) / (disc(expiry) / disc(end))
        nodes.append((expiry, a365.year_fraction(evaluation_date, expiry), a365.year_fraction(evaluation_date, end),
                      tau, F, basis * disc(expiry), (1.0 + tau * F) * disc(end), disc(end)))
    # all the ATM caplets at once: E[D (basis P(t, t) - (1 + tau F) P(t, end))^+]
    t = np.array([n[1] for n in nodes])
    prices = model.exchange_option_values(t, t, [n[2] for n in nodes], [n[5] for n in nodes], [n[6] for n in nodes]).numpy()
    rows = []
    for (expiry, t, _, tau, F, _, _, p_end), price in zip(nodes, prices):
        atm = price / (tau * p_end) * np.sqrt(2.0 * np.pi / t)                          # ATM Bachelier inverted
        smile = sabr_normal_vol(F, CAPLET_STRIKES, t, atm, rho, nu) / sabr_normal_vol(F, [F], t, atm, rho, nu)[0]
        rows += [{"ccy": "EUR", "curve": "6M", "expiry": expiry, "strike": k, "vol": round(float(v), 8),
                  "vol_type": "normal"} for k, v in zip(CAPLET_STRIKES, atm * smile)]
    return pd.DataFrame(rows)


def swaption_vol_table(evaluation_date: date, curves: pd.DataFrame, params: dict, rng) -> pd.DataFrame:
    """EUR 6M swaption normal vols on (expiry x tenor x strike offset).

    ATM vols from the G2++ *params* (the model of :func:`caplet_vol_table`), exact formula
    (Brigo-Mercurio) on swaps starting at expiry (fixed annual Act/360, floating 6M Act/360 with the 6M basis);
    smile from a normal SABR in ``K - F`` (beta = 0: depends on ``K - F`` only).
    """
    from ..timehandles.daycounter import DayCounter, DayCounterConvention

    model, disc, fwd = _rate_model(evaluation_date, curves, params)
    a360, a365 = DayCounter(DayCounterConvention.Actual360), DayCounter(DayCounterConvention.Actual365)
    yf = lambda d: a365.year_fraction(evaluation_date, d)
    rho, nu = rng.uniform(-0.20, 0.20), rng.uniform(0.25, 0.40)
    p0 = disc
    nodes, arrays = [], {k: [] for k in ("te", "p0_te", "pay_t", "pay_w", "p0_pay", "st_t", "st_w", "p0_st",
                                         "en_t", "en_w", "p0_en")}
    for m in SWAPTION_EXPIRIES_M:
        expiry = add_months(evaluation_date, m)
        for tenor in SWAPTION_TENORS:
            years = int(tenor[:-1])
            pay = [add_months(expiry, 12 * k) for k in range(1, years + 1)]
            starts = [add_months(expiry, 6 * k) for k in range(2 * years)]
            ends = [add_months(expiry, 6 * k + 6) for k in range(2 * years)]
            basis = [(fwd(st) / fwd(en)) / (p0(st) / p0(en)) for st, en in zip(starts, ends)]
            row = dict(te=yf(expiry), p0_te=p0(expiry), pay_t=[yf(d) for d in pay],
                       pay_w=[a360.year_fraction(add_months(d, -12), d) for d in pay], p0_pay=[p0(d) for d in pay],
                       st_t=[yf(d) for d in starts], st_w=basis, p0_st=[p0(d) for d in starts],
                       en_t=[yf(d) for d in ends], en_w=[1.0] * len(ends), p0_en=[p0(d) for d in ends])
            for k, v in row.items():
                arrays[k].append(v)
            annuity = float(np.dot(row["pay_w"], row["p0_pay"]))
            nodes.append((expiry, tenor, years, row["te"], annuity,
                          (float(np.dot(row["st_w"], row["p0_st"])) - float(np.dot(row["en_w"], row["p0_en"]))) / annuity))
    # all the ATM swaptions in one batched exact call (flows padded with zero weights, last time repeated)
    width = max(len(v) for v in arrays["pay_t"] + arrays["st_t"])
    pad = lambda rows, times: np.array([r + [r[-1] if times else 0.0] * (width - len(r)) for r in rows])
    batch = {k: (np.array(v) if k in ("te", "p0_te") else pad(v, k.endswith("_t") or k.startswith("p0_")))
             for k, v in arrays.items()}
    prices = model.swaption_exact_values(strike=np.array([n[5] for n in nodes]), **batch).numpy()
    rows = []
    for (expiry, tenor, years, t, annuity, _), price in zip(nodes, prices):
        atm = price / annuity * np.sqrt(2.0 * np.pi / t)                               # ATM Bachelier inverted
        nu_t = nu / np.sqrt(1.0 + years / 10.0)
        smile = sabr_normal_vol(0.0, np.asarray(SWAPTION_OFFSETS), t, atm, rho, nu_t) /             sabr_normal_vol(0.0, [0.0], t, atm, rho, nu_t)[0]
        rows += [{"ccy": "EUR", "curve": "6M", "expiry": expiry, "tenor": tenor, "strike_offset": k,
                  "vol": round(float(v), 8), "vol_type": "normal"} for k, v in zip(SWAPTION_OFFSETS, atm * smile)]
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
        dict: ``ir_curves.csv``, ``ir_vol.csv``, ``ir_swaption_vol.csv``, ``fx_spot.csv``, ``fx_vol.csv``, ``eq_spot.csv``, ``eq_vol.csv``,
            ``eq_dividends.csv``, ``correlations.csv``.
    """
    from .marketenvironment import MarketEnvironment
    from ..timehandles.utils import Settings

    rng = _streams(seed)
    currencies = [ccy for ccy, n in tickers_per_currency for _ in range(n)]
    tickers = [f"TICKER{i}" for i in range(1, len(currencies) + 1)]
    curves = curve_table(evaluation_date, rng["curves"])
    rate_model = _rate_model_parameters(rng["rate_vols"])          # one G2++ behind caplets and swaptions
    ir_vol = caplet_vol_table(evaluation_date, curves, rate_model, rng["rate_vols"])
    sw_vol = swaption_vol_table(evaluation_date, curves, rate_model, rng["swaption_vols"])
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
        "ir_curves.csv": curves, "ir_vol.csv": ir_vol, "ir_swaption_vol.csv": sw_vol, "fx_spot.csv": fx_spot, "fx_vol.csv": pd.DataFrame(fx_rows),
        "eq_spot.csv": eq_spot, "eq_vol.csv": pd.DataFrame(vol_rows), "eq_dividends.csv": eq_div,
        "correlations.csv": pd.DataFrame(corr_rows),
    }
    assert tables["eq_vol.csv"]["vol"].between(0.03, 1.5).all() and tables["fx_vol.csv"]["vol"].between(0.01, 1.0).all()
    return tables


def generate_market(evaluation_date: date, seed: int = 42, **kwargs):
    """Synthetic :class:`MarketEnvironment` (``generate_market_data`` + ``from_frames``)."""
    from .marketenvironment import MarketEnvironment

    return MarketEnvironment.from_frames(generate_market_data(evaluation_date, seed, **kwargs), evaluation_date)
