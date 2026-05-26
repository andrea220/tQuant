"""
market_data.py -- Dati di mercato sintetici e loader per il MarketEnvironment.

Importa questo modulo da qualsiasi notebook o script:

    from examples.market_data import EVALUATION_DATE, build_market_environment
    market_env = build_market_environment()

oppure, se il path non è nel sys.path:

    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from market_data import EVALUATION_DATE, build_market_environment
"""

from __future__ import annotations

import numpy as np
from datetime import date
from pathlib import Path
import sys

# Assicura che la root del progetto sia nel path anche se il modulo viene
# importato direttamente dalla cartella examples/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorquant as tq
from tensorquant.markethandles.dividendcurve import DividendCurve
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

# ══════════════════════════════════════════════════════════════════════════════
# Evaluation date
# ══════════════════════════════════════════════════════════════════════════════

EVALUATION_DATE = date(2026, 5, 15)

# ══════════════════════════════════════════════════════════════════════════════
# Rate curves
# ══════════════════════════════════════════════════════════════════════════════

_RATE_CURVES = {
    "IR:EUR:ESTR:SPOT": {
        "pillars": [0.25, 0.50, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "rates":   [0.026, 0.027, 0.028, 0.029, 0.030, 0.032, 0.033, 0.034],
        "interp":  "LINEAR",
    },
    "IR:EUR:6M:SPOT": {
        "pillars": [0.50, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "rates":   [0.031, 0.032, 0.033, 0.034, 0.036, 0.037, 0.038],
        "interp":  "LINEAR",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Equity data
# ══════════════════════════════════════════════════════════════════════════════

_S0_TICKER1 = 100.0
_S0_TICKER2 = 50.0

def _vol_ticker1() -> list[list[float]]:
    """IV(T,K) = 0.20 + 0.02√T − 0.10·m + 0.30·m²  (m = ln(K/S0))"""
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    mats    = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
    return [
        [float(np.clip(
            0.20 + 0.02 * np.sqrt(T)
            - 0.10 * np.log(K / _S0_TICKER1)
            + 0.30 * np.log(K / _S0_TICKER1) ** 2,
            0.05, 0.80,
        )) for K in strikes]
        for T in mats
    ]

def _vol_ticker2() -> list[list[float]]:
    """IV(T,K) = 0.25 + 0.01√T − 0.05·m + 0.20·m²  (m = ln(K/S0))"""
    strikes = [35, 40, 45, 50, 55, 60, 65]
    mats    = [0.25, 0.50, 1.00, 2.00]
    return [
        [float(np.clip(
            0.25 + 0.01 * np.sqrt(T)
            - 0.05 * np.log(K / _S0_TICKER2)
            + 0.20 * np.log(K / _S0_TICKER2) ** 2,
            0.05, 0.80,
        )) for K in strikes]
        for T in mats
    ]


_EQUITIES = {
    "TICKER1": {
        "currency":  "EUR",
        "spot":      _S0_TICKER1,
        "repo":      -0.005,
        "div_yield":  0.015,
        "vol_surface": {
            "strikes":   [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0],
            "maturities": [0.25, 0.50, 0.75, 1.00, 1.50, 2.00],
            "vol_matrix": _vol_ticker1(),
        },
        "dividends": {
            "ex_dates": [date(2026, 9, 15), date(2027, 3, 15), date(2027, 9, 15)],
            "amounts":  [1.20, 1.25, 1.30],
        },
    },
    "TICKER2": {
        "currency":  "EUR",
        "spot":      _S0_TICKER2,
        "repo":       0.002,
        "div_yield":  0.020,
        "vol_surface": {
            "strikes":   [35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
            "maturities": [0.25, 0.50, 1.00, 2.00],
            "vol_matrix": _vol_ticker2(),
        },
        "dividends": {
            "ex_dates": [date(2026, 12, 10)],
            "amounts":  [0.80],
        },
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Loader
# ══════════════════════════════════════════════════════════════════════════════

def build_market_environment(
    evaluation_date: date = EVALUATION_DATE,
) -> tq.MarketEnvironment:
    """Costruisce e restituisce un MarketEnvironment dai dati sintetici.

    Args:
        evaluation_date: data di valutazione (default: EVALUATION_DATE).

    Returns:
        MarketEnvironment popolato con rate curves, vol surfaces,
        dividend curves e dati equity.
    """
    dc_act365 = DayCounter(DayCounterConvention.Actual365)
    market: dict = {}

    # ── Rate curves ─────────────────────────────────────────────────────────
    for key, rc in _RATE_CURVES.items():
        market[key] = tq.RateCurve(
            reference_date=evaluation_date,
            pillars=rc["pillars"],
            rates=rc["rates"],
            interp=rc["interp"],
            daycounter_convention=DayCounterConvention.Actual365,
        )

    # ── Equities ────────────────────────────────────────────────────────────
    for ticker, eq in _EQUITIES.items():
        ccy = eq["currency"]
        market[f"EQ:{ccy}:{ticker}:SPOT"]     = float(eq["spot"])
        market[f"EQ:{ccy}:{ticker}:REPO"]     = float(eq["repo"])
        market[f"EQ:{ccy}:{ticker}:DIVYIELD"] = float(eq["div_yield"])

        vs = eq["vol_surface"]
        market[f"EQ:{ccy}:{ticker}:VOL"] = tq.VolatilitySurface(
            reference_date=evaluation_date,
            calendar=None,
            daycounter=dc_act365,
            strike=vs["strikes"],
            maturity=vs["maturities"],
            volatility_matrix=vs["vol_matrix"],
        )

        div = eq["dividends"]
        market[f"EQ:{ccy}:{ticker}:DIV"] = DividendCurve(
            reference_date=evaluation_date,
            ex_dates=div["ex_dates"],
            amounts=div["amounts"],
            currency=tq.Currency[ccy],
            daycounter_convention=DayCounterConvention.Actual365,
        )

    return tq.MarketEnvironment(market=market)
