"""Caplet vol stripping from cap quotes (flat cap vols by maturity x strike), and its inverse."""

from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from ..instruments.helpers import CapFloorGenerator
from ..markethandles.marketenvironment import MarketEnvironment
from ..markethandles.utils import Currency
from ..markethandles.volatilitysurface import VolatilitySurface
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..timehandles.utils import Settings, decode_term


def _bachelier_call(F, K, vol, T):
    std = vol * np.sqrt(T)
    d = (F - K) / std
    return (F - K) * norm.cdf(d) + std * norm.pdf(d)


def _caplet_data(market_env: MarketEnvironment, generator: CapFloorGenerator, strike: float, term: str):
    """Per caplet of the cap: fixing time (ACT/365), forward, weight ``N tau P(pay)`` (as ``CapFloorPricer``)."""
    eval_date = Settings.evaluation_date
    cap = generator.build(eval_date, strike, term)
    ccy_str, tenor = cap.index.name.split(":")
    disc = market_env.get_ir_curve(cap.ccy)
    fwd = market_env.get_ir_curve(Currency[ccy_str], ticker=tenor)
    a365 = DayCounter(DayCounterConvention.Actual365)
    rows = []
    for c in cap.caplets:
        if c.fixing_date <= eval_date:
            raise ValueError("cap quotes must be on caplets fixing after the evaluation date")
        tau = c.index.daycounter.year_fraction(c.ref_period_start, c.ref_period_end)
        F = (float(fwd.discount(c.ref_period_start)) / float(fwd.discount(c.ref_period_end)) - 1.0) / tau
        weight = c.nominal * c.accrual_period * float(disc.discount(c.date))
        rows.append((a365.year_fraction(eval_date, c.fixing_date), F, weight, c.fixing_date))
    return rows


def _term_years(term: str) -> float:
    n, unit = decode_term(term)
    return n / 12.0 if unit.name.lower().startswith("month") else float(n)


def cap_flat_vols(market_env: MarketEnvironment, generator: CapFloorGenerator, terms: Sequence[str],
                  strikes: Sequence[float]) -> pd.DataFrame:
    """Flat normal cap vols implied by the caplet vol surface of the market (inverse of :func:`strip_caplet_vols`).

    For each cap (spot start, ``generator`` conventions) the flat vol is the single normal vol that,
    applied to all its caplets, gives the cap price with the market caplet vols.

    Args:
        market_env: Market with the curves and the index caplet vols (normal).
        generator: Cap conventions (index, period, day count, calendar, first period excluded).
        terms: Cap maturities, e.g. ``["1Y", "2Y", "5Y"]``.
        strikes: Absolute strikes.

    Returns:
        pandas.DataFrame: ``term, strike, vol`` (normal flat vols).
    """
    ccy_str, tenor = generator.index.name.split(":")
    surface = market_env.get_ir_vol_surface(Currency[ccy_str], tenor)
    if surface.vol_type != "normal":
        raise ValueError("cap_flat_vols needs normal caplet vols")
    rows = []
    for K in strikes:
        for term in terms:
            data = _caplet_data(market_env, generator, K, term)
            price = sum(w * _bachelier_call(F, K, float(surface.volatility(K, t)), t) for t, F, w, _ in data)
            flat = brentq(lambda v: sum(w * _bachelier_call(F, K, v, t) for t, F, w, _ in data) - price, 1e-7, 0.2,
                          xtol=1e-14)
            rows.append({"term": term, "strike": float(K), "vol": flat})
    return pd.DataFrame(rows)


def strip_caplet_vols(market_env: MarketEnvironment, generator: CapFloorGenerator, quotes: pd.DataFrame,
                      vol_type: str = "normal") -> VolatilitySurface:
    """Caplet vols from cap quotes (flat vols), piecewise constant in expiry between cap maturities.

    For each strike, caps are taken by increasing maturity; the caplets that the cap adds to the
    previous one share one vol, solved so that the cap (with the vols already stripped) reprices
    its quote. Every cap of the quotes is then repriced exactly by :class:`CapFloorPricer`.

    Args:
        market_env: Market with the curves (forwarding and discounting).
        generator: Cap conventions of the quotes (spot start, first period excluded).
        quotes: ``term, strike, vol`` flat normal vols (every term at every strike).
        vol_type: Only ``"normal"`` (Bachelier) quotes are supported.

    Returns:
        VolatilitySurface: Caplet vols on (caplet fixing times of the longest cap, ACT/365) x strikes,
            ``vol_type="normal"``; set it with ``market_env.set_ir_vol_surface``.

    Raises:
        ValueError: If the quotes are not a full term x strike grid or not normal.
    """
    if vol_type != "normal":
        raise ValueError("only normal (Bachelier) cap quotes are supported")
    grid = quotes.pivot(index="term", columns="strike", values="vol")
    if grid.isna().to_numpy().any():
        raise ValueError("cap quotes must be a full term x strike grid")
    terms = sorted(grid.index, key=_term_years)
    strikes = sorted(grid.columns)
    longest = _caplet_data(market_env, generator, strikes[0], terms[-1])
    times = [t for t, _, _, _ in longest]
    matrix = np.zeros((len(times), len(strikes)))
    for j, K in enumerate(strikes):
        stripped: list[float] = []
        for term in terms:
            data = _caplet_data(market_env, generator, K, term)
            target = sum(w * _bachelier_call(F, K, grid.loc[term, K], t) for t, F, w, _ in data)
            known = sum(w * _bachelier_call(F, K, v, t) for (t, F, w, _), v in zip(data, stripped))
            new = data[len(stripped):]
            if not new:
                continue
            vol = brentq(lambda v: known + sum(w * _bachelier_call(F, K, v, t) for t, F, w, _ in new) - target,
                         1e-7, 0.2, xtol=1e-14)
            stripped += [vol] * len(new)
        matrix[:, j] = stripped
    return VolatilitySurface(Settings.evaluation_date, None, DayCounter(DayCounterConvention.Actual365),
                             [float(k) for k in strikes], times, matrix.tolist(), vol_type="normal")
