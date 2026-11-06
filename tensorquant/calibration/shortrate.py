"""Calibration of the Gaussian short rate models (Hull-White / G2++) to the market caplet vols."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from ..instruments.helpers import CapFloorGenerator
from ..markethandles.marketenvironment import MarketEnvironment
from ..markethandles.utils import Currency
from ..models.gaussianshortrate import GaussianShortRateModel
from ..numericalhandles.dtypes import DTYPE
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..timehandles.utils import Settings

DEFAULT_START = {
    1: {"a": 0.05, "sigma": 0.008},
    2: {"a": 0.05, "b": 0.5, "sigma": 0.007, "eta": 0.006, "rho": -0.5},
}


def _bachelier(F, K, vol, T, phi=1.0):
    std = vol * np.sqrt(T)
    d = (F - K) / std
    return phi * (F - K) * norm.cdf(phi * d) + std * norm.pdf(d)


def _instruments(market_env: MarketEnvironment, generator: CapFloorGenerator, term: str,
                 strikes: Optional[Sequence[float]]):
    """Caplets of the *term* cap: ATM (strike = forward) or every caplet at every strike of *strikes*."""
    eval_date = Settings.evaluation_date
    ccy_str, tenor = generator.index.name.split(":")
    disc = market_env.get_ir_curve(Currency[ccy_str])
    fwd = market_env.get_ir_curve(Currency[ccy_str], ticker=tenor)
    surface = market_env.get_ir_vol_surface(Currency[ccy_str], tenor)
    if surface.vol_type != "normal":
        raise ValueError("calibration needs normal caplet vols")
    a365 = DayCounter(DayCounterConvention.Actual365)
    rows = []
    for c in generator.build(eval_date, 0.0, term).caplets:
        s, e = c.ref_period_start, c.ref_period_end
        tau = c.index.daycounter.year_fraction(s, e)
        F = (float(fwd.discount(s)) / float(fwd.discount(e)) - 1.0) / tau
        basis = (float(fwd.discount(s)) / float(fwd.discount(e))) / (float(disc.discount(s)) / float(disc.discount(e)))
        t = a365.year_fraction(eval_date, c.fixing_date)
        df = float(disc.discount(e))
        for K in ([F] if strikes is None else strikes):
            vol = float(surface.volatility(float(K), t))
            rows.append(dict(fixing=c.fixing_date, start=s, end=e, tau=tau, basis=basis, t=t, F=F, K=float(K),
                             df=df, market_vol=vol, market_price=tau * df * _bachelier(F, K, vol, t),
                             vega=tau * df * np.sqrt(t) * norm.pdf((F - K) / (vol * np.sqrt(t)))))
    return rows


NAMES = {1: ("a", "sigma"), 2: ("a", "b", "sigma", "eta", "rho")}
# bounds on the natural parameters: speeds >= 1e-4 (no runaway to 0), vols > 0, |rho| <= 0.99
BOUNDS = {"a": (1e-4, 3.0), "b": (1e-4, 3.0), "sigma": (1e-5, 0.1), "eta": (1e-5, 0.1), "rho": (-0.99, 0.99)}


def _model(curve, factors: int, x: tf.Tensor) -> GaussianShortRateModel:
    """Model from the parameter vector (``NAMES[factors]`` order)."""
    if factors == 1:
        return GaussianShortRateModel.hull_white(curve, x[0], x[1])
    return GaussianShortRateModel.g2pp(curve, x[0], x[1], x[2], x[3], x[4])


def calibrate_gaussian_short_rate(
    market_env: MarketEnvironment,
    generator: CapFloorGenerator,
    factors: int = 1,
    term: str = "10Y",
    strikes: Optional[Sequence[float]] = None,
    initial: Optional[dict] = None,
):
    """Calibrate Hull-White (``factors=1``) or G2++ (``factors=2``) to the index caplet vols of the market.

    Calibration instruments: the caplets of the *term* cap (``generator`` conventions), ATM (strike =
    caplet forward) by default or at every strike of *strikes*. Market prices are Bachelier on the
    market caplet vols; model prices are the closed-form :meth:`GaussianShortRateModel.caplet`
    (the model lives on the discount curve, the index curve through a deterministic basis).
    Least squares on vega-scaled price residuals (≈ normal vol errors), Jacobian by AAD, bounded natural
    parameters (``BOUNDS``; ``params["at_bound"]`` lists those that end on a bound).

    Args:
        market_env: Market with the discount / index curves and the index caplet vols (normal).
        generator: Cap conventions (index, period, day count, calendar).
        factors: 1 (Hull-White: a, sigma) or 2 (G2++: a, b, sigma, eta, rho).
        term: Longest cap whose caplets are used.
        strikes: ``None`` = ATM caplets; otherwise these absolute strikes.
        initial: Starting parameters (default ``DEFAULT_START[factors]``).

    Returns:
        tuple: ``(model, params, report)``: the calibrated :class:`GaussianShortRateModel` on the
            market discount curve, the parameters (dict) and a DataFrame per instrument (fixing, strike,
            market and model normal vols, error in bp).

    Raises:
        ValueError: If *factors* is not 1 or 2, or the caplet vols are not normal.
    """
    if factors not in (1, 2):
        raise ValueError("factors must be 1 (Hull-White) or 2 (G2++)")
    ccy_str, _ = generator.index.name.split(":")
    curve = market_env.get_ir_curve(Currency[ccy_str])
    inst = _instruments(market_env, generator, term, strikes)
    market = np.array([r["market_price"] for r in inst])
    vega = np.array([r["vega"] for r in inst])

    # the curve is fixed during the calibration: legs at time 0 and times precomputed, model vectorized
    a365 = DayCounter(DayCounterConvention.Actual365)
    eval_date = Settings.evaluation_date
    t_fix = [r["t"] for r in inst]
    t_s = [curve.daycounter.year_fraction(eval_date, r["start"]) for r in inst]
    t_e = [curve.daycounter.year_fraction(eval_date, r["end"]) for r in inst]
    leg_a = np.array([r["basis"] * float(curve.discount(r["start"])) for r in inst])
    leg_b = np.array([(1.0 + r["tau"] * r["K"]) * float(curve.discount(r["end"])) for r in inst])

    def model_prices(x: tf.Tensor) -> tf.Tensor:
        return _model(curve, factors, x).exchange_option_values(t_fix, t_s, t_e, leg_a, leg_b)

    @tf.function
    def prices_and_jacobian(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            p = model_prices(x)
        return p, tape.jacobian(p, x)

    cache = {}

    def evaluate(x):
        key = tuple(np.round(x, 15))
        if key not in cache:
            cache.clear()
            p, j = prices_and_jacobian(tf.constant(x, DTYPE))
            cache[key] = ((p.numpy() - market) / vega, j.numpy() / vega[:, None])
        return cache[key]

    def residuals(x):
        return evaluate(x)[0]

    def jacobian(x):
        return evaluate(x)[1]

    names = NAMES[factors]
    start = initial or DEFAULT_START[factors]
    x0 = np.array([start[n] for n in names], dtype=float)
    lower, upper = (np.array([BOUNDS[n][i] for n in names]) for i in (0, 1))
    fit = least_squares(residuals, np.clip(x0, lower, upper), jac=jacobian, bounds=(lower, upper), method="trf",
                        x_scale="jac", xtol=1e-10, ftol=1e-10, gtol=1e-10)
    params = {n: float(v) for n, v in zip(names, fit.x)}
    params["at_bound"] = [n for n, v in zip(names, fit.x) if np.isclose(v, BOUNDS[n][0]) or np.isclose(v, BOUNDS[n][1])]
    params["evaluations"] = int(fit.nfev)
    model = _model(curve, factors, tf.constant(fit.x, DTYPE))

    prices = model_prices(tf.constant(fit.x, DTYPE)).numpy()
    report = []
    for r, p in zip(inst, prices):
        target = p / (r["tau"] * r["df"])
        try:
            vol = brentq(lambda v: _bachelier(r["F"], r["K"], v, r["t"]) - target, 1e-8, 0.2, xtol=1e-14)
        except ValueError:
            vol = float("nan")
        report.append({"fixing": r["fixing"], "strike": r["K"], "market_vol": r["market_vol"], "model_vol": vol,
                       "error_bp": (vol - r["market_vol"]) * 1e4})
    return model, params, pd.DataFrame(report)
