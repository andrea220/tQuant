"""Calibration of the Gaussian short rate models (Hull-White / G2++) to the market caplet and swaption vols."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from ..instruments.helpers import CapFloorGenerator, SwaptionGenerator
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


def _names(factors: int, sigma_times) -> tuple:
    """Parameter names: ``NAMES[factors]``, or ``a, sigma_1..sigma_P`` for a piecewise Hull-White vol."""
    if sigma_times is None:
        return NAMES[factors]
    return ("a",) + tuple(f"sigma_{k + 1}" for k in range(len(sigma_times) + 1))


def _bounds(name: str) -> tuple:
    return BOUNDS["sigma"] if name.startswith("sigma_") else BOUNDS[name]


def _model(curve, factors: int, x: tf.Tensor, sigma_times=None) -> GaussianShortRateModel:
    """Model from the full parameter vector (:func:`_names` order)."""
    if factors == 1:
        if sigma_times is not None:
            return GaussianShortRateModel.hull_white(curve, x[0], x[1:], sigma_times=sigma_times)
        return GaussianShortRateModel.hull_white(curve, x[0], x[1])
    return GaussianShortRateModel.g2pp(curve, x[0], x[1], x[2], x[3], x[4])


def _swaption_instruments(market_env: MarketEnvironment, generator: SwaptionGenerator, pairs, offsets):
    """Swaptions (expiry, tenor) at the forward swap rate + each offset, with the market cube vols."""
    eval_date = Settings.evaluation_date
    swap_gen = generator.swap_generator
    ccy_str, tenor_name = swap_gen.index.name.split(":")
    disc = market_env.get_ir_curve(Currency[ccy_str])
    fwd = market_env.get_ir_curve(Currency[ccy_str], ticker=tenor_name)
    cube = market_env.get_swaption_vol_cube(Currency[ccy_str], tenor_name)
    if cube.vol_type != "normal":
        raise ValueError("calibration needs normal swaption vols")
    rows = []
    for expiry, tenor in pairs:
        probe = generator.build(eval_date, expiry, tenor, 0.0)
        fixed = [(c.date, c.nominal * c.accrual_period) for c in probe.swap.fixed_leg.leg_flows]
        flt = []
        for c in probe.swap.floating_leg.leg_flows:
            st, en = c.ref_period_start, c.ref_period_end
            basis = (float(fwd.discount(st)) / float(fwd.discount(en))) / (float(disc.discount(st)) / float(disc.discount(en)))
            flt.append((st, en, c.nominal, basis))
        annuity = sum(w * float(disc.discount(pd_)) for pd_, w in fixed)
        F = sum(n * (b * float(disc.discount(st)) - float(disc.discount(en))) for st, en, n, b in flt) / annuity
        t = cube.daycounter.year_fraction(eval_date, probe.expiry)
        for off in offsets:
            vol = float(cube.volatility(t, probe.tenor_years, off))
            K = F + off
            arrays = dict(
                te=disc.daycounter.year_fraction(eval_date, probe.expiry),
                pay_t=[disc.daycounter.year_fraction(eval_date, pd_) for pd_, _ in fixed], pay_w=[w for _, w in fixed],
                p0_pay=[float(disc.discount(pd_)) for pd_, _ in fixed],
                st_t=[disc.daycounter.year_fraction(eval_date, st) for st, _, _, _ in flt],
                st_w=[n * b for _, _, n, b in flt], p0_st=[float(disc.discount(st)) for st, _, _, _ in flt],
                en_t=[disc.daycounter.year_fraction(eval_date, en) for _, en, _, _ in flt],
                en_w=[n for _, _, n, _ in flt], p0_en=[float(disc.discount(en)) for _, en, _, _ in flt],
                p0_te=float(disc.discount(probe.expiry)))
            rows.append(dict(expiry=probe.expiry, tenor=tenor, K=K, F=F, t=t, annuity=annuity, arrays=arrays,
                             market_vol=vol, market_price=annuity * _bachelier(F, K, vol, t),
                             vega=annuity * np.sqrt(t) * norm.pdf(-off / (vol * np.sqrt(t)))))
    return rows


def calibrate_gaussian_short_rate(
    market_env: MarketEnvironment,
    generator: Optional[CapFloorGenerator],
    factors: int = 1,
    term: Optional[str] = "10Y",
    strikes: Optional[Sequence[float]] = None,
    initial: Optional[dict] = None,
    swaption_generator: Optional[SwaptionGenerator] = None,
    swaptions: Optional[Sequence[tuple]] = None,
    swaption_offsets: Sequence[float] = (0.0,),
    swaption_method: str = "approx",
    sigma_times=None,
    fixed: Optional[dict] = None,
):
    """Calibrate Hull-White (``factors=1``) or G2++ (``factors=2``) to caplet and / or swaption vols.

    Calibration instruments:

    - caplets of the *term* cap (``generator`` conventions; ``term=None`` for none), ATM (strike = caplet
      forward) by default or at every strike of *strikes*: market Bachelier on the caplet vols, model
      closed form :meth:`GaussianShortRateModel.caplet`;
    - swaptions ``(expiry, tenor)`` of *swaptions* (``swaption_generator`` conventions) at the forward swap
      rate plus each of *swaption_offsets*: market Bachelier on the cube vols, model Gaussian swap-rate
      approximation (``swaption_method="approx"``) or exact formula (``"exact"``: Jamshidian /
      Brigo-Mercurio), :meth:`GaussianShortRateModel.swaption`.

    Hull-White with a piecewise constant vol: ``sigma_times`` = breakpoints, or ``"caplets"`` for the fixing
    times of the calibration caplets (one vol per caplet: the ATM caplet term structure is matched exactly);
    the parameters are then ``a, sigma_1..sigma_P``. Caplets alone do not determine ``a`` with a free
    ``sigma(t)``: hold it with ``fixed={"a": ...}`` (or add swaptions).

    The model lives on the discount curve, the index curve through a deterministic basis. Least squares on
    vega-scaled price residuals (≈ normal vol errors), Jacobian by AAD compiled with ``tf.function``,
    bounded natural parameters (``BOUNDS``; ``params["at_bound"]`` lists those that end on a bound).
    Swaptions (several tenors) pin the factor correlation of G2++, which caplets alone leave undetermined.

    Args:
        market_env: Market with the curves and the (normal) caplet vols / swaption cube.
        generator: Cap conventions (index, period, day count, calendar); may be None without caplets.
        factors: 1 (Hull-White: a, sigma) or 2 (G2++: a, b, sigma, eta, rho).
        term: Longest cap whose caplets are used (None: no caplets).
        strikes: ``None`` = ATM caplets; otherwise these absolute strikes.
        initial: Starting parameters (default ``DEFAULT_START[factors]``).
        swaption_generator: Swaption conventions (needed with *swaptions*).
        swaptions: ``[(expiry, tenor), ...]`` terms, e.g. ``[("1Y", "5Y"), ("5Y", "10Y")]``.
        swaption_offsets: Strike offsets from the forward swap rate (default ATM only).
        swaption_method: ``"approx"`` (Gaussian swap rate, default) or ``"exact"`` model swaptions.
        sigma_times: Hull-White vol breakpoints (year fractions) or ``"caplets"`` (``factors=1`` only).
        fixed: Parameters held at the given values, e.g. ``{"a": 0.03}``.

    Returns:
        tuple: ``(model, params, report)``: the calibrated :class:`GaussianShortRateModel` on the market
            discount curve, the parameters (dict) and a DataFrame per instrument (instrument, date, tenor,
            strike, market and model normal vols, error in bp).

    Raises:
        ValueError: If *factors* is not 1 or 2, there are no instruments, the vols are not normal, or
            *sigma_times* / *fixed* / *swaption_method* are invalid.
    """
    if factors not in (1, 2):
        raise ValueError("factors must be 1 (Hull-White) or 2 (G2++)")
    if swaption_method not in ("approx", "exact"):
        raise ValueError(f"swaption_method must be 'approx' or 'exact', got '{swaption_method}'")
    if sigma_times is not None and factors != 1:
        raise ValueError("sigma_times is for Hull-White (factors=1)")
    index = (generator.index if generator is not None else swaption_generator.swap_generator.index)
    ccy_str, _ = index.name.split(":")
    curve = market_env.get_ir_curve(Currency[ccy_str])
    caps = _instruments(market_env, generator, term, strikes) if term is not None else []
    swpts = _swaption_instruments(market_env, swaption_generator, swaptions, swaption_offsets) if swaptions else []
    if not caps and not swpts:
        raise ValueError("no calibration instruments")
    if isinstance(sigma_times, str):
        if sigma_times != "caplets" or not caps:
            raise ValueError("sigma_times must be breakpoints or 'caplets' (with caplets)")
        sigma_times = sorted({r["t"] for r in caps})[:-1]
    names = _names(factors, sigma_times)
    fixed = dict(fixed or {})
    if any(n not in names for n in fixed):
        raise ValueError(f"fixed parameters must be among {names}")
    free = [n for n in names if n not in fixed]
    market = np.array([r["market_price"] for r in caps] + [r["market_price"] for r in swpts])
    vega = np.array([r["vega"] for r in caps] + [r["vega"] for r in swpts])

    # the curve is fixed during the calibration: caplet legs at time 0 and times precomputed, vectorized
    eval_date = Settings.evaluation_date
    t_fix = [r["t"] for r in caps]
    t_s = [curve.daycounter.year_fraction(eval_date, r["start"]) for r in caps]
    t_e = [curve.daycounter.year_fraction(eval_date, r["end"]) for r in caps]
    leg_a = np.array([r["basis"] * float(curve.discount(r["start"])) for r in caps])
    leg_b = np.array([(1.0 + r["tau"] * r["K"]) * float(curve.discount(r["end"])) for r in caps])

    # swaptions batched: flows padded to the longest swap (zero weights, last time repeated)
    if swpts:
        def padded(key, times=False):
            m = max(len(r["arrays"][key]) for r in swpts)
            return np.array([r["arrays"][key] + [r["arrays"][key][-1] if times else 0.0] * (m - len(r["arrays"][key]))
                             for r in swpts])

        sw_arrays = {k: padded(k, times=k.endswith("_t") or k.startswith("p0_"))
                     for k in ("pay_t", "pay_w", "p0_pay", "st_t", "st_w", "p0_st", "en_t", "en_w", "p0_en")}
        sw_arrays["te"] = np.array([r["arrays"]["te"] for r in swpts])
        sw_arrays["p0_te"] = np.array([r["arrays"]["p0_te"] for r in swpts])
        sw_annuity = np.array([r["annuity"] for r in swpts])
        sw_forward = np.array([r["F"] for r in swpts])
        sw_strike = np.array([r["K"] for r in swpts])
        sw_t = np.array([r["t"] for r in swpts])

    def full_vector(x: tf.Tensor) -> tf.Tensor:
        """Free parameters ``x`` (``free`` order) + the fixed ones, in ``names`` order."""
        it = iter(range(len(free)))
        return tf.stack([tf.constant(np.float64(fixed[n]), DTYPE) if n in fixed else x[next(it)] for n in names])

    def model_prices(x: tf.Tensor) -> tf.Tensor:
        m = _model(curve, factors, full_vector(x), sigma_times)
        parts = []
        if caps:
            parts.append(m.exchange_option_values(t_fix, t_s, t_e, leg_a, leg_b))
        if swpts and swaption_method == "exact":
            parts.append(m.swaption_exact_values(strike=sw_strike, payer=True, **sw_arrays))
        elif swpts:
            std = m.swaption_normal_vols(**sw_arrays) * np.sqrt(sw_t)
            d = (sw_forward - sw_strike) / std
            cdf = 0.5 * (1.0 + tf.math.erf(d / np.sqrt(2.0)))
            pdf = tf.exp(-0.5 * d * d) / np.sqrt(2.0 * np.pi)
            parts.append(sw_annuity * ((sw_forward - sw_strike) * cdf + std * pdf))     # payer, Bachelier
        return tf.concat(parts, axis=0)

    @tf.function
    def prices_and_jacobian(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            p = model_prices(x)
        # one backward pass per instrument in a while loop: the graph builds ~2x faster than with pfor
        return p, tape.jacobian(p, x, experimental_use_pfor=False)

    cache = {}

    def evaluate(x):
        key = tuple(np.round(x, 15))
        if key not in cache:
            cache.clear()
            p, j = prices_and_jacobian(tf.constant(x, DTYPE))
            cache[key] = (p.numpy(), (p.numpy() - market) / vega, j.numpy() / vega[:, None])
        return cache[key]

    start = dict(DEFAULT_START[factors])
    start.update({n: start["sigma"] for n in names if n.startswith("sigma_")})
    start.update(initial or {})
    x0 = np.array([start[n] for n in free], dtype=float)
    lower, upper = (np.array([_bounds(n)[i] for n in free]) for i in (0, 1))
    fit = least_squares(lambda x: evaluate(x)[1], np.clip(x0, lower, upper), jac=lambda x: evaluate(x)[2],
                        bounds=(lower, upper), method="trf", x_scale="jac", xtol=1e-10, ftol=1e-10, gtol=1e-10)
    full = full_vector(tf.constant(fit.x, DTYPE))
    params = {n: float(v) for n, v in zip(names, full.numpy())}
    params["at_bound"] = [n for n, v in zip(free, fit.x) if np.isclose(v, _bounds(n)[0]) or np.isclose(v, _bounds(n)[1])]
    params["evaluations"] = int(fit.nfev)
    if sigma_times is not None:
        params["sigma_times"] = [float(t) for t in sigma_times]
    model = _model(curve, factors, full, sigma_times)

    prices = evaluate(fit.x)[0]
    report = []
    for r, p in zip(caps, prices[:len(caps)]):
        target = p / (r["tau"] * r["df"])
        try:
            vol = brentq(lambda v: _bachelier(r["F"], r["K"], v, r["t"]) - target, 1e-8, 0.2, xtol=1e-14)
        except ValueError:
            vol = float("nan")
        report.append({"instrument": "caplet", "date": r["fixing"], "tenor": None, "strike": r["K"],
                       "market_vol": r["market_vol"], "model_vol": vol, "error_bp": (vol - r["market_vol"]) * 1e4})
    for r, p in zip(swpts, prices[len(caps):]):
        try:
            vol = brentq(lambda v: r["annuity"] * _bachelier(r["F"], r["K"], v, r["t"]) - p, 1e-8, 0.2, xtol=1e-14)
        except ValueError:
            vol = float("nan")
        report.append({"instrument": "swaption", "date": r["expiry"], "tenor": r["tenor"], "strike": r["K"],
                       "market_vol": r["market_vol"], "model_vol": vol, "error_bp": (vol - r["market_vol"]) * 1e4})
    return model, params, pd.DataFrame(report)
