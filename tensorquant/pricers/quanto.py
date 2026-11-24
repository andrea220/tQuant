"""Quanto and composite products in the Monte Carlo pricers: the drift of the process, the FX paths of a composite,
the payment-currency curve."""

import numpy as np
import tensorflow as tf

from ..models.brownian import observation_columns
from ..numericalhandles.dtypes import DTYPE


def quanto_drift(market_env, equity, payment_ccy, process, tenor: float) -> tuple:
    """Quanto drift of one underlying paid in *payment_ccy* (``rho sigma_X``, :meth:`MarketEnvironment.get_quanto`)
    set on *process*.

    Returns:
        tuple: ``(rho, sigma_x)``.

    Raises:
        ValueError: If the model has no quanto drift (local vol and GBM only for now).
    """
    rho, sigma_x = market_env.get_quanto(equity.ticker, equity.currency, payment_ccy, tenor)
    try:
        process.set_quanto(rho * sigma_x)
    except NotImplementedError as exc:
        raise ValueError(f"quanto products: local vol and GBM only ({exc})") from exc
    return rho, sigma_x


def apply_quanto(market_env, product, equity, process, tenor: float):
    """For a quanto *product* (paid in ``product.payment_ccy`` at a fixed rate): the quanto drift on *process*
    (``rho sigma_X``, :meth:`MarketEnvironment.get_quanto` at *tenor*) and the discount curve of the payment currency;
    otherwise the underlying's curve. Sets ``product.quanto_correlation`` / ``product.quanto_fx_vol`` (or ``None``).

    Returns:
        RateCurve: The curve the payments are discounted on.

    Raises:
        ValueError: If the model has no quanto drift (local vol and GBM only for now) or the market lacks the inputs.
    """
    product.quanto_correlation = product.quanto_fx_vol = None
    if not (product.is_quanto or product.is_composite):
        return equity.discount_curve
    rho, sigma_x = quanto_drift(market_env, equity, product.payment_ccy, process, tenor)
    product.quanto_correlation, product.quanto_fx_vol = rho, sigma_x
    return market_env.get_ir_curve(product.payment_ccy)


def composite_fx_paths(market_env, product, equity, time_grid, dw, observation_times, seed: int) -> tf.Tensor:
    """FX paths of a composite product at *observation_times*: ``X`` (one unit of the underlying's currency in the
    payment currency) lognormal around its covered-interest-parity forward, ``X_t = F_X(t) exp(-sigma_X^2 t / 2 +
    sigma_X W_X(t))``, ``W_X = rho W_S + sqrt(1 - rho^2) W_Z``: ``W_S`` the Brownian motion of the underlying (the
    normals *dw* of its process on *time_grid*, one per step), ``W_Z`` independent (stateless seed ``[seed, 1]``).
    Call after :func:`apply_quanto` (under the payment-currency measure the underlying has the quanto drift).

    Returns:
        tf.Tensor: ``[n_paths, len(observation_times)]``.

    Raises:
        ValueError: If the process draws more than one normal per step (Heston, SLV).
    """
    if len(dw.shape) != 2:
        raise ValueError("composite products: models with one normal per step only (local vol, GBM)")
    rho, sigma_x = product.quanto_correlation, product.quanto_fx_vol
    grid = np.asarray(time_grid, dtype=float)
    dt = tf.constant(np.diff(np.concatenate([[0.0], grid])), DTYPE)
    dz = tf.random.stateless_normal(tf.shape(dw), seed=[seed, 1], dtype=DTYPE)
    w_s = tf.cumsum(tf.sqrt(dt)[None, :] * dw, axis=1)
    w_z = tf.cumsum(tf.sqrt(dt)[None, :] * dz, axis=1)
    cols = observation_columns(grid, observation_times)
    w_x = rho * tf.gather(w_s, cols, axis=1) + tf.sqrt(1.0 - rho * rho) * tf.gather(w_z, cols, axis=1)
    t_obs = tf.constant(grid[cols], DTYPE)
    forwards = tf.stack([tf.cast(market_env.get_fx_forward(equity.currency, product.payment_ccy, float(t)), DTYPE)
                         for t in grid[cols]])
    return forwards[None, :] * tf.exp(-0.5 * sigma_x * sigma_x * t_obs[None, :] + sigma_x * w_x)
