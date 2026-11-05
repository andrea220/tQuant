from __future__ import annotations

from datetime import date
from typing import Union

import numpy as np
import tensorflow as tf

from .stochasticprocess import StochasticProcess
from ..numericalhandles.dtypes import DTYPE, as_variable
from ..timehandles.utils import Settings


def _as_input(x) -> tf.Tensor:
    """Model parameter: tensors/variables are kept in the graph (cast to float64), numbers become variables."""
    if isinstance(x, (tf.Tensor, tf.Variable)):
        return tf.cast(x, DTYPE)
    return as_variable(x)


def _equity_carry_inputs(equity, maturity: Union[date, float]):
    """Spot, rate, carry and tenor of an :class:`Equity` up to *maturity* (market tensors, float64).

    Discrete dividends: escrowed spot ``S* = S - PV(div)`` and carry ``repo``
    (as :class:`BlackScholesPricer` with ``dividend_model="discrete"``).
    Continuous yield: spot ``S`` and carry ``q + repo``. ``r = -ln P(0,T) / T``; drift ``r - carry``.

    Returns:
        tuple: ``(x0, r, carry, tenor)``.
    """
    # tenor as in BlackScholesPricer: vol-surface day count from the evaluation date
    tenor = equity.vol_surface.daycounter.year_fraction(Settings.evaluation_date, maturity) \
        if isinstance(maturity, date) else float(maturity)
    df = tf.cast(equity.discount_curve.discount(maturity), DTYPE)
    r = -tf.math.log(df) / tenor
    carry = equity.repo
    if equity.dividend_model == "continuous":
        carry = carry + equity.div_yield
    x0 = equity.spot - equity.pv_dividends(maturity)
    return x0, r, carry, tenor


def observation_columns(t_grid, observation_times):
    """Column indices of *observation_times* in *t_grid* (each must be a grid point, 1e-10 tolerance).

    Raises:
        ValueError: If an observation time is not on the grid.
    """
    grid = np.asarray(t_grid, dtype=float)
    obs = np.atleast_1d(np.asarray(observation_times, dtype=float))
    idx = np.clip(np.searchsorted(grid, obs - 1e-10), 0, len(grid) - 1)
    if np.any(np.abs(grid[idx] - obs) > 1e-10):
        raise ValueError("observation_times must be points of t_grid")
    return idx


class GeometricBrownianMotion(StochasticProcess):
    """GBM ``dS = mu S dt + sigma S dW``, simulated exactly on a time grid (float64).

    Parameters given as tensors/``tf.Variable``s are used as they are, so gradients
    flow to them (e.g. market risk factors via :meth:`from_equity`); numbers are
    wrapped into new float64 variables.
    """

    def __init__(self, mu, sigma, x0):
        self._x0 = _as_input(x0)
        self._mu = _as_input(mu)
        self._sigma = _as_input(sigma)

    @classmethod
    def from_equity(cls, equity, maturity: Union[date, float], strike: float) -> "GeometricBrownianMotion":
        """GBM for one option on *equity*, from the market risk factors (differentiable).

        ``x0`` and ``mu`` as in :func:`_equity_carry_inputs`; ``sigma`` is the surface
        vol at ``(strike, maturity)``, so the MC price converges to Black-Scholes.

        Args:
            equity (Equity): The underlying (``market_env.get_equity``).
            maturity (Union[date, float]): Option maturity.
            strike (float): Option strike (vol lookup).

        Returns:
            GeometricBrownianMotion: Model whose inputs are market tensors.

        Raises:
            ValueError: If the underlying has no vol surface.
        """
        if equity.vol_surface is None:
            raise ValueError(f"No volatility surface for underlying '{equity.ticker}'")
        x0, r, carry, tenor = _equity_carry_inputs(equity, maturity)
        sigma = equity.vol_surface.volatility(strike=float(strike), tenor=tenor)
        return cls(mu=r - carry, sigma=sigma, x0=x0)

    def drift(self, dt):
        return (self._mu - (self._sigma**2) / 2) * dt

    def diffusion(self, dt):
        return self._sigma * tf.math.sqrt(dt)

    def initial_values(self):
        return self._x0

    def size(self):
        return 1

    def evolve(self, t_grid, dw, observation_times=None):
        """
        Args:
            t_grid: 1D tensor of shape [n_steps] with observation times, e.g. [0.25, 0.5, 0.75, 1.0]
            dw:     tensor of shape [n_paths, n_steps], standard normals
            observation_times: optional subset of t_grid; if given, only those columns are returned

        Returns:
            tensor of shape [n_paths, n_steps] (or [n_paths, len(observation_times)]) with simulated process values
        """
        t_grid = tf.cast(t_grid, DTYPE)
        dw = tf.cast(dw, DTYPE)
        t_full = tf.concat([tf.zeros([1], dtype=DTYPE), t_grid], axis=0)  # prepend 0
        dt = tf.reshape(t_full[1:] - t_full[:-1], [1, -1])  # [1, n_steps], one dt per step
        exp_factor = tf.math.exp(self.drift(dt) + self.diffusion(dt) * dw)
        paths = self._x0 * tf.math.cumprod(exp_factor, axis=1)
        if observation_times is not None:
            paths = tf.gather(paths, observation_columns(t_grid.numpy(), observation_times), axis=1)
        return paths


class ArithmeticBrownianMotion(StochasticProcess):
    """ABM ``dX = mu dt + sigma dW`` (float64; tensor parameters kept in the graph)."""

    def __init__(self, mu, sigma, x0):
        self._x0 = _as_input(x0)
        self._mu = _as_input(mu)
        self._sigma = _as_input(sigma)

    def drift(self, dt):
        return self._mu * dt

    def diffusion(self, dt):
        return self._sigma * tf.math.sqrt(dt)

    def initial_values(self):
        return self._x0

    def size(self):
        return 1

    def evolve(self, t_grid, dw, observation_times=None):
        """
        Args:
            t_grid: 1D tensor of shape [n_steps] with observation times, e.g. [0.25, 0.5, 0.75, 1.0]
            dw:     tensor of shape [n_paths, n_steps], standard normals
            observation_times: optional subset of t_grid; if given, only those columns are returned

        Returns:
            tensor of shape [n_paths, n_steps] (or [n_paths, len(observation_times)]) with simulated process values
        """
        t_grid = tf.cast(t_grid, DTYPE)
        dw = tf.cast(dw, DTYPE)
        t_full = tf.concat([tf.zeros([1], dtype=DTYPE), t_grid], axis=0)  # prepend 0
        dt = tf.reshape(t_full[1:] - t_full[:-1], [1, -1])  # [1, n_steps], one dt per step
        paths = self._x0 + tf.math.cumsum(self.drift(dt) + self.diffusion(dt) * dw, axis=1)
        if observation_times is not None:
            paths = tf.gather(paths, observation_columns(t_grid.numpy(), observation_times), axis=1)
        return paths
