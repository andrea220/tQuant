from __future__ import annotations

from datetime import date
from typing import Optional, Sequence, Union

import numpy as np
import tensorflow as tf

from .stochasticprocess import StochasticProcess
from ..markethandles.ircurve import RateCurve
from ..numericalhandles.dtypes import DTYPE
from ..timehandles.utils import Settings


def _as_param(x) -> tf.Tensor:
    """Model parameter as a float64 tensor (variables stay in the graph; numbers go straight to float64,
    not through float32)."""
    if isinstance(x, (tf.Tensor, tf.Variable)):
        return tf.cast(x, DTYPE)
    return tf.convert_to_tensor(np.asarray(x, dtype=np.float64))


class GaussianShortRateModel(StochasticProcess):
    """n-factor Gaussian short rate model fitted to a discount curve (G1++ = Hull-White, G2++).

    ``r(t) = x_1(t) + ... + x_n(t) + phi(t)``, ``dx_i = -kappa_i x_i dt + sigma_i dW_i``,
    ``x_i(0) = 0``, ``d<W_i, W_j> = rho_ij dt``; ``phi`` fits the initial curve ``P(0, T)`` exactly.
    With ``V(t, T) = Var(int_t^T sum_i x_i du | F_t)`` (Brigo-Mercurio):

    - bond: ``P(t, T) = P(0,T)/P(0,t) exp(1/2 [V(t,T) - V(0,T) + V(0,t)] - sum_i B_i(T - t) x_i(t))``,
      ``B_i(tau) = (1 - e^{-kappa_i tau}) / kappa_i``;
    - bank-account discount along a path: ``D(0, t) = P(0, t) exp(-sum_i I_i(t) - V(0, t)/2)``,
      ``I_i(t) = int_0^t x_i``, so ``E[D(0, t)] = P(0, t)``.

    :meth:`simulate` draws ``(x_i, I_i)`` exactly (joint Gaussian transition: no time-step error).
    The curve enters only through ``P(0, .)``, so prices are differentiable w.r.t. its pillars
    (rho by AAD) and w.r.t. the model parameters if passed as variables.

    Args:
        curve (RateCurve): Initial discount curve (e.g. ``market_env.get_ir_curve(ccy)``).
        kappas (Sequence[float]): Mean-reversion speeds ``[n]``.
        sigmas (Sequence[float]): Volatilities ``[n]``.
        correlation (array-like, optional): ``[n, n]`` correlation of the Brownians (identity if None).
    """

    def __init__(self, curve: RateCurve, kappas, sigmas, correlation=None):
        self._curve = curve
        self._kappa = _as_param(kappas)
        self._sigma = _as_param(sigmas)
        n = int(self._kappa.shape[0])
        if int(self._sigma.shape[0]) != n:
            raise ValueError("kappas and sigmas must have the same length")
        self._rho = tf.eye(n, dtype=DTYPE) if correlation is None else _as_param(correlation)
        if tuple(self._rho.shape) != (n, n):
            raise ValueError(f"correlation must be [{n}, {n}]")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def hull_white(cls, curve: RateCurve, a, sigma) -> "GaussianShortRateModel":
        """One factor (G1++ / Hull-White): ``dx = -a x dt + sigma dW``."""
        return cls(curve, tf.reshape(_as_param(a), [1]), tf.reshape(_as_param(sigma), [1]))

    @classmethod
    def g2pp(cls, curve: RateCurve, a, b, sigma, eta, rho) -> "GaussianShortRateModel":
        """Two factors (G2++): speeds ``a, b``, vols ``sigma, eta``, correlation ``rho``."""
        rho = _as_param(rho)
        one = tf.constant(1.0, DTYPE)
        corr = tf.stack([tf.stack([one, rho]), tf.stack([rho, one])])
        return cls(curve, tf.stack([_as_param(a), _as_param(b)]), tf.stack([_as_param(sigma), _as_param(eta)]), corr)

    # ------------------------------------------------------------------
    # StochasticProcess interface
    # ------------------------------------------------------------------

    def size(self) -> int:
        return int(self._kappa.shape[0])

    def initial_values(self) -> tf.Tensor:
        return tf.zeros([self.size()], DTYPE)

    @property
    def curve(self) -> RateCurve:
        return self._curve

    # ------------------------------------------------------------------
    # Closed forms
    # ------------------------------------------------------------------

    def time(self, d: Union[date, float]) -> float:
        """Year fraction from the evaluation date with the curve's day count."""
        if isinstance(d, date):
            return self._curve.daycounter.year_fraction(Settings.evaluation_date, d)
        return float(d)

    def _B(self, tau) -> tf.Tensor:
        """``B_i(tau) = (1 - e^{-kappa_i tau}) / kappa_i`` for every factor (last axis)."""
        tau = _as_param(tau)[..., None]
        return (1.0 - tf.exp(-self._kappa * tau)) / self._kappa

    def variance(self, t, T) -> tf.Tensor:
        """``V(t, T) = Var(int_t^T sum_i x_i du | F_t)``."""
        tau = _as_param(T) - _as_param(t)
        k_i, k_j = self._kappa[:, None], self._kappa[None, :]
        s = self._sigma[:, None] * self._sigma[None, :] * self._rho
        b_i = (1.0 - tf.exp(-k_i * tau)) / k_i
        b_j = (1.0 - tf.exp(-k_j * tau)) / k_j
        b_ij = (1.0 - tf.exp(-(k_i + k_j) * tau)) / (k_i + k_j)
        return tf.reduce_sum(s / (k_i * k_j) * (tau - b_i - b_j + b_ij))

    def zero_bond(self, t: Union[date, float], T: Union[date, float], state: tf.Tensor) -> tf.Tensor:
        """``P(t, T)`` for every path given the factors at ``t``.

        Args:
            t: Observation date / year fraction (``t <= T``).
            T: Bond maturity (date / year fraction).
            state: ``[n_paths, n_factors]`` factors ``x(t)``.

        Returns:
            tf.Tensor: ``[n_paths]`` float64 bond prices.
        """
        tt, TT = self.time(t), self.time(T)
        p0t = tf.cast(self._curve.discount(t if isinstance(t, date) else tt), DTYPE)
        p0T = tf.cast(self._curve.discount(T if isinstance(T, date) else TT), DTYPE)
        conv = 0.5 * (self.variance(tt, TT) - self.variance(0.0, TT) + self.variance(0.0, tt))
        return p0T / p0t * tf.exp(conv - tf.reduce_sum(self._B(TT - tt) * tf.cast(state, DTYPE), axis=-1))

    def short_rate(self, t: Union[date, float], state: tf.Tensor) -> tf.Tensor:
        """``r(t) = sum_i x_i(t) + phi(t)``, ``phi(t) = f(0, t) + 1/2 dV(0, t)/dt``."""
        tt = _as_param(self.time(t))
        with tf.GradientTape() as tape:
            tape.watch(tt)
            v = self.variance(0.0, tt)
        phi = tf.cast(self._curve.inst_fwd(float(tt)), DTYPE) + 0.5 * tape.gradient(v, tt)
        return tf.reduce_sum(tf.cast(state, DTYPE), axis=-1) + phi

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _transition(self, dt: float):
        """Mean matrix ``M [2n, 2n]`` and covariance ``C [2n, 2n]`` of ``z = (x, I)`` over ``dt``."""
        k = self._kappa
        k_i, k_j = k[:, None], k[None, :]
        s = self._sigma[:, None] * self._sigma[None, :] * self._rho
        e = tf.exp(-k * dt)
        b = (1.0 - e) / k                                                    # B_i(dt)
        b_ij = (1.0 - tf.exp(-(k_i + k_j) * dt)) / (k_i + k_j)
        c_xx = s * b_ij
        c_xi = s / k_j * (b[:, None] - b_ij)                                 # Cov(x_i, I_j)
        c_ii = s / (k_i * k_j) * (dt - b[:, None] - b[None, :] + b_ij)
        cov = tf.concat([tf.concat([c_xx, c_xi], 1), tf.concat([tf.transpose(c_xi), c_ii], 1)], 0)
        n = self.size()
        zero = tf.zeros([n, n], DTYPE)
        mean = tf.concat([tf.concat([tf.linalg.diag(e), zero], 1),
                          tf.concat([tf.linalg.diag(b), tf.eye(n, dtype=DTYPE)], 1)], 0)
        return mean, cov

    def _path(self, times: np.ndarray, dw: tf.Tensor):
        """Exact transitions of ``z = (x, I)``: factors ``[n_paths, n_times, n]`` and ``sum_i I_i`` ``[n_paths, n_times]``."""
        if np.any(np.diff(np.concatenate([[0.0], times])) <= 0):
            raise ValueError("times must be increasing and > 0")
        n = self.size()
        dw = tf.cast(dw, DTYPE)
        z = tf.zeros([tf.shape(dw)[0], 2 * n], DTYPE)
        xs, integrals = [], []
        t_prev = 0.0
        for k, t in enumerate(times):
            mean, cov = self._transition(t - t_prev)
            chol = tf.linalg.cholesky(cov + 1e-18 * tf.eye(2 * n, dtype=DTYPE))
            z = tf.matmul(z, mean, transpose_b=True) + tf.matmul(dw[:, k], chol, transpose_b=True)
            xs.append(z[:, :n])
            integrals.append(tf.reduce_sum(z[:, n:], axis=1))
            t_prev = t
        return tf.stack(xs, axis=1), tf.stack(integrals, axis=1)

    def evolve(self, t_grid, dw, observation_times=None) -> tf.Tensor:
        """Factors on *t_grid* (exact simulation; common :class:`StochasticProcess` interface).

        Args:
            t_grid: ``[n_steps]`` increasing times (> 0).
            dw: ``[n_paths, n_steps, 2 n_factors]`` standard normals (shocks of the factors and their
                integrals at each step).
            observation_times: Optional subset of ``t_grid``.

        Returns:
            tf.Tensor: ``[n_paths, n_steps, n_factors]`` (or the observed columns).
        """
        times = np.asarray(tf.convert_to_tensor(t_grid), dtype=float)
        xs, _ = self._path(times, dw)
        if observation_times is not None:
            from .brownian import observation_columns
            xs = tf.gather(xs, observation_columns(times, observation_times), axis=1)
        return xs

    def simulate(self, times: Sequence[float], n_paths: int, seed: int = 42):
        """Draw (stateless seed) and simulate the factors and the bank-account discount on *times*.

        Args:
            times: Increasing year fractions (> 0) from the evaluation date.
            n_paths: Number of paths.
            seed: Seed of the stateless Gaussian draws.

        Returns:
            tuple: ``(x [n_paths, n_times, n_factors], discount [n_paths, n_times])`` with
                ``discount = D(0, t) = P(0, t) exp(-sum_i I_i(t) - V(0, t)/2)``.
        """
        times = np.asarray(times, dtype=float)
        dw = tf.random.stateless_normal([n_paths, len(times), 2 * self.size()], seed=[seed, 0], dtype=DTYPE)
        xs, integrals = self._path(times, dw)
        p0 = tf.stack([tf.cast(self._curve.discount(float(t)), DTYPE) for t in times])
        var = tf.stack([self.variance(0.0, t) for t in times])
        return xs, p0[None, :] * tf.exp(-integrals - 0.5 * var[None, :])
