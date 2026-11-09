"""Heston stochastic local volatility (SLV) equity model on the market forward."""

from __future__ import annotations

from datetime import date
from typing import Union

import numpy as np
import tensorflow as tf

from .brownian import observation_columns
from .equitymodel import EquityModel
from .heston import HestonModel, HestonProcess, _f64
from .localvolatility import LocalVolatilityModel, LocalVolatilityProcess, _rows_at
from .stochasticprocess import StochasticProcess
from ..numericalhandles.dtypes import DTYPE


def _quantiles(x: tf.Tensor, lo: float, hi: float) -> tuple:
    """Two empirical quantiles (order statistics of one sort; no gradient needed)."""
    last = tf.cast(tf.shape(x)[0] - 1, DTYPE)
    ordered = tf.sort(x)
    return ordered[tf.cast(tf.round(lo * last), tf.int32)], ordered[tf.cast(tf.round(hi * last), tf.int32)]


SHRINK = 0.05            # sparse smoother nodes shrink to the mean variance with weight SHRINK * (paths per node)
VARIANCE_FLOOR = 1e-4    # floor on E[v | X] (a 1% vol): bounds the leverage


@tf.function(reduce_retracing=True)
def _simulate_slv(rows, sigma_rows, lv_y, v0, dw, n_nodes, max_leverage, keep_all: bool):
    """Particle simulation of ``X`` (and, with ``keep_all``, ``v`` and ``L``) ``[n_paths, n_steps]``.

    Per step (``rows [n_steps, 9]``: ``m0, e, s1, s2`` of the variance, ``dt, kappa theta, rho / xi,
    kappa rho / xi, 1 - rho^2``; ``sigma_rows [n_steps, nY]``: local vol at the step start on ``lv_y``):

    1. ``E[v | ln X]`` from the particles by a tent-kernel smoother on ``n_nodes`` nodes spanning the 0.1% - 99.9%
       quantiles of ``ln X`` (node set held fixed in the step): each particle spreads its weight linearly over its
       two nodes, ``E_j = (sum w v + c mean v) / (sum w + c)`` (``c = SHRINK n / n_nodes`` shrinks sparse nodes to
       the mean), read back with the same linear weights, floored at ``VARIANCE_FLOOR``; held out of the gradients
       (see :class:`HestonSLVModel`);
    2. leverage per particle ``L = sigma_LV(t, ln X) / sqrt(E[v | ln X])`` (capped at ``max_leverage``);
    3. Andersen QE for ``v'`` and, with ``L`` frozen on the step, ``ln X' = ln X + K0* + K1 v + K2 v' +
       sqrt(K3 v + K4 v') Z`` with the per-particle coefficients scaled by ``L`` and the exact martingale
       correction ``K0*`` (fallback ``K0`` where the moment condition fails).

    Gradients flow through the local vol rows only (the forward is applied outside); the variance and
    ``E[v | X]`` carry none (Heston parameters held: SLV risk convention).
    """
    n_paths, n_steps = tf.shape(dw)[0], tf.shape(dw)[1]
    coef = tf.TensorArray(DTYPE, size=n_steps).unstack(rows)
    lv_rows = tf.TensorArray(DTYPE, size=n_steps).unstack(sigma_rows)
    z_v = tf.TensorArray(DTYPE, size=n_steps).unstack(tf.transpose(dw[..., 0]))
    z_x = tf.TensorArray(DTYPE, size=n_steps).unstack(tf.transpose(dw[..., 1]))
    n_y = tf.shape(lv_y)[0]
    shrink = SHRINK * tf.cast(n_paths, DTYPE) / tf.cast(n_nodes, DTYPE)
    v = tf.fill([n_paths], v0)
    log_x = tf.zeros([n_paths], DTYPE)
    xs, vs, ls = (tf.TensorArray(DTYPE, size=n_steps) for _ in range(3))
    for i in tf.range(n_steps):
        r = coef.read(i)
        m0, e, s1, s2, dt, k_theta, r_x, k_r, q = (r[j] for j in range(9))
        # 1. conditional expectation of the variance given ln X (tent-kernel smoother on fixed nodes)
        y_sg = tf.stop_gradient(log_x)
        lo, hi = _quantiles(y_sg, 0.001, 0.999)
        span = tf.maximum(hi - lo, 1e-12)
        pos = tf.clip_by_value((log_x - lo) / span * tf.cast(n_nodes - 1, DTYPE), 0.0, tf.cast(n_nodes - 1, DTYPE))
        j = tf.clip_by_value(tf.cast(tf.floor(tf.stop_gradient(pos)), tf.int32), 0, n_nodes - 2)
        w = pos - tf.cast(j, DTYPE)                                                   # weight of node j + 1
        num = tf.math.unsorted_segment_sum(v * (1.0 - w), j, n_nodes) + tf.math.unsorted_segment_sum(v * w, j + 1, n_nodes)
        den = tf.math.unsorted_segment_sum(1.0 - w, j, n_nodes) + tf.math.unsorted_segment_sum(w, j + 1, n_nodes)
        node_v = (num + shrink * tf.reduce_mean(v)) / (den + shrink)
        cond_v = tf.stop_gradient(tf.gather(node_v, j) * (1.0 - w) + tf.gather(node_v, j + 1) * w)
        # 2. leverage
        row = lv_rows.read(i)
        yc = tf.clip_by_value(log_x, lv_y[0], lv_y[-1])
        k = tf.clip_by_value(tf.searchsorted(lv_y, tf.stop_gradient(yc), side="right") - 1, 0, n_y - 2)
        y0, y1 = tf.gather(lv_y, k), tf.gather(lv_y, k + 1)
        wy = (yc - y0) / (y1 - y0)
        sigma = tf.gather(row, k) * (1.0 - wy) + tf.gather(row, k + 1) * wy
        lev = tf.minimum(sigma / tf.sqrt(tf.maximum(cond_v, VARIANCE_FLOOR)), max_leverage)
        # 3. QE variance step
        zv = z_v.read(i)
        m = m0 + e * v
        psi = (s1 * v + s2) / (m * m)
        pq = tf.minimum(psi, 1.5)
        b2 = 2.0 / pq - 1.0 + tf.sqrt(2.0 / pq) * tf.sqrt(2.0 / pq - 1.0)
        aq = m / (1.0 + b2)
        v_q = aq * tf.square(tf.sqrt(b2) + zv)
        pe = tf.maximum(psi, 1.5)
        p = (pe - 1.0) / (pe + 1.0)
        beta = (1.0 - p) / m
        tail = 0.5 * tf.math.erfc(zv / np.sqrt(2.0))
        v_e = tf.maximum(tf.math.log((1.0 - p) / tf.maximum(tail, 1e-300)), 0.0) / beta
        quad = psi <= 1.5
        v_new = tf.where(quad, v_q, v_e)
        # log step with the leverage frozen on the step (Andersen's central weights)
        half = 0.5 * dt * (k_r * lev - 0.5 * lev * lev)
        k0 = -r_x * k_theta * dt * lev
        k1, k2 = half - r_x * lev, half + r_x * lev
        k3 = 0.5 * dt * lev * lev * q
        big_a = k2 + 0.5 * k3
        den_q = 1.0 - 2.0 * big_a * aq
        log_mq = big_a * b2 * aq / tf.maximum(den_q, 1e-12) - 0.5 * tf.math.log(tf.maximum(den_q, 1e-12))
        log_me = tf.math.log(tf.maximum(p + beta * (1.0 - p) / tf.maximum(beta - big_a, 1e-12), 1e-300))
        ok = tf.where(quad, den_q > 0.0, beta > big_a)
        k0_star = tf.where(ok, -tf.where(quad, log_mq, log_me) - (k1 + 0.5 * k3) * v, k0)
        log_x = log_x + k0_star + k1 * v + k2 * v_new + tf.sqrt(tf.maximum(k3 * (v + v_new), 1e-300)) * z_x.read(i)
        v = v_new
        xs = xs.write(i, tf.exp(log_x))
        if keep_all:                      # (unused outputs still cost gradient memory: only on request)
            vs, ls = vs.write(i, v), ls.write(i, lev)
    if keep_all:
        return tuple(tf.transpose(a.stack()) for a in (xs, vs, ls))
    return (tf.transpose(xs.stack()),)


class HestonSLVProcess(StochasticProcess):
    """Heston stochastic local volatility process on the market forward: the :class:`HestonSLVModel` bound to an
    :class:`Equity` (or built from the market inside the caller's tape by :meth:`from_equity`).

    ``S_t = F(t) X_t`` with ``F(t) = equity.forward(t)`` and::

        dX / X = L(t, X) sqrt(v) dW_1,   dv = kappa (theta - v) dt + eta xi sqrt(v) dW_2,   d<W_1, W_2> = rho dt,

    ``v`` the Heston variance (:class:`HestonModel` parameters, vol of vol scaled by the mixing fraction
    ``eta``) and the leverage ``L^2(t, x) = sigma_LV^2(t, x) / E[v_t | X_t = x]`` with ``sigma_LV`` the local vol
    of the market surface (:func:`gatheral_local_vol`, as :class:`LocalVolatilityModel`). By Gyongy's theorem
    the one-dimensional marginals are those of the local vol model, so the vanilla smile is repriced, while the
    forward smile / path dependence follow the stochastic variance (``eta = 1``: the calibrated Heston dynamics;
    ``eta -> 0``: local vol).

    ``E[v | X]`` is estimated on the simulated particles at every step (kernel smoother in ``ln X``): the
    leverage is built inside :meth:`evolve` on the pricing paths (particle method, one pass).

    Greeks (AAD) follow the SLV risk convention: the Heston parameters are held and the leverage moves with the
    market through the local vol (vol nodes) and the forward (spot, repo, dividends, curve). In exact SLV the
    vanilla prices do not depend on the Heston parameters at all. Differentiating the particle estimator through
    them and through ``E[v | X]`` was tried and rejected: the derivative of the smoother w.r.t. a particle's own
    position is the slope between noisy nodes, which, multiplied over hundreds of steps, blew up (5Y autocallable:
    delta ~1e18); freezing only ``E`` left the Heston channel uncompensated (vanilla delta 16% off). With the
    parameters held, ``E`` depends on the market only through the particle positions (not differentiated).
    """

    def __init__(self, equity, heston: HestonProcess, local_vol: LocalVolatilityProcess, mixing: float = 1.0,
                 n_nodes: int = 100, martingale_correction: bool = True, max_leverage: float = 10.0):
        """Built by :meth:`HestonSLVModel.process` or :meth:`from_equity`, or from the two bound components.

        Args:
            equity (Equity): The underlying (its ``forward`` drives the dynamics).
            heston (HestonProcess): Stochastic variance parameters.
            local_vol (LocalVolatilityProcess): Local vol of the market surface.
            mixing (float): Vol-of-vol scale ``eta`` in ``(0, 1]``.
            n_nodes (int): Nodes of the conditional-expectation smoother.
            martingale_correction (bool): Rescale ``X_t`` to sample mean 1 (MC forward = ``F(t)``).
            max_leverage (float): Cap on the leverage.

        Raises:
            ValueError: If ``mixing`` is not in ``(0, 1]`` or ``n_nodes < 2``.
        """
        if not 0.0 < mixing <= 1.0:
            raise ValueError(f"mixing must be in (0, 1], got {mixing}")
        if n_nodes < 2:
            raise ValueError("n_nodes must be >= 2")
        self._equity = equity
        self.heston, self.local_vol = heston, local_vol
        self._mixing, self._n_nodes = float(mixing), int(n_nodes)
        self._martingale_correction = martingale_correction
        self._max_leverage = float(max_leverage)
        self._S0 = tf.cast(equity.spot, DTYPE)

    @classmethod
    def from_equity(cls, equity, maturity: Union[date, float] = None, strike: float = None, mixing: float = 1.0,
                    n_nodes: int = 100, martingale_correction: bool = True) -> "HestonSLVProcess":
        """Heston calibrated to the surface (:meth:`HestonProcess.from_equity`) + local vol of the same surface.

        Args:
            equity (Equity): The underlying (``market_env.get_equity``).
            maturity, strike: Unused (same factory signature as the other MC models).
            mixing (float): Vol-of-vol scale ``eta`` in ``(0, 1]``.
            n_nodes (int): Nodes of the conditional-expectation smoother.
            martingale_correction (bool): Rescale the simulated ``X_t`` to sample mean 1 at each time.

        Returns:
            HestonSLVProcess: Process built from the market (differentiable w.r.t. it).
        """
        heston = HestonProcess.from_equity(equity)
        local_vol = LocalVolatilityProcess.from_equity(equity)
        return cls(equity, heston, local_vol, mixing, n_nodes, martingale_correction)

    @property
    def mixing(self) -> float:
        return self._mixing

    # ------------------------------------------------------------------
    # StochasticProcess interface
    # ------------------------------------------------------------------

    def size(self) -> int:
        """State dimensions: spot and variance."""
        return 2

    def noise_size(self) -> int:
        """Two normals per step: variance shock and (independent) log-spot shock."""
        return 2

    def initial_values(self) -> tf.Tensor:
        return tf.stack([self._S0, self.heston.parameters["v0"]])

    def _step_rows(self, dt: np.ndarray) -> tf.Tensor:
        """Per-step coefficients ``[n_steps, 9]`` (Heston parameters held out of the gradients)."""
        p = {k: tf.stop_gradient(v) for k, v in self.heston.parameters.items()}
        kappa, theta, rho = p["kappa"], p["theta"], p["rho"]
        xi = self._mixing * p["xi"]
        dt = _f64(dt)
        e = tf.exp(-kappa * dt)
        ones = tf.ones_like(dt)
        return tf.stack([theta * (1.0 - e), e, xi * xi * e * (1.0 - e) / kappa,
                         theta * xi * xi * tf.square(1.0 - e) / (2.0 * kappa), dt, kappa * theta * ones,
                         rho / xi * ones, kappa * rho / xi * ones, (1.0 - rho * rho) * ones], axis=1)

    def simulate(self, t_grid, dw, keep_all: bool = True) -> tuple:
        """``(X, v, L)`` ``[n_paths, n_steps]``: forward ratio, variance and the leverage used on each step
        (``(X,)`` only with ``keep_all=False``).

        Args:
            t_grid: ``[n_steps]`` increasing times (> 0).
            dw: ``[n_paths, n_steps, 2]`` standard normals (``n_paths >= n_nodes``).
            keep_all (bool): Also return the variance and leverage paths.
        """
        times = np.asarray(tf.convert_to_tensor(t_grid), dtype=float)
        if times[0] <= 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("t_grid must be increasing and > 0")
        dw = tf.cast(dw, DTYPE)
        if len(dw.shape) != 3 or int(dw.shape[-1]) != 2:
            raise ValueError("dw must be [n_paths, n_steps, 2]")
        if int(dw.shape[0]) < self._n_nodes:
            raise ValueError(f"at least n_nodes = {self._n_nodes} paths are needed")
        starts = np.concatenate([[0.0], times[:-1]])
        sigma_ty, lv_t, lv_y = self.local_vol.local_vol_grid
        sigma_rows = _rows_at(sigma_ty, lv_t, tf.constant(starts, DTYPE))
        return _simulate_slv(self._step_rows(np.diff(np.concatenate([[0.0], times]))), sigma_rows, lv_y,
                             tf.stop_gradient(self.heston.parameters["v0"]), dw, tf.constant(self._n_nodes),
                             tf.constant(self._max_leverage, DTYPE), keep_all)

    def evolve(self, t_grid, dw, observation_times=None) -> tf.Tensor:
        """Simulate spot paths ``S_t = F(t) X_t`` (particle method on the given paths).

        Args:
            t_grid: ``[n_steps]`` increasing times (year fractions), without 0.
            dw: ``[n_paths, n_steps, 2]`` standard normals.
            observation_times: Optional subset of ``t_grid``: only those columns are returned and the
                market forward is computed only there.

        Returns:
            tf.Tensor: ``[n_paths, n_steps]`` (or ``[n_paths, len(observation_times)]``) spots.
        """
        times = np.asarray(tf.convert_to_tensor(t_grid), dtype=float)
        x = self.simulate(times, dw, keep_all=False)[0]
        if observation_times is not None:
            cols = observation_columns(times, observation_times)
            x, times = tf.gather(x, cols, axis=1), times[cols]
        if self._martingale_correction:
            x = x / tf.reduce_mean(x, axis=0, keepdims=True)
        forwards = tf.stack([tf.cast(self._equity.forward(float(v)), DTYPE) for v in times])
        return x * forwards[None, :]


class HestonSLVModel(EquityModel):
    """Heston SLV model instance: a :class:`HestonModel` (stochastic variance, held), a :class:`LocalVolatilityModel`
    (settings: the local vol is rebuilt from the market at every pricing) and the SLV settings (``mixing`` in
    ``(0, 1]``, ``n_nodes`` of the conditional-expectation smoother, ``max_leverage``, ``martingale_correction``).

    The leverage ``L = sigma_LV / sqrt(E[v | X])`` is built by :meth:`process` on the pricing paths (particle method),
    with the risk convention of :class:`HestonSLVProcess`: Heston parameters held, so :attr:`parameters` is empty and
    the greeks are the market greeks through the local vol and the forward (vega per node direct, no Jacobian).

    Example::

        model = tq.HestonSLVModel.calibrate(equity, mixing=0.5)           # or HestonSLVModel(tq.HestonModel(...))
        tq.VanillaMCPricer(model=model).price(option, market_env)
    """

    SETTINGS = {"mixing": 1.0, "n_nodes": 100, "max_leverage": 10.0, "martingale_correction": True}

    def __init__(self, heston: HestonModel, local_vol: LocalVolatilityModel = None, **settings):
        """Args:
            heston (HestonModel): Stochastic variance (calibrated or by hand).
            local_vol (LocalVolatilityModel, optional): Local vol settings (default settings if omitted).
            **settings: ``mixing``, ``n_nodes``, ``max_leverage``, ``martingale_correction``.

        Raises:
            ValueError: If ``mixing`` is not in ``(0, 1]`` or ``n_nodes < 2``.
        """
        super().__init__(**settings)
        if not 0.0 < float(self.settings["mixing"]) <= 1.0:
            raise ValueError(f"mixing must be in (0, 1], got {self.settings['mixing']}")
        if int(self.settings["n_nodes"]) < 2:
            raise ValueError("n_nodes must be >= 2")
        self.heston = heston
        self.local_vol = local_vol if local_vol is not None else LocalVolatilityModel()

    def process(self, equity) -> HestonSLVProcess:
        s = self.settings
        return HestonSLVProcess(equity, self.heston.process(equity), self.local_vol.process(equity),
                                float(s["mixing"]), int(s["n_nodes"]), bool(s["martingale_correction"]),
                                float(s["max_leverage"]))

    @classmethod
    def calibrate(cls, equity, initial: dict = None, local_vol_settings: dict = None, **settings) -> "HestonSLVModel":
        """Heston calibrated to the surface of *equity* (:meth:`HestonModel.calibrate`) + local vol settings.

        Args:
            equity (Equity): The underlying.
            initial (dict, optional): Starting parameters of the Heston search.
            local_vol_settings (dict, optional): Settings of the :class:`LocalVolatilityModel`.
            **settings: SLV settings (``mixing``, ...).
        """
        heston = HestonModel.calibrate(equity, initial=initial)
        model = cls(heston, LocalVolatilityModel.calibrate(equity, **(local_vol_settings or {})), **settings)
        model._record_calibration(equity, {}, None, report=heston.report)
        return model

    @classmethod
    def from_equity(cls, equity, maturity: Union[date, float] = None, strike: float = None,
                    **settings) -> HestonSLVProcess:
        """Process built from the market inside the caller's tape (:meth:`HestonSLVProcess.from_equity`)."""
        return HestonSLVProcess.from_equity(equity, maturity, strike, **settings)

    def to_frames(self) -> dict:
        frames = super().to_frames()
        frames.update({f"heston.{k}": v for k, v in self.heston.to_frames().items()})
        frames.update({f"local_vol.{k}": v for k, v in self.local_vol.to_frames().items()})
        return frames

    @classmethod
    def _from_values(cls, values: dict, settings: dict, frames: dict) -> "HestonSLVModel":
        sub = lambda prefix: {k[len(prefix):]: v for k, v in frames.items() if k.startswith(prefix)}
        return cls(HestonModel.from_frames(sub("heston.")), LocalVolatilityModel.from_frames(sub("local_vol.")),
                   **settings)
