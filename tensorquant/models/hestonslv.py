"""Heston stochastic local volatility (SLV) equity model on the market forward.

The leverage ``L(t, x) = sigma_LV(t, x) / sqrt(E[v_t | ln X_t = x])`` has two parts:

- ``sigma_LV``: the local vol of the market surface, rebuilt from the market at every pricing (explicit and
  differentiable, as :class:`LocalVolatilityModel`);
- ``E[v | X]``: the conditional expectation of the variance, **calibrated** by a particle simulation (tent-kernel
  smoother on fixed nodes in ``ln X`` per time row) and stored on a grid, the model's parameter (like the Heston
  parameters of :class:`HestonModel`); the pricing reads it from the table.

Greeks: the pricing tape gives the direct market greeks (local vol and forward, including the move of each path on
the stored table) and the model greeks ``dV/dE``; :func:`par_greeks` maps the latter to the market by the **adjoint of
the calibration** (the particle simulation re-run on the stored nodes inside a tape, one backward pass with
``dV/dE`` as output gradient; for a portfolio the sum of the trades' ``dV/dE``, one pass). The class path
(:meth:`HestonSLVModel.from_equity`) calibrates inside the pricing tape and gives the same greeks in one pass. The
Heston parameters are held (SLV risk convention).
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from .brownian import observation_columns
from .equitymodel import EquityModel
from .heston import HestonModel, HestonProcess, _f64
from .localvolatility import LocalVolatilityModel, LocalVolatilityProcess, _rows_at
from .stochasticprocess import StochasticProcess
from ..numericalhandles.dtypes import DTYPE

SHRINK = 0.05            # sparse smoother nodes shrink to the mean variance with weight SHRINK * (particles per node)
VARIANCE_FLOOR = 1e-4    # floor on E[v | X] (a 1% vol): bounds the leverage
NODE_PAD = 0.01          # end nodes 1% of the range beyond the 0.1% / 99.9% quantiles (a node on a particle is a kink
                         # of the estimator at the calibration point: one-sided AAD slope)


def _quantiles(x: tf.Tensor, lo: float, hi: float) -> tuple:
    """Two empirical quantiles (order statistics of one sort; no gradient needed)."""
    last = tf.cast(tf.shape(x)[0] - 1, DTYPE)
    ordered = tf.sort(x)
    return ordered[tf.cast(tf.round(lo * last), tf.int32)], ordered[tf.cast(tf.round(hi * last), tf.int32)]


def _tent(log_x, nodes):
    """Node index ``j`` and weight ``w`` of node ``j + 1`` of each path on the equally spaced *nodes* (``w`` is
    differentiable in ``log_x``; outside the nodes the weight is clipped)."""
    n_nodes = tf.shape(nodes)[0]
    last = tf.cast(n_nodes - 1, DTYPE)
    span = tf.maximum(nodes[-1] - nodes[0], 1e-12)
    pos = tf.clip_by_value((log_x - nodes[0]) / span * last, 0.0, last)
    j = tf.clip_by_value(tf.cast(tf.floor(tf.stop_gradient(pos)), tf.int32), 0, n_nodes - 2)
    return j, pos - tf.cast(j, DTYPE)


def _read(values, j, w):
    return tf.gather(values, j) * (1.0 - w) + tf.gather(values, j + 1) * w


def _node_values(log_x, v, nodes, shrink):
    """Tent-kernel estimate of ``E[v | ln X]`` at *nodes*: ``(sum w v + c mean v) / (sum w + c)``."""
    n_nodes = tf.shape(nodes)[0]
    j, w = _tent(log_x, nodes)
    num = tf.math.unsorted_segment_sum(v * (1.0 - w), j, n_nodes) + tf.math.unsorted_segment_sum(v * w, j + 1, n_nodes)
    den = tf.math.unsorted_segment_sum(1.0 - w, j, n_nodes) + tf.math.unsorted_segment_sum(w, j + 1, n_nodes)
    return (num + shrink * tf.reduce_mean(v)) / (den + shrink)


def _conditional(values, nodes, log_x, bias_correction: bool):
    """``E[v | ln X]`` at the paths from a table row ``values = [fine (on nodes), coarse (on nodes[::2])]``: the fine
    read, or with *bias_correction* the Richardson combination ``(4 fine - coarse) / 3`` (the tent smoother's bias is
    ``O(h^2)`` in the node spacing ``h``: the combination removes its leading term)."""
    n_nodes = tf.shape(nodes)[0]
    fine = _read(values[:n_nodes], *_tent(log_x, nodes))
    if not bias_correction:
        return fine
    coarse = _read(values[n_nodes:], *_tent(log_x, nodes[::2]))
    return (4.0 * fine - coarse) / 3.0


def _local_vol_at(row, lv_y, log_x):
    """Local vol row (on ``lv_y``) at the paths' ``ln X``, linear, flat outside."""
    yc = tf.clip_by_value(log_x, lv_y[0], lv_y[-1])
    k = tf.clip_by_value(tf.searchsorted(lv_y, tf.stop_gradient(yc), side="right") - 1, 0, tf.shape(lv_y)[0] - 2)
    y0, y1 = tf.gather(lv_y, k), tf.gather(lv_y, k + 1)
    wy = (yc - y0) / (y1 - y0)
    return tf.gather(row, k) * (1.0 - wy) + tf.gather(row, k + 1) * wy


def _step(r, v, log_x, lev, zv, zx):
    """One step: Andersen QE for ``v'`` and, with ``L`` frozen on the step, ``ln X' = ln X + K0* + K1 v + K2 v' +
    sqrt(K3 v + K4 v') Z`` with per-path coefficients scaled by ``L`` and the exact martingale correction ``K0*``
    (fallback ``K0`` where the moment condition fails). ``r``: ``m0, e, s1, s2`` of the variance, ``dt,
    kappa theta, rho / xi, kappa rho / xi, 1 - rho^2``."""
    m0, e, s1, s2, dt, k_theta, r_x, k_r, q = (r[j] for j in range(9))
    m = m0 + e * v
    psi = (s1 * v + s2) / (m * m)
    pq = tf.minimum(psi, 1.5)          # quadratic branch inputs kept in range: finite gradients of the unused branch
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
    log_x = log_x + k0_star + k1 * v + k2 * v_new + tf.sqrt(tf.maximum(k3 * (v + v_new), 1e-300)) * zx
    return v_new, log_x


def _checkpointed(bias_correction: bool):
    """The calibration and pricing steps wrapped in ``tf.recompute_grad`` (check-pointing, Savine 2018, ch. 13): the
    tape keeps only the state ``(v, ln X)`` at each step and recomputes the step's intermediates in the backward pass.
    Same derivatives; without it reverse mode stores every intermediate of every step for every path (memory bound:
    the 1Y put, 50k paths x 252 steps, took 25-40 s with AAD, about 6 s with it)."""

    @tf.recompute_grad
    def calibration_step(coef, lv_row, lv_y, nk, max_leverage, shrink, shrink_coarse, v, log_x, z_v, z_x):
        values = tf.concat([_node_values(log_x, v, nk, shrink), _node_values(log_x, v, nk[::2], shrink_coarse)], 0)
        sigma = _local_vol_at(lv_row, lv_y, log_x)
        lev = tf.minimum(sigma / tf.sqrt(tf.maximum(_conditional(values, nk, log_x, bias_correction),
                                                    VARIANCE_FLOOR)), max_leverage)
        v_new, x_new = _step(coef, v, log_x, lev, z_v, z_x)
        return values, v_new, x_new

    @tf.recompute_grad
    def pricing_step(coef, lv_row, lv_y, e0, e1, n0, n1, a_i, max_leverage, v, log_x, z_v, z_x):
        cond = ((1.0 - a_i) * _conditional(e0, n0, log_x, bias_correction)
                + a_i * _conditional(e1, n1, log_x, bias_correction))
        sigma = _local_vol_at(lv_row, lv_y, log_x)
        lev = tf.minimum(sigma / tf.sqrt(tf.maximum(cond, VARIANCE_FLOOR)), max_leverage)
        v_new, x_new = _step(coef, v, log_x, lev, z_v, z_x)
        return v_new, x_new, lev

    return calibration_step, pricing_step


_STEPS = {flag: _checkpointed(flag) for flag in (False, True)}


@tf.function(reduce_retracing=True)
def _calibrate_expectation(rows, sigma_rows, lv_y, v0, n_particles, seed, nodes, given_nodes: bool, max_leverage,
                           bias_correction: bool):
    """Particle calibration of ``E[v | ln X]`` at the row times ``t_k`` (``rows [n, 9]``, ``sigma_rows [n, nY]``).

    At each row: the nodes (0.1% - 99.9% quantiles of ``ln X``, or the given ``nodes[k]``; held constant), the tent-
    kernel estimates ``E_j = (sum w v + c mean v) / (sum w + c)`` (``c = SHRINK n / nodes``) on the nodes and on every
    other node, the leverage of every particle read from them (:func:`_conditional`), then one step with the normals
    of seed ``(seed, k)`` (a longer horizon extends the table with the same first rows). Differentiable w.r.t. the
    local vol rows (through the particle positions).

    Returns:
        tuple: ``(nodes [n, n_nodes], E [n, n_nodes + n_coarse])`` (fine then coarse estimates).
    """
    step = _STEPS[bias_correction][0]
    n_rows, n_nodes = tf.shape(rows)[0], tf.shape(nodes)[1]
    coef = tf.TensorArray(DTYPE, size=n_rows).unstack(rows)
    lv_rows = tf.TensorArray(DTYPE, size=n_rows).unstack(sigma_rows)
    given = tf.TensorArray(DTYPE, size=n_rows).unstack(nodes)
    shrink = SHRINK * tf.cast(n_particles, DTYPE) / tf.cast(n_nodes, DTYPE)
    shrink_coarse = SHRINK * tf.cast(n_particles, DTYPE) / tf.cast((n_nodes + 1) // 2, DTYPE)
    unit = tf.linspace(tf.constant(0.0, DTYPE), tf.constant(1.0, DTYPE), n_nodes)
    v = tf.fill([n_particles], v0)
    log_x = tf.zeros([n_particles], DTYPE)
    out_nodes, out_e = tf.TensorArray(DTYPE, size=n_rows), tf.TensorArray(DTYPE, size=n_rows)
    for k in tf.range(n_rows):
        if given_nodes:
            nk = given.read(k)
        else:
            lo, hi = _quantiles(tf.stop_gradient(log_x), 0.001, 0.999)
            pad = NODE_PAD * (hi - lo)      # the quantiles are particles: no particle exactly on the end nodes
            nk = lo - pad + (hi - lo + 2.0 * pad) * unit
        z = tf.random.stateless_normal([n_particles, 2], seed=tf.stack([seed, k]), dtype=DTYPE)
        values, v, log_x = step(coef.read(k), lv_rows.read(k), lv_y, nk, max_leverage, shrink, shrink_coarse, v,
                                log_x, z[:, 0], z[:, 1])
        out_nodes, out_e = out_nodes.write(k, nk), out_e.write(k, values)
    return out_nodes.stack(), out_e.stack()


@tf.function(reduce_retracing=True)
def _simulate_slv(rows, sigma_rows, lv_y, v0, dw, e_lo, e_hi, nodes_lo, nodes_hi, a, max_leverage, keep_all: bool,
                  bias_correction: bool):
    """Pricing simulation of ``X`` (and, with ``keep_all``, ``v`` and ``L``) ``[n_paths, n_steps]`` on the stored
    table: at each step start ``E[v | X]`` is read at the paths' ``ln X`` on the two bracketing rows
    (``e_lo``/``nodes_lo``, ``e_hi``/``nodes_hi``, linear in time with weight ``a``), differentiable in ``ln X`` and in
    the table; leverage ``sigma_LV / sqrt(E)`` capped at ``max_leverage``; :func:`_step`."""
    step = _STEPS[bias_correction][1]
    n_paths, n_steps = tf.shape(dw)[0], tf.shape(dw)[1]
    coef = tf.TensorArray(DTYPE, size=n_steps).unstack(rows)
    lv_rows = tf.TensorArray(DTYPE, size=n_steps).unstack(sigma_rows)
    tables = [tf.TensorArray(DTYPE, size=n_steps).unstack(x) for x in (e_lo, e_hi, nodes_lo, nodes_hi)]
    weights = tf.TensorArray(DTYPE, size=n_steps).unstack(a)
    z_v = tf.TensorArray(DTYPE, size=n_steps).unstack(tf.transpose(dw[..., 0]))
    z_x = tf.TensorArray(DTYPE, size=n_steps).unstack(tf.transpose(dw[..., 1]))
    v = tf.fill([n_paths], v0)
    log_x = tf.zeros([n_paths], DTYPE)
    xs, vs, ls = (tf.TensorArray(DTYPE, size=n_steps) for _ in range(3))
    for i in tf.range(n_steps):
        v, log_x, lev = step(coef.read(i), lv_rows.read(i), lv_y, tables[0].read(i), tables[1].read(i),
                             tables[2].read(i), tables[3].read(i), weights.read(i), max_leverage, v, log_x,
                             z_v.read(i), z_x.read(i))
        xs = xs.write(i, tf.exp(log_x))
        if keep_all:                      # (unused outputs still cost gradient memory: only on request)
            vs, ls = vs.write(i, v), ls.write(i, lev)
    if keep_all:
        return tuple(tf.transpose(x.stack()) for x in (xs, vs, ls))
    return (tf.transpose(xs.stack()),)


def _heston_rows(heston: HestonProcess, mixing: float, dt: np.ndarray) -> tf.Tensor:
    """Per-step coefficients ``[n_steps, 9]`` of :func:`_step` (Heston parameters held out of the gradients)."""
    p = {k: tf.stop_gradient(v) for k, v in heston.parameters.items()}
    kappa, theta, rho = p["kappa"], p["theta"], p["rho"]
    xi = mixing * p["xi"]
    dt = _f64(dt)
    e = tf.exp(-kappa * dt)
    ones = tf.ones_like(dt)
    return tf.stack([theta * (1.0 - e), e, xi * xi * e * (1.0 - e) / kappa,
                     theta * xi * xi * tf.square(1.0 - e) / (2.0 * kappa), dt, kappa * theta * ones,
                     rho / xi * ones, kappa * rho / xi * ones, (1.0 - rho * rho) * ones], axis=1)


def calibrate_expectation(heston: HestonProcess, local_vol: LocalVolatilityProcess, horizon: float,
                          mixing: float = 1.0, n_nodes: int = 41, n_particles: int = 50_000,
                          steps_per_year: int = 50, seed: int = 0, max_leverage: float = 10.0,
                          nodes: Optional[np.ndarray] = None, bias_correction: bool = False,
                          differentiable: bool = True) -> dict:
    """Calibrate ``E[v | ln X]`` on the rows ``t_k = k / steps_per_year``, ``k = 0..ceil(horizon steps_per_year)``.

    Args:
        heston (HestonProcess): Stochastic variance (parameters held).
        local_vol (LocalVolatilityProcess): Local vol of the market (differentiable: run inside a tape for greeks).
        horizon (float): Last time needed (years).
        mixing (float): Vol-of-vol scale.
        n_nodes, n_particles, steps_per_year, seed: Smoother nodes, particles, rows per year, seed of the normals.
        max_leverage (float): Cap on the leverage.
        nodes (np.ndarray, optional): ``[n_rows, n_nodes]`` nodes to use (default: quantiles of the particles).
        bias_correction (bool): Richardson combination of the estimates on the nodes and on every other node
            (:func:`_conditional`; ``n_nodes`` odd).
        differentiable (bool): ``False`` holds the local vol out of the caller's tape (the price table of the
            smoothing for risk: value only).

    Returns:
        dict: ``times [n_rows]``, ``nodes [n_rows, n_nodes]`` (NumPy), ``values`` (float64 tensor ``[n_rows,
        n_nodes]``; with *bias_correction* ``[n_rows, n_nodes + (n_nodes + 1) // 2]``: fine then coarse estimates) and
        ``bias_correction``.
    """
    n_rows = int(math.ceil(horizon * steps_per_year - 1e-9)) + 1
    times = np.arange(n_rows) / float(steps_per_year)
    if nodes is not None:
        nodes = np.asarray(nodes, dtype=float)
        if nodes.shape[0] < n_rows or nodes.shape[1] != n_nodes:
            raise ValueError(f"nodes must cover {n_rows} rows of {n_nodes} nodes, got {nodes.shape}")
        nodes = nodes[:n_rows]
    sigma_ty, lv_t, lv_y = local_vol.local_vol_grid
    sigma_rows = _rows_at(sigma_ty, lv_t, tf.constant(times, DTYPE))
    if not differentiable:
        sigma_rows, lv_y = tf.stop_gradient(sigma_rows), tf.stop_gradient(lv_y)
    rows = _heston_rows(heston, mixing, np.full(n_rows, 1.0 / steps_per_year))
    given = nodes is not None
    node_input = tf.constant(nodes if given else np.zeros((n_rows, n_nodes)), DTYPE)
    out_nodes, values = _calibrate_expectation(rows, sigma_rows, lv_y, tf.stop_gradient(heston.parameters["v0"]),
                                               int(n_particles), tf.constant(int(seed), tf.int32), node_input, given,
                                               tf.constant(max_leverage, DTYPE), bool(bias_correction))
    if not bias_correction:
        values = values[:, :n_nodes]                  # the coarse estimates serve the correction only
    return {"times": times, "nodes": out_nodes.numpy(), "values": values, "bias_correction": bool(bias_correction)}


class HestonSLVProcess(StochasticProcess):
    """Heston stochastic local volatility process on the market forward: a :class:`HestonSLVModel` bound to an
    :class:`Equity` (or built from the market inside the caller's tape by :meth:`from_equity`).

    ``S_t = F(t) X_t`` with ``F(t) = equity.forward(t)`` and::

        dX / X = L(t, X) sqrt(v) dW_1,   dv = kappa (theta - v) dt + eta xi sqrt(v) dW_2,   d<W_1, W_2> = rho dt,

    ``v`` the Heston variance (vol of vol scaled by the mixing ``eta``) and ``L^2(t, x) = sigma_LV^2(t, x) /
    E[v_t | X_t = x]``: ``sigma_LV`` the local vol of the market surface, ``E`` the calibrated table
    (:func:`calibrate_expectation`), read at each step start at the paths' ``ln X`` (linear on the nodes and in time).
    By Gyongy's theorem the marginals are those of the local vol model.

    Smoothing for risk (Savine 2018): with a price table (``table["price_values"]``, the Richardson-corrected
    calibration on the same nodes) the paths are ``X_risk + stop_gradient(X_price - X_risk)``: the value of the
    unbiased estimator, the derivative of the smooth one (whose adjoint through the calibration matches bumps; the
    derivative of the corrected estimator follows the particle noise and blows up along the calibration).
    """

    def __init__(self, equity, heston: HestonProcess, local_vol: LocalVolatilityProcess, table: dict,
                 mixing: float = 1.0, martingale_correction: bool = True, max_leverage: float = 10.0):
        """Built by :meth:`HestonSLVModel.process` or :meth:`from_equity`.

        Args:
            equity (Equity): The underlying (its ``forward`` drives the dynamics).
            heston (HestonProcess): Stochastic variance parameters.
            local_vol (LocalVolatilityProcess): Local vol of the market surface.
            table (dict): ``times``, ``nodes``, ``values`` (risk table) of :func:`calibrate_expectation`, optional
                ``price_values`` (price table, Richardson-corrected on the same nodes).
            mixing (float): Vol-of-vol scale ``eta`` in ``(0, 1]``.
            martingale_correction (bool): Rescale ``X_t`` to sample mean 1 (MC forward = ``F(t)``).
            max_leverage (float): Cap on the leverage.

        Raises:
            ValueError: If ``mixing`` is not in ``(0, 1]``.
        """
        if not 0.0 < mixing <= 1.0:
            raise ValueError(f"mixing must be in (0, 1], got {mixing}")
        self._equity = equity
        self.heston, self.local_vol, self.table = heston, local_vol, table
        self._mixing = float(mixing)
        self._martingale_correction = martingale_correction
        self._max_leverage = float(max_leverage)
        self._S0 = tf.cast(equity.spot, DTYPE)

    @classmethod
    def from_equity(cls, equity, maturity: Union[date, float] = None, strike: float = None,
                    heston: HestonProcess = None, mixing: float = 1.0, n_nodes: int = 41, max_leverage: float = 10.0,
                    martingale_correction: bool = True, n_particles: int = 50_000, steps_per_year: int = 50,
                    seed: int = 0, horizon: float = None, bias_correction: bool = True) -> "HestonSLVProcess":
        """Heston calibrated to the surface (:meth:`HestonProcess.from_equity`, unless *heston* is given), local vol
        of the same surface and ``E[v | X]`` calibrated up to the maturity, all inside the caller's tape (greeks
        through the leverage calibration in one backward pass; Heston parameters held).

        Args:
            equity (Equity): The underlying.
            maturity: Horizon of the table (date or year fraction; default the last maturity of the surface).
            strike: Unused (factory signature of the MC models).
            heston (HestonProcess, optional): Given stochastic variance (no Heston calibration).
            mixing, n_nodes, max_leverage, martingale_correction, n_particles, steps_per_year, seed,
                bias_correction: As :class:`HestonSLVModel`.
            horizon (float, optional): Horizon of the table in years (overrides *maturity*).
        """
        heston = heston if heston is not None else HestonProcess.from_equity(equity)
        local_vol = LocalVolatilityProcess.from_equity(equity)
        if horizon is None:
            horizon = (float(equity.year_fraction(maturity)) if maturity is not None
                       else float(np.max(equity.vol_surface.maturity)))
        table = calibrate_expectation(heston, local_vol, horizon, mixing, n_nodes, n_particles, steps_per_year, seed,
                                      max_leverage)
        if bias_correction:
            table["price_values"] = calibrate_expectation(heston, local_vol, horizon, mixing, n_nodes, n_particles,
                                                          steps_per_year, seed, max_leverage, table["nodes"], True,
                                                          differentiable=False)["values"]
        return cls(equity, heston, local_vol, table, mixing, martingale_correction, max_leverage)

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

    def simulate(self, t_grid, dw, keep_all: bool = True) -> tuple:
        """``(X, v, L)`` ``[n_paths, n_steps]``: forward ratio, variance and the leverage used on each step
        (``(X,)`` only with ``keep_all=False``).

        Args:
            t_grid: ``[n_steps]`` increasing times (> 0).
            dw: ``[n_paths, n_steps, 2]`` standard normals.
            keep_all (bool): Also return the variance and leverage paths.
        """
        times = np.asarray(tf.convert_to_tensor(t_grid), dtype=float)
        if times[0] <= 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("t_grid must be increasing and > 0")
        dw = tf.cast(dw, DTYPE)
        if len(dw.shape) != 3 or int(dw.shape[-1]) != 2:
            raise ValueError("dw must be [n_paths, n_steps, 2]")
        starts = np.concatenate([[0.0], times[:-1]])
        t_rows = np.asarray(self.table["times"], dtype=float)
        last = len(t_rows) - 1
        if starts[-1] > t_rows[-1] + 1e-12:
            raise ValueError(f"the SLV table covers up to {t_rows[-1]:.4f}y, the grid starts a step at "
                             f"{starts[-1]:.4f}y: recalibrate the leverage with a longer horizon")
        lo = np.clip(np.searchsorted(t_rows, starts, side="right") - 1, 0, last)
        hi = np.minimum(lo + 1, last)
        a = np.where(hi > lo, (starts - t_rows[lo]) / np.maximum(t_rows[hi] - t_rows[lo], 1e-300), 0.0)
        nodes = tf.constant(self.table["nodes"], DTYPE)
        # rows picked by one-hot products (dense gradient to the table, not IndexedSlices)
        pick = lambda idx: tf.constant(np.eye(len(t_rows))[idx], DTYPE)
        sigma_ty, lv_t, lv_y = self.local_vol.local_vol_grid
        sigma_rows = _rows_at(sigma_ty, lv_t, tf.constant(starts, DTYPE))
        rows = _heston_rows(self.heston, self._mixing, np.diff(np.concatenate([[0.0], times])))
        v0 = tf.stop_gradient(self.heston.parameters["v0"])

        def run(values, sigma, y, corrected):
            return _simulate_slv(rows, sigma, y, v0, dw, pick(lo) @ values, pick(hi) @ values, tf.gather(nodes, lo),
                                 tf.gather(nodes, hi), tf.constant(a, DTYPE), tf.constant(self._max_leverage, DTYPE),
                                 keep_all, corrected)

        risk = run(self.table["values"], sigma_rows, lv_y, False)
        price_values = self.table.get("price_values")
        if price_values is None:
            return risk
        price = run(tf.stop_gradient(tf.convert_to_tensor(price_values, DTYPE)), tf.stop_gradient(sigma_rows),
                    tf.stop_gradient(lv_y), True)
        return tuple(r + tf.stop_gradient(p - r) for r, p in zip(risk, price))

    def evolve(self, t_grid, dw, observation_times=None) -> tf.Tensor:
        """Simulate spot paths ``S_t = F(t) X_t`` on the stored ``E[v | X]`` table.

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
    (settings: the local vol is rebuilt from the market at every pricing), the calibrated ``E[v | X]`` table (the
    parameter ``expectation``) and the SLV settings:

    - ``mixing`` in ``(0, 1]`` (vol-of-vol scale), ``max_leverage``, ``martingale_correction``;
    - the leverage calibration: ``n_particles``, ``steps_per_year`` (table rows), ``n_nodes`` of the smoother
      (41: narrower nodes bias less but make the adjoint of the calibration unstable), ``bias_correction``
      (smoothing for risk: a second, Richardson-corrected table on the same nodes gives the price, the 41-node
      table the greeks; odd ``n_nodes``), ``seed``, ``horizon`` (years; default the last maturity of the surface).

    Two speeds: :meth:`HestonModel.calibrate` now and then, :meth:`calibrate_leverage` whenever the surface moves.
    Greeks: model greeks ``dV/dE`` on :attr:`parameters`, mapped to the market by :func:`par_greeks` (``equity``
    required: adjoint of the leverage calibration).

    Example::

        model = tq.HestonSLVModel.calibrate(equity, mixing=0.5)           # Heston + leverage
        model.calibrate_leverage(equity)                                  # later: the leverage only
        tq.VanillaMCPricer(model=model).price(option, market_env, autodiff=True)
    """

    SETTINGS = {"mixing": 1.0, "n_nodes": 41, "max_leverage": 10.0, "martingale_correction": True,
                "n_particles": 50_000, "steps_per_year": 50, "seed": 0, "horizon": None, "bias_correction": True}

    def __init__(self, heston: HestonModel, local_vol: LocalVolatilityModel = None, **settings):
        """Args:
            heston (HestonModel): Stochastic variance (calibrated or by hand).
            local_vol (LocalVolatilityModel, optional): Local vol settings (default settings if omitted).
            **settings: See the class docstring.

        Raises:
            ValueError: If ``mixing`` is not in ``(0, 1]``, ``n_nodes < 2`` or ``n_particles < n_nodes``.
        """
        super().__init__(**settings)
        s = self.settings
        if not 0.0 < float(s["mixing"]) <= 1.0:
            raise ValueError(f"mixing must be in (0, 1], got {s['mixing']}")
        if int(s["n_nodes"]) < 2:
            raise ValueError("n_nodes must be >= 2")
        if int(s["n_particles"]) < int(s["n_nodes"]):
            raise ValueError("n_particles must be >= n_nodes")
        if s["bias_correction"] and int(s["n_nodes"]) % 2 == 0:
            raise ValueError("bias_correction needs an odd n_nodes (the coarse estimate is on every other node)")
        self.heston = heston
        self.local_vol = local_vol if local_vol is not None else LocalVolatilityModel()
        self.leverage_table: Optional[dict] = None
        self._expectation: Optional[tf.Variable] = None

    @property
    def parameters(self) -> dict:
        """``{"expectation": E[v | X] table [n_rows, n_nodes]}`` once the leverage is calibrated."""
        return {} if self._expectation is None else {"expectation": self._expectation}

    def _table(self, equity, nodes=None, horizon=None, corrected=False) -> dict:
        s = self.settings
        if horizon is None:
            horizon = s["horizon"] if s["horizon"] is not None else float(np.max(equity.vol_surface.maturity))
        return calibrate_expectation(self.heston.process(equity), self.local_vol.process(equity), float(horizon),
                                     float(s["mixing"]), int(s["n_nodes"]), int(s["n_particles"]),
                                     int(s["steps_per_year"]), int(s["seed"]), float(s["max_leverage"]), nodes,
                                     corrected, differentiable=not corrected)

    def calibrate_leverage(self, equity, nodes: np.ndarray = None) -> "HestonSLVModel":
        """Calibrate the ``E[v | X]`` table on the market of *equity* (Heston parameters held): the step to repeat when
        the surface moves.

        Args:
            equity (Equity): The underlying.
            nodes (np.ndarray, optional): Smoother nodes to reuse (e.g. the stored ones, for a recalibration on the same
                particles and nodes).

        Returns:
            HestonSLVModel: *self*.
        """
        table = self._table(equity, nodes)
        self.leverage_table = {"times": table["times"], "nodes": table["nodes"]}
        if self.settings["bias_correction"]:
            self.leverage_table["price_values"] = self._table(equity, table["nodes"], corrected=True)["values"].numpy()
        values = table["values"].numpy()
        if self._expectation is not None and tuple(self._expectation.shape) == values.shape:
            self._expectation.assign(values)
        else:
            self._expectation = self._variable(values, "expectation")
        self._record_calibration(equity, {}, None, report=self.report if self.report is not None else self.heston.report)
        return self

    def process(self, equity) -> HestonSLVProcess:
        """Bind to *equity* on the stored table (calibrated here first if the model has none)."""
        if self._expectation is None:
            self.calibrate_leverage(equity)
        s = self.settings
        table = {**self.leverage_table, "values": self._expectation}
        return HestonSLVProcess(equity, self.heston.process(equity), self.local_vol.process(equity), table,
                                float(s["mixing"]), bool(s["martingale_correction"]), float(s["max_leverage"]))

    def map_model_greeks(self, model_greeks: dict, equity=None) -> dict:
        """``dV/dE`` mapped to the market by the adjoint of the leverage calibration: the particle simulation of the
        risk table re-run on the stored nodes inside a tape (up to the last row with a non-zero ``dV/dE``), one
        backward pass with ``dV/dE`` as output gradient.

        Raises:
            ValueError: If *equity* is missing or is not the calibrated underlying.
        """
        g = model_greeks.get("expectation")
        if g is None or self._expectation is None:
            return {}
        if equity is None:
            raise ValueError("HestonSLVModel maps its model greeks by the adjoint of the leverage calibration: "
                             "pass equity to par_greeks")
        if self.ticker is not None and equity.ticker != self.ticker:
            raise ValueError(f"HestonSLVModel calibrated on '{self.ticker}', mapped on '{equity.ticker}'")
        g = np.asarray(g, dtype=float)
        used = np.nonzero(np.any(g != 0.0, axis=1))[0]
        factors = equity.risk_factors()
        if len(used) == 0:
            return {k: [np.zeros(v.shape) for v in vs] for k, vs in factors.items()}
        n = int(used[-1]) + 1
        horizon = (n - 1) / float(self.settings["steps_per_year"])
        with tf.GradientTape() as tape:
            values = self._table(equity, self.leverage_table["nodes"], horizon)["values"]
        grads = tape.gradient(values, factors, output_gradients=tf.constant(g[:n], DTYPE))
        return {k: [np.zeros(v.shape) if x is None else x.numpy() for x, v in zip(xs, factors[k])]
                for k, xs in grads.items()}

    @classmethod
    def calibrate(cls, equity, initial: dict = None, local_vol_settings: dict = None, **settings) -> "HestonSLVModel":
        """Heston calibrated to the surface of *equity* (:meth:`HestonModel.calibrate`), then the leverage
        (:meth:`calibrate_leverage`).

        Args:
            equity (Equity): The underlying.
            initial (dict, optional): Starting parameters of the Heston search.
            local_vol_settings (dict, optional): Settings of the :class:`LocalVolatilityModel`.
            **settings: SLV settings (``mixing``, ...).
        """
        heston = HestonModel.calibrate(equity, initial=initial)
        model = cls(heston, LocalVolatilityModel.calibrate(equity, **(local_vol_settings or {})), **settings)
        model.report = heston.report
        return model.calibrate_leverage(equity)

    @classmethod
    def from_equity(cls, equity, maturity: Union[date, float] = None, strike: float = None,
                    **settings) -> HestonSLVProcess:
        """Process calibrated inside the caller's tape (Heston and leverage, :meth:`HestonSLVProcess.from_equity`)."""
        return HestonSLVProcess.from_equity(equity, maturity, strike, **settings)

    def to_frames(self) -> dict:
        frames = super().to_frames()
        frames.update({f"heston.{k}": v for k, v in self.heston.to_frames().items()})
        frames.update({f"local_vol.{k}": v for k, v in self.local_vol.to_frames().items()})
        if self.leverage_table is not None:
            nodes = self.leverage_table["nodes"]
            frames["leverage_nodes"] = pd.DataFrame(
                [{"row": k, "time": float(self.leverage_table["times"][k]), "node": j, "log_x": float(nodes[k, j])}
                 for k in range(nodes.shape[0]) for j in range(nodes.shape[1])])
            price = self.leverage_table.get("price_values")
            if price is not None:
                frames["leverage_price_values"] = pd.DataFrame(
                    [{"row": k, "index": j, "value": float(price[k, j])}
                     for k in range(price.shape[0]) for j in range(price.shape[1])])
        return frames

    @classmethod
    def _from_values(cls, values: dict, settings: dict, frames: dict) -> "HestonSLVModel":
        sub = lambda prefix: {k[len(prefix):]: v for k, v in frames.items() if k.startswith(prefix)}
        model = cls(HestonModel.from_frames(sub("heston.")), LocalVolatilityModel.from_frames(sub("local_vol.")),
                    **settings)
        grid = frames.get("leverage_nodes")
        if grid is not None and "expectation" in values:
            n_rows, n_nodes = int(grid["row"].max()) + 1, int(grid["node"].max()) + 1
            grid = grid.sort_values(["row", "node"])
            model.leverage_table = {"times": grid.groupby("row")["time"].first().to_numpy(),
                                    "nodes": grid["log_x"].to_numpy().reshape(n_rows, n_nodes)}
            price = frames.get("leverage_price_values")
            if price is not None:
                model.leverage_table["price_values"] = (price.sort_values(["row", "index"])["value"].to_numpy()
                                                        .reshape(n_rows, -1))
            model._expectation = cls._variable(np.asarray(values["expectation"]).reshape(n_rows, n_nodes),
                                               "expectation")
        return model
