from datetime import date
from typing import Sequence, Union

import numpy as np
import tensorflow as tf

from .equitymodel import EquityModel
from .stochasticprocess import StochasticProcess
from .brownian import observation_columns
from ..numericalhandles.dtypes import DTYPE


# ============================================================
# Local vol from total implied variance (Gatheral), forward moneyness
# ============================================================

def _time_operators(t_nodes: np.ndarray, t_new: np.ndarray):
    """``w(t) = A @ w_nodes`` and ``dw/dt = B @ w_nodes``: total variance linear in T between node
    maturities at fixed forward moneyness; flat implied vol before the first / after the last node."""
    A, B = np.zeros((len(t_new), len(t_nodes))), np.zeros((len(t_new), len(t_nodes)))
    for a, t in enumerate(t_new):
        if t <= t_nodes[0]:
            A[a, 0], B[a, 0] = t / t_nodes[0], 1.0 / t_nodes[0]
        elif t >= t_nodes[-1]:
            A[a, -1], B[a, -1] = t / t_nodes[-1], 1.0 / t_nodes[-1]
        else:
            i = int(np.searchsorted(t_nodes, t, side="right")) - 1
            h = t_nodes[i + 1] - t_nodes[i]
            u = (t - t_nodes[i]) / h
            A[a, i], A[a, i + 1] = 1.0 - u, u
            B[a, i], B[a, i + 1] = -1.0 / h, 1.0 / h
    return A, B


def gatheral_local_vol(
    iv_matrix,
    t_nodes: Sequence[float],
    log_moneyness,
    y_grid,
    time_step: float = 1.0 / 24.0,
    max_wing_slope: float = 1.9,
    g_floor: float = 1e-3,
    sigma_cap: float = 2.0,
):
    """Local vol on a (t, y) grid from the implied vol nodes, ``y = ln(K / F(T))``.

    Gatheral's formula in total implied variance ``w = iv^2 T``::

        sigma_loc^2 = (dw/dT) / g,
        g = 1 - (y/w) w_y + 1/4 (-1/4 - 1/w + y^2/w^2) w_y^2 + 1/2 w_yy

    (``g`` is the butterfly-arbitrage density factor, ``dw/dT`` at fixed y the calendar
    one). Per node maturity, ``w`` is a natural cubic spline in y through the node strikes,
    continued linearly outside (Lee: wing slope clipped to ``max_wing_slope`` < 2); in T it
    is linear between nodes at fixed y. Derivatives come from the spline (no finite
    differences on call prices, which are tiny at short maturities and in the far wings).
    Differentiable w.r.t. the node vols and the node log-moneyness (i.e. the forwards).

    Args:
        iv_matrix: ``[nT, nK]`` node implied vols.
        t_nodes: ``[nT]`` node maturities (year fractions, increasing).
        log_moneyness: ``[nT, nK]`` ``ln(K_j / F(T_i))`` (tensor; increasing along K).
        y_grid: ``[nY]`` forward log-moneyness grid of the local vol.
        time_step: Time step of the local vol grid (points before the first node included).
        max_wing_slope: Clip on ``|dw/dy|`` used outside the strike nodes.
        g_floor: Floor on ``g`` (cells below it have butterfly arbitrage).
        sigma_cap: Cap on the local vol.

    Returns:
        tuple: ``(sigma [nT', nY] tf.Tensor, t_grid np.ndarray [nT'], diagnostics dict)``;
            diagnostics on the data region (y covered by the strike nodes of every maturity):
            ``min_g``, ``min_dw_dt``, ``arbitrage_fraction`` (share of cells with ``g < g_floor`` or
            ``dw/dT <= 0``); ``wing_arbitrage_fraction``: same share in the extrapolated wings.
    """
    from ..numericalhandles.interpolation import natural_cubic_spline

    t_nodes = np.asarray(t_nodes, dtype=float)
    iv = tf.cast(iv_matrix, DTYPE)
    y = tf.cast(y_grid, DTYPE)
    rows = [natural_cubic_spline(log_moneyness[i], tf.square(iv[i]) * t_nodes[i], y, max_wing_slope)
            for i in range(len(t_nodes))]
    W, W_y, W_yy = (tf.stack([r[j] for r in rows]) for j in range(3))       # [nT, nY] each

    n_short = max(2, int(np.ceil(t_nodes[0] / time_step)))
    steps = np.arange(time_step, t_nodes[-1] + 1e-12, time_step)
    t_grid = np.unique(np.concatenate([t_nodes[0] * np.arange(1, n_short + 1) / n_short,
                                       steps[steps > t_nodes[0]], t_nodes]))
    A, B = (tf.constant(m, DTYPE) for m in _time_operators(t_nodes, t_grid))
    w = tf.maximum(tf.matmul(A, W), 1e-12)
    w_y, w_yy, w_t = tf.matmul(A, W_y), tf.matmul(A, W_yy), tf.matmul(B, W)
    g = 1.0 - y / w * w_y + 0.25 * (-0.25 - 1.0 / w + tf.square(y / w)) * tf.square(w_y) + 0.5 * w_yy
    sig2 = tf.maximum(w_t, 1e-12) / tf.maximum(g, g_floor)
    sigma = tf.sqrt(tf.clip_by_value(sig2, 1e-12, sigma_cap ** 2))

    # data region: y covered by the strike nodes of every maturity (outside, w is extrapolated and the
    # wings of two maturities can cross: floored, reported apart as wing_arbitrage_fraction)
    lm = tf.stop_gradient(tf.cast(log_moneyness, DTYPE))
    in_data = tf.logical_and(y >= tf.reduce_max(lm[:, 0]), y <= tf.reduce_min(lm[:, -1]))
    bad = tf.logical_or(g < g_floor, w_t <= 0.0)

    def share(mask):
        cells = tf.boolean_mask(tf.cast(bad, DTYPE), mask, axis=1)
        return float(tf.reduce_mean(cells)) if int(tf.size(cells)) else 0.0

    g_data, wt_data = tf.boolean_mask(g, in_data, axis=1), tf.boolean_mask(w_t, in_data, axis=1)
    diagnostics = {
        "min_g": float(tf.reduce_min(g_data)),
        "min_dw_dt": float(tf.reduce_min(wt_data)),
        "arbitrage_fraction": share(in_data),
        "wing_arbitrage_fraction": share(tf.logical_not(in_data)),
    }
    return sigma, t_grid, diagnostics


def _interp_ty(sigma_ty, t_grid, y_grid, t, y):
    """Bilinear ``sigma(t, y)`` at a scalar time ``t`` for a vector ``y`` (clamped to the grid)."""
    n_t, n_y = tf.shape(t_grid)[0], tf.shape(y_grid)[0]
    tc = tf.clip_by_value(t, t_grid[0], t_grid[-1])
    yc = tf.clip_by_value(y, y_grid[0], y_grid[-1])
    i = tf.clip_by_value(tf.searchsorted(t_grid, tf.reshape(tc, [1]), side="right")[0] - 1, 0, n_t - 2)
    j = tf.clip_by_value(tf.searchsorted(y_grid, yc, side="right") - 1, 0, n_y - 2)
    wt = (tc - t_grid[i]) / (t_grid[i + 1] - t_grid[i])
    y0, y1 = tf.gather(y_grid, j), tf.gather(y_grid, j + 1)
    wy = (yc - y0) / (y1 - y0)
    r0, r1 = sigma_ty[i], sigma_ty[i + 1]
    s0 = tf.gather(r0, j) * (1.0 - wy) + tf.gather(r0, j + 1) * wy
    s1 = tf.gather(r1, j) * (1.0 - wy) + tf.gather(r1, j + 1) * wy
    return s0 * (1.0 - wt) + s1 * wt


def _rows_at(sigma_ty, lv_t, t_start):
    """Local vol rows ``sigma(t_k, .)`` at every step start, linear in t: ``[n_steps, nY]`` (dense ops)."""
    tc = tf.clip_by_value(t_start, lv_t[0], lv_t[-1])
    i = tf.clip_by_value(tf.searchsorted(lv_t, tc, side="right") - 1, 0, tf.shape(lv_t)[0] - 2)
    t0, t1 = tf.gather(lv_t, i), tf.gather(lv_t, i + 1)
    w = ((tc - t0) / (t1 - t0))[:, None]
    return tf.gather(sigma_ty, i) * (1.0 - w) + tf.gather(sigma_ty, i + 1) * w


@tf.function(reduce_retracing=True)
def _simulate_forward_ratio_rows(sig_rows, lv_y, t_grid, dw):
    """Log-Euler loop on ``ln X`` with one local vol row per step (rows read from a TensorArray).

    The rows are not captured tensors gathered inside the loop: a ``gather`` on a captured tensor
    has a sparse (IndexedSlices) gradient that the while-loop accumulates by concatenation, which
    made the backward pass quadratic in the number of steps (16 s instead of 2 s at 260 steps).
    """
    n_paths, n_steps = tf.shape(dw)[0], tf.shape(dw)[1]
    t_full = tf.concat([tf.zeros([1], DTYPE), t_grid], axis=0)
    rows = tf.TensorArray(DTYPE, size=n_steps).unstack(sig_rows)
    normals = tf.TensorArray(DTYPE, size=n_steps).unstack(tf.transpose(dw))
    steps = tf.TensorArray(DTYPE, size=n_steps).unstack(t_full[1:] - t_full[:-1])
    n_y = tf.shape(lv_y)[0]
    log_x = tf.zeros([n_paths], DTYPE)
    paths = tf.TensorArray(dtype=DTYPE, size=n_steps)
    for k in tf.range(n_steps):
        row, dt = rows.read(k), steps.read(k)
        yc = tf.clip_by_value(log_x, lv_y[0], lv_y[-1])
        j = tf.clip_by_value(tf.searchsorted(lv_y, yc, side="right") - 1, 0, n_y - 2)
        y0, y1 = tf.gather(lv_y, j), tf.gather(lv_y, j + 1)
        wy = (yc - y0) / (y1 - y0)
        sig = tf.gather(row, j) * (1.0 - wy) + tf.gather(row, j + 1) * wy
        log_x = log_x - 0.5 * tf.square(sig) * dt + sig * tf.sqrt(dt) * normals.read(k)
        paths = paths.write(k, tf.exp(log_x))
    return tf.transpose(paths.stack(), perm=[1, 0])


def _simulate_forward_ratio(sigma_ty, lv_t, lv_y, t_grid, dw):
    """Paths of ``X_t = S_t / F(t)`` ``[n_paths, n_steps]``: ``d ln X = -sigma^2/2 dt + sigma dW``,
    ``sigma = sigma_loc(t, ln X)`` bilinear in (t, y) (log-Euler, exact martingale step by step)."""
    t_grid = tf.cast(t_grid, DTYPE)
    t_start = tf.concat([tf.zeros([1], DTYPE), t_grid[:-1]], axis=0)
    return _simulate_forward_ratio_rows(_rows_at(sigma_ty, lv_t, t_start), lv_y, t_grid, dw)


# ============================================================
# LocalVolatilityModel
# ============================================================

class LocalVolatilityProcess(StochasticProcess):
    """Local volatility process on the market forward (Gatheral's formula, forward moneyness): the
    :class:`LocalVolatilityModel` bound to an :class:`Equity`.

    ``S_t = F(t) X_t`` with ``F(t) = equity.forward(t)`` and
    ``d ln X = -sigma^2/2 dt + sigma dW``, ``sigma = sigma_loc(t, ln X)`` from
    :func:`gatheral_local_vol` on the market vol nodes. Built by :meth:`LocalVolatilityModel.process` (or
    :meth:`from_equity`) inside the pricing tape; everything is differentiable w.r.t. the vol nodes, spot, repo,
    dividends and discount curve.
    """

    def __init__(self, equity, sigma_ty, lv_t, lv_y, martingale_correction: bool = True, arbitrage: dict = None):
        """Use :meth:`from_equity`.

        Args:
            equity (Equity): The underlying (its ``forward`` drives the dynamics).
            sigma_ty: ``[nT, nY]`` local vol on the (t, y) grid.
            lv_t: ``[nT]`` times of the grid.
            lv_y: ``[nY]`` forward log-moneyness of the grid.
            martingale_correction (bool): Rescale ``X_t`` to sample mean 1 (MC forward = F(t)).
            arbitrage (dict, optional): Static-arbitrage diagnostics of the surface.
        """
        self._equity = equity
        self._sigma_ty = tf.cast(sigma_ty, DTYPE)
        self._lv_t = tf.constant(np.asarray(lv_t, dtype=float), DTYPE)
        self._lv_y = tf.constant(np.asarray(lv_y, dtype=float), DTYPE)
        self._S0 = tf.cast(equity.spot, DTYPE)
        self._martingale_correction = martingale_correction
        self.arbitrage = arbitrage or {}

    @classmethod
    def from_equity(
        cls,
        equity,
        maturity: Union[date, float] = None,
        strike: float = None,
        n_y: int = 300,
        y_range: tuple = (np.log(0.05), np.log(5.0)),
        time_step: float = 1.0 / 96.0,
        max_wing_slope: float = 1.9,
        g_floor: float = 1e-3,
        martingale_correction: bool = True,
    ) -> "LocalVolatilityProcess":
        """Local vol process of *equity* on the market forward (differentiable w.r.t. the market).

        Simulates ``X_t = S_t / F(t)`` with ``F(t) = equity.forward(t)`` (spot, repo, dividends,
        discount curve) and returns ``S_t = F(t) X_t``: the forward is the market one at every
        date (discrete dividends act as proportional drops in the dynamics). The local vol
        comes from :func:`gatheral_local_vol` on the market vol nodes in forward moneyness
        ``y = ln(K / F(T))``, so gradients reach the vol nodes, spot, repo, dividends and curve.

        If the vol surface has static arbitrage where it has data (``g < g_floor`` or
        ``dw/dT <= 0``), no local vol can reprice it: a warning is issued. Diagnostics are on
        ``model.arbitrage`` (see :func:`gatheral_local_vol`).

        Args:
            equity (Equity): The underlying (``market_env.get_equity``).
            maturity, strike: Unused (same factory signature as the GBM).
            n_y (int): Points of the forward log-moneyness grid.
            y_range (tuple): Range of that grid (``ln(K/F)``).
            time_step (float): Time step of the local vol grid.
            max_wing_slope (float): Lee wing-slope clip on ``|dw/dy|`` outside the strike nodes.
            g_floor (float): Floor on the butterfly factor ``g``.
            martingale_correction (bool): Rescale the simulated ``X_t`` to sample mean 1 at each
                time, so the MC forward equals ``F(t)`` exactly (removes the forward noise).

        Returns:
            LocalVolatilityProcess: Process in forward mode.

        Raises:
            ValueError: If the underlying has no vol surface.
        """
        import warnings
        from ..timehandles.utils import Settings
        from datetime import timedelta

        if equity.vol_surface is None:
            raise ValueError(f"No volatility surface for underlying '{equity.ticker}'")
        surface = equity.vol_surface
        t_nodes = np.asarray(surface.maturity, dtype=float)
        eval_date = Settings.evaluation_date
        f_nodes = tf.stack([
            tf.cast(equity.forward(eval_date + timedelta(days=max(1, int(round(t * 365))))), DTYPE)
            for t in t_nodes
        ])
        log_k = tf.math.log(tf.constant(surface.strike, DTYPE))
        log_moneyness = log_k[None, :] - tf.math.log(f_nodes)[:, None]
        y_grid = np.linspace(y_range[0], y_range[1], n_y)
        sigma_ty, lv_t, diagnostics = gatheral_local_vol(
            surface.volatility_matrix, t_nodes, log_moneyness, y_grid,
            time_step=time_step, max_wing_slope=max_wing_slope, g_floor=g_floor,
        )
        if diagnostics["arbitrage_fraction"] > 0.0:
            warnings.warn(
                f"Vol surface of '{equity.ticker}' has static arbitrage on "
                f"{diagnostics['arbitrage_fraction']:.1%} of the local vol grid where it has data (min g "
                f"{diagnostics['min_g']:+.3f}, min dw/dT {diagnostics['min_dw_dt']:+.3g}): the local vol "
                "is floored there and cannot reprice the surface exactly.",
                stacklevel=2,
            )
        return cls(equity, sigma_ty, lv_t, y_grid, martingale_correction, diagnostics)

    @property
    def local_vol_grid(self) -> tuple:
        """``(sigma [nT, nY], t [nT], y [nY])``: the local vol on its (time, forward log-moneyness) grid."""
        return self._sigma_ty, self._lv_t, self._lv_y

    def _forward(self, t) -> tf.Tensor:
        """Market forward ``F(t)`` (forward mode), ``t`` in years."""
        return tf.cast(self._equity.forward(float(t)), DTYPE)

    # ------------------------------------------------------------------
    # StochasticProcess interface
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Number of state dimensions (1: spot)."""
        return 1

    def initial_values(self) -> tf.Tensor:
        """Spot ``S_0``."""
        return self._S0

    def sigma_loc(self, t, S) -> tf.Tensor:
        """Local vol at ``(t, S)`` pairs (``y = ln(S / F(t))``, bilinear on the grid).

        Args:
            t: ``[n]`` or scalar times.
            S: ``[n]`` spots.

        Returns:
            tf.Tensor: ``[n]`` local vols.
        """
        S = tf.reshape(tf.cast(S, DTYPE), [-1])
        t_v = tf.broadcast_to(tf.cast(t, DTYPE), tf.shape(S)).numpy()
        y = tf.math.log(S / tf.stack([self._forward(float(v)) for v in t_v]))
        return tf.stack([_interp_ty(self._sigma_ty, self._lv_t, self._lv_y, tf.constant(t_v[i], DTYPE), y[i:i + 1])[0]
                         for i in range(len(t_v))])

    def evolve(self, t_grid, dw, observation_times=None) -> tf.Tensor:
        """Simulate spot paths ``S_t = F(t) X_t`` (log-Euler on ``ln X``).

        Args:
            t_grid: ``[n_steps]`` increasing times (year fractions), without 0.
            dw: ``[n_paths, n_steps]`` standard normals.
            observation_times: Optional subset of ``t_grid``: only those columns are returned and the
                market forward is computed only there.

        Returns:
            tf.Tensor: ``[n_paths, n_steps]`` (or ``[n_paths, len(observation_times)]``) spots.
        """
        dw = tf.cast(tf.convert_to_tensor(dw), DTYPE)
        t_grid = tf.cast(tf.convert_to_tensor(t_grid), DTYPE)
        x = _simulate_forward_ratio(self._sigma_ty, self._lv_t, self._lv_y, t_grid, dw)
        times = t_grid.numpy()
        if observation_times is not None:          # forwards (and the correction) only where observed
            cols = observation_columns(times, observation_times)
            x, times = tf.gather(x, cols, axis=1), times[cols]
        if self._martingale_correction:
            x = x / tf.reduce_mean(x, axis=0, keepdims=True)
        forwards = tf.stack([self._forward(t) for t in times])
        return x * forwards[None, :]


class LocalVolatilityModel(EquityModel):
    """Local volatility model (Dupire-Gatheral on the market forward): settings only, no calibrated parameters.

    The local vol is a function of the market vol surface, rebuilt by :meth:`process` from the vol nodes of the
    equity at every pricing (inside the pricing tape): the vega per node comes straight from the AAD, already a par
    greek, and :attr:`jacobian` stays ``None``.

    Settings: ``n_y`` (points of the forward log-moneyness grid), ``y_range`` (its range, ``ln(K/F)``), ``time_step``,
    ``max_wing_slope`` (Lee clip outside the strike nodes), ``g_floor`` (floor on the butterfly factor),
    ``martingale_correction`` (sample mean of ``X_t`` rescaled to 1). See :meth:`LocalVolatilityProcess.from_equity`.

    Example::

        model = tq.LocalVolatilityModel()                                  # or LocalVolatilityModel(n_y=400)
        tq.VanillaMCPricer(model=model).price(option, market_env)
    """

    SETTINGS = {"n_y": 300, "y_range": [float(np.log(0.05)), float(np.log(5.0))], "time_step": 1.0 / 96.0,
                "max_wing_slope": 1.9, "g_floor": 1e-3, "martingale_correction": True}

    def process(self, equity) -> LocalVolatilityProcess:
        s = self.settings
        return LocalVolatilityProcess.from_equity(
            equity, n_y=int(s["n_y"]), y_range=tuple(s["y_range"]), time_step=float(s["time_step"]),
            max_wing_slope=float(s["max_wing_slope"]), g_floor=float(s["g_floor"]),
            martingale_correction=bool(s["martingale_correction"]))

    @classmethod
    def calibrate(cls, equity, **settings) -> "LocalVolatilityModel":
        """Nothing to fit: records the settings and the underlying (the local vol is rebuilt at every pricing)."""
        model = cls(**settings)
        model._record_calibration(equity, {}, None)
        return model

    @classmethod
    def from_equity(cls, equity, maturity: Union[date, float] = None, strike: float = None,
                    **settings) -> LocalVolatilityProcess:
        """Local vol process of *equity* (``maturity`` and ``strike`` unused: factory signature of the pricers)."""
        return cls(**settings).process(equity)
