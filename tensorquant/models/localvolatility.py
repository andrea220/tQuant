from datetime import date
from typing import Sequence, Union

import numpy as np
import tensorflow as tf

from .stochasticprocess import StochasticProcess
from .brownian import _equity_carry_inputs
from ..numericalhandles.dtypes import DTYPE


# ============================================================
# Black-Scholes helpers  (used for calibration from IV surface)
# ============================================================

def _norm_cdf(x):
    """Standard-normal CDF via TF erf."""
    return 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(tf.constant(2.0, dtype=x.dtype))))


def _bs_call(S0, K, T, vol, r=0.0, q=0.0):
    """
    Black-Scholes call price with continuous rate r and dividend yield q.
    Supports broadcasting over (T, K) grids.

    Args:
        S0:  spot (scalar or tensor)
        K:   strike grid [nK] or broadcastable
        T:   maturity grid [nT] or broadcastable
        vol: implied-vol grid [nT, nK] or broadcastable
        r:   continuously-compounded risk-free rate (scalar)
        q:   continuous dividend yield (scalar)

    Returns:
        discounted call price C = S·e^{-qT}·N(d1) - K·e^{-rT}·N(d2)
    """
    dtype = DTYPE
    T   = tf.maximum(tf.cast(T,   dtype), tf.constant(1e-12, dtype))
    vol = tf.maximum(tf.cast(vol, dtype), tf.constant(1e-12, dtype))
    S0  = tf.cast(S0, dtype)
    K   = tf.cast(K,  dtype)
    r   = tf.cast(r,  dtype)
    q   = tf.cast(q,  dtype)
    sigT = vol * tf.sqrt(T)
    d1   = (tf.math.log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / sigT
    d2   = d1 - sigT
    return (
        S0 * tf.exp(-q * T) * _norm_cdf(d1)
        - K * tf.exp(-r * T) * _norm_cdf(d2)
    )


# ============================================================
# Dupire helpers  (pure TF ops — fully differentiable)
# ============================================================

def _dC_dT(C, T):
    """
    Partial derivative dC/dT on a (T, K) grid.

    Args:
        C: [nT, nK]  call prices
        T: [nT]      maturities (year fractions)

    Returns:
        dC/dT [nT, nK] — forward/backward at boundaries, central inside.
    """
    dt_f = T[1:] - T[:-1]                                         # [nT-1]

    d0   = (C[1, :]  - C[0, :])  / dt_f[0]                       # forward  at i=0
    dn   = (C[-1, :] - C[-2, :]) / dt_f[-1]                      # backward at i=nT-1

    slopes_f = (C[2:, :]   - C[1:-1, :]) / dt_f[1:,  None]       # [nT-2, nK]
    slopes_b = (C[1:-1, :] - C[:-2, :])  / dt_f[:-1, None]       # [nT-2, nK]
    w_f = dt_f[:-1, None]
    w_b = dt_f[1:,  None]
    dmid = (slopes_f * w_f + slopes_b * w_b) / (w_f + w_b)       # weighted central

    return tf.concat([d0[None, :], dmid, dn[None, :]], axis=0)


def _d2C_dK2(C, K):
    """
    Second derivative d²C/dK² on a (T, K) grid — non-uniform K spacing.

    Uses the 3-point non-uniform central difference formula for interior nodes:
        d²C/dK²|_i = 2 [ C_{i-1}/(h1·(h1+h2)) - C_i/(h1·h2) + C_{i+1}/(h2·(h1+h2)) ]
    where h1 = K_i - K_{i-1}, h2 = K_{i+1} - K_i.

    Boundary values are extrapolated from the nearest interior estimate.

    Args:
        C: [nT, nK]
        K: [nK]

    Returns:
        d²C/dK² [nT, nK]
    """
    h1 = K[1:-1] - K[:-2]   # [nK-2]
    h2 = K[2:]   - K[1:-1]  # [nK-2]
    d2mid = 2.0 * (
        C[:, :-2] / (h1 * (h1 + h2))
        - C[:, 1:-1] / (h1 * h2)
        + C[:, 2:] / (h2 * (h1 + h2))
    )  # [nT, nK-2]
    return tf.concat([d2mid[:, 0:1], d2mid, d2mid[:, -1:]], axis=1)


def _dC_dK(C, K):
    """
    First derivative dC/dK on a (T, K) grid — non-uniform K spacing.

    Interior: weighted central difference on non-uniform grid.
    Boundaries: one-sided (forward/backward).

    Args:
        C: [nT, nK]
        K: [nK]

    Returns:
        dC/dK [nT, nK]
    """
    # Interior: (C_{i+1} - C_{i-1}) / (K_{i+1} - K_{i-1})
    dK_span = K[2:] - K[:-2]                               # [nK-2]
    dmid    = (C[:, 2:] - C[:, :-2]) / dK_span[None, :]   # [nT, nK-2]
    # Boundaries
    d0 = (C[:, 1:2] - C[:, 0:1]) / (K[1:2] - K[0:1])
    dn = (C[:, -1:]  - C[:, -2:-1]) / (K[-1:] - K[-2:-1])
    return tf.concat([d0, dmid, dn], axis=1)


def _dupire_local_vol(C, T, K, r=0.0, q=0.0, eps=1e-10, sig_cap=4.0):
    """
    Build the local volatility surface σ_loc(T, K) via the full Dupire formula.

    For discounted call prices C(T,K) = e^{-rT}·E[(S_T - K)^+]:

        σ²_loc = [∂C/∂T + (r-q)·K·∂C/∂K + q·C] / [½·K²·∂²C/∂K²]

    All operations are pure TF — the result is differentiable w.r.t. C.

    Args:
        C:       [nT, nK]  discounted call price matrix (tf.Variable or tf.Tensor)
        T:       [nT]      maturities (year fractions)
        K:       [nK]      strikes (non-uniform grid supported)
        r:       scalar    risk-free rate
        q:       scalar    continuous dividend yield (repo included)
        eps:     float     numerical floor to avoid division by zero
        sig_cap: float     cap on σ_loc (for numerical robustness)

    Returns:
        sigma_loc [nT, nK]
    """
    r = tf.cast(r, DTYPE)
    q = tf.cast(q, DTYPE)

    dT  = _dC_dT(C, T)   # [nT, nK]
    dK  = _dC_dK(C, K)   # [nT, nK]
    d2K = _d2C_dK2(C, K) # [nT, nK]

    K_row = K[None, :]                                          # [1, nK]
    numerator = dT + (r - q) * K_row * dK + q * C
    denom     = tf.maximum(0.5 * tf.square(K_row) * tf.maximum(d2K, eps), eps)
    sig2      = tf.maximum(numerator, 0.0) / denom

    # floor inside the sqrt: sqrt'(0) is infinite and 0 * inf = nan in the gradient
    return tf.sqrt(tf.clip_by_value(sig2, 1e-12, sig_cap))


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


@tf.function(reduce_retracing=True)
def _simulate_forward_ratio(sigma_ty, lv_t, lv_y, t_grid, dw):
    """Paths of ``X_t = S_t / F(t)`` ``[n_paths, n_steps]``: ``d ln X = -sigma^2/2 dt + sigma dW``,
    ``sigma = sigma_loc(t, ln X)`` (log-Euler, exact martingale step by step)."""
    n_paths, n_steps = tf.shape(dw)[0], tf.shape(dw)[1]
    t_full = tf.concat([tf.zeros([1], DTYPE), t_grid], axis=0)
    log_x = tf.zeros([n_paths], DTYPE)
    paths = tf.TensorArray(dtype=DTYPE, size=n_steps)
    for k in tf.range(n_steps):
        dt = t_full[k + 1] - t_full[k]
        sig = _interp_ty(sigma_ty, lv_t, lv_y, t_full[k], log_x)
        log_x = log_x - 0.5 * tf.square(sig) * dt + sig * tf.sqrt(dt) * dw[:, k]
        paths = paths.write(k, tf.exp(log_x))
    return tf.transpose(paths.stack(), perm=[1, 0])


def _bilinear_interp(sigma_TK, T_grid, K_grid, t, S):
    """
    Differentiable bilinear interpolation of σ_loc(T, K) evaluated at (t, S).

    Args:
        sigma_TK: [nT, nK]
        T_grid:   [nT]
        K_grid:   [nK]
        t:        [nPaths] or scalar — current time for each path
        S:        [nPaths]           — current spot for each path

    Returns:
        sigma(t, S): [nPaths]
    """
    nT = tf.shape(T_grid)[0]
    nK = tf.shape(K_grid)[0]

    t_c = tf.clip_by_value(t, T_grid[0],  T_grid[-1])
    S_c = tf.clip_by_value(S, K_grid[0], K_grid[-1])

    i = tf.clip_by_value(
        tf.cast(tf.searchsorted(T_grid, t_c, side="right"), tf.int32) - 1,
        0, nT - 2,
    )
    j = tf.clip_by_value(
        tf.cast(tf.searchsorted(K_grid, S_c, side="right"), tf.int32) - 1,
        0, nK - 2,
    )

    T0 = tf.gather(T_grid, i);     T1 = tf.gather(T_grid, i + 1)
    K0 = tf.gather(K_grid, j);     K1 = tf.gather(K_grid, j + 1)

    wt = (t_c - T0) / tf.maximum(T1 - T0, 1e-12)
    wk = (S_c - K0) / tf.maximum(K1 - K0, 1e-12)

    s00 = tf.gather_nd(sigma_TK, tf.stack([i,     j    ], axis=1))
    s01 = tf.gather_nd(sigma_TK, tf.stack([i,     j + 1], axis=1))
    s10 = tf.gather_nd(sigma_TK, tf.stack([i + 1, j    ], axis=1))
    s11 = tf.gather_nd(sigma_TK, tf.stack([i + 1, j + 1], axis=1))

    s0 = s00 * (1.0 - wk) + s01 * wk
    s1 = s10 * (1.0 - wk) + s11 * wk
    return s0 * (1.0 - wt) + s1 * wt


@tf.function(reduce_retracing=True)
def _simulate_log_euler(sigma_TK, T_grid, K_grid, S0, r, q, t_grid, dw):
    """Log-Euler paths ``[n_paths, n_steps]`` with local vol ``sigma_TK`` (bilinear in (t, S)).

    Module-level ``tf.function`` on tensors only: compiled once per input signature
    (not per model instance) and differentiable w.r.t. ``sigma_TK``, ``S0``, ``r``, ``q``.
    """
    n_paths = tf.shape(dw)[0]
    n_steps = tf.shape(dw)[1]
    t_full = tf.concat([tf.zeros([1], DTYPE), t_grid], axis=0)  # [n_steps+1]
    S = tf.fill([n_paths], S0)
    paths = tf.TensorArray(dtype=DTYPE, size=n_steps)
    for k in tf.range(n_steps):
        t0 = t_full[k]
        dt = t_full[k + 1] - t_full[k]
        sig = _bilinear_interp(sigma_TK, T_grid, K_grid, tf.fill([n_paths], t0), S)
        S = S * tf.exp((r - q - 0.5 * tf.square(sig)) * dt + sig * tf.sqrt(dt) * dw[:, k])
        paths = paths.write(k, S)
    return tf.transpose(paths.stack(), perm=[1, 0])


# ============================================================
# LocalVolatilityModel
# ============================================================

class LocalVolatilityModel(StochasticProcess):
    """
    One-dimensional local volatility (Dupire) equity model.

    The SDE is:

        dS_t = (r - q) S_t dt  +  σ_loc(t, S_t) S_t dW_t

    where σ_loc is obtained by applying Dupire's formula to a call-price
    matrix C[nT, nK].  All operations are pure TensorFlow, so the entire
    pipeline (calibration → simulation) is differentiable w.r.t. C and
    therefore w.r.t. any implied-vol parameters that feed into C.

    Construction
    ------------
    Direct from call-price matrix::

        model = LocalVolatilityModel(C_surface, T_grid, K_grid, S0)

    From an implied-vol surface (convenience classmethod)::

        model = LocalVolatilityModel.from_implied_vol(iv_matrix, T_grid, K_grid, S0)

    Usage
    -----
    paths = model.evolve(t_grid, dw)   # [n_paths, n_steps]

    For AAD, pass a tf.Variable as C_surface (or iv_matrix) and wrap
    model.evolve inside a tf.GradientTape.
    """

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    def __init__(
        self,
        C_surface,
        T_grid,
        K_grid,
        S0,
        r=0.0,
        q=0.0,
        dupire_eps=1e-10,
        sigma_cap=4.0,
    ):
        """
        Args:
            C_surface:  [nT, nK] call price matrix.  Pass a tf.Variable to
                        enable AAD differentiation w.r.t. prices.
            T_grid:     [nT] maturities (year fractions).
            K_grid:     [nK] strikes.
            S0:         Initial spot level (scalar).
            r:          Risk-free rate (scalar or tf.Variable).
            q:          Dividend yield (scalar or tf.Variable).
            dupire_eps: Numerical floor in the Dupire denominator.
            sigma_cap:  Upper cap on σ_loc (for numerical robustness).
        """
        self._C      = C_surface
        self._T_grid = tf.cast(tf.convert_to_tensor(T_grid), DTYPE)
        self._K_grid = tf.cast(tf.convert_to_tensor(K_grid), DTYPE)
        self._S0     = tf.cast(tf.convert_to_tensor(S0),     DTYPE)
        self._r      = tf.cast(tf.convert_to_tensor(r),      DTYPE)
        self._q      = tf.cast(tf.convert_to_tensor(q),      DTYPE)
        self._eps    = dupire_eps
        self._cap    = sigma_cap

    @classmethod
    def from_implied_vol(
        cls,
        iv_matrix,
        T_grid,
        K_grid,
        S0,
        r=0.0,
        q=0.0,
        dupire_eps=1e-10,
        sigma_cap=4.0,
    ):
        """
        Build a LocalVolatilityModel from an implied-volatility surface.

        The call-price matrix is computed internally via the Black-Scholes
        formula (r = q = 0 convention for Dupire) and then passed to the
        standard constructor.

        Args:
            iv_matrix:  [nT, nK] implied volatility surface (tf.Variable for
                        AAD differentiation w.r.t. vol parameters).
            T_grid:     [nT] maturities (year fractions).
            K_grid:     [nK] strikes.
            S0:         Initial spot level (scalar).
            r, q:       Risk-free rate / dividend yield.
            dupire_eps: Numerical floor in the Dupire denominator.
            sigma_cap:  Upper cap on σ_loc.

        Returns:
            LocalVolatilityModel instance.

        Example::

            TT, KK = tf.meshgrid(T_grid, K_grid, indexing="ij")
            iv_matrix = tf.clip_by_value(
                0.18 + 0.03 * tf.sqrt(TT) - 0.10 * tf.math.log(KK / S0)
                     + 0.35 * tf.square(tf.math.log(KK / S0)),
                0.05, 1.0,
            )
            model = LocalVolatilityModel.from_implied_vol(
                iv_matrix, T_grid, K_grid, S0=100.0
            )
        """
        T_tf = tf.cast(tf.convert_to_tensor(T_grid), DTYPE)
        K_tf = tf.cast(tf.convert_to_tensor(K_grid), DTYPE)
        TT, KK = tf.meshgrid(T_tf, K_tf, indexing="ij")   # [nT, nK]

        C_surface = _bs_call(
            tf.cast(S0, DTYPE), KK, TT,
            tf.cast(iv_matrix, DTYPE),
            r=tf.cast(r, DTYPE),
            q=tf.cast(q, DTYPE),
        )
        return cls(C_surface, T_grid, K_grid, S0, r, q, dupire_eps, sigma_cap)

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
    ) -> "LocalVolatilityModel":
        """Local vol model of *equity* on the market forward (differentiable w.r.t. the market).

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
            LocalVolatilityModel: Model in forward mode.

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
        model = cls.__new__(cls)
        model._mode = "forward"
        model._equity = equity
        model._sigma_ty = sigma_ty
        model._lv_t = tf.constant(lv_t, DTYPE)
        model._lv_y = tf.constant(y_grid, DTYPE)
        model._S0 = tf.cast(equity.spot, DTYPE)
        model._martingale_correction = martingale_correction
        model.arbitrage = diagnostics
        return model

    def _forward(self, t) -> tf.Tensor:
        """Market forward ``F(t)`` (forward mode), ``t`` in years."""
        return tf.cast(self._equity.forward(float(t)), DTYPE)

    # ------------------------------------------------------------------
    # StochasticProcess interface
    # ------------------------------------------------------------------

    def size(self):
        """Returns the number of state dimensions (1 — spot only)."""
        return 1

    def initial_values(self):
        """Returns S0 as a scalar tf.Tensor."""
        return self._S0

    def drift(self, t0, x0, dt):
        """
        GBM-like drift term:  (r - q) · S · dt

        Args:
            t0: current time (unused — drift is time-homogeneous)
            x0: current spot [n_paths]
            dt: time increment (scalar)

        Returns:
            drift increment [n_paths]
        """
        return (self._r - self._q) * x0 * dt

    def diffusion(self, t0, x0, dt):
        """
        Local-vol diffusion term:  σ_loc(t0, S) · S · √dt

        Requires the precomputed sigma surface stored in self._sigma_TK
        (populated automatically by evolve).  Calling this method outside
        of evolve will raise a RuntimeError.

        Args:
            t0: current time [n_paths] or scalar
            x0: current spot [n_paths]
            dt: time increment (scalar)

        Returns:
            diffusion increment [n_paths]
        """
        if not hasattr(self, "_sigma_TK"):
            raise RuntimeError(
                "diffusion() requires the precomputed local-vol surface. "
                "Call evolve() to simulate paths — it manages the surface internally."
            )
        t_v = tf.fill([tf.shape(x0)[0]], tf.cast(t0, DTYPE))
        sig = _bilinear_interp(self._sigma_TK, self._T_grid, self._K_grid, t_v, x0)
        return sig * x0 * tf.sqrt(tf.cast(dt, DTYPE))

    def sigma_loc(self, t, S):
        """
        Evaluate σ_loc(t, S) at arbitrary (t, S) pairs using bilinear
        interpolation on the Dupire surface.

        The surface is computed once from self._C on the first call and
        cached until the call-price matrix changes.

        Args:
            t: [n_paths] or scalar — time
            S: [n_paths]           — spot

        Returns:
            σ_loc(t, S): [n_paths]
        """
        if getattr(self, "_mode", None) == "forward":
            S = tf.cast(S, DTYPE)
            t = tf.cast(t, DTYPE)
            t_v = tf.broadcast_to(t, tf.shape(S))
            y = tf.math.log(S / tf.stack([self._forward(float(v)) for v in t_v.numpy()]))
            return tf.stack([_interp_ty(self._sigma_ty, self._lv_t, self._lv_y, t_v[i], y[i:i + 1])[0]
                             for i in range(int(tf.shape(S)[0]))])
        sigma_TK = _dupire_local_vol(
            tf.cast(self._C, DTYPE),
            self._T_grid, self._K_grid,
            r=self._r, q=self._q,
            eps=self._eps, sig_cap=self._cap,
        )
        t_v = tf.cast(tf.broadcast_to(t, [tf.shape(tf.cast(S, DTYPE))[0]]), DTYPE)
        return _bilinear_interp(sigma_TK, self._T_grid, self._K_grid, t_v, tf.cast(S, DTYPE))

    # ------------------------------------------------------------------
    # Main simulation
    # ------------------------------------------------------------------

    def evolve(self, t_grid, dw):
        """
        Simulate paths using a log-Euler (Milstein-order-0) scheme with
        Dupire local volatility.  Interface is consistent with GBM in
        brownian.py — paths are returned at every observation time in t_grid.

        All operations are pure TF — wrap inside tf.GradientTape for AAD.

        Args:
            t_grid: [n_steps] observation times (year fractions), e.g.
                    tf.linspace(0.0, 1.0, 52)[1:] for weekly steps to T=1.
            dw:     [n_paths, n_steps] standard-normal increments.

        Returns:
            paths [n_paths, n_steps]  — spot at each observation time
            (does NOT include S0 at t=0).

        Example::

            n_paths, n_steps = 20_000, 252
            t_grid = tf.linspace(0.0, 1.0, n_steps + 1)[1:]          # skip t=0
            dw     = tf.random.normal([n_paths, n_steps])
            paths  = model.evolve(t_grid, dw)
        """
        dw = tf.cast(tf.convert_to_tensor(dw), DTYPE)
        t_grid = tf.cast(tf.convert_to_tensor(t_grid), DTYPE)
        if getattr(self, "_mode", None) == "forward":
            x = _simulate_forward_ratio(self._sigma_ty, self._lv_t, self._lv_y, t_grid, dw)
            if self._martingale_correction:
                x = x / tf.reduce_mean(x, axis=0, keepdims=True)
            forwards = tf.stack([self._forward(t) for t in t_grid.numpy()])
            return x * forwards[None, :]
        # Local vol surface once, eagerly (differentiable w.r.t. self._C)
        sigma_TK = _dupire_local_vol(
            tf.cast(self._C, DTYPE),
            self._T_grid, self._K_grid,
            r=self._r, q=self._q,
            eps=self._eps, sig_cap=self._cap,
        )
        return _simulate_log_euler(sigma_TK, self._T_grid, self._K_grid, self._S0, self._r, self._q, t_grid, dw)

