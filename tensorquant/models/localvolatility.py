from datetime import date
from typing import Sequence, Union

import numpy as np
import tensorflow as tf
from scipy.interpolate import CubicSpline

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
# Re-gridding of the implied vol surface before Dupire
# ============================================================

def _strike_weights(k_nodes: np.ndarray, k_new: np.ndarray) -> np.ndarray:
    """Weights ``W [n_new, n_nodes]`` with ``iv(k_new) = W @ iv(k_nodes)``.

    Natural cubic spline in log-strike inside the node range; outside, linear in
    log-strike with the slope of the two edge nodes (the skew is continued). Both
    are linear in the node values, so ``W`` is built once by splining unit vectors.
    """
    x, y = np.log(k_nodes), np.log(k_new)
    n = len(x)
    W = np.empty((len(y), n))
    lo, hi = y < x[0], y > x[-1]
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        col = CubicSpline(x, e, bc_type="natural")(np.clip(y, x[0], x[-1]))
        col[lo] = e[0] + (e[1] - e[0]) / (x[1] - x[0]) * (y[lo] - x[0])
        col[hi] = e[-1] + (e[-1] - e[-2]) / (x[-1] - x[-2]) * (y[hi] - x[-1])
        W[:, j] = col
    return W


def _time_weights(t_nodes: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Weights ``W [n_new, n_nodes]`` with ``w(t_new) = W @ w(t_nodes)`` for total variance ``w = iv^2 T``.

    Linear in total variance between node maturities; flat implied vol before the
    first and after the last node (``w`` proportional to ``T``).
    """
    W = np.zeros((len(t_new), len(t_nodes)))
    for a, t in enumerate(t_new):
        if t <= t_nodes[0]:
            W[a, 0] = t / t_nodes[0]
        elif t >= t_nodes[-1]:
            W[a, -1] = t / t_nodes[-1]
        else:
            i = int(np.searchsorted(t_nodes, t)) - 1
            u = (t - t_nodes[i]) / (t_nodes[i + 1] - t_nodes[i])
            W[a, i], W[a, i + 1] = 1.0 - u, u
    return W


def regrid_implied_vol(
    iv_matrix,
    t_nodes: Sequence[float],
    k_nodes: Sequence[float],
    strike_range: tuple = (0.1, 5.0),
    n_strikes: int = 60,
    time_step: float = 1.0 / 12.0,
    iv_bounds: tuple = (0.05, 1.5),
):
    """Implied vol on a wide, fine (T, K) grid for Dupire, differentiable w.r.t. the node vols.

    Dupire needs dC/dT and d2C/dK2 on a grid that covers where paths go. The market
    grid is usually narrow (e.g. 60%-140% of spot) and coarse: paths outside it get the
    local vol of the edge nodes and the skew is cut, which underprices the wings and,
    at long maturities, the whole smile. Here the surface is re-gridded with linear
    operators on the node values (``tf.matmul``), so gradients reach every node:

    - strikes: ``n_strikes`` log-spaced points over ``strike_range`` times the central
      node strike ``sqrt(K_min K_max)``; natural cubic spline in log K inside the nodes,
      skew continued linearly in log K outside;
    - maturities: points before the first node (flat implied vol), the first node, then
      every ``time_step`` up to the last node; total variance linear in T.

    Args:
        iv_matrix: ``[nT, nK]`` node implied vols (e.g. ``VolatilitySurface.volatility_matrix``).
        t_nodes: ``[nT]`` node maturities (year fractions, increasing).
        k_nodes: ``[nK]`` node strikes (increasing).
        strike_range: Strike range as multiples of the central node strike.
        n_strikes: Number of strikes of the new grid.
        time_step: Maturity step of the new grid (years).
        iv_bounds: Floor and cap on the re-gridded implied vols.

    Returns:
        tuple: ``(iv_new [nT', nK'] tf.Tensor, t_new np.ndarray, k_new np.ndarray)``.
    """
    t_nodes, k_nodes = np.asarray(t_nodes, dtype=float), np.asarray(k_nodes, dtype=float)
    k_ref = np.sqrt(k_nodes[0] * k_nodes[-1])
    k_new = k_ref * np.exp(np.linspace(np.log(strike_range[0]), np.log(strike_range[1]), n_strikes))
    steps = np.arange(time_step, t_nodes[-1], time_step)
    # points before the first node (flat implied vol there): otherwise the first Dupire row,
    # used for every t < T_0, holds the forward vol between the first two nodes
    n_short = max(2, int(np.ceil(t_nodes[0] / time_step)))
    short = t_nodes[0] * np.arange(1, n_short) / n_short
    t_new = np.unique(np.concatenate([short, [t_nodes[0]], steps[steps > t_nodes[0]], [t_nodes[-1]]]))

    iv = tf.cast(iv_matrix, DTYPE)
    w_nodes = tf.square(iv) * tf.constant(t_nodes[:, None], DTYPE)                 # [nT, nK]
    w_t = tf.matmul(tf.constant(_time_weights(t_nodes, t_new), DTYPE), w_nodes)    # [nT', nK]
    iv_t = tf.sqrt(w_t / tf.constant(t_new[:, None], DTYPE))
    iv_new = tf.matmul(iv_t, tf.constant(_strike_weights(k_nodes, k_new).T, DTYPE))  # [nT', nK']
    return tf.clip_by_value(iv_new, iv_bounds[0], iv_bounds[1]), t_new, k_new


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
        maturity: Union[date, float],
        strike: float = None,
        dupire_eps: float = 1e-10,
        sigma_cap: float = 4.0,
        regrid: bool = True,
        strike_range: tuple = (0.1, 5.0),
        n_strikes: int = 60,
        time_step: float = 1.0 / 12.0,
    ) -> "LocalVolatilityModel":
        """Dupire model for one option on *equity*, from the market risk factors (differentiable).

        Implied vols are the surface nodes (``volatility_matrix``, a market variable), so
        gradients reach every node. With ``regrid`` (default) Dupire runs on the wide, fine
        grid of :func:`regrid_implied_vol` instead of the market nodes (TICKER1 snapshot,
        MC implied vols at the nodes vs market: 5Y ATM -4.7 vol points on the market grid,
        within 0.3 re-gridded). Spot and carry are flat up to *maturity*, as in
        :meth:`GeometricBrownianMotion.from_equity`: escrowed ``S* = S - PV(div)`` with
        ``q = repo`` (discrete dividends), or ``S`` with ``q = div_yield + repo``;
        ``r = -ln P(0,T)/T``. The forward is exact at *maturity* only (flat r, q).

        Args:
            equity (Equity): The underlying (``market_env.get_equity``).
            maturity (Union[date, float]): Option maturity.
            strike (float, optional): Unused (same signature as the GBM factory).
            dupire_eps (float): Numerical floor in the Dupire denominator.
            sigma_cap (float): Upper cap on the local vol.
            regrid (bool): Re-grid the implied vols before Dupire (see :func:`regrid_implied_vol`).
            strike_range, n_strikes, time_step: Re-gridding parameters.

        Returns:
            LocalVolatilityModel: Model whose inputs are market tensors.

        Raises:
            ValueError: If the underlying has no vol surface.
        """
        if equity.vol_surface is None:
            raise ValueError(f"No volatility surface for underlying '{equity.ticker}'")
        x0, r, q, _ = _equity_carry_inputs(equity, maturity)
        surface = equity.vol_surface
        iv, t_grid, k_grid = surface.volatility_matrix, surface.maturity, surface.strike
        if regrid:
            iv, t_grid, k_grid = regrid_implied_vol(iv, t_grid, k_grid, strike_range, n_strikes, time_step)
        return cls.from_implied_vol(
            iv, t_grid, k_grid, S0=x0, r=r, q=q, dupire_eps=dupire_eps, sigma_cap=sigma_cap,
        )

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
        # Local vol surface once, eagerly (differentiable w.r.t. self._C)
        sigma_TK = _dupire_local_vol(
            tf.cast(self._C, DTYPE),
            self._T_grid, self._K_grid,
            r=self._r, q=self._q,
            eps=self._eps, sig_cap=self._cap,
        )
        return _simulate_log_euler(sigma_TK, self._T_grid, self._K_grid, self._S0, self._r, self._q, t_grid, dw)

