from .stochasticprocess import StochasticProcess
import tensorflow as tf


# ============================================================
# Black-Scholes helpers  (used for calibration from IV surface)
# ============================================================

def _norm_cdf(x):
    """Standard-normal CDF via TF erf."""
    return 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(tf.constant(2.0, dtype=x.dtype))))


def _bs_call(S0, K, T, vol):
    """
    Black-Scholes call price (r = q = 0).  Supports broadcasting.

    Args:
        S0:  spot (scalar or tensor)
        K:   strike grid [nK] or broadcastable
        T:   maturity grid [nT] or broadcastable
        vol: implied-vol grid [nT, nK] or broadcastable

    Returns:
        call price — same shape as inputs after broadcasting
    """
    dtype = tf.float32
    T   = tf.maximum(tf.cast(T,   dtype), tf.constant(1e-12, dtype))
    vol = tf.maximum(tf.cast(vol, dtype), tf.constant(1e-12, dtype))
    S0  = tf.cast(S0, dtype)
    K   = tf.cast(K,  dtype)
    sigT = vol * tf.sqrt(T)
    d1   = (tf.math.log(S0 / K) + tf.constant(0.5, dtype) * vol * vol * T) / sigT
    d2   = d1 - sigT
    return S0 * _norm_cdf(d1) - K * _norm_cdf(d2)


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
    Second derivative d²C/dK² on a (T, K) grid (uniform K spacing assumed).

    Args:
        C: [nT, nK]
        K: [nK]

    Returns:
        d²C/dK² [nT, nK] — central differences inside, boundary copy at edges.
    """
    dK    = K[1] - K[0]
    d2mid = (C[:, 2:] - 2.0 * C[:, 1:-1] + C[:, :-2]) / (dK * dK)  # [nT, nK-2]
    return tf.concat([d2mid[:, 0:1], d2mid, d2mid[:, -1:]], axis=1)


def _dupire_local_vol(C, T, K, r=0.0, q=0.0, eps=1e-10, sig_cap=4.0):
    """
    Build the local volatility surface σ_loc(T, K) via Dupire's formula.

    For r = q = 0:
        σ²_loc = 2 · (∂C/∂T) / (K² · ∂²C/∂K²)

    All operations are pure TF — the result is differentiable w.r.t. C.

    Args:
        C:       [nT, nK]  call price matrix (tf.Variable or tf.Tensor)
        T:       [nT]      maturities
        K:       [nK]      strikes
        r, q:    scalars   risk-free rate / dividend yield
        eps:     float     numerical floor to avoid division by zero
        sig_cap: float     cap on σ_loc (for numerical robustness)

    Returns:
        sigma_loc [nT, nK]
    """
    dT  = _dC_dT(C, T)    # [nT, nK]
    d2K = _d2C_dK2(C, K)  # [nT, nK]

    K2    = tf.square(K)[None, :]                          # [1, nK]
    denom = tf.maximum(K2 * tf.maximum(d2K, eps), eps)
    sig2  = 2.0 * tf.maximum(dT, 0.0) / denom

    return tf.sqrt(tf.clip_by_value(sig2, 0.0, sig_cap))


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
        self._T_grid = tf.cast(tf.convert_to_tensor(T_grid), tf.float32)
        self._K_grid = tf.cast(tf.convert_to_tensor(K_grid), tf.float32)
        self._S0     = tf.cast(tf.convert_to_tensor(S0),     tf.float32)
        self._r      = tf.cast(tf.convert_to_tensor(r),      tf.float32)
        self._q      = tf.cast(tf.convert_to_tensor(q),      tf.float32)
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
        T_tf = tf.cast(tf.convert_to_tensor(T_grid), tf.float32)
        K_tf = tf.cast(tf.convert_to_tensor(K_grid), tf.float32)
        TT, KK = tf.meshgrid(T_tf, K_tf, indexing="ij")   # [nT, nK]

        C_surface = _bs_call(
            tf.cast(S0, tf.float32), KK, TT,
            tf.cast(iv_matrix, tf.float32),
        )
        return cls(C_surface, T_grid, K_grid, S0, r, q, dupire_eps, sigma_cap)

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
        t_v = tf.fill([tf.shape(x0)[0]], tf.cast(t0, tf.float32))
        sig = _bilinear_interp(self._sigma_TK, self._T_grid, self._K_grid, t_v, x0)
        return sig * x0 * tf.sqrt(tf.cast(dt, tf.float32))

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
            tf.cast(self._C, tf.float32),
            self._T_grid, self._K_grid,
            r=self._r, q=self._q,
            eps=self._eps, sig_cap=self._cap,
        )
        t_v = tf.cast(tf.broadcast_to(t, [tf.shape(tf.cast(S, tf.float32))[0]]), tf.float32)
        return _bilinear_interp(sigma_TK, self._T_grid, self._K_grid, t_v, tf.cast(S, tf.float32))

    # ------------------------------------------------------------------
    # Main simulation
    # ------------------------------------------------------------------

    @tf.function
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
        dw     = tf.cast(tf.convert_to_tensor(dw),     tf.float32)
        t_grid = tf.cast(tf.convert_to_tensor(t_grid), tf.float32)

        n_paths = tf.shape(dw)[0]
        n_steps = tf.shape(dw)[1]

        # Prepend t=0 so we can compute dt for the first step
        t_full = tf.concat([tf.zeros([1], tf.float32), t_grid], axis=0)  # [n_steps+1]

        # Precompute σ_loc surface once (differentiable w.r.t. self._C)
        sigma_TK = _dupire_local_vol(
            tf.cast(self._C, tf.float32),
            self._T_grid, self._K_grid,
            r=self._r, q=self._q,
            eps=self._eps, sig_cap=self._cap,
        )

        S     = tf.fill([n_paths], self._S0)
        paths = tf.TensorArray(dtype=tf.float32, size=n_steps)

        for k in tf.range(n_steps):
            t0 = t_full[k]
            dt = t_full[k + 1] - t_full[k]

            t_v = tf.fill([n_paths], t0)
            sig = _bilinear_interp(sigma_TK, self._T_grid, self._K_grid, t_v, S)

            # Log-Euler step
            drift = (self._r - self._q - 0.5 * tf.square(sig)) * dt
            diff  = sig * tf.sqrt(dt) * dw[:, k]
            S     = S * tf.exp(drift + diff)

            paths = paths.write(k, S)

        # [n_steps, n_paths] → [n_paths, n_steps]
        return tf.transpose(paths.stack(), perm=[1, 0])

