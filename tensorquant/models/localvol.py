from .stochasticprocess import StochasticProcess
import tensorflow as tf


# ============================================================
# Dupire helpers (pure TF ops — fully differentiable)
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
    dt_f = T[1:] - T[:-1]                               # [nT-1]

    d0 = (C[1, :] - C[0, :]) / dt_f[0]                  # forward  at i=0
    dn = (C[-1, :] - C[-2, :]) / dt_f[-1]               # backward at i=nT-1

    slopes_f = (C[2:, :]   - C[1:-1, :]) / dt_f[1:, None]   # [nT-2, nK]
    slopes_b = (C[1:-1, :] - C[:-2, :])  / dt_f[:-1, None]  # [nT-2, nK]
    w_f = dt_f[:-1, None]
    w_b = dt_f[1:, None]
    dmid = (slopes_f * w_f + slopes_b * w_b) / (w_f + w_b)   # weighted central

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
    dK = K[1] - K[0]
    d2_mid = (C[:, 2:] - 2.0 * C[:, 1:-1] + C[:, :-2]) / (dK * dK)  # [nT, nK-2]
    return tf.concat([d2_mid[:, 0:1], d2_mid, d2_mid[:, -1:]], axis=1)


def dupire_local_vol(C, T, K, r=0.0, q=0.0, eps=1e-10, sig_cap=4.0):
    """
    Build the local volatility surface σ_loc(T, K) via Dupire's formula.

    For r=q=0:
        σ²_loc = 2 * (∂C/∂T) / (K² · ∂²C/∂K²)

    All operations are pure TF — the result is differentiable w.r.t. C.

    Args:
        C:       [nT, nK]  call price matrix  (tf.Variable or tf.Tensor)
        T:       [nT]      maturities
        K:       [nK]      strikes
        r, q:    scalars   risk-free rate / dividend yield
        eps:     float     numerical floor to avoid division by zero
        sig_cap: float     cap on σ_loc (for numerical robustness)

    Returns:
        sigma_loc [nT, nK]
    """
    dT  = _dC_dT(C, T)          # [nT, nK]
    d2K = _d2C_dK2(C, K)        # [nT, nK]

    K2    = tf.square(K)[None, :]                         # [1, nK]
    denom = tf.maximum(K2 * tf.maximum(d2K, eps), eps)
    sig2  = 2.0 * tf.maximum(dT, 0.0) / denom

    return tf.sqrt(tf.clip_by_value(sig2, 0.0, sig_cap))


def bilinear_interp(sigma_TK, T_grid, K_grid, t, S):
    """
    Differentiable bilinear interpolation of σ_loc(T, K) evaluated at (t, S).

    Args:
        sigma_TK: [nT, nK]
        T_grid:   [nT]
        K_grid:   [nK]
        t:        [nPaths] or scalar  — current time for each path
        S:        [nPaths]            — current spot for each path

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
# LocalVolatilityProcess
# ============================================================

class LocalVolatilityProcess(StochasticProcess):
    """
    One-dimensional local volatility process (Dupire).

    The SDE is

        dS_t = (r - q) S_t dt  +  σ_loc(t, S_t) S_t dW_t

    where σ_loc is obtained by applying Dupire's formula to a call-price
    matrix C[nT, nK].  All operations are pure TensorFlow, so the entire
    path simulation is differentiable w.r.t. C (and therefore w.r.t. any
    implied-vol parameters that feed into C).

    Usage
    -----
    proc = LocalVolatilityProcess(C_surface, T_grid, K_grid, S0, r, q)
    paths = proc.evolve(T, dw)          # [n_paths, n_steps]

    For AAD use a tf.Variable for C_surface and wrap proc.evolve inside
    a GradientTape.
    """

    def __init__(self, C_surface, T_grid, K_grid, S0,
                 r=0.0, q=0.0,
                 dupire_eps=1e-10, sigma_cap=4.0):
        """
        Args:
            C_surface:  [nT, nK] call price matrix — tf.Variable for AAD,
                        tf.Tensor otherwise.
            T_grid:     [nT] maturities (year fractions).
            K_grid:     [nK] strikes.
            S0:         Initial spot level.
            r:          Risk-free rate (scalar or tf.Variable).
            q:          Dividend yield (scalar or tf.Variable).
            dupire_eps: Numerical floor in Dupire denominator.
            sigma_cap:  Cap on local volatility value.
        """
        self._C       = C_surface
        self._T_grid  = tf.cast(tf.convert_to_tensor(T_grid), tf.float32)
        self._K_grid  = tf.cast(tf.convert_to_tensor(K_grid), tf.float32)
        self._S0      = tf.cast(tf.convert_to_tensor(S0),     tf.float32)
        self._r       = tf.cast(tf.convert_to_tensor(r),      tf.float32)
        self._q       = tf.cast(tf.convert_to_tensor(q),      tf.float32)
        self._eps     = dupire_eps
        self._cap     = sigma_cap

    # ------------------------------------------------------------------
    # StochasticProcess interface
    # ------------------------------------------------------------------

    def size(self):
        return 1

    def initial_values(self):
        return self._S0

    def drift(self, t0, x0, dt):
        """GBM-like drift:  (r - q) · S · dt"""
        return (self._r - self._q) * x0 * dt

    def diffusion(self, t0, x0, dt):
        """σ_loc(t0, S) · S · √dt  (uses precomputed sigma_TK)."""
        raise NotImplementedError(
            "Call evolve() directly — diffusion requires the precomputed "
            "sigma surface passed through evolve()."
        )

    # ------------------------------------------------------------------
    # Main simulation
    # ------------------------------------------------------------------

    @tf.function
    def evolve(self, T, dw):
        """
        Simulate paths using a log-Euler scheme with Dupire local volatility.

        All operations are pure TF — wrap inside GradientTape for AAD.

        Args:
            T:   Final time horizon (scalar).
            dw:  [n_paths, n_steps] standard-normal increments.

        Returns:
            paths [n_paths, n_steps]  (spot at each step, NOT including S0).
        """
        dw = tf.cast(tf.convert_to_tensor(dw), tf.float32)

        n_paths = tf.shape(dw)[0]
        n_steps = tf.shape(dw)[1]
        T_f     = tf.cast(T, tf.float32)
        dt      = T_f / tf.cast(n_steps, tf.float32)

        # Precompute σ_loc surface once (differentiable w.r.t. self._C)
        C_f       = tf.cast(self._C, tf.float32)
        sigma_TK  = dupire_local_vol(C_f, self._T_grid, self._K_grid,
                                     r=self._r, q=self._q,
                                     eps=self._eps, sig_cap=self._cap)

        # Time grid for each step
        t_steps = tf.linspace(tf.constant(0.0), T_f, n_steps + 1)  # [n_steps+1]

        S     = tf.fill([n_paths], self._S0)
        paths = tf.TensorArray(dtype=tf.float32, size=n_steps)

        for k in tf.range(n_steps):
            t   = t_steps[k]
            t_v = tf.fill([n_paths], t)

            sig   = bilinear_interp(sigma_TK, self._T_grid, self._K_grid, t_v, S)
            drift = (self._r - self._q - 0.5 * tf.square(sig)) * dt
            diff  = sig * tf.sqrt(dt) * dw[:, k]
            S     = S * tf.exp(drift + diff)

            paths = paths.write(k, S)

        # [n_steps, n_paths] → [n_paths, n_steps]
        return tf.transpose(paths.stack(), perm=[1, 0])
