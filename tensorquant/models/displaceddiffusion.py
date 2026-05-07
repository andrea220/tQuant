from .stochasticprocess import StochasticProcess
import tensorflow as tf
import numpy as np
from scipy import optimize
from scipy.stats import norm as _scipy_norm


# ============================================================
# Numpy helpers  (calibration — not TF-traced)
# ============================================================

def _bs_call_np(S: float, K: float, T: float, vol: float, r: float, q: float) -> float:
    """Black-Scholes call price (numpy scalar)."""
    T   = max(T,   1e-12)
    vol = max(vol, 1e-12)
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return S * np.exp(-q * T) * _scipy_norm.cdf(d1) - K * np.exp(-r * T) * _scipy_norm.cdf(d2)


def _dd_call_np(S0: float, K: float, T: float, beta: float, sigma: float,
                r: float, q: float) -> float:
    """
    Displaced Diffusion call price (numpy scalar).

    The shifted process Z_t = S_t + a·e^{(r-q)t} follows GBM with drift (r-q):
        Z_T = Z_0 · exp((r-q - ½σ_a²)T + σ_a·W_T)

    where  Z_0 = S_0 + a = S_0/β,  a = S_0·(1-β)/β,  σ_a = σ·β.

    C^DD = BS_call(S_0/β, K + a·e^{(r-q)T}, T, σ_a, r, q)
    """
    a        = S0 * (1.0 - beta) / beta
    sigma_a  = sigma * beta
    S_sh     = S0 + a                                       # = S0 / beta
    K_sh     = K + a * np.exp((r - q) * T)
    return _bs_call_np(S_sh, K_sh, T, sigma_a, r, q)


def _atm_vol_correction_np(atm_vol: float, beta: float, T: float) -> float:
    """
    Correct ATM implied vol so that the DD ATM price matches Black-Scholes:

        σ = σ_ATM · [1 - σ_ATM²·T/24] / [1 - (β·σ_ATM)²·T/24]

    Returns corrected σ (model vol), from which σ_a = β·σ.
    """
    numer = 1.0 - atm_vol ** 2 * T / 24.0
    denom = 1.0 - (beta * atm_vol) ** 2 * T / 24.0
    denom = max(denom, 1e-6)
    return atm_vol * numer / denom


def _compute_skew_average_np(fwd_betas, fwd_vols, maturities) -> float:
    """
    Discrete skew-averaging formula: recovers the spot β_T from forward betas.

        β_T = Σ_j β_j · σ_j² · (Σ_{i≤j} σ_i²·Δt_i) · Δt_j
              / Σ_j σ_j² · (Σ_{i≤j} σ_i²·Δt_i) · Δt_j
    """
    n    = len(fwd_betas)
    mats = np.concatenate([[0.0], maturities[:n]])
    dts  = np.diff(mats)
    vols = np.asarray(fwd_vols[:n], dtype=float)
    betas = np.asarray(fwd_betas,   dtype=float)

    var    = vols ** 2 * dts
    cumvar = np.cumsum(var)
    num    = np.sum(betas * var * cumvar)
    den    = np.sum(var * cumvar)
    return float(num / den)


def _calibrate_forward_betas_np(spot_betas, atm_vols, maturities):
    """
    Extract forward betas from calibrated spot betas using the closed-form
    inverse of the skew-averaging formula.

    At each step j the relation is LINEAR in β_j^fwd:

        spot_β_{T_j} = (A_j + β_j · B_j) / (C_j + B_j)

    where A_j, C_j are known from β_0, ..., β_{j-1}, so:

        β_j = [spot_β_{T_j} · (C_j + B_j) - A_j] / B_j

    Returns
    -------
    fwd_betas : ndarray [nT]
    fwd_vols  : ndarray [nT]  (forward ATM vols per interval)
    """
    nT       = len(maturities)
    fwd_vols = np.zeros(nT)
    mats_arr = np.asarray(maturities, dtype=float)
    atm_arr  = np.asarray(atm_vols,   dtype=float)

    # --- Forward ATM vols ---
    fwd_vols[0] = atm_arr[0]
    for j in range(1, nT):
        T_j    = mats_arr[j]
        T_prev = mats_arr[j - 1]
        var_diff = max(atm_arr[j] ** 2 * T_j - atm_arr[j - 1] ** 2 * T_prev, 1e-8)
        fwd_vols[j] = np.sqrt(var_diff / (T_j - T_prev))

    # --- Precompute variance weights ---
    dts    = np.diff(np.concatenate([[0.0], mats_arr]))   # [nT]
    var    = fwd_vols ** 2 * dts                           # [nT]
    cumvar = np.cumsum(var)                                 # [nT]

    # --- Closed-form forward beta extraction ---
    fwd_betas = np.zeros(nT)
    for j in range(nT):
        B_j = var[j] * cumvar[j]
        if j == 0:
            fwd_betas[0] = spot_betas[0]
        else:
            A_j = float(np.sum(fwd_betas[:j] * var[:j] * cumvar[:j]))
            C_j = float(np.sum(var[:j] * cumvar[:j]))
            if abs(B_j) < 1e-12:
                fwd_betas[j] = spot_betas[j]
            else:
                fwd_betas[j] = (spot_betas[j] * (C_j + B_j) - A_j) / B_j

    return fwd_betas, fwd_vols


def _bs_implied_vol_np(C_mkt: float, S: float, K: float, T: float,
                        r: float, q: float) -> float:
    """Invert Black-Scholes via Brent on [1e-4, 10]."""
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if C_mkt <= intrinsic + 1e-10:
        return float("nan")
    try:
        return optimize.brentq(
            lambda v: _bs_call_np(S, K, T, v, r, q) - C_mkt,
            1e-4, 10.0, xtol=1e-7,
        )
    except ValueError:
        return float("nan")


# ============================================================
# TensorFlow helpers  (pricing / MC — differentiable)
# ============================================================

def _norm_cdf_tf(x):
    """Standard-normal CDF via TF erf."""
    return 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(tf.constant(2.0, dtype=x.dtype))))


def _dd_call_tf(S0, K, T, beta, sigma, r, q):
    """
    Displaced Diffusion call price (TensorFlow).

    C^DD = BS_call(S0 + a, K + a·e^{(r-q)T}, T, σ_a, r, q)
    where a = S0·(1-β)/β, σ_a = σ·β.

    All arguments broadcastable; dtype follows S0.
    """
    dtype = tf.float32
    S0    = tf.cast(S0,    dtype)
    K     = tf.cast(K,     dtype)
    T     = tf.maximum(tf.cast(T,     dtype), tf.constant(1e-12, dtype))
    beta  = tf.cast(beta,  dtype)
    sigma = tf.cast(sigma, dtype)
    r     = tf.cast(r,     dtype)
    q     = tf.cast(q,     dtype)

    a        = S0 * (1.0 - beta) / beta
    sigma_a  = sigma * beta
    S_sh     = S0 + a
    K_sh     = K + a * tf.exp((r - q) * T)

    sigT = sigma_a * tf.sqrt(T)
    d1   = (tf.math.log(S_sh / K_sh) + (r - q + 0.5 * sigma_a ** 2) * T) / sigT
    d2   = d1 - sigT

    return (S_sh * tf.exp(-q * T) * _norm_cdf_tf(d1)
            - K_sh * tf.exp(-r * T) * _norm_cdf_tf(d2))


# ============================================================
# DisplacedDiffusionModel
# ============================================================

class DisplacedDiffusionModel(StochasticProcess):
    """
    Displaced Diffusion (Shifted Lognormal) equity model.

    The driftless SDE is:
        dX_t = (X_t + a_t) · σ_a · dZ_t

    Under risk-neutral measure (continuous dividend yield q):
        d ln(S_t + a_t·e^{(r-q)t}) ≈ (r-q)·dt − ½σ_a²·dt + σ_a·dZ_t

    where:
        a_t  = S_t · (1−β_t)/β_t    (path-dependent displacement)
        σ_a  = σ_t · β_t             (shifted vol)

    β=1 recovers standard GBM; β→0 approaches normal diffusion.

    Construction
    ------------
    From an implied-vol surface (calibrates β slice-by-slice + forward skew)::

        model = DisplacedDiffusionModel.from_implied_vol(
            spot, vol_surface, disc_curve, q=0.0, forward_skew=True
        )

    Direct construction (supply calibrated curves)::

        model = DisplacedDiffusionModel(S0, r, q, T_grid,
                                         fwd_beta_curve, fwd_sigma_curve)

    Usage
    -----
    paths = model.evolve(t_grid, dw)       # [n_paths, n_steps]
    price = model.dd_call(K, T_idx=0)      # DD call at maturity T_grid[0]
    iv    = model.implied_vol_surface()    # [nT, nK] IV from DD prices
    model.print_calibration_summary()
    """

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    def __init__(
        self,
        S0,
        r,
        q,
        T_grid,
        fwd_beta_curve,
        fwd_sigma_curve,
        spot_beta_curve=None,
        spot_sigma_curve=None,
        K_grid=None,
        r_vec=None,
    ):
        """
        Args:
            S0:               Initial spot (scalar).
            r:                Flat risk-free rate used for MC simulation (scalar).
            q:                Continuous dividend yield (scalar).
            T_grid:           [nT] calibrated maturities (year fractions).
            fwd_beta_curve:   [nT] forward β per interval [0,T_0],[T_0,T_1],...
            fwd_sigma_curve:  [nT] forward ATM vol per interval (raw, not corrected).
                              σ_a for each step = fwd_sigma[j] · fwd_beta[j].
            spot_beta_curve:  [nT] spot β per maturity (stored for pricing reference).
            spot_sigma_curve: [nT] spot σ (ATM-corrected) per maturity.
            K_grid:           [nK] strike grid (stored for implied_vol_surface).
            r_vec:            [nT] per-maturity zero rates (stored for pricing).
        """
        self._S0    = tf.constant(float(S0),  dtype=tf.float32)
        self._r     = tf.constant(float(r),   dtype=tf.float32)
        self._q     = tf.constant(float(q),   dtype=tf.float32)

        T_np   = np.asarray(T_grid,          dtype=np.float32)
        fb_np  = np.asarray(fwd_beta_curve,  dtype=np.float32)
        fs_np  = np.asarray(fwd_sigma_curve, dtype=np.float32)

        self._T_grid    = tf.constant(T_np)
        self._fwd_beta  = tf.constant(fb_np)
        self._fwd_sigma = tf.constant(fs_np)

        self._spot_beta  = (tf.constant(np.asarray(spot_beta_curve,  np.float32))
                            if spot_beta_curve  is not None else self._fwd_beta)
        self._spot_sigma = (tf.constant(np.asarray(spot_sigma_curve, np.float32))
                            if spot_sigma_curve is not None else self._fwd_sigma)

        self._K_grid = (np.asarray(K_grid, dtype=np.float64)
                        if K_grid is not None else None)
        self._r_vec  = (np.asarray(r_vec,  dtype=np.float64)
                        if r_vec  is not None else None)

    # ------------------------------------------------------------------
    # Factory: calibrate from implied-vol surface
    # ------------------------------------------------------------------

    @classmethod
    def from_implied_vol(
        cls,
        spot,
        vol_surface,
        disc_curve,
        q: float = 0.0,
        forward_skew: bool = True,
        weights=None,
    ):
        """
        Calibrate a DisplacedDiffusionModel from a VolatilitySurface.

        For each maturity T_j:
          1. Interpolate ATM vol at K = spot.
          2. Compute Black-Scholes market call prices.
          3. Grid-search + Nelder-Mead minimisation of weighted squared
             price residuals w.r.t. β_T (the single free parameter per slice).
          4. Apply ATM vol correction to derive σ and σ_a.

        If forward_skew=True, extract forward β via closed-form inversion
        of the discrete skew-averaging formula (Section 13-15 of the spec).

        Args:
            spot:         Initial spot level (float).
            vol_surface:  VolatilitySurface (.maturity, .strike, .volatility_matrix).
            disc_curve:   RateCurve with .zero_rate(T) method.
            q:            Continuous dividend yield / repo rate (scalar).
            forward_skew: Extract forward betas (default True).
            weights:      [nK] optional strike weights (equal if None).

        Returns:
            DisplacedDiffusionModel instance with calibrated curves stored.
        """
        S0     = float(spot)
        T_grid = np.asarray(vol_surface.maturity,          dtype=np.float64)
        K_grid = np.asarray(vol_surface.strike,            dtype=np.float64)
        iv_mat = np.asarray(vol_surface.volatility_matrix, dtype=np.float64)
        nT, nK = iv_mat.shape

        r_vec = np.array([float(disc_curve.zero_rate(T)) for T in T_grid])

        spot_betas  = np.zeros(nT)
        spot_sigmas = np.zeros(nT)
        atm_vols    = np.zeros(nT)

        for i, T in enumerate(T_grid):
            r_i      = float(r_vec[i])
            q_f      = float(q)
            iv_slice = iv_mat[i, :]

            atm_vol = float(np.interp(S0, K_grid, iv_slice))
            atm_vols[i] = atm_vol

            mkt_prices = np.array([
                _bs_call_np(S0, K_grid[k], T, iv_slice[k], r_i, q_f)
                for k in range(nK)
            ])

            w = (np.ones(nK) if weights is None else np.asarray(weights, dtype=float))
            w = w / w.sum()

            def _obj(beta_arr, _T=T, _r=r_i, _mkt=mkt_prices, _w=w, _atm=atm_vol):
                beta = float(beta_arr[0])
                if beta < 1e-4:
                    return 1e10
                a = S0 * (1.0 - beta) / beta
                # All shifted strikes must be strictly positive
                if K_grid.min() + a * np.exp((_r - q_f) * _T) <= 0.0:
                    return 1e10
                sigma = _atm_vol_correction_np(_atm, beta, _T)
                dd    = np.array([_dd_call_np(S0, K_grid[k], _T, beta, sigma, _r, q_f)
                                  for k in range(nK)])
                return float(np.sum(_w * (dd - _mkt) ** 2))

            # Coarse grid to find a good starting bracket
            betas0    = np.linspace(0.05, 2.5, 30)
            obj_vals  = [_obj([b]) for b in betas0]
            beta_init = betas0[int(np.argmin(obj_vals))]

            res = optimize.minimize(
                _obj, x0=[beta_init], method="Nelder-Mead",
                options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 3000},
            )
            beta_star  = float(res.x[0])
            sigma_star = _atm_vol_correction_np(atm_vol, beta_star, T)

            spot_betas[i]  = beta_star
            spot_sigmas[i] = sigma_star

        # --- Forward skew calibration ---
        if forward_skew:
            fwd_betas, fwd_vols = _calibrate_forward_betas_np(
                spot_betas, atm_vols, T_grid
            )
        else:
            fwd_betas = spot_betas.copy()
            fwd_vols  = atm_vols.copy()

        r_flat = float(r_vec[-1])   # representative flat rate for MC

        return cls(
            S0=S0, r=r_flat, q=q,
            T_grid=T_grid,
            fwd_beta_curve=fwd_betas,
            fwd_sigma_curve=fwd_vols,
            spot_beta_curve=spot_betas,
            spot_sigma_curve=spot_sigmas,
            K_grid=K_grid,
            r_vec=r_vec,
        )

    # ------------------------------------------------------------------
    # StochasticProcess interface
    # ------------------------------------------------------------------

    def size(self) -> int:
        return 1

    def initial_values(self):
        return self._S0

    def drift(self, t0, x0, dt):
        """GBM-like drift  (r−q)·S·dt — same as standard equity GBM."""
        return (self._r - self._q) * x0 * dt

    def diffusion(self, t0, x0, dt):
        raise NotImplementedError(
            "DisplacedDiffusionModel uses a path-dependent diffusion. "
            "Use evolve() to simulate paths."
        )

    # ------------------------------------------------------------------
    # Vanilla pricing (TF, differentiable)
    # ------------------------------------------------------------------

    def dd_call(self, K, T_idx: int = 0):
        """
        Displaced Diffusion call price using spot-calibrated params.

        Args:
            K:     Strike (scalar or tensor).
            T_idx: Maturity index into T_grid.

        Returns:
            Call price as tf.Tensor (float32).
        """
        beta  = self._spot_beta[T_idx]
        sigma = self._spot_sigma[T_idx]
        T     = self._T_grid[T_idx]
        r_val = (tf.constant(float(self._r_vec[T_idx]), tf.float32)
                 if self._r_vec is not None else self._r)
        return _dd_call_tf(self._S0, K, T, beta, sigma, r_val, self._q)

    def implied_vol_surface(self):
        """
        Compute the implied-vol surface implied by the DD model by
        inverting Black-Scholes on DD call prices.

        Returns
        -------
        iv_dd : ndarray [nT, nK]
            Implied vols from DD pricing (NaN where inversion fails).
        """
        if self._K_grid is None:
            raise RuntimeError("K_grid not set — pass K_grid to __init__ or use from_implied_vol.")

        T_np  = self._T_grid.numpy().astype(float)
        K_np  = self._K_grid.astype(float)
        b_np  = self._spot_beta.numpy().astype(float)
        s_np  = self._spot_sigma.numpy().astype(float)
        r_vec = self._r_vec if self._r_vec is not None else np.full(len(T_np), float(self._r.numpy()))
        q_f   = float(self._q.numpy())
        S0    = float(self._S0.numpy())

        nT, nK = len(T_np), len(K_np)
        iv_dd = np.full((nT, nK), float("nan"))
        for i in range(nT):
            for k in range(nK):
                price = _dd_call_np(S0, K_np[k], T_np[i],
                                    b_np[i], s_np[i], float(r_vec[i]), q_f)
                iv_dd[i, k] = _bs_implied_vol_np(price, S0, K_np[k], T_np[i],
                                                  float(r_vec[i]), q_f)
        return iv_dd

    # ------------------------------------------------------------------
    # Calibration summary
    # ------------------------------------------------------------------

    def print_calibration_summary(self):
        """Print a table of calibrated parameters (spot and forward)."""
        T_np   = self._T_grid.numpy()
        sb_np  = self._spot_beta.numpy()
        ss_np  = self._spot_sigma.numpy()
        fb_np  = self._fwd_beta.numpy()
        fs_np  = self._fwd_sigma.numpy()
        S0     = float(self._S0.numpy())

        print("\n" + "=" * 72)
        print("Displaced Diffusion — Calibration Summary")
        print("=" * 72)
        print(f"  S0 = {S0:.4f}   r = {float(self._r.numpy()):.4f}   q = {float(self._q.numpy()):.4f}")
        print()
        print(f"  {'T':>6}  {'b_spot':>8}  {'s_spot':>8}  {'a_spot':>8}  {'sa_spot':>10}"
              f"  {'b_fwd':>8}  {'s_fwd':>8}  {'sa_fwd':>8}")
        print("  " + "-" * 70)
        for i, T in enumerate(T_np):
            a_spot   = S0 * (1 - sb_np[i]) / sb_np[i]
            siga_spt = ss_np[i] * sb_np[i]
            siga_fwd = fs_np[i] * fb_np[i]
            print(f"  {T:6.2f}  {sb_np[i]:8.4f}  {ss_np[i]:8.4f}  {a_spot:8.4f}"
                  f"  {siga_spt:10.4f}  {fb_np[i]:8.4f}  {fs_np[i]:8.4f}  {siga_fwd:8.4f}")
        print()

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    @tf.function
    def evolve(self, t_grid, dw):
        """
        Simulate displaced-diffusion paths with piecewise-constant forward
        parameters and path-dependent displacement.

        Euler step (per interval j with forward params β_j, σ_j):
            a_t      = S_t · (1−β_j) / β_j
            σ_a      = σ_j · β_j
            S_{t+dt} = (S_t + a_t) · exp((r−q)·dt − ½σ_a²·dt + σ_a·√dt·Z)
                       − a_t · exp((r−q)·dt)

        This preserves the correct martingale forward E[S_T] = S_0·e^{(r−q)T}.

        Args:
            t_grid: [n_steps] observation times (year fractions, t > 0).
            dw:     [n_paths, n_steps] standard-normal increments.

        Returns:
            paths: [n_paths, n_steps]  spot at each observation time.
        """
        dw     = tf.cast(tf.convert_to_tensor(dw),     tf.float32)
        t_grid = tf.cast(tf.convert_to_tensor(t_grid), tf.float32)

        n_paths = tf.shape(dw)[0]
        n_steps = tf.shape(dw)[1]
        t_full  = tf.concat([tf.zeros([1], tf.float32), t_grid], axis=0)

        S     = tf.fill([n_paths], self._S0)
        paths = tf.TensorArray(dtype=tf.float32, size=n_steps)

        carry = self._r - self._q

        for k in tf.range(n_steps):
            t0 = t_full[k]
            dt = t_full[k + 1] - t_full[k]

            # Piecewise-constant forward params: interval index j
            # j = number of T_grid entries strictly < t0 (so at t0=T_j we enter next interval)
            j = tf.reduce_sum(tf.cast(self._T_grid <= t0, tf.int32))
            j = tf.clip_by_value(j, 0, tf.shape(self._fwd_beta)[0] - 1)

            beta_j  = self._fwd_beta[j]
            sigma_j = self._fwd_sigma[j]
            sigma_a = sigma_j * beta_j

            # Path-dependent displacement
            a = S * (1.0 - beta_j) / beta_j

            # DD Euler step
            drift = (carry - 0.5 * sigma_a * sigma_a) * dt
            diff  = sigma_a * tf.sqrt(dt) * dw[:, k]
            exp_carry = tf.exp(carry * dt)
            S = (S + a) * tf.exp(drift + diff) - a * exp_carry

            # Absorbing barrier at zero to avoid negative spots
            S = tf.maximum(S, tf.constant(1e-8, tf.float32))

            paths = paths.write(k, S)

        return tf.transpose(paths.stack(), perm=[1, 0])
