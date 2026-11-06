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


def _call_on_bond_sum(a: tf.Tensor, b: tf.Tensor, mean: tf.Tensor, sd: tf.Tensor, phi: float) -> tf.Tensor:
    """``E[(phi sum_k a_k exp(-b_k y))^+]`` for ``y ~ N(mean, sd^2)``, batched (``a, b [..., m]``, ``mean, sd [...]``).

    ``V(y) = sum_k a_k e^{-b_k y}`` is assumed to change sign once, from negative to positive (payer side), at
    ``y*`` (without gradient: 16 bisections on ``[mean - 12 sd, mean + 12 sd]``, then 4 Newton steps kept in the
    bracket); then ``E[e^{-b y} 1{y > y*}] = e^{-b mean + b^2 sd^2 / 2} N((mean - b sd^2 - y*) / sd)``.
    """
    lo, hi = tf.stop_gradient(mean - 12.0 * sd), tf.stop_gradient(mean + 12.0 * sd)
    a_c, b_c = tf.stop_gradient(a), tf.stop_gradient(b)
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        positive = tf.reduce_sum(a_c * tf.exp(-b_c * mid[..., None]), axis=-1) > 0.0
        lo, hi = tf.where(positive, lo, mid), tf.where(positive, mid, hi)
    root = 0.5 * (lo + hi)
    for _ in range(4):
        terms = a_c * tf.exp(-b_c * root[..., None])
        slope = -tf.reduce_sum(b_c * terms, axis=-1)
        step = tf.math.divide_no_nan(tf.reduce_sum(terms, axis=-1), slope)
        root = tf.clip_by_value(root - step, lo, hi)
    cdf = lambda v: 0.5 * (1.0 + tf.math.erf(v / np.sqrt(2.0)))
    mean, sd = mean[..., None], sd[..., None]
    moments = a * tf.exp(-b * mean + 0.5 * b * b * sd * sd)
    return tf.reduce_sum(moments * cdf(phi * (mean - b * sd * sd - root[..., None]) / sd), axis=-1) * phi


def _implied_bachelier(price: float, forward: float, strike: float, t: float, phi: float) -> float:
    """Bachelier vol of an undiscounted price (Newton from the ATM approximation)."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    def bachelier(v):
        std = v * np.sqrt(t)
        d = (forward - strike) / std
        return phi * (forward - strike) * norm.cdf(phi * d) + std * norm.pdf(d)

    intrinsic = max(phi * (forward - strike), 0.0)
    if price <= intrinsic + 1e-18:
        return 0.0
    return brentq(lambda v: bachelier(v) - price, 1e-10, 1.0, xtol=1e-15)


class GaussianShortRateModel(StochasticProcess):
    """n-factor Gaussian short rate model fitted to a discount curve (G1++ = Hull-White, G2++).

    ``r(t) = x_1(t) + ... + x_n(t) + phi(t)``, ``dx_i = -kappa_i x_i dt + sigma_i(t) dW_i``,
    ``x_i(0) = 0``, ``d<W_i, W_j> = rho_ij dt``; ``phi`` fits the initial curve ``P(0, T)`` exactly.
    With ``V(t, T) = Var(int_t^T sum_i x_i du | F_t)`` (Brigo-Mercurio):

    - bond: ``P(t, T) = P(0,T)/P(0,t) exp(1/2 [V(t,T) - V(0,T) + V(0,t)] - sum_i B_i(T - t) x_i(t))``,
      ``B_i(tau) = (1 - e^{-kappa_i tau}) / kappa_i``;
    - bank-account discount along a path: ``D(0, t) = P(0, t) exp(-sum_i I_i(t) - V(0, t)/2)``,
      ``I_i(t) = int_0^t x_i``, so ``E[D(0, t)] = P(0, t)``.

    The vols are constant or piecewise constant in time (``sigma_times``: the same breakpoints for every
    factor); every variance / covariance is an exact integral over the pieces.

    :meth:`simulate` draws ``(x_i, I_i)`` exactly (joint Gaussian transition: no time-step error).
    The curve enters only through ``P(0, .)``, so prices are differentiable w.r.t. its pillars
    (rho by AAD) and w.r.t. the model parameters if passed as variables.

    Args:
        curve (RateCurve): Initial discount curve (e.g. ``market_env.get_ir_curve(ccy)``).
        kappas (Sequence[float]): Mean-reversion speeds ``[n]``.
        sigmas: Volatilities ``[n]`` (constant) or ``[n_pieces, n]`` (piecewise constant on *sigma_times*).
        correlation (array-like, optional): ``[n, n]`` correlation of the Brownians (identity if None).
        sigma_times (Sequence[float], optional): Increasing breakpoints ``[n_pieces - 1]`` (year fractions > 0):
            piece ``k`` holds on ``[t_{k-1}, t_k)``, the last one from the last breakpoint on.

    Raises:
        ValueError: If the shapes of the parameters do not match or the breakpoints are not increasing.
    """

    def __init__(self, curve: RateCurve, kappas, sigmas, correlation=None,
                 sigma_times: Optional[Sequence[float]] = None):
        self._curve = curve
        self._kappa = _as_param(kappas)
        n = int(self._kappa.shape[0])
        sigma = _as_param(sigmas)
        self._sigma = sigma[None, :] if len(sigma.shape) == 1 else sigma                       # [n_pieces, n]
        knots = [] if sigma_times is None else [float(t) for t in sigma_times]
        if int(self._sigma.shape[-1]) != n:
            raise ValueError("kappas and sigmas must have the same number of factors")
        if int(self._sigma.shape[0]) != len(knots) + 1:
            raise ValueError(f"{int(self._sigma.shape[0])} vol pieces need {int(self._sigma.shape[0]) - 1} sigma_times")
        if knots and (knots[0] <= 0.0 or np.any(np.diff(knots) <= 0.0)):
            raise ValueError("sigma_times must be increasing and > 0")
        self._sigma_times = knots
        self._lo = tf.constant([0.0] + knots, DTYPE)                                           # piece starts
        self._hi = tf.constant(knots + [1e9], DTYPE)                                           # piece ends
        self._rho = tf.eye(n, dtype=DTYPE) if correlation is None else _as_param(correlation)
        if tuple(self._rho.shape) != (n, n):
            raise ValueError(f"correlation must be [{n}, {n}]")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def hull_white(cls, curve: RateCurve, a, sigma,
                   sigma_times: Optional[Sequence[float]] = None) -> "GaussianShortRateModel":
        """One factor (G1++ / Hull-White): ``dx = -a x dt + sigma(t) dW``.

        Args:
            curve: Initial discount curve.
            a: Mean-reversion speed.
            sigma: Constant vol, or ``[n_pieces]`` vols piecewise constant on *sigma_times*.
            sigma_times: Breakpoints ``[n_pieces - 1]`` (None: constant vol).
        """
        sigma = _as_param(sigma)
        sigma = tf.reshape(sigma, [1]) if sigma_times is None else tf.reshape(sigma, [-1, 1])
        return cls(curve, tf.reshape(_as_param(a), [1]), sigma, sigma_times=sigma_times)

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

    @property
    def sigma_times(self) -> list[float]:
        """Breakpoints of the piecewise constant vols (empty: constant)."""
        return list(self._sigma_times)

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

    def _integrals(self, t, lo, kernels: str = "xx,xi,ii"):
        """Exact integrals over ``v`` in ``[lo, t]`` of ``rho_ij sigma_i(v) sigma_j(v) f_ij(t - v)``.

        Inputs of any (broadcastable) shape; returns ``[..., n, n]`` tensors for the kernels (*kernels*, comma
        separated, in this order) ``"xx"``: ``f = e^{-(k_i + k_j) w}`` (factor covariance), ``"xi"``:
        ``e^{-k_i w} B_j(w)`` (factor / integral covariance), ``"ii"``: ``B_i(w) B_j(w)`` (integral covariance),
        summed over the vol pieces. With ``b_k(w) = (1 - e^{-k w}) / k`` the antiderivatives vanishing at 0 are
        ``b_ij``, ``(b_i - b_ij) / k_j`` and ``(w - b_i - b_j + b_ij) / (k_i k_j)`` (expm1: no cancellation for
        small speeds). Constant vols: one piece, ``F(t - lo)``.
        """
        k_i, k_j = self._kappa[:, None], self._kappa[None, :]
        k_ij = k_i + k_j
        b = lambda k, w: -tf.math.expm1(-k * w) / k
        f = {"xx": lambda w: b(k_ij, w),
             "xi": lambda w: (b(k_i, w) - b(k_ij, w)) / k_j,
             "ii": lambda w: (w - b(k_i, w) - b(k_j, w) + b(k_ij, w)) / (k_i * k_j)}
        kinds = kernels.split(",")
        t, lo = _as_param(t), _as_param(lo)
        if not self._sigma_times:
            s = self._sigma[0][:, None] * self._sigma[0][None, :] * self._rho                # [n, n]
            w = (t - lo)[..., None, None]
            out = [s * f[k](w) for k in kinds]
        else:
            t, lo = t[..., None], lo[..., None]                                              # [..., 1]
            u_lo = tf.minimum(tf.maximum(self._lo, lo), t)                                   # [..., P]
            u_hi = tf.minimum(tf.maximum(self._hi, lo), t)
            w_lo, w_hi = (t - u_hi)[..., None, None], (t - u_lo)[..., None, None]            # [..., P, 1, 1]
            s = self._sigma[:, :, None] * self._sigma[:, None, :] * self._rho               # [P, n, n]
            out = [tf.reduce_sum(s * (f[k](w_hi) - f[k](w_lo)), axis=-3) for k in kinds]
        return out[0] if len(out) == 1 else tuple(out)

    def factor_covariance(self, t) -> tf.Tensor:
        """``Cov(x(t))`` ``[..., n, n]`` (factors start at 0)."""
        return self._integrals(t, 0.0, "xx")

    def variance(self, t, T) -> tf.Tensor:
        """``V(t, T) = Var(int_t^T sum_i x_i du | F_t)`` (broadcast over the shapes of *t* and *T*)."""
        return tf.reduce_sum(self._integrals(T, t, "ii"), axis=[-2, -1])

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
        v = self.variance(np.array([tt, 0.0, 0.0]), np.array([TT, TT, tt]))           # V(t,T), V(0,T), V(0,t)
        conv = 0.5 * (v[0] - v[1] + v[2])
        return p0T / p0t * tf.exp(conv - tf.reduce_sum(self._B(TT - tt) * tf.cast(state, DTYPE), axis=-1))

    def short_rate(self, t: Union[date, float], state: tf.Tensor) -> tf.Tensor:
        """``r(t) = sum_i x_i(t) + phi(t)``, ``phi(t) = f(0, t) + 1/2 dV(0, t)/dt``."""
        tt = _as_param(self.time(t))
        with tf.GradientTape() as tape:
            tape.watch(tt)
            v = self.variance(0.0, tt)
        phi = tf.cast(self._curve.inst_fwd(float(tt)), DTYPE) + 0.5 * tape.gradient(v, tt)
        return tf.reduce_sum(tf.cast(state, DTYPE), axis=-1) + phi

    def exchange_option_values(self, t, s, e, A, B) -> tf.Tensor:
        """Vectorized ``E[D(0,t) (a P(t,s) - b P(t,e))^+]`` given ``A = a P(0,s)`` and ``B = b P(0,e)``.

        Args:
            t, s, e: ``[n]`` year fractions (``t <= s, e``).
            A, B: ``[n]`` forward values of the two legs at time 0 (tensors or arrays).

        Returns:
            tf.Tensor: ``[n]`` option values (Margrabe on the Gaussian log bond ratio).
        """
        t, s, e = (np.asarray(v, dtype=float) for v in (t, s, e))
        cov = self.factor_covariance(t)                                                       # [n, nf, nf]
        db = self._B(e - t) - self._B(s - t)                                                  # [n, nf]
        v = tf.sqrt(tf.maximum(tf.einsum("ni,nij,nj->n", db, cov, db), 1e-300))
        A, B = _as_param(A), _as_param(B)
        d1 = (tf.math.log(A / B) + 0.5 * v * v) / v
        cdf = lambda x: 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0)))
        return A * cdf(d1) - B * cdf(d1 - v)

    def bond_exchange_option(self, t, s, e, a, b) -> tf.Tensor:
        """``E[D(0, t) (a P(t, s) - b P(t, e))^+]`` in closed form (``t <= s, e``).

        ``ln(P(t,s) / P(t,e))`` is Gaussian with variance ``v^2 = dB^T Cov(x(t)) dB``, ``dB = B(e-t) - B(s-t)``
        (constant vols: ``Cov_ij = rho_ij sigma_i sigma_j (1 - e^{-(k_i+k_j) t}) / (k_i + k_j)``),
        so the value is ``A N(d1) - B N(d2)`` with ``A = a P(0, s)``, ``B = b P(0, e)``,
        ``d1 = (ln(A/B) + v^2/2) / v``, ``d2 = d1 - v`` (Margrabe). Zero-bond options are the special
        cases ``s = t`` (put on ``P(t, e)``) or ``e = t`` (call on ``P(t, s)``).
        """
        tt, ss, ee = self.time(t), self.time(s), self.time(e)
        A = _as_param(a) * tf.cast(self._curve.discount(s if isinstance(s, date) else ss), DTYPE)
        B = _as_param(b) * tf.cast(self._curve.discount(e if isinstance(e, date) else ee), DTYPE)
        return self.exchange_option_values([tt], [ss], [ee], tf.reshape(A, [1]), tf.reshape(B, [1]))[0]

    def caplet(self, fixing, start, end, strike: float, tau: float, basis=1.0, is_cap: bool = True) -> tf.Tensor:
        """Caplet / floorlet per unit notional, paid at *end*, rate fixed at *fixing* on ``[start, end]``.

        With a deterministic multiplicative basis between the index curve and the model (discount)
        curve, ``1 + tau L = basis P(fix, start) / P(fix, end)``,
        ``basis = [P_idx(0,start)/P_idx(0,end)] / [P(0,start)/P(0,end)]``, so the caplet is an exchange
        option: ``E[D (basis P(fix, start) - (1 + tau K) P(fix, end))^+]`` (floor: the other way round).

        Args:
            fixing, start, end: Dates (or year fractions) of the period.
            strike (float): Strike rate.
            tau (float): Index accrual year fraction of the period.
            basis: Index-vs-discount curve ratio (1 for a single curve).
            is_cap (bool): Caplet (True) or floorlet.

        Returns:
            tf.Tensor: float64 value per unit notional (``tau`` included).
        """
        k_tau = 1.0 + tau * strike
        cap = self.bond_exchange_option(fixing, start, end, basis, k_tau)
        if is_cap:
            return cap
        # floor = cap - (basis P(0,start) - (1 + tau K) P(0,end))   (parity)
        p_s = tf.cast(self._curve.discount(start if isinstance(start, date) else self.time(start)), DTYPE)
        p_e = tf.cast(self._curve.discount(end if isinstance(end, date) else self.time(end)), DTYPE)
        return cap - (_as_param(basis) * p_s - k_tau * p_e)

    def _swap_legs(self, te, pay_t, p0_pay, st_t, p0_st, en_t, p0_en, p0_te):
        """Coefficients ``c_T = P(0,T)/P(0,te) exp((V(te,T) - V(0,T) + V(0,te))/2)`` and ``B(T - te)`` of the
        swap bonds at expiry (``P(te, T; x) = c_T exp(-B(T - te) . x)``), batched ``[n, m]`` / ``[n, m, nf]``."""
        te = np.asarray(te, dtype=float)
        p0_te = tf.constant(np.asarray(p0_te, dtype=float), DTYPE)[..., None]
        v_te = self.variance(0.0, te)[..., None]

        def legs(times, p0):
            times = np.maximum(np.asarray(times, dtype=float), te[..., None])
            conv = 0.5 * (self.variance(te[..., None], times) - self.variance(0.0, times) + v_te)
            return tf.constant(np.asarray(p0, dtype=float), DTYPE) / p0_te * tf.exp(conv), self._B(times - te[..., None])

        return legs(pay_t, p0_pay), legs(st_t, p0_st), legs(en_t, p0_en)

    def swaption_normal_vols(self, te, pay_t, pay_w, p0_pay, st_t, st_w, p0_st, en_t, en_w, p0_en, p0_te) -> tf.Tensor:
        """Batched Gaussian swap-rate normal vols ``[n]`` (see :meth:`swaption_from_arrays`).

        Flow arrays are ``[n, m]`` (pad shorter swaps with zero weights), ``te`` and ``p0_te`` ``[n]``.
        """
        (c_p, b_p), (c_s, b_s), (c_e, b_e) = self._swap_legs(te, pay_t, p0_pay, st_t, p0_st, en_t, p0_en, p0_te)
        w_p, w_s, w_e = (tf.constant(np.asarray(w, dtype=float), DTYPE) for w in (pay_w, st_w, en_w))
        a_x = tf.reduce_sum(w_p * c_p, axis=-1)
        f_x = tf.reduce_sum(w_s * c_s, axis=-1) - tf.reduce_sum(w_e * c_e, axis=-1)
        da = -tf.reduce_sum((w_p * c_p)[..., None] * b_p, axis=-2)
        df = -tf.reduce_sum((w_s * c_s)[..., None] * b_s, axis=-2) + tf.reduce_sum((w_e * c_e)[..., None] * b_e, axis=-2)
        g = (df * a_x[..., None] - f_x[..., None] * da) / (a_x * a_x)[..., None]
        cov = self.factor_covariance(np.asarray(te, dtype=float))
        var = tf.einsum("...i,...ij,...j->...", g, cov, g)
        return tf.sqrt(tf.maximum(var, 1e-300) / tf.constant(np.asarray(te, dtype=float), DTYPE))

    def swaption_exact_values(self, te, pay_t, pay_w, p0_pay, st_t, st_w, p0_st, en_t, en_w, p0_en, p0_te,
                              strike, payer: bool = True, nodes: int = 64) -> tf.Tensor:
        """Batched exact swaption values ``[n]`` (physical settlement) under the model.

        At expiry the swap is a bond portfolio ``V(x) = sum_k a_k exp(-B_k . x)`` (start bonds ``+st_w``, end
        bonds ``-en_w``, fixed flows ``-K pay_w``) and the value is ``P(0,te) E^te[V(x)^+]`` with ``x(te)``
        Gaussian under the ``te``-forward measure: mean ``-Cov(x(te), I(te)) 1``, covariance ``Cov(x(te))``.

        - one factor: Jamshidian, root ``x*`` of ``V`` and ``E[e^{-B x} 1{x > x*}]`` in closed form;
        - two factors (Brigo-Mercurio): the first factor integrated by Gauss-Hermite (*nodes*), the second
          one in closed form given the first (root of the conditional ``V``).

        ``V`` is assumed to change sign once along the last factor (true for a swap: the start bond has the
        shortest maturity). The roots are found by bisection outside the tape: the value does not depend on
        them to first order (the payoff vanishes at the root), so the AAD gradients are exact.

        Args:
            te, p0_te: ``[n]`` expiry year fractions and ``P(0, te)``.
            pay_t, pay_w, p0_pay, st_t, st_w, p0_st, en_t, en_w, p0_en: ``[n, m]`` flows (pad with zero weights).
            strike: ``[n]`` fixed rates.
            payer (bool): Payer or receiver swaptions.
            nodes (int): Gauss-Hermite nodes of the outer integral (two factors).

        Returns:
            tf.Tensor: ``[n]`` float64 values.

        Raises:
            ValueError: With more than two factors.
        """
        if self.size() > 2:
            raise ValueError("exact swaptions for one or two factors only")
        te = np.atleast_1d(np.asarray(te, dtype=float))
        (c_p, b_p), (c_s, b_s), (c_e, b_e) = self._swap_legs(te, pay_t, p0_pay, st_t, p0_st, en_t, p0_en, p0_te)
        k = tf.constant(np.asarray(strike, dtype=float), DTYPE)[:, None]
        w_p, w_s, w_e = (tf.constant(np.asarray(w, dtype=float), DTYPE) for w in (pay_w, st_w, en_w))
        a = tf.concat([w_s * c_s, -w_e * c_e, -k * w_p * c_p], axis=1)                          # [n, m]
        bb = tf.concat([b_s, b_e, b_p], axis=1)                                                  # [n, m, nf]
        c_xx, c_xi = self._integrals(te, 0.0, "xx,xi")
        mean = -tf.reduce_sum(c_xi, axis=-1)                                                     # [n, nf]
        phi = 1.0 if payer else -1.0
        if self.size() == 1:
            value = _call_on_bond_sum(a, bb[..., 0], mean[:, 0], tf.sqrt(c_xx[:, 0, 0]), phi)
        else:
            z, w = np.polynomial.hermite_e.hermegauss(nodes)
            z, w = tf.constant(z, DTYPE), tf.constant(w / np.sqrt(2.0 * np.pi), DTYPE)
            s1, s2 = tf.sqrt(c_xx[:, 0, 0]), tf.sqrt(c_xx[:, 1, 1])
            r = c_xx[:, 0, 1] / (s1 * s2)
            x1 = mean[:, 0:1] + s1[:, None] * z[None, :]                                         # [n, q]
            m2 = mean[:, 1:2] + (r * s2)[:, None] * z[None, :]
            sd2 = tf.sqrt(tf.maximum(1.0 - r * r, 1e-300)) * s2
            a_cond = a[:, None, :] * tf.exp(-bb[:, None, :, 0] * x1[..., None])                  # [n, q, m]
            inner = _call_on_bond_sum(a_cond, tf.broadcast_to(bb[:, None, :, 1], tf.shape(a_cond)), m2,
                                      tf.broadcast_to(sd2[:, None], tf.shape(m2)), phi)
            value = tf.reduce_sum(w[None, :] * inner, axis=1)
        return tf.constant(np.asarray(p0_te, dtype=float), DTYPE) * value

    def swaption_from_arrays(self, te: float, pay_t, pay_w, p0_pay, st_t, st_w, p0_st, en_t, en_w, p0_en,
                             p0_te: float, strike: float, payer: bool = True, method: str = "approx"):
        """Swaption from time-0 bond values: Gaussian swap-rate approximation (default) or exact.

        ``S(x) = (sum st_w P(te, st) - sum en_w P(te, en)) / sum pay_w P(te, pay)`` with
        ``P(te, T; x) = c_T exp(-B(T - te) x)``, ``c_T = P(0,T)/P(0,te) exp((V(te,T) - V(0,T) + V(0,te))/2)``.

        - ``"approx"``: ``g = dS/dx`` at ``x = 0`` analytically (``dP/dx = -B P``), normal vol
          ``sigma_N^2 te = g^T Cov(x(te)) g``, price ``A(0) Bachelier(S(0), K, sigma_N, te)``;
        - ``"exact"``: :meth:`swaption_exact_values`; ``normal_vol`` is then the implied Bachelier vol
          (Newton on the price, outside the tape).

        Returns:
            tuple: ``(price, forward_swap_rate, annuity, normal_vol)``.

        Raises:
            ValueError: If *method* is unknown.
        """
        annuity = float(np.dot(pay_w, p0_pay))
        forward = (float(np.dot(st_w, p0_st)) - float(np.dot(en_w, p0_en))) / annuity
        phi = 1.0 if payer else -1.0
        cdf = lambda v: 0.5 * (1.0 + tf.math.erf(v / np.sqrt(2.0)))
        pdf = lambda v: tf.exp(-0.5 * v * v) / np.sqrt(2.0 * np.pi)
        batch = lambda v: np.asarray(v, dtype=float)[None]
        arrays = dict(te=batch(te), pay_t=batch(pay_t), pay_w=batch(pay_w), p0_pay=batch(p0_pay), st_t=batch(st_t),
                      st_w=batch(st_w), p0_st=batch(p0_st), en_t=batch(en_t), en_w=batch(en_w), p0_en=batch(p0_en),
                      p0_te=batch(p0_te))
        if method == "approx":
            normal_vol = self.swaption_normal_vols(**arrays)[0]
            std = normal_vol * np.sqrt(te)
            d = (forward - strike) / std
            price = annuity * (phi * (forward - strike) * cdf(phi * d) + std * pdf(d))
        elif method == "exact":
            price = self.swaption_exact_values(strike=batch(strike), payer=payer, **arrays)[0]
            normal_vol = tf.constant(_implied_bachelier(float(price) / annuity, forward, strike, te, phi), DTYPE)
        else:
            raise ValueError(f"method must be 'approx' or 'exact', got '{method}'")
        return price, tf.constant(forward, DTYPE), tf.constant(annuity, DTYPE), normal_vol

    def swaption(self, expiry, fixed_flows, floating_flows, strike: float, payer: bool = True,
                 method: str = "approx"):
        """European swaption (physical settlement): Gaussian swap-rate approximation or exact formula.

        Swap rate at expiry ``S(x) = sum_j n_j (basis_j P(te, s_j) - P(te, e_j)) / sum_i w_i P(te, p_i)`` with the
        model bonds (floating periods paid at their end, deterministic index basis).

        - ``"approx"`` (default): ``S`` linearised in the factors at ``x = 0``: ``g = dS/dx``, normal vol
          ``sigma_N^2 te = g^T Cov(x(te)) g``; price ``A(0) * Bachelier(S(0), K, sigma_N, te)`` with
          ``A(0) = sum_i w_i P(0, p_i)`` and ``S(0)`` the forward swap rate. The frozen-weight vol is within
          ~0.4bp of the exact ATM vol and has no smile.
        - ``"exact"``: Jamshidian (one factor) / Brigo-Mercurio one-dimensional integral (G2++), see
          :meth:`swaption_exact_values`.

        The curve values are read as numbers (no gradient to the curve).

        Args:
            expiry: Exercise date / year fraction.
            fixed_flows: ``[(pay, tau_times_notional), ...]`` of the fixed leg.
            floating_flows: ``[(start, end, notional, basis), ...]`` of the floating leg (paid at ``end``).
            strike (float): Fixed rate.
            payer (bool): Payer (call on the swap rate) or receiver.
            method (str): ``"approx"`` or ``"exact"``.

        Returns:
            tuple: ``(price, forward_swap_rate, annuity, normal_vol)`` float64 tensors.
        """
        te = self.time(expiry)
        p0 = lambda T: float(self._curve.discount(T if isinstance(T, date) else self.time(T)))
        pay_t = [self.time(p) for p, _ in fixed_flows]
        st_t = [self.time(st) for st, _, _, _ in floating_flows]
        en_t = [self.time(en) for _, en, _, _ in floating_flows]
        return self.swaption_from_arrays(
            te, pay_t, [w for _, w in fixed_flows], [p0(p) for p, _ in fixed_flows],
            st_t, [n * float(b) for _, _, n, b in floating_flows], [p0(st) for st, _, _, _ in floating_flows],
            en_t, [n for _, _, n, _ in floating_flows], [p0(en) for _, en, _, _ in floating_flows],
            p0(expiry), strike, payer, method)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _transition(self, t0: float, t1: float):
        """Mean matrix ``M [2n, 2n]`` and covariance ``C [2n, 2n]`` of ``z = (x, I)`` from *t0* to *t1*."""
        k = self._kappa
        dt = t1 - t0
        e = tf.exp(-k * dt)
        b = -tf.math.expm1(-k * dt) / k                                      # B_i(dt)
        c_xx, c_xi, c_ii = self._integrals(t1, t0)                       # c_xi = Cov(x_i, I_j)
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
            mean, cov = self._transition(t_prev, t)
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
        var = self.variance(0.0, times)
        return xs, p0[None, :] * tf.exp(-integrals - 0.5 * var[None, :])
