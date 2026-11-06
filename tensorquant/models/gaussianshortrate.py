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

    def exchange_option_values(self, t, s, e, A, B) -> tf.Tensor:
        """Vectorized ``E[D(0,t) (a P(t,s) - b P(t,e))^+]`` given ``A = a P(0,s)`` and ``B = b P(0,e)``.

        Args:
            t, s, e: ``[n]`` year fractions (``t <= s, e``).
            A, B: ``[n]`` forward values of the two legs at time 0 (tensors or arrays).

        Returns:
            tf.Tensor: ``[n]`` option values (Margrabe on the Gaussian log bond ratio).
        """
        t, s, e = (np.asarray(v, dtype=float) for v in (t, s, e))
        k_i, k_j = self._kappa[:, None], self._kappa[None, :]
        tt = tf.constant(t, DTYPE)[:, None, None]
        cov = (self._sigma[:, None] * self._sigma[None, :] * self._rho)[None] *             (1.0 - tf.exp(-(k_i + k_j)[None] * tt)) / (k_i + k_j)[None]                    # [n, nf, nf]
        db = self._B(e - t) - self._B(s - t)                                                  # [n, nf]
        v = tf.sqrt(tf.maximum(tf.einsum("ni,nij,nj->n", db, cov, db), 1e-300))
        A, B = _as_param(A), _as_param(B)
        d1 = (tf.math.log(A / B) + 0.5 * v * v) / v
        cdf = lambda x: 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0)))
        return A * cdf(d1) - B * cdf(d1 - v)

    def bond_exchange_option(self, t, s, e, a, b) -> tf.Tensor:
        """``E[D(0, t) (a P(t, s) - b P(t, e))^+]`` in closed form (``t <= s, e``).

        ``ln(P(t,s) / P(t,e))`` is Gaussian with variance
        ``v^2 = sum_ij rho_ij sigma_i sigma_j (B_i(e-t) - B_i(s-t)) (B_j(e-t) - B_j(s-t)) (1 - e^{-(k_i+k_j) t}) / (k_i + k_j)``,
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

    def _variance_tau(self, tau) -> tf.Tensor:
        """``V`` for horizons ``tau`` of any shape (``V(t, t + tau)`` does not depend on ``t``)."""
        tau = tf.constant(np.asarray(tau, dtype=float), DTYPE)[..., None, None]
        k_i, k_j = self._kappa[:, None], self._kappa[None, :]
        s = self._sigma[:, None] * self._sigma[None, :] * self._rho
        b_i, b_j = (1.0 - tf.exp(-k_i * tau)) / k_i, (1.0 - tf.exp(-k_j * tau)) / k_j
        b_ij = (1.0 - tf.exp(-(k_i + k_j) * tau)) / (k_i + k_j)
        return tf.reduce_sum(s / (k_i * k_j) * (tau - b_i - b_j + b_ij), axis=[-2, -1])

    def swaption_normal_vols(self, te, pay_t, pay_w, p0_pay, st_t, st_w, p0_st, en_t, en_w, p0_en, p0_te) -> tf.Tensor:
        """Batched Gaussian swap-rate normal vols ``[n]`` (see :meth:`swaption_from_arrays`).

        Flow arrays are ``[n, m]`` (pad shorter swaps with zero weights), ``te`` and ``p0_te`` ``[n]``.
        """
        te = np.asarray(te, dtype=float)
        v_te = self._variance_tau(te)[:, None]
        p0_te = tf.constant(np.asarray(p0_te, dtype=float), DTYPE)[:, None]

        def legs(times, p0):
            horizon = np.maximum(np.asarray(times, dtype=float) - te[:, None], 0.0)
            conv = 0.5 * (self._variance_tau(horizon) - self._variance_tau(np.asarray(times, dtype=float)) + v_te)
            c = tf.constant(np.asarray(p0, dtype=float), DTYPE) / p0_te * tf.exp(conv)
            return c, self._B(horizon)

        c_p, b_p = legs(pay_t, p0_pay)
        c_s, b_s = legs(st_t, p0_st)
        c_e, b_e = legs(en_t, p0_en)
        w_p, w_s, w_e = (tf.constant(np.asarray(w, dtype=float), DTYPE) for w in (pay_w, st_w, en_w))
        a_x = tf.reduce_sum(w_p * c_p, axis=1)
        f_x = tf.reduce_sum(w_s * c_s, axis=1) - tf.reduce_sum(w_e * c_e, axis=1)
        da = -tf.reduce_sum((w_p * c_p)[..., None] * b_p, axis=1)
        df = -tf.reduce_sum((w_s * c_s)[..., None] * b_s, axis=1) + tf.reduce_sum((w_e * c_e)[..., None] * b_e, axis=1)
        g = (df * a_x[:, None] - f_x[:, None] * da) / (a_x * a_x)[:, None]
        k_i, k_j = self._kappa[:, None], self._kappa[None, :]
        tt = tf.constant(te, DTYPE)[:, None, None]
        cov = (self._sigma[:, None] * self._sigma[None, :] * self._rho) * (1.0 - tf.exp(-(k_i + k_j) * tt)) / (k_i + k_j)
        return tf.sqrt(tf.maximum(tf.einsum("ni,nij,nj->n", g, cov, g), 1e-300) / tf.constant(te, DTYPE))

    def swaption_from_arrays(self, te: float, pay_t, pay_w, p0_pay, st_t, st_w, p0_st, en_t, en_w, p0_en,
                             p0_te: float, strike: float, payer: bool = True):
        """Vectorized Gaussian swap-rate swaption (see :meth:`swaption`) from time-0 bond values.

        ``S(x) = (sum st_w P(te, st) - sum en_w P(te, en)) / sum pay_w P(te, pay)`` with
        ``P(te, T; x) = c_T exp(-B(T - te) x)``, ``c_T = P(0,T)/P(0,te) exp((V(te,T) - V(0,T) + V(0,te))/2)``;
        ``g = dS/dx`` at ``x = 0`` analytically (``dP/dx = -B P``).

        Returns:
            tuple: ``(price, forward_swap_rate, annuity, normal_vol)``.
        """
        def legs(times, p0):
            times = np.asarray(times, dtype=float)
            v_te = self._variance_tau([te])[0]
            conv = 0.5 * (self._variance_tau(times - te) - self._variance_tau(times) + v_te)
            c = tf.constant(np.asarray(p0, dtype=float), DTYPE) / p0_te * tf.exp(conv)        # [m]
            return c, self._B(times - te)                                                       # [m], [m, nf]

        c_p, b_p = legs(pay_t, p0_pay)
        c_s, b_s = legs(st_t, p0_st)
        c_e, b_e = legs(en_t, p0_en)
        w_p, w_s, w_e = (tf.constant(np.asarray(w, dtype=float), DTYPE) for w in (pay_w, st_w, en_w))
        a_x = tf.reduce_sum(w_p * c_p)
        f_x = tf.reduce_sum(w_s * c_s) - tf.reduce_sum(w_e * c_e)
        da = -tf.reduce_sum((w_p * c_p)[:, None] * b_p, axis=0)                                # [nf]
        df = -tf.reduce_sum((w_s * c_s)[:, None] * b_s, axis=0) + tf.reduce_sum((w_e * c_e)[:, None] * b_e, axis=0)
        g = (df * a_x - f_x * da) / (a_x * a_x)
        k_i, k_j = self._kappa[:, None], self._kappa[None, :]
        cov = self._sigma[:, None] * self._sigma[None, :] * self._rho * (1.0 - tf.exp(-(k_i + k_j) * te)) / (k_i + k_j)
        normal_vol = tf.sqrt(tf.maximum(tf.einsum("i,ij,j->", g, cov, g), 1e-300) / te)
        annuity = float(np.dot(pay_w, p0_pay))
        forward = (float(np.dot(st_w, p0_st)) - float(np.dot(en_w, p0_en))) / annuity
        phi = 1.0 if payer else -1.0
        std = normal_vol * np.sqrt(te)
        d = (forward - strike) / std
        cdf = lambda v: 0.5 * (1.0 + tf.math.erf(v / np.sqrt(2.0)))
        pdf = lambda v: tf.exp(-0.5 * v * v) / np.sqrt(2.0 * np.pi)
        price = annuity * (phi * (forward - strike) * cdf(phi * d) + std * pdf(d))
        return price, tf.constant(forward, DTYPE), tf.constant(annuity, DTYPE), normal_vol

    def swaption(self, expiry, fixed_flows, floating_flows, strike: float, payer: bool = True):
        """European swaption (physical settlement) by the Gaussian swap-rate approximation.

        Swap rate at expiry ``S(x) = sum_j n_j (basis_j P(te, s_j) - P(te, e_j)) / sum_i w_i P(te, p_i)`` with the
        model bonds (floating periods paid at their end, deterministic index basis). ``S`` is linearised in the
        factors at ``x = 0``: ``g = dS/dx``, normal vol ``sigma_N^2 te = g^T Cov(x(te)) g``; price
        ``A(0) * Bachelier(S(0), K, sigma_N, te)`` with ``A(0) = sum_i w_i P(0, p_i)`` and ``S(0)`` the forward
        swap rate. The forward is exact; the frozen-weight vol is within ~0.4bp of the exact (MC) ATM vol.
        The curve values are read as numbers (no gradient to the curve).

        Args:
            expiry: Exercise date / year fraction.
            fixed_flows: ``[(pay, tau_times_notional), ...]`` of the fixed leg.
            floating_flows: ``[(start, end, notional, basis), ...]`` of the floating leg (paid at ``end``).
            strike (float): Fixed rate.
            payer (bool): Payer (call on the swap rate) or receiver.

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
            p0(expiry), strike, payer)

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
