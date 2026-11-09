"""Heston stochastic volatility equity model on the market forward."""

from __future__ import annotations

from datetime import date
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import least_squares

from .brownian import observation_columns
from .displaceddiffusion import _black_implied_vol
from .equitymodel import EquityModel
from .stochasticprocess import StochasticProcess
from ..numericalhandles.dtypes import DTYPE

HESTON_PARAMS = ("v0", "kappa", "theta", "xi", "rho")
HESTON_BOUNDS = {"v0": (1e-4, 1.0), "kappa": (1e-2, 20.0), "theta": (1e-4, 1.0), "xi": (1e-2, 5.0), "rho": (-0.99, 0.99)}
COS_TERMS = 1024          # pricing: |COS - Lewis quadrature| ~1e-15 for usual parameters
COS_WIDTH = 30.0
SOLVE_TERMS = 256         # calibration search (~1e-7): the final Newton step uses the full precision
SOLVE_WIDTH = 12.0
CTYPE = tf.complex128


def _f64(x) -> tf.Tensor:
    """float64 tensor; numbers / lists go straight to float64 (``tf.cast`` of a Python float goes through float32)."""
    if isinstance(x, (tf.Tensor, tf.Variable)):
        return tf.cast(x, DTYPE)
    return tf.constant(np.asarray(x, dtype=np.float64), DTYPE)


# ============================================================
# Characteristic function, cumulants, COS prices (per unit forward, undiscounted)
# ============================================================

def heston_characteristic_function(u, t, v0, kappa, theta, xi, rho) -> tf.Tensor:
    """``E[exp(i u ln X_t)]`` of the Heston forward ratio ``X = S / F`` (``X_0 = 1``, martingale).

    ``dX / X = sqrt(v) dW_1``, ``dv = kappa (theta - v) dt + xi sqrt(v) dW_2``, ``d<W_1, W_2> = rho dt``.
    Stable form (Albrecher et al., "little Heston trap"): with ``b = kappa - rho xi i u``,
    ``d = sqrt(b^2 + xi^2 (i u + u^2))``, ``g = (b - d) / (b + d)``::

        C = kappa theta / xi^2 [(b - d) t - 2 ln((1 - g e^{-d t}) / (1 - g))]
        D = (b - d) / xi^2 (1 - e^{-d t}) / (1 - g e^{-d t}),     phi = exp(C + D v0)

    Args:
        u: Real frequencies (broadcastable with *t*).
        t: Maturities (years).
        v0, kappa, theta, xi, rho: Heston parameters (scalars, tensors keep their gradients).

    Returns:
        tf.Tensor: complex128 values.
    """
    c = lambda x: tf.complex(_f64(x), tf.zeros_like(_f64(x)))        # (a real -> complex cast warns in the gradient)
    u, t = c(u), c(t)
    v0, kappa, theta, xi, rho = (c(p) for p in (v0, kappa, theta, xi, rho))
    iu = tf.complex(tf.constant(0.0, DTYPE), tf.constant(1.0, DTYPE)) * u
    b = kappa - rho * xi * iu
    d = tf.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    e = tf.exp(-d * t)
    big_c = kappa * theta / (xi * xi) * ((b - d) * t - 2.0 * tf.math.log((1.0 - g * e) / (1.0 - g)))
    big_d = (b - d) / (xi * xi) * (1.0 - e) / (1.0 - g * e)
    return tf.exp(big_c + big_d * v0)


def heston_cumulants(t, v0, kappa, theta, xi, rho) -> tuple:
    """Mean and variance of ``ln X_t`` (the COS truncation range).

    ``c1 = E[ln X_t] = -1/2 int_0^t E[v]`` in closed form; ``c2`` from the characteristic function,
    ``ln phi(u) = i c1 u - c2 u^2 / 2 + O(u^3)``, by a central second difference (``h = 1e-3``: relative error
    ~1e-6, plenty for a truncation range).
    """
    t, v0, kappa, theta, xi, rho = (_f64(v) for v in (t, v0, kappa, theta, xi, rho))
    e = tf.exp(-kappa * t)
    c1 = (1.0 - e) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * t
    h = 1e-3
    log_cf = tf.math.log(heston_characteristic_function(tf.constant([h, -h], DTYPE), t, v0, kappa, theta, xi, rho))
    c2 = -tf.reduce_sum(tf.math.real(log_cf), axis=-1, keepdims=True) / (h * h)            # ln phi(0) = 0
    return c1, c2


def heston_price(k, t, v0, kappa, theta, xi, rho, phi: float = 1.0, n_terms: int = COS_TERMS,
                 width: float = COS_WIDTH) -> tf.Tensor:
    """European options per unit forward, ``E[(phi (X_t - k))^+]``, by the COS method (Fang-Oosterlee).

    ``y = ln(X_t / k)`` is truncated to ``[a, b] = -ln k + c1 -/+ width sqrt(c2)`` (held out of the gradients); the put
    ``k (1 - e^y)^+`` has closed-form cosine coefficients on ``[a, 0]``, the call follows by parity
    (``C - P = 1 - k``). Differentiable w.r.t. the strikes and the parameters (TF, float64 / complex128).

    Args:
        k: ``[n, m]`` strikes over forward (rows: maturities).
        t: ``[n]`` maturities.
        v0, kappa, theta, xi, rho: Heston parameters.
        phi: 1 call, -1 put.
        n_terms: Cosine terms.
        width: Truncation width in standard deviations.

    Returns:
        tf.Tensor: ``[n, m]`` float64 prices.
    """
    k, t = _f64(k), tf.reshape(_f64(t), [-1, 1])
    c1, c2 = heston_cumulants(t, v0, kappa, theta, xi, rho)                                  # [n, 1]
    # the range is a numerical choice: held fixed (the gradient is that of the COS sum on it)
    c1, half = tf.stop_gradient(c1), tf.stop_gradient(width * tf.sqrt(tf.abs(c2) + 1e-12))
    x = -tf.math.log(k)                                                                     # [n, m]
    a, b = x + c1 - half, x + c1 + half
    span = 2.0 * half                                                                       # [n, 1]
    j = tf.range(n_terms, dtype=DTYPE)                                                      # [N]
    w = j[None, :] * np.pi / span                                                           # [n, N]
    cf = heston_characteristic_function(w, t, v0, kappa, theta, xi, rho)                    # [n, N]
    w3 = w[:, None, :]                                                                      # [n, 1, N]
    a3, b3 = a[..., None], b[..., None]                                                     # [n, m, 1]
    d = tf.maximum(tf.minimum(b3, 0.0), a3)                                                 # put region [a, min(b, 0)]
    cos_d, sin_d = tf.cos(w3 * (d - a3)), tf.sin(w3 * (d - a3))
    chi = (cos_d * tf.exp(d) - tf.exp(a3) + w3 * sin_d * tf.exp(d)) / (1.0 + w3 * w3)
    first = tf.equal(j, 0.0)[None, None, :]
    psi = tf.where(first, d - a3, sin_d / tf.where(first, tf.ones_like(w3), w3))
    coeff = 2.0 / span[..., None] * (psi - chi)                                             # [n, m, N]
    phase = tf.exp(tf.complex(tf.zeros_like(a3 * w3), w3 * (x - a)[..., None]))             # e^{i w (x - a)}
    terms = tf.math.real(cf[:, None, :] * phase) * coeff
    terms = tf.where(first, 0.5 * terms, terms)
    put = k * tf.reduce_sum(terms, axis=-1)
    return put if phi == -1.0 else put + 1.0 - k


# ============================================================
# Simulation (Andersen QE with martingale correction)
# ============================================================

@tf.function(reduce_retracing=True)
def _simulate_qe(rows, v0, dw):
    """Paths of ``(X, v)`` ``[n_paths, n_steps]`` each by Andersen's QE-M scheme.

    ``rows [n_steps, 10]`` per step: ``m0, e, s1, s2`` (``E[v'] = m0 + e v``, ``Var[v'] = s1 v + s2``),
    ``K0, K1, K2, K3, K4`` and ``A = K2 + K4 / 2``; ``dw [n_paths, n_steps, 2]`` (variance shock, independent
    log shock). ``ln X' = ln X + K0* + K1 v + K2 v' + sqrt(K3 v + K4 v') Z`` with ``K0*`` making ``E[X' | X] = X``
    exactly (falls back to ``K0`` where the moment condition fails). Rows read from TensorArrays.
    """
    n_paths, n_steps = tf.shape(dw)[0], tf.shape(dw)[1]
    coef = tf.TensorArray(DTYPE, size=n_steps).unstack(rows)
    z_v = tf.TensorArray(DTYPE, size=n_steps).unstack(tf.transpose(dw[..., 0]))
    z_x = tf.TensorArray(DTYPE, size=n_steps).unstack(tf.transpose(dw[..., 1]))
    v = tf.fill([n_paths], v0)
    log_x = tf.zeros([n_paths], DTYPE)
    xs, vs = tf.TensorArray(DTYPE, size=n_steps), tf.TensorArray(DTYPE, size=n_steps)
    for i in tf.range(n_steps):
        r = coef.read(i)
        m0, e, s1, s2, k0, k1, k2, k3, k4, big_a = (r[j] for j in range(10))
        zv = z_v.read(i)
        m = m0 + e * v
        psi = (s1 * v + s2) / (m * m)
        # quadratic branch (psi <= 1.5), inputs kept in range so the unused branch has finite gradients
        pq = tf.minimum(psi, 1.5)
        b2 = 2.0 / pq - 1.0 + tf.sqrt(2.0 / pq) * tf.sqrt(2.0 / pq - 1.0)
        aq = m / (1.0 + b2)
        v_q = aq * tf.square(tf.sqrt(b2) + zv)
        # exponential branch (psi > 1.5)
        pe = tf.maximum(psi, 1.5)
        p = (pe - 1.0) / (pe + 1.0)
        beta = (1.0 - p) / m
        tail = 0.5 * tf.math.erfc(zv / np.sqrt(2.0))                                       # 1 - Phi(z)
        v_e = tf.maximum(tf.math.log((1.0 - p) / tf.maximum(tail, 1e-300)), 0.0) / beta     # 0 where Phi(z) <= p
        quad = psi <= 1.5
        v_new = tf.where(quad, v_q, v_e)
        # martingale correction: E[exp(A v') | v] per branch
        den_q = 1.0 - 2.0 * big_a * aq
        log_mq = big_a * b2 * aq / tf.maximum(den_q, 1e-12) - 0.5 * tf.math.log(tf.maximum(den_q, 1e-12))
        log_me = tf.math.log(tf.maximum(p + beta * (1.0 - p) / tf.maximum(beta - big_a, 1e-12), 1e-300))
        ok = tf.where(quad, den_q > 0.0, beta > big_a)
        k0_star = tf.where(ok, -tf.where(quad, log_mq, log_me) - (k1 + 0.5 * k3) * v, k0)
        log_x = log_x + k0_star + k1 * v + k2 * v_new + tf.sqrt(tf.maximum(k3 * v + k4 * v_new, 1e-300)) * z_x.read(i)
        v = v_new
        xs, vs = xs.write(i, tf.exp(log_x)), vs.write(i, v)
    return tf.transpose(xs.stack()), tf.transpose(vs.stack())


# ============================================================
# HestonProcess
# ============================================================

class HestonProcess(StochasticProcess):
    """Heston stochastic volatility process on the market forward: the :class:`HestonModel` bound to an
    :class:`Equity` (or calibrated inside the caller's tape by :meth:`from_equity`).

    ``S_t = F(t) X_t`` with ``F(t) = equity.forward(t)`` (spot, repo, dividends, discount curve) and::

        dX / X = sqrt(v) dW_1,   dv = kappa (theta - v) dt + xi sqrt(v) dW_2,   d<W_1, W_2> = rho dt,

    ``X_0 = 1``, ``v_0 = v0`` (time-homogeneous parameters; discrete dividends act as proportional drops).
    Built by :meth:`HestonModel.process` (parameters held: model greeks) or by :meth:`from_equity` (calibrated to the
    vol surface inside the caller's tape: greeks through the calibration).

    - closed form: :func:`heston_price` (characteristic function in the stable form, COS method);
    - calibration: least squares on the prices of the whole surface, each slice scaled by its ATM vega (a
      vol error near the money, the wings count by their vega), bounded parameters, Feller condition
      reported; sensitivities to the vol nodes and forwards through the calibration (implicit function
      theorem at the optimum, exact Hessian held constant);
    - Monte Carlo: Andersen's QE scheme for the variance with the exact martingale correction of ``X``
      (``noise_size() = 2``).
    """

    def __init__(self, equity, v0, kappa, theta, xi, rho, martingale_correction: bool = True,
                 calibration: tuple = None):
        """Use :meth:`from_equity`, or give the parameters (numbers or tensors).

        Args:
            equity (Equity): The underlying (its ``forward`` drives the dynamics).
            v0, kappa, theta, xi, rho: Heston parameters.
            martingale_correction (bool): Rescale ``X_t`` to sample mean 1 (MC forward = ``F(t)``).
            calibration (tuple, optional): Calibration inputs of the report (:attr:`calibration`).

        Raises:
            ValueError: If a parameter is outside its natural range.
        """
        self._equity = equity
        self._p = {n: _f64(v) for n, v in zip(HESTON_PARAMS, (v0, kappa, theta, xi, rho))}
        values = {n: float(v) for n, v in self._p.items()}
        if min(values["v0"], values["kappa"], values["theta"], values["xi"]) <= 0.0 or abs(values["rho"]) >= 1.0:
            raise ValueError(f"invalid Heston parameters {values}")
        self._S0 = tf.cast(equity.spot, DTYPE)
        self._martingale_correction = martingale_correction
        self._calibration_inputs, self._calibration = calibration, None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @classmethod
    def from_equity(cls, equity, maturity: Union[date, float] = None, strike: float = None,
                    martingale_correction: bool = True, initial: dict = None) -> "HestonProcess":
        """Calibrate the Heston parameters to the vol surface of *equity* (differentiable w.r.t. the market).

        Args:
            equity (Equity): The underlying (``market_env.get_equity``).
            maturity, strike: Unused (same factory signature as the other MC models).
            martingale_correction (bool): Rescale the simulated ``X_t`` to sample mean 1 at each time.
            initial (dict, optional): Starting parameters (default from the ATM vols, ``kappa = 1.5``,
                ``xi = 0.5``, ``rho = -0.6``).

        Returns:
            HestonProcess: Calibrated process; :attr:`parameters`, :attr:`feller`, :attr:`calibration` (report).

        Raises:
            ValueError: If the underlying has no lognormal vol surface.
        """
        surface = equity.vol_surface
        if surface is None:
            raise ValueError(f"No volatility surface for underlying '{equity.ticker}'")
        if surface.vol_type != "lognormal":
            raise ValueError("Heston calibrates to lognormal (Black) vols")
        t = np.asarray(surface.maturity, dtype=float)
        forwards = tf.stack([tf.cast(equity.forward(float(v)), DTYPE) for v in t])
        k = _f64(np.asarray(surface.strike, dtype=float))[None, :] / forwards[:, None]                 # [n, m]
        iv = tf.cast(surface.volatility_matrix, DTYPE)
        market, weight = _market_targets(k, iv, t)
        k_c, t_c, market_c, weight_c = (tf.constant(v.numpy()) for v in (k, _f64(t), market, weight))
        atm = float(np.mean(iv.numpy()[:, np.argmin(np.abs(np.log(k.numpy())), axis=1)].diagonal()))
        start = {"v0": float(iv.numpy()[0, np.argmin(np.abs(np.log(k.numpy()[0])))]) ** 2,
                 "kappa": 1.5, "theta": atm ** 2, "xi": 0.5, "rho": -0.6}
        start.update(initial or {})
        lower, upper = (np.array([HESTON_BOUNDS[n][i] for n in HESTON_PARAMS]) for i in (0, 1))
        x0 = np.clip(np.array([start[n] for n in HESTON_PARAMS], dtype=float), lower, upper)
        args = (k_c, t_c, market_c, weight_c)
        fit = least_squares(lambda x: _residuals(tf.constant(x, DTYPE), *args).numpy(), x0, jac="2-point",
                            bounds=(lower, upper), method="trf", x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        # polish on the full-precision COS, inside the bounds (from the cheap optimum: a few evaluations)
        fit = least_squares(lambda x: _residuals(tf.constant(x, DTYPE), *args, precise=True).numpy(), fit.x,
                            jac="2-point", bounds=(lower, upper), method="trf", x_scale="jac",
                            xtol=1e-14, ftol=1e-14, gtol=1e-14)
        params = _implicit_step(fit.x, lower, upper, k, _f64(t), market, weight)
        report = (k.numpy(), iv.numpy(), t, params.numpy(), int(fit.nfev))
        return cls(equity, *[params[i] for i in range(5)], martingale_correction=martingale_correction,
                   calibration=report)

    @property
    def parameters(self) -> dict:
        """``{v0, kappa, theta, xi, rho}`` as float64 tensors (differentiable through the calibration)."""
        return dict(self._p)

    @property
    def feller(self) -> float:
        """``2 kappa theta - xi^2`` (>= 0: the variance does not reach 0)."""
        p = self._p
        return float(2.0 * p["kappa"] * p["theta"] - p["xi"] ** 2)

    @property
    def calibration(self) -> pd.DataFrame:
        """Calibration report per maturity (computed on first access; None if not calibrated): model - market
        vol errors in vol points (ATM, rms and max over the strikes); parameters on ``attrs``."""
        if self._calibration is None and self._calibration_inputs is not None:
            k, iv, t, x, nfev = self._calibration_inputs
            prices = heston_price(k, t, *x).numpy()
            rows = []
            for i in range(len(t)):
                errors = 100.0 * np.array([_black_implied_vol(float(p), kk, t[i]) - v
                                           for p, kk, v in zip(prices[i], k[i], iv[i])])
                finite = errors[np.isfinite(errors)]
                rows.append({"maturity": t[i], "atm_error_volpts": float(np.interp(0.0, np.log(k[i]), errors)),
                             "rms_error_volpts": float(np.sqrt(np.mean(finite ** 2))) if finite.size else np.nan,
                             "max_error_volpts": float(np.max(np.abs(finite))) if finite.size else np.nan})
            self._calibration = pd.DataFrame(rows)
            self._calibration.attrs.update({n: float(v) for n, v in zip(HESTON_PARAMS, x)})
            self._calibration.attrs.update(
                feller=float(2 * x[1] * x[2] - x[3] ** 2), evaluations=nfev,
                at_bound=[n for n, v in zip(HESTON_PARAMS, x) if np.isclose(v, HESTON_BOUNDS[n][0]) or np.isclose(v, HESTON_BOUNDS[n][1])])
        return self._calibration

    # ------------------------------------------------------------------
    # Closed forms
    # ------------------------------------------------------------------

    def _args(self):
        return [self._p[n] for n in HESTON_PARAMS]

    def vanilla_price(self, strike: float, maturity: Union[date, float], option_type: float = 1.0) -> tf.Tensor:
        """Closed-form European price ``P(0,T) F(T) E[(phi (X_T - K/F))^+]`` (COS).

        Args:
            strike (float): Strike.
            maturity: Date or year fraction.
            option_type (float): 1 call, -1 put (or an :class:`OptionType`).

        Returns:
            tf.Tensor: float64 price, differentiable w.r.t. the market (through the calibration).
        """
        phi = float(getattr(option_type, "value", option_type))
        T = self._equity.year_fraction(maturity)
        F = tf.cast(self._equity.forward(maturity), DTYPE)
        df = tf.cast(self._equity.discount_curve.discount(maturity), DTYPE)
        k = tf.reshape(np.float64(strike) / F, [1, 1])
        return df * F * heston_price(k, [T], *self._args(), phi=phi)[0, 0]

    def implied_vol(self, strike: float, maturity: Union[date, float]) -> float:
        """Black vol of the closed-form price (number, not differentiable)."""
        T = self._equity.year_fraction(maturity)
        k = float(np.float64(strike) / float(self._equity.forward(maturity)))
        return _black_implied_vol(float(heston_price([[k]], [T], *self._args())[0, 0]), k, T)

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
        return tf.stack([self._S0, self._p["v0"]])

    def _step_rows(self, dt: np.ndarray) -> tf.Tensor:
        """QE-M coefficients per step ``[n_steps, 10]`` (differentiable w.r.t. the parameters)."""
        p = self._p
        kappa, theta, xi, rho = p["kappa"], p["theta"], p["xi"], p["rho"]
        dt = _f64(dt)
        e = tf.exp(-kappa * dt)
        m0 = theta * (1.0 - e)
        s1 = xi * xi * e * (1.0 - e) / kappa
        s2 = theta * xi * xi * tf.square(1.0 - e) / (2.0 * kappa)
        k0 = -rho * kappa * theta / xi * dt
        k1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
        k2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
        k3 = k4 = 0.5 * dt * (1.0 - rho * rho)
        return tf.stack([m0, e, s1, s2, k0, k1, k2, k3, k4, k2 + 0.5 * k4], axis=1)

    def simulate(self, t_grid, dw) -> tuple:
        """Forward ratio and variance paths ``(X, v)`` ``[n_paths, n_steps]`` (QE-M, no rescaling).

        Args:
            t_grid: ``[n_steps]`` increasing times (> 0).
            dw: ``[n_paths, n_steps, 2]`` standard normals.
        """
        times = np.asarray(tf.convert_to_tensor(t_grid), dtype=float)
        if times[0] <= 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("t_grid must be increasing and > 0")
        dw = tf.cast(dw, DTYPE)
        if len(dw.shape) != 3 or int(dw.shape[-1]) != 2:
            raise ValueError("dw must be [n_paths, n_steps, 2]")
        return _simulate_qe(self._step_rows(np.diff(np.concatenate([[0.0], times]))), self._p["v0"], dw)

    def evolve(self, t_grid, dw, observation_times=None) -> tf.Tensor:
        """Simulate spot paths ``S_t = F(t) X_t``.

        Args:
            t_grid: ``[n_steps]`` increasing times (year fractions), without 0.
            dw: ``[n_paths, n_steps, 2]`` standard normals.
            observation_times: Optional subset of ``t_grid``: only those columns are returned and the
                market forward is computed only there.

        Returns:
            tf.Tensor: ``[n_paths, n_steps]`` (or ``[n_paths, len(observation_times)]``) spots.
        """
        times = np.asarray(tf.convert_to_tensor(t_grid), dtype=float)
        x, _ = self.simulate(times, dw)
        if observation_times is not None:
            cols = observation_columns(times, observation_times)
            x, times = tf.gather(x, cols, axis=1), times[cols]
        if self._martingale_correction:
            x = x / tf.reduce_mean(x, axis=0, keepdims=True)
        forwards = tf.stack([tf.cast(self._equity.forward(float(v)), DTYPE) for v in times])
        return x * forwards[None, :]


class HestonModel(EquityModel):
    """Heston model instance: the five parameters as float64 ``tf.Variable``s, no market.

    ``dX / X = sqrt(v) dW_1``, ``dv = kappa (theta - v) dt + xi sqrt(v) dW_2``, ``d<W_1, W_2> = rho dt`` on the
    market forward of the equity given to :meth:`process`. Build it by hand (``HestonModel(v0=..., ...)``: no
    Jacobian) or with :meth:`calibrate` (least squares on the whole vol surface, see :meth:`HestonProcess.from_equity`;
    stores the report and the calibration Jacobian ``dp / d(market inputs)`` for :func:`par_greeks`).

    Example::

        model = tq.HestonModel.calibrate(market_env.get_equity("SX5E"))
        pricer = tq.VanillaMCPricer(model=model)
        pricer.price(option, market_env)
        model_greeks = pricer.tape.gradient(option.price, model.parameters)
    """

    SETTINGS = {"martingale_correction": True}

    def __init__(self, v0, kappa, theta, xi, rho, **settings):
        """Args:
            v0, kappa, theta, xi, rho: Heston parameters (numbers or tensors: their values are taken).
            **settings: ``martingale_correction`` (sample mean of ``X_t`` rescaled to 1).

        Raises:
            ValueError: If a parameter is outside its natural range.
        """
        super().__init__(**settings)
        values = {n: float(np.asarray(tf.stop_gradient(v)) if tf.is_tensor(v) else v)
                  for n, v in zip(HESTON_PARAMS, (v0, kappa, theta, xi, rho))}
        if min(values["v0"], values["kappa"], values["theta"], values["xi"]) <= 0.0 or abs(values["rho"]) >= 1.0:
            raise ValueError(f"invalid Heston parameters {values}")
        self._p = {n: self._variable(v, f"heston_{n}") for n, v in values.items()}

    @property
    def parameters(self) -> dict:
        """``{v0, kappa, theta, xi, rho}`` as float64 ``tf.Variable``s (model greeks: ``tape.gradient(price,
        model.parameters)``)."""
        return dict(self._p)

    @property
    def feller(self) -> float:
        """``2 kappa theta - xi^2`` (>= 0: the variance does not reach 0)."""
        p = {n: float(v) for n, v in self._p.items()}
        return 2.0 * p["kappa"] * p["theta"] - p["xi"] ** 2

    def process(self, equity) -> HestonProcess:
        return HestonProcess(equity, *[self._p[n] for n in HESTON_PARAMS],
                             martingale_correction=bool(self.settings["martingale_correction"]))

    @classmethod
    def calibrate(cls, equity, initial: dict = None, **settings) -> "HestonModel":
        """Calibrate to the vol surface of *equity* (as :meth:`HestonProcess.from_equity`) and store the parameters,
        the report and the Jacobian of the parameters w.r.t. the market risk factors of *equity*.

        Args:
            equity (Equity): The underlying.
            initial (dict, optional): Starting parameters of the search.
            **settings: Model settings.
        """
        with tf.GradientTape(persistent=True) as tape:
            fitted = HestonProcess.from_equity(equity, initial=initial)
            params = fitted.parameters
        model = cls(**params, **settings)
        model._record_calibration(equity, params, tape, report=fitted.calibration)
        return model

    @classmethod
    def from_equity(cls, equity, maturity: Union[date, float] = None, strike: float = None, initial: dict = None,
                    **settings) -> HestonProcess:
        """Process calibrated inside the caller's tape (greeks through the calibration): :meth:`HestonProcess.from_equity`."""
        return HestonProcess.from_equity(equity, maturity, strike, initial=initial,
                                         martingale_correction=settings.get("martingale_correction", True))

    @classmethod
    def _from_values(cls, values: dict, settings: dict, frames: dict) -> "HestonModel":
        return cls(**{n: float(values[n][0]) for n in HESTON_PARAMS}, **settings)


# ============================================================
# Calibration helpers
# ============================================================

def _market_targets(k, iv, t):
    """Market call prices per unit forward ``[n, m]`` and the slice weights ``1 / ATM vega`` ``[n, 1]`` (TF).

    The ATM vol of a slice is the node nearest to the forward (one-hot: dense gradient)."""
    tt = _f64(t)[:, None]
    sd = iv * tf.sqrt(tt)
    d1 = -tf.math.log(k) / sd + 0.5 * sd
    cdf = lambda x: 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0)))
    market = cdf(d1) - k * cdf(d1 - sd)
    nearest = tf.one_hot(np.argmin(np.abs(np.log(k.numpy())), axis=1), int(k.shape[1]), dtype=DTYPE)
    atm = tf.reduce_sum(iv * nearest, axis=1, keepdims=True)
    vega = tf.sqrt(tt) * tf.exp(-0.125 * tf.square(atm) * tt) / np.sqrt(2.0 * np.pi)
    return market, 1.0 / vega


def _objective(x, k, t, market, weight):
    return 0.5 * tf.reduce_sum(tf.square((heston_price(k, t, *tf.unstack(x)) - market) * weight))


# one trace per session for any surface shape: (params [5], k [n, m], t [n], market [n, m], weight [n, 1])
_SIGNATURE = [tf.TensorSpec([len(HESTON_PARAMS)], DTYPE), tf.TensorSpec([None, None], DTYPE),
              tf.TensorSpec([None], DTYPE), tf.TensorSpec([None, None], DTYPE), tf.TensorSpec([None, 1], DTYPE)]


def _make_residuals(precise: bool):
    @tf.function(input_signature=_SIGNATURE)
    def residuals(x, k, t, market, weight):
        if precise:
            model = heston_price(k, t, *tf.unstack(x))
        else:
            model = heston_price(k, t, *tf.unstack(x), n_terms=SOLVE_TERMS, width=SOLVE_WIDTH)
        return tf.reshape((model - market) * weight, [-1])
    return residuals


_RESIDUALS = {False: _make_residuals(False), True: _make_residuals(True)}


def _residuals(x, k, t, market, weight, precise=False):
    """Residuals of the search: cheaper COS, or the pricing one with ``precise`` (compiled, traced once)."""
    return _RESIDUALS[precise](x, k, t, market, weight)


@tf.function(input_signature=_SIGNATURE)
def _gradient_and_hessian(x, k, t, market, weight):
    """Gradient of the objective (differentiable w.r.t. the market inputs) and its Hessian (held constant).

    The Hessian rows are gradients of the gradient components (one backward pass each)."""
    with tf.GradientTape(persistent=True) as outer:
        outer.watch(x)
        with tf.GradientTape() as inner:
            inner.watch(x)
            loss = _objective(x, k, t, market, weight)
        grad = inner.gradient(loss, x)
        rows = [grad[j] for j in range(len(HESTON_PARAMS))]
    hess = tf.stack([outer.gradient(r, x) for r in rows])
    return grad, tf.stop_gradient(hess)


def _implicit_step(theta: np.ndarray, lower, upper, k, t, market, weight) -> tf.Tensor:
    """Parameters as differentiable functions of the market (implicit function theorem).

    1. Value: one Newton step on the full-precision objective from the solver's point (which stops ~1e-8 away
       with a finite-difference Jacobian), kept only if small and inside the bounds (a near-degenerate Hessian,
       e.g. a flat smile, gives large steps along directions the prices do not see).
    2. Derivative: ``s = H^{-1} grad_p L(p*, m)``, ``p(m) = p* - (s - stop_gradient(s))``: the value stays ``p*``
       and ``dp/dm = -H^{-1} d grad / dm``, ``H`` the exact Hessian held constant. Parameters on a bound stay fixed.
    """
    free = ~(np.isclose(theta, lower) | np.isclose(theta, upper))
    f = tf.constant(free.astype(float), DTYPE)

    def newton_step(x, grad, hess):
        hess = hess * f[:, None] * f[None, :] + tf.linalg.diag(1.0 - f)
        return tf.linalg.solve(hess, (grad * f)[:, None])[:, 0]

    stopped = [tf.stop_gradient(v) for v in (k, t, market, weight)]
    step = newton_step(tf.constant(theta, DTYPE), *_gradient_and_hessian(tf.constant(theta, DTYPE), *stopped)).numpy()
    moved = theta - step
    if np.all(moved >= lower) and np.all(moved <= upper) and np.all(np.abs(step) <= 1e-4 * np.maximum(np.abs(theta), 1e-2)):
        theta = moved
    x = tf.constant(theta, DTYPE)
    step = newton_step(x, *_gradient_and_hessian(x, k, t, market, weight))
    return x - (step - tf.stop_gradient(step))
