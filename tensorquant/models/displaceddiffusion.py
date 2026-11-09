"""Displaced diffusion (shifted lognormal) equity model on the market forward."""

from __future__ import annotations

import warnings
from datetime import date
from typing import Sequence, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from .brownian import observation_columns
from .equitymodel import EquityModel
from .stochasticprocess import StochasticProcess
from ..numericalhandles.dtypes import DTYPE
from ..numericalhandles.implicit import implicit_sensitivities, implicit_solution, newton_step, optimality_derivatives

BETA_BOUNDS = (0.02, 1.5)
SIGMA_BOUNDS = (1e-4, 3.0)


def _f64(x) -> tf.Tensor:
    """float64 tensor; numbers / lists go straight to float64 (``tf.cast`` of a Python float goes through float32)."""
    if isinstance(x, (tf.Tensor, tf.Variable)):
        return tf.cast(x, DTYPE)
    return tf.constant(np.asarray(x, dtype=np.float64), DTYPE)


# ============================================================
# Closed form (per unit forward, undiscounted)
# ============================================================

def displaced_diffusion_price(k, beta, sigma, t, phi=1.0) -> tf.Tensor:
    """``E[(phi (X_T - k))^+]`` for ``dX = sigma (beta X + 1 - beta) dW``, ``X_0 = 1`` (constant parameters).

    ``Y = beta X + 1 - beta`` is lognormal with mean 1 and vol ``beta sigma``, so the call is
    ``Black(1, k*, beta sigma, t) / beta`` with the shifted strike ``k* = beta k + 1 - beta``
    (``k* <= 0``: always exercised, ``1 - k``); the put by parity ``C - P = 1 - k``. ``beta = 1`` is Black,
    ``beta -> 0`` Bachelier with normal vol ``sigma``. Broadcasts; differentiable in every argument.

    Args:
        k: Strike over forward ``K / F(T)``.
        beta: Skew parameter (> 0).
        sigma: Vol parameter (the ATM local vol).
        t: Maturity (years, > 0).
        phi: 1 call, -1 put.

    Returns:
        tf.Tensor: float64 price per unit forward (multiply by ``P(0,T) F(T)``).
    """
    k, beta, sigma, t = (_f64(v) for v in (k, beta, sigma, t))
    ks = beta * k + 1.0 - beta
    safe = tf.maximum(ks, 1e-300)
    sd = beta * sigma * tf.sqrt(t)
    d1 = -tf.math.log(safe) / sd + 0.5 * sd
    cdf = lambda x: 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0)))
    call = tf.where(ks > 0.0, (cdf(d1) - safe * cdf(d1 - sd)) / beta, 1.0 - k)
    return call if phi == 1.0 else call - (1.0 - k)


def _price_np(k, beta, sigma, t):
    """Numpy call of :func:`displaced_diffusion_price` (calibration objective)."""
    ks = beta * k + 1.0 - beta
    safe = np.maximum(ks, 1e-300)
    sd = beta * sigma * np.sqrt(t)
    d1 = -np.log(safe) / sd + 0.5 * sd
    return np.where(ks > 0.0, (norm.cdf(d1) - safe * norm.cdf(d1 - sd)) / beta, 1.0 - k)


def _black_tf(k, vol, t):
    """Black call per unit forward (TF)."""
    sd = vol * tf.sqrt(t)
    d1 = -tf.math.log(k) / sd + 0.5 * sd
    cdf = lambda x: 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0)))
    return cdf(d1) - k * cdf(d1 - sd)


def _black_implied_vol(price: float, k: float, t: float) -> float:
    """Black vol of an undiscounted call per unit forward (NaN outside the no-arbitrage bounds)."""
    def black(v):
        sd = v * np.sqrt(t)
        d1 = -np.log(k) / sd + 0.5 * sd
        return norm.cdf(d1) - k * norm.cdf(d1 - sd)
    lo, hi = 1e-6, 5.0
    if not black(lo) < price < black(hi):
        return float("nan")
    return brentq(lambda v: black(v) - price, lo, hi, xtol=1e-14)


# ============================================================
# Skew averaging (Piterbarg): forward pieces <-> effective (slice) parameters
# ============================================================

def _averaging_weights(piece_var):
    """``W_j = int_piece sigma^2(t) v(t) dt = var_j (cum_{j-1} + var_j / 2)`` with ``v(t) = int_0^t sigma^2``."""
    cum = tf.cumsum(piece_var, axis=-1)
    return piece_var * (cum - piece_var + 0.5 * piece_var), cum


def forward_from_slices(t_nodes, beta_slice, sigma_slice):
    """Piecewise constant forward parameters on ``[T_{j-1}, T_j)`` from the slice (effective) ones.

    ``sigma``: total variance ``sigma_T^2 T`` differenced; ``beta``: inverse of the skew averaging
    ``beta_T = sum_j beta_j W_j / sum_j W_j`` (``W_j = sigma_j^2 int_piece v(t) dt``, ``v`` the cumulative
    variance), solved maturity by maturity (linear). Differentiable (TF).

    Returns:
        tuple: ``(beta_fwd [n], sigma_fwd [n], calendar_floor_hits)``.
    """
    t = np.asarray(t_nodes, dtype=float)
    dt = tf.constant(np.diff(np.concatenate([[0.0], t])), DTYPE)
    total = tf.square(sigma_slice) * tf.constant(t, DTYPE)
    piece_var = total - tf.concat([tf.zeros([1], DTYPE), total[:-1]], axis=0)
    floored = int(tf.reduce_sum(tf.cast(piece_var <= 1e-10, tf.int32)))
    piece_var = tf.maximum(piece_var, 1e-10)
    sigma_fwd = tf.sqrt(piece_var / dt)
    w, cum = _averaging_weights(piece_var)
    total_w = 0.5 * tf.square(cum)                                            # sum_{i<=j} W_i
    betas = []
    for j in range(len(t)):
        acc = tf.add_n([betas[i] * w[i] for i in range(j)]) if j else tf.constant(0.0, DTYPE)
        betas.append((beta_slice[j] * total_w[j] - acc) / w[j])
    return tf.stack(betas), sigma_fwd, floored


def _piece_overlaps(knots: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """``[n_intervals, n_pieces]`` durations of ``[start, end]`` inside each piece (last piece open-ended)."""
    lo = np.concatenate([[0.0], knots[:-1]])
    hi = np.concatenate([knots[:-1], [np.inf]])
    return np.maximum(np.minimum(ends[:, None], hi[None, :]) - np.maximum(starts[:, None], lo[None, :]), 0.0)


# ============================================================
# Simulation
# ============================================================

@tf.function(reduce_retracing=True)
def _simulate_ratio(beta, step_sd, dw):
    """Paths of ``X`` ``[n_paths, n_steps]``: exact shifted-lognormal step with the step's ``beta`` and the
    standard deviation ``step_sd`` of ``ln Y`` (``Y = beta X + 1 - beta``, ``Y' = Y exp(sd Z - sd^2 / 2)``); rows read from
    TensorArrays (dense gradients, see :mod:`localvolatility`)."""
    n_paths, n_steps = tf.shape(dw)[0], tf.shape(dw)[1]
    betas = tf.TensorArray(DTYPE, size=n_steps).unstack(beta)
    sds = tf.TensorArray(DTYPE, size=n_steps).unstack(step_sd)
    normals = tf.TensorArray(DTYPE, size=n_steps).unstack(tf.transpose(dw))
    x = tf.ones([n_paths], DTYPE)
    paths = tf.TensorArray(DTYPE, size=n_steps)
    for k in tf.range(n_steps):
        b, sd = betas.read(k), sds.read(k)
        growth = tf.math.expm1(sd * normals.read(k) - 0.5 * sd * sd)
        x = x + (b * x + 1.0 - b) * growth / b
        paths = paths.write(k, x)
    return tf.transpose(paths.stack(), perm=[1, 0])


# ============================================================
# DisplacedDiffusionModel
# ============================================================

class DisplacedDiffusionProcess(StochasticProcess):
    """Displaced diffusion (shifted lognormal) process on the market forward: the :class:`DisplacedDiffusionModel`
    bound to an :class:`Equity` (or calibrated inside the caller's tape by :meth:`from_equity`).

    ``S_t = F(t) X_t`` with ``F(t) = equity.forward(t)`` and ``dX = sigma(t) (beta(t) X + 1 - beta(t)) dW``,
    ``X_0 = 1``: ``beta = 1`` is lognormal, ``beta -> 0`` normal, ``beta < 1`` a negative equity skew.
    ``beta(t)``, ``sigma(t)`` are piecewise constant between the maturities of the vol surface (forward
    parameters). :meth:`from_equity` calibrates it:

    1. slice calibration: per surface maturity, ``(beta_T, sigma_T)`` of the closed form
       :func:`displaced_diffusion_price` fitted to the market smile by least squares on the prices per unit
       forward (a vol error weighted by its vega: the strikes near the money dominate, the wings count little);
    2. forward parameters by inverting Piterbarg's skew averaging (:func:`forward_from_slices`);
    3. Monte Carlo with exact shifted-lognormal steps (a step across a piece boundary uses the
       variance-weighted ``beta`` and the summed variance ``sum sigma_j^2 dt_j``: the ATM variance is kept).

    The calibration is differentiable: the slice parameters are the solution of the first-order
    conditions, so a Newton step on them at the optimum (implicit function theorem) carries the exact
    sensitivities to the vol nodes and the forwards (spot, repo, dividends, curve). Vanilla prices at the
    nodes are the fitted closed form; with time-dependent ``beta`` the MC reprices the slices only up to the
    skew-averaging approximation.

    Skew limit: the steepest ATM skew is the normal one (``beta -> 0``), ``d sigma_imp / d ln K ~ -sigma / 2``.
    Steeper equity skews (index smiles) are not reached: ``beta`` ends on its lower bound and the wings are
    off (``calibration["beta_at_bound"]``, vol errors per slice). For full smiles use the local vol.
    """

    def __init__(self, equity, t_nodes, beta_fwd, sigma_fwd, beta_slice=None, sigma_slice=None,
                 martingale_correction: bool = True, calibration: tuple = None):
        """Use :meth:`from_equity`, or give the forward parameters directly.

        Args:
            equity (Equity): The underlying (its ``forward`` drives the dynamics).
            t_nodes: ``[n]`` piece ends (year fractions, increasing; the last piece continues after it).
            beta_fwd, sigma_fwd: ``[n]`` forward parameters per piece (tensors keep their gradients).
            beta_slice, sigma_slice: ``[n]`` effective slice parameters (default: averaged from the pieces).
            martingale_correction (bool): Rescale ``X_t`` to sample mean 1 (MC forward = ``F(t)``).
            calibration (tuple, optional): Calibration inputs of the report (:attr:`calibration`).

        Raises:
            ValueError: If the shapes differ, the nodes are not increasing or a ``beta`` is not positive.
        """
        self._equity = equity
        self._t = np.asarray(t_nodes, dtype=float)
        self._beta = tf.reshape(_f64(beta_fwd), [-1])
        self._sigma = tf.reshape(_f64(sigma_fwd), [-1])
        if not len(self._t) == int(self._beta.shape[0]) == int(self._sigma.shape[0]):
            raise ValueError("t_nodes, beta_fwd and sigma_fwd must have the same length")
        if self._t[0] <= 0.0 or np.any(np.diff(self._t) <= 0.0):
            raise ValueError("t_nodes must be increasing and > 0")
        if float(tf.reduce_min(self._beta)) <= 0.0:
            raise ValueError("beta must be positive")
        if beta_slice is None or sigma_slice is None:
            beta_slice, sigma_slice = self.effective_parameters(self._t)
        self._beta_slice, self._sigma_slice = _f64(beta_slice), _f64(sigma_slice)
        self._S0 = tf.cast(equity.spot, DTYPE)
        self._martingale_correction = martingale_correction
        self._calibration_inputs, self._calibration = calibration, None

    @property
    def calibration(self) -> pd.DataFrame:
        """Calibration report per maturity (computed on first access; None if not calibrated): slice and forward
        parameters, ``beta_at_bound``, model - market vol errors in vol points (ATM, rms, max over the strikes)."""
        if self._calibration is None and self._calibration_inputs is not None:
            self._calibration = _report(*self._calibration_inputs)
        return self._calibration

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @classmethod
    def from_equity(cls, equity, maturity: Union[date, float] = None, strike: float = None,
                    martingale_correction: bool = True) -> "DisplacedDiffusionProcess":
        """Calibrate the model to the vol surface of *equity* (differentiable w.r.t. the market).

        Args:
            equity (Equity): The underlying (``market_env.get_equity``).
            maturity, strike: Unused (same factory signature as the other MC models).
            martingale_correction (bool): Rescale the simulated ``X_t`` to sample mean 1 at each time.

        Returns:
            DisplacedDiffusionProcess: Calibrated process; :attr:`calibration` reports the fit per maturity.

        Raises:
            ValueError: If the underlying has no lognormal vol surface, or the forward betas are not positive.
        """
        surface = equity.vol_surface
        if surface is None:
            raise ValueError(f"No volatility surface for underlying '{equity.ticker}'")
        if surface.vol_type != "lognormal":
            raise ValueError("the displaced diffusion calibrates to lognormal (Black) vols")
        t = np.asarray(surface.maturity, dtype=float)
        forwards = tf.stack([tf.cast(equity.forward(float(v)), DTYPE) for v in t])                 # [n]
        k = tf.constant(np.asarray(surface.strike, dtype=float), DTYPE)[None, :] / forwards[:, None]   # [n, m]
        iv = tf.cast(surface.volatility_matrix, DTYPE)
        theta = _fit_slices(k.numpy(), iv.numpy(), t)                                             # [n, 2]
        theta = _implicit_step(theta, k, iv, t)
        beta_slice, sigma_slice = theta[:, 0], theta[:, 1]
        beta_fwd, sigma_fwd, floored = forward_from_slices(t, beta_slice, sigma_slice)
        if floored:
            warnings.warn(f"Vol surface of '{equity.ticker}': total variance decreasing at {floored} maturities "
                          "(calendar arbitrage): forward variance floored.", stacklevel=2)
        if float(tf.reduce_min(beta_fwd)) <= 0.0:
            raise ValueError(f"Vol surface of '{equity.ticker}': the skew averaging gives a non-positive forward beta "
                             f"({beta_fwd.numpy().round(4)}): the slice skews are not consistent in time")
        report = (k.numpy(), iv.numpy(), t, theta.numpy(), beta_fwd.numpy(), sigma_fwd.numpy())
        return cls(equity, t, beta_fwd, sigma_fwd, beta_slice, sigma_slice, martingale_correction, report)

    # ------------------------------------------------------------------
    # Parameters and closed forms
    # ------------------------------------------------------------------

    @property
    def t_nodes(self) -> np.ndarray:
        return self._t.copy()

    @property
    def forward_parameters(self) -> tuple:
        """``(beta_fwd, sigma_fwd)`` ``[n]`` per piece."""
        return self._beta, self._sigma

    @property
    def slice_parameters(self) -> tuple:
        """``(beta_T, sigma_T)`` ``[n]`` at the nodes (the calibrated closed-form parameters)."""
        return self._beta_slice, self._sigma_slice

    def effective_parameters(self, maturities) -> tuple:
        """Effective ``(beta_T, sigma_T)`` at any maturities from the forward pieces (skew averaging).

        Args:
            maturities: ``[m]`` year fractions (> 0).

        Returns:
            tuple: ``(beta [m], sigma [m])`` tensors.
        """
        T = np.atleast_1d(np.asarray(maturities, dtype=float))
        overlap = tf.constant(_piece_overlaps(self._t, np.zeros_like(T), T), DTYPE)            # [m, n]
        piece_var = overlap * tf.square(self._sigma)[None, :]
        # W_j = sigma_j^2 int_piece v(t) dt, v linear inside the piece: var_j (cum_{j-1} + var_j / 2)
        w, cum = _averaging_weights(piece_var)
        total = cum[:, -1]
        beta = tf.reduce_sum(w * self._beta[None, :], axis=1) / tf.reduce_sum(w, axis=1)
        return beta, tf.sqrt(total / tf.constant(T, DTYPE))

    def vanilla_price(self, strike: float, maturity: Union[date, float], option_type: float = 1.0) -> tf.Tensor:
        """Closed-form European price ``P(0,T) F(T) E[(phi (X_T - K/F))^+]`` with the effective parameters.

        Args:
            strike (float): Strike.
            maturity: Date or year fraction.
            option_type (float): 1 call, -1 put (or an :class:`OptionType`).

        Returns:
            tf.Tensor: float64 price, differentiable w.r.t. the market (through the calibration).
        """
        phi = float(getattr(option_type, "value", option_type))
        T = self._equity.year_fraction(maturity)
        beta, sigma = self.effective_parameters([T])
        F = tf.cast(self._equity.forward(maturity), DTYPE)
        df = tf.cast(self._equity.discount_curve.discount(maturity), DTYPE)
        return df * F * displaced_diffusion_price(np.float64(strike) / F, beta[0], sigma[0], T, phi)

    def implied_vol(self, strike: float, maturity: Union[date, float]) -> float:
        """Black vol of the closed-form price (number, not differentiable)."""
        T = self._equity.year_fraction(maturity)
        k = float(np.float64(strike) / float(self._equity.forward(maturity)))
        beta, sigma = (float(v[0]) for v in self.effective_parameters([T]))
        return _black_implied_vol(float(_price_np(k, beta, sigma, T)), k, T)

    # ------------------------------------------------------------------
    # StochasticProcess interface
    # ------------------------------------------------------------------

    def size(self) -> int:
        return 1

    def initial_values(self) -> tf.Tensor:
        return self._S0

    def evolve(self, t_grid, dw, observation_times=None) -> tf.Tensor:
        """Simulate spot paths ``S_t = F(t) X_t`` (exact shifted-lognormal steps).

        Args:
            t_grid: ``[n_steps]`` increasing times (year fractions), without 0.
            dw: ``[n_paths, n_steps]`` standard normals.
            observation_times: Optional subset of ``t_grid``: only those columns are returned and the
                market forward is computed only there.

        Returns:
            tf.Tensor: ``[n_paths, n_steps]`` (or ``[n_paths, len(observation_times)]``) spots.
        """
        times = np.asarray(tf.convert_to_tensor(t_grid), dtype=float)
        if times[0] <= 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("t_grid must be increasing and > 0")
        starts = np.concatenate([[0.0], times[:-1]])
        overlap = tf.constant(_piece_overlaps(self._t, starts, times), DTYPE)                   # [n_steps, n]
        step_var = overlap * tf.square(self._sigma)[None, :]
        var = tf.reduce_sum(step_var, axis=1)                                   # ATM variance of X over the step
        beta = tf.reduce_sum(step_var * self._beta[None, :], axis=1) / var
        x = _simulate_ratio(beta, beta * tf.sqrt(var), tf.cast(dw, DTYPE))      # sd of ln Y = beta sigma sqrt(dt)
        if observation_times is not None:
            cols = observation_columns(times, observation_times)
            x, times = tf.gather(x, cols, axis=1), times[cols]
        if self._martingale_correction:
            x = x / tf.reduce_mean(x, axis=0, keepdims=True)
        forwards = tf.stack([tf.cast(self._equity.forward(float(v)), DTYPE) for v in times])
        return x * forwards[None, :]


class DisplacedDiffusionModel(EquityModel):
    """Displaced diffusion model instance: forward parameters ``beta(t)``, ``sigma(t)`` piecewise constant on
    ``t_nodes`` (float64 ``tf.Variable``s ``[n]``), no market.

    ``dX = sigma(t) (beta(t) X + 1 - beta(t)) dW`` on the market forward of the equity given to :meth:`process`.
    Build it by hand (no Jacobian) or with :meth:`calibrate` (slice fit + skew averaging, see
    :meth:`DisplacedDiffusionProcess.from_equity`; stores the report, the slice parameters of the closed form and the
    calibration Jacobian for :func:`par_greeks`).
    """

    SETTINGS = {"t_nodes": None, "beta_slice": None, "sigma_slice": None, "martingale_correction": True}

    def __init__(self, t_nodes, beta_fwd, sigma_fwd, **settings):
        """Args:
            t_nodes: ``[n]`` piece ends (year fractions, increasing).
            beta_fwd, sigma_fwd: ``[n]`` forward parameters per piece (numbers or tensors: their values are taken).
            **settings: ``beta_slice`` / ``sigma_slice`` (slice parameters of the closed form, default averaged from
                the pieces), ``martingale_correction``.
        """
        t_nodes = [float(v) for v in np.ravel(t_nodes)]
        for key in ("beta_slice", "sigma_slice"):
            if settings.get(key) is not None:
                value = settings[key]
                value = np.asarray(tf.stop_gradient(value)) if tf.is_tensor(value) else np.asarray(value, dtype=float)
                settings[key] = [float(v) for v in np.ravel(value)]
        super().__init__(t_nodes=t_nodes, **settings)
        beta = np.ravel(np.asarray(tf.stop_gradient(beta_fwd) if tf.is_tensor(beta_fwd) else beta_fwd, dtype=float))
        sigma = np.ravel(np.asarray(tf.stop_gradient(sigma_fwd) if tf.is_tensor(sigma_fwd) else sigma_fwd, dtype=float))
        self._beta = self._variable(beta, "dd_beta_fwd")
        self._sigma = self._variable(sigma, "dd_sigma_fwd")

    @property
    def parameters(self) -> dict:
        """``{beta_fwd, sigma_fwd}`` ``[n]`` as float64 ``tf.Variable``s."""
        return {"beta_fwd": self._beta, "sigma_fwd": self._sigma}

    def process(self, equity) -> DisplacedDiffusionProcess:
        s = self.settings
        return DisplacedDiffusionProcess(equity, s["t_nodes"], self._beta, self._sigma, s["beta_slice"],
                                         s["sigma_slice"], bool(s["martingale_correction"]))

    @classmethod
    def calibrate(cls, equity, **settings) -> "DisplacedDiffusionModel":
        """Calibrate to the vol surface of *equity* (as :meth:`DisplacedDiffusionProcess.from_equity`) and store the
        forward and slice parameters, the report and the Jacobian w.r.t. the market risk factors of *equity*."""
        with tf.GradientTape(persistent=True) as tape:
            fitted = DisplacedDiffusionProcess.from_equity(equity)
            beta_fwd, sigma_fwd = fitted.forward_parameters
        beta_slice, sigma_slice = fitted.slice_parameters
        model = cls(fitted.t_nodes, beta_fwd, sigma_fwd, beta_slice=beta_slice, sigma_slice=sigma_slice, **settings)
        model._record_calibration(equity, {"beta_fwd": beta_fwd, "sigma_fwd": sigma_fwd}, tape,
                                  report=fitted.calibration)
        return model

    @classmethod
    def from_equity(cls, equity, maturity: Union[date, float] = None, strike: float = None,
                    **settings) -> DisplacedDiffusionProcess:
        """Process calibrated inside the caller's tape (greeks through the calibration)."""
        return DisplacedDiffusionProcess.from_equity(equity, maturity, strike,
                                                     martingale_correction=settings.get("martingale_correction", True))

    @classmethod
    def _from_values(cls, values: dict, settings: dict, frames: dict) -> "DisplacedDiffusionModel":
        settings = dict(settings)
        t_nodes = settings.pop("t_nodes")
        return cls(t_nodes, values["beta_fwd"], values["sigma_fwd"], **settings)


# ============================================================
# Calibration helpers
# ============================================================

def _fit_slices(k: np.ndarray, iv: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Least squares per slice on the price residuals per unit forward (numbers): ``[n, 2]`` (beta, sigma)."""
    out = []
    for i in range(len(t)):
        sd = iv[i] * np.sqrt(t[i])
        d1 = -np.log(k[i]) / sd + 0.5 * sd
        market = norm.cdf(d1) - k[i] * norm.cdf(d1 - sd)
        atm = float(np.interp(0.0, np.log(k[i]), iv[i]))
        fit = least_squares(lambda p: _price_np(k[i], p[0], p[1], t[i]) - market, [0.5, atm],
                            bounds=([BETA_BOUNDS[0], SIGMA_BOUNDS[0]], [BETA_BOUNDS[1], SIGMA_BOUNDS[1]]),
                            method="trf", x_scale=[0.1, 0.01], xtol=1e-14, ftol=1e-14, gtol=1e-14)
        out.append(fit.x)
    return np.array(out)


def _implicit_step(theta: np.ndarray, k: tf.Tensor, iv: tf.Tensor, t: np.ndarray) -> tf.Tensor:
    """Slice parameters as differentiable functions of the market (implicit function theorem).

    ``theta(m) = theta* - H^{-1} grad_theta L(theta*, m)`` with ``L`` the half sum of squared residuals and
    ``H`` its exact Hessian at the optimum: the value is ``theta*`` (``grad = 0``, up to the solver tolerance)
    and ``d theta / d m = -H^{-1} d grad / d m`` for ``m = (k, iv)``, computed once as numbers
    (:mod:`~tensorquant.numericalhandles.implicit`) and attached as the gradient. Parameters on a bound stay fixed.
    """
    tt = tf.constant(t, DTYPE)[:, None]

    def gradient(x, k_, iv_):
        market = _black_tf(k_, iv_, tt)
        with tf.GradientTape() as tape:
            tape.watch(x)
            model = displaced_diffusion_price(k_, x[:, 0:1], x[:, 1:2], tt)
            loss = 0.5 * tf.reduce_sum(tf.square(model - market))                          # separable slices
        return tape.gradient(loss, x)                                                     # [n, 2]

    lower = np.array([BETA_BOUNDS[0], SIGMA_BOUNDS[0]])
    upper = np.array([BETA_BOUNDS[1], SIGMA_BOUNDS[1]])
    free = ~(np.isclose(theta, lower) | np.isclose(theta, upper))                         # [n, 2]
    g, hess, dg_dz = optimality_derivatives(gradient, theta, (k, iv))
    value = theta - newton_step(g, hess, free).reshape(theta.shape)
    return implicit_solution(value, (k, iv), implicit_sensitivities(hess, dg_dz, free, theta.shape))


def _report(k, iv, t, theta, beta_fwd, sigma_fwd) -> pd.DataFrame:
    rows = []
    for i in range(len(t)):
        errors = []
        for kk, v in zip(k[i], iv[i]):
            model = _black_implied_vol(float(_price_np(kk, theta[i, 0], theta[i, 1], t[i])), kk, t[i])
            errors.append(model - v)
        errors = np.array(errors) * 100.0                                                 # vol points
        if np.isnan(errors).any():
            errors = np.where(np.isnan(errors), np.nanmax(np.abs(errors)), errors)
        rows.append({"maturity": t[i], "beta": theta[i, 0], "sigma": theta[i, 1],
                     "beta_at_bound": bool(np.isclose(theta[i, 0], BETA_BOUNDS[0]) or np.isclose(theta[i, 0], BETA_BOUNDS[1])),
                     "atm_error_volpts": float(np.interp(0.0, np.log(k[i]), errors)), "beta_fwd": beta_fwd[i],
                     "sigma_fwd": sigma_fwd[i], "rms_error_volpts": float(np.sqrt(np.nanmean(errors ** 2))),
                     "max_error_volpts": float(np.nanmax(np.abs(errors)))})
    return pd.DataFrame(rows)
