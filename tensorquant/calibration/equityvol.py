"""Equity implied volatility surface from listed options.

Construction in two stages, as in the industry practice for listed equity options: first the forward of each listed
expiry, implied from the option prices by put-call parity (American options de-Americanized), which gives the
repo term structure on top of the forecast dividends; then the smile of each expiry on that forward.

Stage 1 (forward and repo):

- :func:`implied_volatility`: vectorised implied vol (bisection) of European (Black-Scholes) or American
  (Bjerksund-Stensland 1993) prices, with the discrete-dividend inputs of :class:`BlackScholesPricer`
  (escrowed spot ``S - PV(div)``, rate ``r``, carry ``q`` = repo).
- :func:`implied_forward`: forward of one expiry. Start from the market forward; at the two admissible strikes
  around it (call and put with a two-sided quote), de-Americanize the mid prices (implied vol of the American price
  on the current forward, European price at that vol), take the parity forward ``K + (C - P) / P(0, T)`` at each
  strike and their average as the new forward; repeat until it lies between the two strikes and moves less than
  ``tol``. The dividends are kept (forecasts), the rest of the carry is the repo
  ``z(T) = -ln(F P(0, T) / (S - PV_div(T))) / T``. Bid / ask parity forwards bound it.
- :func:`implied_repo_curve`: the repo of every expiry as a :class:`RateCurve` (linear zero repo rates, ACT/365)
  for ``EQ:<ccy>:<ticker>:REPO``.

Stage 2 (smile of an expiry):

- :class:`LognormalMixture`: risk-neutral density of ``S_T`` as a mixture of lognormals plus a default mass at zero
  (Brigo-Mercurio lognormal mixture; the point mass helps the steep short-dated put skews). Weights ``p_l``,
  multiplicative means ``xi_l`` (``sum p_l xi_l = 1``: the forward is repriced) and total standard deviations
  ``Sigma_l`` of the log; ``Q + sum p_l = 1``. Prices are sums of Black prices (``Q K`` more for a put): a legitimate
  density, so the smile is free of strike arbitrage. Fit to the out-of-the-money quotes of one expiry by least
  squares on vega-weighted prices (about the vol errors), numpy closed forms.

Stage 3 (surface, :func:`build_equity_vol`): quotes filtered (two-sided, ask <= 3 bid; settlement prices ``last``
used where there is no bid / ask, with a lower weight), forward and repo of every expiry, Black vols of the
out-of-the-money quotes (the American implied vol on the implied forward is the Black vol), outliers dropped
(outside 0.2-5 x the median of the expiry), a mixture fitted per expiry (standard start; above 0.2 vol points also a warm start from the previous
expiry, the better fit kept). On the
grid (``GRID_TENORS`` x ``GRID_MONEYNESS`` of the spot), at the forward moneyness ``K / F(t)``:

- between the expiries: monotone (PCHIP) interpolation of the total variance in time;
- before the first: the vol of the first expiry;
- after the last: the last mixture extrapolated (vols of the lognormals flat, means scaled by ``sqrt(t / T_n)``,
  survival ``P(t) = (P(T_n) + exp(-lambda t)) / 2``), its at-the-money vol raised to the Heston-like reference level
  ``sigma_H^2(t) = theta + (V0 - theta)(1 - exp(-kappa t)) / (kappa t)`` fitted to the at-the-money vols of the
  expiries when it is below it (all the lognormal vols scaled);
- extreme strikes where the vol cannot be inverted: linear in log-moneyness from the neighbours (Lee).
"""

from __future__ import annotations

import datetime as _dt
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr

from ..markethandles.equity import Equity
from ..markethandles.ircurve import RateCurve
from ..markethandles.utils import Currency, OptionType
from ..markethandles.volatilitysurface import VolatilitySurface
from ..numericalhandles.dtypes import DTYPE
from ..pricers.black import bjerksund_stensland_calc, blackscholes_calc
from ..timehandles.daycounter import DayCounter, DayCounterConvention

__all__ = ["implied_volatility", "implied_forward", "implied_repo_curve", "black_implied_vol", "LognormalMixture",
           "build_equity_vol", "calibrate_snapshot_vols", "GRID_TENORS", "GRID_MONEYNESS"]

# Bjerksund-Stensland is finite and increasing in the vol on this range (NaN below ~1.5%, below the European
# price at very high vols); equity vols are well inside it
_VOL_LOW, _VOL_HIGH = 0.02, 3.0


def _option_type(option_type: Union[str, OptionType]) -> OptionType:
    if isinstance(option_type, OptionType):
        return option_type
    return OptionType.Call if str(option_type).upper().startswith("C") else OptionType.Put


_COMPILED: dict = {}


def _compiled(option_type: OptionType, american: bool):
    """The library price function (Black-Scholes or Bjerksund-Stensland) compiled once per (type, exercise) on flat
    vectors: no eager overhead per operation (many evaluations in the vol inversions)."""
    key = (option_type, american)
    if key not in _COMPILED:
        calc = bjerksund_stensland_calc if american else blackscholes_calc
        vector, scalar = tf.TensorSpec([None], DTYPE), tf.TensorSpec([], DTYPE)

        @tf.function(input_signature=[scalar, vector, scalar, vector, scalar, scalar])
        def price(s, k, r, sigma, t, q):
            return calc(s, k, r, sigma, t, q, option_type)

        _COMPILED[key] = price
    return _COMPILED[key]


def _prices(sigma, strikes, option_type: OptionType, s_net: float, r: float, q: float, t: float,
            american: bool) -> np.ndarray:
    if american and option_type == OptionType.Call and q <= 1e-12:
        american = False                 # no carry: an American call is never exercised early (and BS 1993 overflows)
    sigma, strikes = np.broadcast_arrays(np.asarray(sigma, dtype=float), np.asarray(strikes, dtype=float))
    c = lambda x: tf.constant(x, DTYPE)
    out = _compiled(option_type, american)(c(s_net), c(strikes.ravel()), c(r), c(sigma.ravel()), c(t), c(q))
    return out.numpy().reshape(sigma.shape)


def implied_volatility(prices, strikes, option_type: Union[str, OptionType], s_net: float, r: float, q: float,
                       t: float, american: bool = False, grid: int = 121, iterations: int = 15) -> np.ndarray:
    """Implied volatilities of option prices on ``[0.02, 3]``: the prices on a vol grid (one vectorised call)
    bracket the root, then Illinois regula falsi steps (vectorised) refine it below 1e-12.

    Args:
        prices: Option prices ``[n]``.
        strikes: Strikes ``[n]``.
        option_type (str | OptionType): ``"C"`` / ``"P"`` or an :class:`OptionType` (same for all).
        s_net (float): Escrowed spot ``S - PV(div to T)`` (the spot for a continuous yield).
        r (float): Continuously compounded rate to ``T``.
        q (float): Carry (repo, plus the yield in the continuous model).
        t (float): Year fraction to the expiry.
        american (bool): Bjerksund-Stensland American prices; European Black-Scholes otherwise.
        grid (int): Points of the bracketing grid (geometric).
        iterations (int): Regula falsi steps.

    Returns:
        np.ndarray: Vols ``[n]``; NaN where the price is outside the model range (below ``sigma = 0.02``, above
        ``sigma = 3``).
    """
    prices = np.atleast_1d(np.asarray(prices, dtype=float))
    strikes = np.broadcast_to(np.asarray(strikes, dtype=float), prices.shape).copy()
    ot = _option_type(option_type)
    sig = np.geomspace(_VOL_LOW, _VOL_HIGH, grid)
    table = _prices(sig[:, None] * np.ones_like(strikes)[None, :], strikes[None, :] * np.ones((grid, 1)), ot,
                    s_net, r, q, t, american)
    # Bjerksund-Stensland can overflow (NaN) at extreme vols with a tiny positive carry: low end -inf, high end +inf
    table = np.where(np.isnan(table), np.where(sig[:, None] < 0.5, -np.inf, np.inf), table)
    j = (table < prices[None, :]).sum(axis=0) - 1
    valid = (j >= 0) & (j < grid - 1)
    j = np.clip(j, 0, grid - 2)
    cols = np.arange(len(prices))
    lo, hi = sig[j], sig[j + 1]
    f_lo, f_hi = table[j, cols] - prices, table[j + 1, cols] - prices
    x = 0.5 * (lo + hi)
    for _ in range(iterations):
        finite = np.isfinite(f_lo) & np.isfinite(f_hi) & (f_hi != f_lo)
        x = np.where(finite, hi - f_hi * (hi - lo) / np.where(finite, f_hi - f_lo, 1.0), 0.5 * (lo + hi))
        x = np.clip(x, np.minimum(lo, hi), np.maximum(lo, hi))
        f_x = _prices(x, strikes, ot, s_net, r, q, t, american) - prices
        f_x = np.where(np.isnan(f_x), np.where(x < 0.5, -np.inf, np.inf), f_x)
        flip = f_x * f_hi < 0                             # root between x and hi: hi becomes the other end
        lo, f_lo = np.where(flip, hi, lo), np.where(flip, f_hi, 0.5 * f_lo)   # Illinois: halve the kept end
        hi, f_hi = x, f_x
    return np.where(valid, x, np.nan)


def _carry_inputs(equity: Equity, expiry: Union[date, float]):
    """``t``, discount factor, ``r`` and escrowed spot of *equity* to *expiry* (floats)."""
    t = equity.year_fraction(expiry)
    if t <= 0.0:
        raise ValueError(f"Expiry must be after the reference date (T = {t})")
    df = float(equity.discount_curve.discount(expiry))
    return t, df, -np.log(df) / t, float(equity.spot) - float(equity.pv_dividends(expiry))


def implied_forward(equity: Equity, expiry: Union[date, float], quotes: pd.DataFrame, tol: float = 1e-10,
                    max_iter: int = 50, american: Optional[bool] = None, dividend_model: str = "continuous") -> dict:
    """Forward of one expiry implied from call and put quotes (put-call parity, de-Americanized).

    Args:
        equity (Equity): Underlying (spot, dividends, discount curve; its repo only gives the starting forward).
        expiry (date | float): Expiry of the quotes.
        quotes (pd.DataFrame): Quotes of this expiry: ``strike``, ``option_type`` (``C`` / ``P``), ``bid``,
            ``ask``, optional ``exercise`` (``American`` / ``European``).
        tol (float): Relative change of the forward at convergence.
        max_iter (int): Maximum iterations.
        american (bool, optional): Exercise style; default from the ``exercise`` column (European if absent).
        dividend_model (str): Model of the American prices, as in :class:`BlackScholesPricer` (same default):
            ``"continuous"`` (spot ``S``, total carry ``q = r - ln(F / S) / T``) or ``"discrete"`` (escrowed spot
            ``S - PV(div)``, carry = repo). The European parity and the returned repo do not depend on it.

    Returns:
        dict: ``forward``, ``repo`` (zero repo rate to the expiry), ``forward_bid`` / ``forward_ask`` (parity
        forwards from bid / ask, intersection over the strikes), ``strike_low`` / ``strike_high``, ``iterations``,
        ``converged`` (``False``: the iteration jumped between two strike pairs, whose parity forwards lie each in
        the other pair; the forward is the mean of the two).

    Raises:
        ValueError: If there are no strikes with two-sided call and put quotes.
    """
    if american is None:
        american = "exercise" in quotes and quotes["exercise"].astype(str).str.lower().eq("american").any()
    if dividend_model not in ("continuous", "discrete"):
        raise ValueError(f"dividend_model must be 'continuous' or 'discrete', got {dividend_model!r}")
    t, df, r, s_net = _carry_inputs(equity, expiry)
    base = float(equity.spot) if dividend_model == "continuous" else s_net    # spot of the American model
    q_frame = quotes.assign(option_type=quotes["option_type"].astype(str).str.upper().str[0])
    ok = q_frame[(q_frame["bid"] > 0) & (q_frame["ask"] >= q_frame["bid"])]
    calls = ok[ok["option_type"] == "C"].set_index("strike")
    puts = ok[ok["option_type"] == "P"].set_index("strike")
    strikes = np.array(sorted(set(calls.index) & set(puts.index)), dtype=float)
    if len(strikes) == 0:
        raise ValueError(f"No strike with two-sided call and put quotes for expiry {expiry}")

    def parity_forward(forward: float):
        """Parity forward at the two strikes around *forward* (de-Americanized on *forward*), and the inputs."""
        below, above = strikes[strikes <= forward], strikes[strikes > forward]
        pair = np.unique([below[-1] if len(below) else above[0], above[0] if len(above) else below[-1]])
        c_bid, c_ask = calls.loc[pair, "bid"].to_numpy(float), calls.loc[pair, "ask"].to_numpy(float)
        p_bid, p_ask = puts.loc[pair, "bid"].to_numpy(float), puts.loc[pair, "ask"].to_numpy(float)
        c_mid, p_mid = 0.5 * (c_bid + c_ask), 0.5 * (p_bid + p_ask)
        if american:                                   # European prices at the American implied vols
            q = r - np.log(forward / base) / t
            c_eu = _prices(implied_volatility(c_mid, pair, "C", base, r, q, t, True), pair, OptionType.Call,
                           base, r, q, t, False)
            p_eu = _prices(implied_volatility(p_mid, pair, "P", base, r, q, t, True), pair, OptionType.Put,
                           base, r, q, t, False)
            if np.isnan(c_eu).any() or np.isnan(p_eu).any():
                raise ValueError(f"Expiry {expiry}: American mid prices out of the model range at {pair}")
            c_adj, p_adj = c_eu - c_mid, p_eu - p_mid    # minus the early exercise premia (applied to bid / ask)
        else:
            c_adj = p_adj = np.zeros_like(pair)
        parity = pair + ((c_mid + c_adj) - (p_mid + p_adj)) / df
        return float(parity.mean()), pair, (c_bid, c_ask, p_bid, p_ask, c_adj, p_adj)

    # fixed point F = g(F) (g: parity forward de-Americanized on F), solved by secant steps on g(F) - F
    forward, previous, history, converged = float(equity.forward(expiry)), None, [], True
    for iteration in range(1, max_iter + 1):
        new, pair, inputs = parity_forward(forward)
        history.append((new, pair, inputs))
        h = new - forward
        if abs(h) <= tol * forward and pair[0] <= new <= pair[-1]:
            forward = new
            break
        step = new                                     # plain fixed-point step, or secant when it is defined
        if previous is not None and previous[1] != h:
            step = forward - h * (forward - previous[0]) / (h - previous[1])
        previous, forward = (forward, h), step
    else:
        # noisy quotes: the parity forward of the pair below lies above it and that of the pair above below it (g
        # jumps at a strike): the mean of the last two parity forwards, over both pairs
        converged = False
        (f1, p1, in1), (f2, p2, in2) = history[-2], history[-1]
        if not np.array_equal(p1, p2):
            forward = 0.5 * (f1 + f2)
            pair = np.unique(np.concatenate([p1, p2]))
            inputs = tuple(np.concatenate([a, b]) for a, b in zip(in1, in2))
            order = np.argsort(np.concatenate([p1, p2]), kind="stable")
            keep = np.unique(np.concatenate([p1, p2])[order], return_index=True)[1]
            inputs = tuple(x[order][keep] for x in inputs)
        else:
            forward = f2
    c_bid, c_ask, p_bid, p_ask, c_adj, p_adj = inputs
    f_bid = pair + ((c_bid + c_adj) - (p_ask + p_adj)) / df
    f_ask = pair + ((c_ask + c_adj) - (p_bid + p_adj)) / df
    return {"forward": forward, "repo": float(-np.log(forward * df / s_net) / t),
            "forward_bid": float(f_bid.max()), "forward_ask": float(f_ask.min()),
            "strike_low": float(pair[0]), "strike_high": float(pair[-1]), "iterations": iteration,
            "converged": converged}


def implied_repo_curve(equity: Equity, quotes: pd.DataFrame, **kwargs) -> tuple[RateCurve, pd.DataFrame]:
    """Repo term structure from the implied forward of every listed expiry.

    Args:
        equity (Equity): Underlying (spot, dividends, discount curve).
        quotes (pd.DataFrame): Quotes with an ``expiry`` column (dates) and the columns of :func:`implied_forward`.
        **kwargs: Passed to :func:`implied_forward`.

    Returns:
        tuple: :class:`RateCurve` of zero repo rates at the expiries (linear, ACT/365, for
        ``EQ:<ccy>:<ticker>:REPO``) and the report (one row per expiry: the :func:`implied_forward` results).
    """
    rows = []
    for expiry, grp in quotes.groupby("expiry", sort=True):
        expiry = pd.Timestamp(expiry).date()
        rows.append({"expiry": expiry, **implied_forward(equity, expiry, grp, **kwargs)})
    report = pd.DataFrame(rows)
    reference = equity.discount_curve.reference_date
    curve = RateCurve(reference, list(report["expiry"]), [float(z) for z in report["repo"]], "LINEAR",
                      DayCounterConvention.Actual365)
    return curve, report


# ----------------------------------------------------------------------------------------------------------------------
# Stage 2: smile of one expiry
# ----------------------------------------------------------------------------------------------------------------------

def _black(forward, strikes, stdev, option_type: str) -> np.ndarray:
    """Undiscounted Black price with total standard deviation ``stdev`` (= ``sigma sqrt(T)``) of the log."""
    stdev = np.maximum(stdev, 1e-300)
    d1 = (np.log(forward / strikes) + 0.5 * stdev ** 2) / stdev
    d2 = d1 - stdev
    if option_type == "C":
        return forward * ndtr(d1) - strikes * ndtr(d2)
    return strikes * ndtr(-d2) - forward * ndtr(-d1)


def black_implied_vol(prices, forward: float, strikes, t: float, option_type: str,
                      iterations: int = 100) -> np.ndarray:
    """Black implied vols of undiscounted prices (vectorised bisection on ``[1e-4, 5]``, numpy).

    Args:
        prices: Undiscounted option prices ``[n]``.
        forward (float): Forward.
        strikes: Strikes ``[n]``.
        t (float): Year fraction to the expiry.
        option_type (str): ``"C"`` or ``"P"`` (same for all).
        iterations (int): Bisection steps.

    Returns:
        np.ndarray: Vols ``[n]``; NaN outside the range.
    """
    prices = np.atleast_1d(np.asarray(prices, dtype=float))
    strikes = np.broadcast_to(np.asarray(strikes, dtype=float), prices.shape)
    cp = _option_type(option_type).name[0]
    root_t = np.sqrt(t)
    lo, hi = np.full(prices.shape, 1e-4), np.full(prices.shape, 5.0)
    valid = (prices >= _black(forward, strikes, lo * root_t, cp)) & (prices <= _black(forward, strikes, hi * root_t, cp))
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        above = _black(forward, strikes, mid * root_t, cp) > prices
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    return np.where(valid, 0.5 * (lo + hi), np.nan)


class LognormalMixture:
    """Smile of one expiry: ``S_T`` is ``0`` with probability ``Q`` (default) and
    ``xi_l F exp(Sigma_l Z - Sigma_l^2 / 2)`` with probability ``p_l``.

    Args:
        weights: ``p_l`` (rescaled so that ``Q + sum p_l = 1``).
        means: ``xi_l`` (rescaled so that ``sum p_l xi_l = 1``: the forward is repriced).
        stdevs: ``Sigma_l`` = total standard deviation of ``ln S_T`` of each lognormal (``sigma_l sqrt(T)``).
        default (float): ``Q``.
    """

    def __init__(self, weights, means, stdevs, default: float = 0.0):
        weights = np.asarray(weights, dtype=float)
        if not 0.0 <= default < 1.0 or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("weights must be non-negative and the default probability in [0, 1)")
        self.default = float(default)
        self.weights = weights * (1.0 - self.default) / weights.sum()
        means = np.asarray(means, dtype=float)
        self.means = means / np.dot(self.weights, means)
        self.stdevs = np.asarray(stdevs, dtype=float)

    def price(self, forward: float, strikes, discount: float, option_type: str) -> np.ndarray:
        """Option prices ``discount x E[payoff]`` (a put also pays ``K`` in default)."""
        strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
        cp = _option_type(option_type).name[0]
        value = (self.weights[:, None] * _black(self.means[:, None] * forward, strikes[None, :], self.stdevs[:, None],
                                                cp)).sum(axis=0)
        if cp == "P":
            value = value + self.default * strikes
        return discount * value

    def implied_vols(self, forward: float, strikes, t: float) -> np.ndarray:
        """Black implied vols of the out-of-the-money options (puts below the forward)."""
        strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
        put = strikes < forward
        vols = np.empty(strikes.shape)
        for mask, cp in ((put, "P"), (~put, "C")):
            if mask.any():
                vols[mask] = black_implied_vol(self.price(forward, strikes[mask], 1.0, cp), forward, strikes[mask],
                                               t, cp)
        return vols

    @classmethod
    def _from_params(cls, x: np.ndarray, n: int, default: bool) -> "LognormalMixture":
        logits = x[: n + 1] if default else np.concatenate([[-np.inf], x[:n]])
        w = np.exp(logits - logits.max())
        w = w / w.sum()
        m = x[n + 1: 2 * n + 1] if default else x[n: 2 * n]
        return cls(w[1:], np.exp(m), np.exp(x[-n:]), w[0])

    @classmethod
    def fit(cls, forward: float, t: float, strikes, vols, n: int = 4, default: bool = True,
            weights=None, x0: Optional[np.ndarray] = None) -> "LognormalMixture":
        """Least-squares fit to the implied vols of one expiry (out-of-the-money prices divided by the Black vega:
        about the vol errors).

        Args:
            forward (float): Forward of the expiry.
            t (float): Year fraction to the expiry.
            strikes: Strikes ``[k]``.
            vols: Market implied vols ``[k]`` (mid).
            n (int): Number of lognormals (4).
            default (bool): With the default mass.
            weights: Optional weights of the quotes ``[k]`` (e.g. lower for settlement-only quotes).
            x0 (np.ndarray, optional): Start parameters (e.g. the fit of the previous expiry, ``fit.params``).

        Returns:
            LognormalMixture: The fitted mixture (``params``: unconstrained parameters, ``rms``: weighted rms of
            the vol errors).
        """
        strikes, vols = np.asarray(strikes, dtype=float), np.asarray(vols, dtype=float)
        w = np.ones_like(strikes) if weights is None else np.asarray(weights, dtype=float)
        root_t = np.sqrt(t)
        put = strikes < forward
        target = np.where(put, _black(forward, strikes, vols * root_t, "P"), _black(forward, strikes, vols * root_t, "C"))
        d1 = (np.log(forward / strikes) + 0.5 * (vols * root_t) ** 2) / (vols * root_t)
        vega = np.maximum(forward * np.exp(-0.5 * d1 ** 2) / np.sqrt(2.0 * np.pi) * root_t, 1e-8 * forward)

        def residuals(x):
            m = cls._from_params(x, n, default)
            model = np.where(put, m.price(forward, strikes, 1.0, "P"), m.price(forward, strikes, 1.0, "C"))
            return w * (model - target) / vega

        if x0 is None:                                    # ordered start: low mean / high vol first (no exchange)
            order = np.argsort(strikes)
            atm = float(np.interp(forward, strikes[order], vols[order]))
            span = np.linspace(-1.0, 1.0, n) if n > 1 else np.zeros(1)
            logits = -span ** 2
            means = 0.12 * span * max(atm * root_t, 0.05)
            stdevs = np.log(atm * root_t * (1.0 - 0.35 * span))
            x0 = np.concatenate([[np.log(1e-3)] if default else [], logits, means, stdevs])
        lower = np.full(x0.shape, -np.inf)
        upper = np.full(x0.shape, np.inf)
        n_logits = n + 1 if default else n
        lower[:n_logits], upper[:n_logits] = -15.0, 15.0     # weights (and the default mass) away from 0 and 1
        if default:
            upper[0] = np.log(0.2)                         # Q up to about 17%
        lower[-n:], upper[-n:] = np.log(0.005 * root_t), np.log(3.0 * max(root_t, 0.1))
        first_mean = n + 1 if default else n
        lower[first_mean: first_mean + n], upper[first_mean: first_mean + n] = -3.0, 3.0   # log means
        fit = least_squares(residuals, np.clip(x0, lower + 1e-9, upper - 1e-9), bounds=(lower, upper),
                            method="trf", x_scale="jac", max_nfev=600, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        mixture = cls._from_params(fit.x, n, default)
        mixture.params = fit.x
        mixture.rms = float(np.sqrt(np.mean(fit.fun ** 2)))
        return mixture


# ----------------------------------------------------------------------------------------------------------------------
# Stage 3: surface
# ----------------------------------------------------------------------------------------------------------------------

GRID_TENORS = (1 / 12, 2 / 12, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
GRID_MONEYNESS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5)


def _prepare_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Normalised quotes: ``option_type`` C / P, ``settlement`` flag (no two-sided quote, ``last`` used as bid and
    ask), filters of the bid / ask (positive, ask <= 3 bid)."""
    q = quotes.copy()
    q["option_type"] = q["option_type"].astype(str).str.upper().str[0]
    q["expiry"] = pd.to_datetime(q["expiry"]).dt.date
    for column in ("bid", "ask"):
        q[column] = pd.to_numeric(q[column], errors="coerce")
    two_sided = (q["bid"] > 0) & (q["ask"] >= q["bid"]) & (q["ask"] <= 3.0 * q["bid"])
    q["settlement"] = False
    if "last" in q:
        last = pd.to_numeric(q["last"], errors="coerce")
        settle = ~two_sided & q["bid"].isna() & q["ask"].isna() & (last > 0)
        q.loc[settle, ["bid", "ask"]] = np.column_stack([last[settle], last[settle]])
        q.loc[settle, "settlement"] = True
        two_sided = two_sided | settle
    return q[two_sided].reset_index(drop=True)


def _heston_atm(times: np.ndarray, atm: np.ndarray):
    """``sigma_H(t)`` fitted to the at-the-money vols (``kappa``, ``V0``, ``theta``); flat at the last one with fewer
    than three expiries."""
    if len(times) < 3:
        return lambda t: float(atm[-1])

    def model(x, t):
        kappa, v0, theta = x
        return np.sqrt(theta + (v0 - theta) * (1.0 - np.exp(-kappa * t)) / (kappa * t))

    v = atm[-1] ** 2
    fit = least_squares(lambda x: model(x, times) - atm, [1.0, atm[0] ** 2, v], bounds=([0.05, 1e-4, 1e-4], [10.0, 4.0, 4.0]))
    return lambda t: float(model(fit.x, t))


def _extrapolate(mixture: LognormalMixture, t_n: float, t: float) -> LognormalMixture:
    """Mixture of the last expiry carried to ``t > t_n``: lognormal vols flat, means scaled by ``sqrt(t / t_n)``,
    survival halfway between ``P(t_n)`` and ``exp(-lambda t)``."""
    survival_n = 1.0 - mixture.default
    lam = -np.log(survival_n) / t_n
    survival = 0.5 * survival_n + 0.5 * np.exp(-lam * t)
    scale = np.sqrt(t / t_n)
    return LognormalMixture(mixture.weights * survival / survival_n, np.exp(np.log(mixture.means) * scale),
                            mixture.stdevs * scale, 1.0 - survival)


def _fill_wings(log_m: np.ndarray, vols: np.ndarray) -> np.ndarray:
    """NaN vols (extreme strikes) linear in log-moneyness from the two nearest valid ones on that side."""
    out = vols.copy()
    ok = np.flatnonzero(np.isfinite(vols))
    if len(ok) < 2:
        return out
    for i in np.flatnonzero(~np.isfinite(vols)):
        a, b = (ok[0], ok[1]) if i < ok[0] else (ok[-2], ok[-1])
        slope = (vols[b] - vols[a]) / (log_m[b] - log_m[a])
        out[i] = vols[a] + slope * (log_m[i] - log_m[a])
    return out


def build_equity_vol(equity: Equity, quotes: pd.DataFrame, dividend_model: str = "continuous", n: int = 4,
                     settlement_weight: float = 0.25, min_quotes: int = 3, min_repo_days: int = 14, tenors=GRID_TENORS,
                     moneyness=GRID_MONEYNESS) -> tuple[VolatilitySurface, RateCurve, pd.DataFrame]:
    """Implied vol surface and repo term structure of *equity* from its listed option quotes.

    Args:
        equity (Equity): Underlying (spot, dividends, discount curve; its repo only starts the forwards).
        quotes (pd.DataFrame): ``expiry``, ``strike``, ``option_type`` (C / P), ``bid``, ``ask``, optional
            ``last`` (settlement, used without a two-sided quote) and ``exercise`` (American / European).
        dividend_model (str): Model of the American prices (as :class:`BlackScholesPricer`).
        n (int): Lognormals of the smile mixtures.
        settlement_weight (float): Fit weight of the settlement-only quotes.
        min_quotes (int): Minimum out-of-the-money quotes of an expiry.
        min_repo_days (int): Expiries closer than this are left out of the repo curve (a small forward error is a
            large rate over a few days); their smiles keep their own implied forward.
        tenors: Maturities of the output grid (years).
        moneyness: Strikes of the output grid as fractions of the spot.

    Returns:
        tuple: :class:`VolatilitySurface` (``tenors`` x ``moneyness`` x spot, ACT/365), repo :class:`RateCurve`
        (zero repo rates at the calibrated expiries from ``min_repo_days``) and the report (one row per expiry: ``expiry``, ``t``,
        ``forward``, ``repo``, ``forward_bid`` / ``forward_ask``, ``quotes``, ``source`` (market / settlement),
        ``fit_rms`` (vol), ``atm_vol``, ``mixture``, ``quote_vols`` (per out-of-the-money quote: ``strike``,
        ``option_type``, ``iv_bid`` / ``iv_mid`` / ``iv_ask`` on the implied forward, ``settlement``, ``used`` (kept
        by the filters), ``model_vol``); skipped expiries with ``error``).

    Raises:
        ValueError: If no expiry can be calibrated.
    """
    q = _prepare_quotes(quotes)
    rows, x0 = [], None
    spot = float(equity.spot)
    for expiry, grp in q.groupby("expiry", sort=True):
        t = equity.year_fraction(expiry)
        if t < 1.0 / 365.0:
            continue
        market = grp[~grp["settlement"]]
        source = "market" if len(market) else "settlement"
        try:
            res = implied_forward(equity, expiry, market if len(market) else grp, dividend_model=dividend_model)
        except ValueError as exc:
            rows.append({"expiry": expiry, "t": t, "error": str(exc)})
            continue
        forward = res["forward"]
        _, df, r, s_net = _carry_inputs(equity, expiry)
        base = spot if dividend_model == "continuous" else s_net
        carry = r - np.log(forward / base) / t
        american = "exercise" in grp and grp["exercise"].astype(str).str.lower().eq("american").any()
        otm = grp[np.where(grp["option_type"] == "P", grp["strike"] < forward, grp["strike"] >= forward)]
        strikes, vols, weights, bids, asks, types, settles = [], [], [], [], [], [], []
        for cp in ("P", "C"):
            side = otm[otm["option_type"] == cp]
            if len(side):
                bid, ask = side["bid"].to_numpy(float), side["ask"].to_numpy(float)
                strikes.append(side["strike"].to_numpy(float))
                vols.append(implied_volatility(0.5 * (bid + ask), strikes[-1], cp, base, r, carry, t, american))
                bids.append(implied_volatility(bid, strikes[-1], cp, base, r, carry, t, american))
                asks.append(implied_volatility(ask, strikes[-1], cp, base, r, carry, t, american))
                weights.append(np.where(side["settlement"], settlement_weight, 1.0))
                types.append(np.full(len(side), cp))
                settles.append(side["settlement"].to_numpy(bool))
        k, v, w = (np.concatenate(x) for x in (strikes, vols, weights))
        keep = np.isfinite(v)
        if keep.any():
            median = np.median(v[keep])
            keep &= (v > 0.2 * median) & (v < 5.0 * median)
        quote_vols = pd.DataFrame({"strike": k, "option_type": np.concatenate(types), "iv_bid": np.concatenate(bids),
                                   "iv_mid": v, "iv_ask": np.concatenate(asks),
                                   "settlement": np.concatenate(settles), "used": keep})
        if keep.sum() < min_quotes:
            rows.append({"expiry": expiry, "t": t, "error": f"{int(keep.sum())} usable quotes",
                         "quote_vols": quote_vols.assign(model_vol=np.nan)})
            continue
        mixture = LognormalMixture.fit(forward, t, k[keep], v[keep], n=n, weights=w[keep])
        if x0 is not None and mixture.rms > 2e-3:      # warm start from the previous expiry: keep the better fit
            try:
                warm = LognormalMixture.fit(forward, t, k[keep], v[keep], n=n, weights=w[keep], x0=x0)
                mixture = warm if warm.rms < mixture.rms else mixture
            except ValueError:
                pass
        x0 = mixture.params
        rows.append({"expiry": expiry, "t": t, **res, "quotes": int(keep.sum()), "source": source,
                     "fit_rms": mixture.rms, "atm_vol": float(mixture.implied_vols(forward, [forward], t)[0]),
                     "mixture": mixture, "quote_vols": quote_vols.assign(model_vol=mixture.implied_vols(forward, k, t))})
    report = pd.DataFrame(rows)
    good = report[report["mixture"].notna()] if "mixture" in report else report.iloc[0:0]
    if good.empty:
        raise ValueError(f"No expiry of {equity.ticker} could be calibrated")
    reference = equity.discount_curve.reference_date
    pillars = good[good["t"] >= min_repo_days / 365.0]
    pillars = pillars if len(pillars) else good.iloc[-1:]
    repo = RateCurve(reference, list(pillars["expiry"]), [float(z) for z in pillars["repo"]], "LINEAR",
                     DayCounterConvention.Actual365)

    # surface on the grid, at the forward moneyness of the new forwards (repo curve)
    view = Equity(equity.ticker, equity.currency, equity.spot, equity.discount_curve, repo, equity.dividends,
                  equity.div_yield, None)
    times = good["t"].to_numpy(float)
    mixtures = list(good["mixture"])
    sigma_h = _heston_atm(times, good["atm_vol"].to_numpy(float))
    m_grid = np.asarray(moneyness, dtype=float)
    matrix = np.empty((len(tenors), len(m_grid)))
    for i, t in enumerate(tenors):
        x = spot * m_grid / float(view.forward(t))                       # forward moneyness K / F(t)
        if t <= times[0]:
            vols = mixtures[0].implied_vols(1.0, x, times[0])
        elif t >= times[-1]:
            ext = _extrapolate(mixtures[-1], times[-1], t) if t > times[-1] else mixtures[-1]
            target, atm = sigma_h(t), float(ext.implied_vols(1.0, [1.0], t)[0])
            if t > times[-1] and np.isfinite(atm) and atm < target:
                def gap(beta):
                    scaled = LognormalMixture(ext.weights, ext.means, beta * ext.stdevs, ext.default)
                    return float(scaled.implied_vols(1.0, [1.0], t)[0]) - target
                beta = brentq(gap, 1.0, 5.0)
                ext = LognormalMixture(ext.weights, ext.means, beta * ext.stdevs, ext.default)
            vols = ext.implied_vols(1.0, x, t)
        else:
            w = np.array([m.implied_vols(1.0, x, tt) ** 2 * tt for m, tt in zip(mixtures, times)])
            vols = np.sqrt(PchipInterpolator(times, w, axis=0)(t) / t)
        matrix[i] = _fill_wings(np.log(x), vols)
    surface = VolatilitySurface(reference_date=reference, calendar=None,
                                daycounter=DayCounter(DayCounterConvention.Actual365),
                                strike=[float(k) for k in spot * m_grid], maturity=[float(t) for t in tenors],
                                volatility_matrix=matrix.tolist())
    return surface, repo, report


def calibrate_snapshot_vols(folder) -> list:
    """Vol surfaces and repo curves of the equities of a snapshot folder (``<yyyymmdd>/<set>/`` with
    ``eq_option_quotes.csv``, curves already there): writes ``eq_vol.csv``, ``eq_repo.csv`` (read by
    ``MarketEnvironment.from_folder``), ``eq_vol_calibration.csv`` (one row per expiry with the mixture parameters,
    or the error of an underlying) and ``eq_vol_quotes.csv`` (bid / mid / ask and model vols of the quotes).

    Each underlying is calibrated on the reference spot of its quotes (``ref_spot``: the close of the quote session
    when the options are not of today), the surface strikes are that spot times ``GRID_MONEYNESS``.

    Returns:
        list: The files written.
    """
    from ..markethandles.marketenvironment import MarketEnvironment
    from ..timehandles.utils import Settings

    folder = Path(folder)
    evaluation_date = _dt.datetime.strptime(folder.parent.name, "%Y%m%d").date()
    quotes = pd.read_csv(folder / "eq_option_quotes.csv")
    previous = Settings.evaluation_date
    Settings.evaluation_date = evaluation_date
    vols, repos, reports, quote_rows = [], [], [], []
    try:
        mkt = MarketEnvironment.from_folder(folder.parent.parent, evaluation_date, folder.name)
        for (ticker, ccy), grp in quotes.groupby(["ticker", "ccy"], sort=True):
            info = {"ticker": ticker, "ccy": ccy, "quote_date": grp["quote_date"].iloc[0],
                    "ref_spot": float(grp["ref_spot"].iloc[0]), "spot_source": grp["spot_source"].iloc[0]}
            try:
                eq = mkt.get_equity(ticker, ccy=Currency[ccy])
                view = Equity(ticker, eq.currency, tf.Variable(info["ref_spot"], dtype=DTYPE), eq.discount_curve,
                              None, eq.dividends, eq.div_yield, None)
                surface, repo, report = build_equity_vol(view, grp)
            except (ValueError, KeyError) as exc:
                reports.append({**info, "error": str(exc)})
                continue
            matrix = surface.volatility_matrix.numpy()
            for i, t in enumerate(surface.maturity):
                expiry = evaluation_date + timedelta(days=int(round(float(t) * 365)))
                vols += [{"ticker": ticker, "ccy": ccy, "expiry": expiry.isoformat(), "strike": float(k),
                          "vol": float(matrix[i, j])} for j, k in enumerate(surface.strike)]
            repos += [{"ticker": ticker, "ccy": ccy, "pillar": d.isoformat(), "rate": float(r)} for d, r in repo.nodes]
            for row in report.to_dict("records"):
                mixture, qv = row.pop("mixture", None), row.pop("quote_vols", None)
                expiry = str(row["expiry"])
                if isinstance(mixture, LognormalMixture):          # parameters: the smile is redrawn from them
                    row.update({"weights": json.dumps(mixture.weights.tolist()),
                                "means": json.dumps(mixture.means.tolist()),
                                "stdevs": json.dumps(mixture.stdevs.tolist()), "default": mixture.default})
                reports.append({**info, **row, "expiry": expiry})
                if isinstance(qv, pd.DataFrame):
                    quote_rows += [{"ticker": ticker, "ccy": ccy, "expiry": expiry, **q} for q in qv.to_dict("records")]
    finally:
        Settings.evaluation_date = previous
    pd.DataFrame(vols, columns=["ticker", "ccy", "expiry", "strike", "vol"]).to_csv(folder / "eq_vol.csv", index=False)
    pd.DataFrame(repos, columns=["ticker", "ccy", "pillar", "rate"]).to_csv(folder / "eq_repo.csv", index=False)
    pd.DataFrame(reports).to_csv(folder / "eq_vol_calibration.csv", index=False)
    pd.DataFrame(quote_rows, columns=["ticker", "ccy", "expiry", "strike", "option_type", "iv_bid", "iv_mid", "iv_ask",
                                      "settlement", "used", "model_vol"]).to_csv(folder / "eq_vol_quotes.csv",
                                                                                 index=False)
    return ["eq_vol.csv", "eq_repo.csv", "eq_vol_calibration.csv", "eq_vol_quotes.csv"]

