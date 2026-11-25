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
  squares on vega-weighted prices (about the vol errors), numpy closed forms. The vols of the lognormals are kept
  above ``MIN_VOL_FRACTION`` (1/4) of the at-the-money vol: without that floor the least squares, freer than the
  quotes need where the strikes are sparse, would put a lognormal on a vol near zero, an almost point mass in the
  density (up to 12% of probability on real snapshots) for a gain of hundredths of a vol point.

Stage 3 (surface, :func:`build_equity_vol`): quotes filtered (two-sided, ask <= 3 bid; settlement prices ``last``
only for an underlying without a bid / ask market, Bloomberg's rule), forward and repo of every expiry, Black vols of the
out-of-the-money quotes (the American implied vol on the implied forward is the Black vol), outliers dropped
(outside 0.2-5 x the median of the expiry), a mixture fitted per expiry (standard start; above 0.2 vol points also a warm start from the previous
expiry, the better fit kept). On the
grid (by default the calibrated listed expiries, then one point a year to 6 years, x ``GRID_MONEYNESS`` of the
spot, 30% to 200%, every 5% below 95%; ``GRID_TENORS`` is a fixed alternative), at the forward moneyness ``K / F(t)``.
Each smile is the mixture between the outermost quotes of its expiry; beyond them no quote constrains the mixture's
tail, so the total variance ``w = sigma^2 t`` continues linear in ``k = ln(K/F)`` (slope of the mixture at the
junction, within Lee's bounds and never decreasing away from the money). The straight wing drops the curvature of the
smile: where that would make the density negative (Gatheral's ``g(k) < 0``) the junction moves outward, the mixture
kept up to it; if no junction works the mixture's tail is kept (a density, free of arbitrage) and marked unreliable.
An expiry quoted only near the money (none outside 90%-110% of the forward: a few days) or on one side of the forward
only (a long expiry with puts only) has nothing that shapes a wing: the smile of the next complete expiry (the
previous one for the last expiries), carried to its maturity (Bloomberg's short-term extrapolation:
the vols of the lognormals kept, the log relative forwards scaled by ``sqrt(t / T)``) and scaled to its at-the-money
vol, guides them (points beyond its quotes, low weight, and the start of the fit), its own quotes fitted as usual.
Then:

- between the expiries: monotone (PCHIP) interpolation of the total variance in time, after the calendar step: at the
  same forward moneyness the total variance of an expiry is raised to that of the earlier ones where it is lower
  (raising keeps the smile free of butterfly arbitrage, a lowering would not); a reliable point (quoted or on a
  straight wing) only by earlier reliable points, so an unreliable tail (very short expiries quoted near the money
  only) never moves a quote nor a reliable wing;
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

from ..markethandles.dividendcurve import DividendCurve
from ..markethandles.equity import Equity
from ..markethandles.ircurve import RateCurve
from ..markethandles.utils import Currency, OptionType
from ..markethandles.volatilitysurface import VolatilitySurface
from ..numericalhandles.dtypes import DTYPE
from ..pricers.black import bjerksund_stensland_calc, blackscholes_calc
from ..timehandles.daycounter import DayCounter, DayCounterConvention

__all__ = ["implied_volatility", "implied_forward", "implied_repo_curve", "black_implied_vol", "LognormalMixture",
           "build_equity_vol", "calibrate_snapshot_vols", "implied_spot", "listed_tenors", "GRID_TENORS",
           "GRID_MONEYNESS"]

# Bjerksund-Stensland is finite and increasing in the vol on this range (NaN below ~1.5%, below the European
# price at very high vols); equity vols are well inside it
_VOL_LOW, _VOL_HIGH = 0.02, 3.0
# smile mixture: vols of the lognormals at least this fraction of the at-the-money vol of the expiry (and 0.5%)
MIN_VOL_FRACTION = 0.25


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
            weights=None, x0: Optional[np.ndarray] = None,
            min_vol_fraction: float = MIN_VOL_FRACTION) -> "LognormalMixture":
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
            min_vol_fraction (float): Lower bound of the vols of the lognormals as a fraction of the at-the-money
                vol (linear in the strike between the quotes around the forward), never below 0.5%; ``0``: 0.5% only.

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

        order = np.argsort(strikes)
        atm = float(np.interp(forward, strikes[order], vols[order]))
        if x0 is None:                                    # ordered start: low mean / high vol first (no exchange)
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
        vol_floor = max(0.005, min_vol_fraction * atm)    # no lognormal near a point mass
        lower[-n:], upper[-n:] = np.log(vol_floor * root_t), np.log(3.0 * max(root_t, 0.1))
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
GRID_MONEYNESS = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05,
                  1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0)   # of the spot: the autocallable barriers sit at 50-70%
WING_SLOPE_MAX = 2.0         # Lee: beyond the quotes the total variance grows at most 2 |ln(K/F)|
WING_RANGE = (0.1, 4.0)      # forward moneyness over which a straight wing must keep a positive density
WING_G_MIN = 0.25            # density margin of a straight wing: Gatheral's g >= this on it (a flat smile has g = 1)
WING_INWARD = 0.2            # how far inside the quotes (log moneyness) the junction may move to find that margin
WING_QUOTE_TOL = 0.0025      # largest vol gap between a wing moved inside the quotes and the smile it replaces
MIN_QUOTES = 3               # quotes of a calibrated expiry (and two-sided quotes of a good bid / ask expiry)
NARROW_RANGE = (0.9, 1.1)    # an expiry without quotes outside this forward moneyness borrows its wings' shape
SHORT_DAYS = 7               # expiries closer than this keep their quotes within NARROW_RANGE only (wings borrowed)
NARROW_ANCHORS = (0.5, 0.6, 0.7, 0.8, 0.85, 1.15, 1.2, 1.3, 1.5)   # forward moneyness of the borrowed wing points
NARROW_ANCHOR_WEIGHT = 0.1   # their fit weight (a two-sided quote weighs 1)
DIVIDEND_YEARS = 10.0        # equity indices: dividends extrapolated at a constant yield to this horizon
DIVIDEND_KEEP_YEARS = 2.0    # equity indices: projected dividends kept to this horizon, implied by the options beyond


def listed_tenors(times, max_years: float = 6.0) -> list:
    """Maturities of the surface: the listed expiries (``times``, years) and one a year after the last one up to
    *max_years* (whole years)."""
    times = sorted(float(t) for t in times)
    return times + [float(y) for y in range(int(np.floor(times[-1])) + 1, int(max_years) + 1)]


def _prepare_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Normalised quotes: ``option_type`` C / P, ``settlement`` flag, filters of the bid / ask (positive, ask <= 3 bid).

    Settlement prices (``last`` used as bid and ask, no two-sided quote) only for an underlying without a bid / ask
    market (Bloomberg's rule): an expiry has a good bid / ask market with at least ``MIN_QUOTES`` two-sided quotes and
    10% of its quotes; with at least 3 such expiries (or 10% of the expiries) the settlement prices are dropped
    (forwards, repo, dividends and smiles from the quoted expiries only), otherwise only they are used. A settlement
    price is not a market: an exchange mark that may be stale against the spot of the quotes."""
    q = quotes.copy()
    q["option_type"] = q["option_type"].astype(str).str.upper().str[0]
    q["expiry"] = pd.to_datetime(q["expiry"]).dt.date
    for column in ("bid", "ask"):
        q[column] = pd.to_numeric(q[column], errors="coerce")
    two_sided = (q["bid"] > 0) & (q["ask"] >= q["bid"]) & (q["ask"] <= 3.0 * q["bid"])
    per_expiry = two_sided.groupby(q["expiry"]).agg(["sum", "mean"])
    good = (per_expiry["sum"] >= MIN_QUOTES) & (per_expiry["mean"] >= 0.10)
    q["settlement"] = False
    if good.sum() >= 3 or (len(good) and good.mean() >= 0.10):
        return q[two_sided].reset_index(drop=True)
    if "last" in q:                                       # no bid / ask market: the settlement prices only
        last = pd.to_numeric(q["last"], errors="coerce")
        settle = q["bid"].isna() & q["ask"].isna() & (last > 0)
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


def _carry_smile(mixture: LognormalMixture, t_from: float, t: float) -> LognormalMixture:
    """The smile of *t_from* carried to *t* (Bloomberg's short-term extrapolation): the vols of the lognormals and the
    default probability kept, the log relative forwards ``ln xi_l`` scaled by ``sqrt(t / t_from)``."""
    scale = np.sqrt(t / t_from)
    return LognormalMixture(mixture.weights, np.exp(np.log(mixture.means) * scale), mixture.stdevs * scale,
                            mixture.default)


def _borrow_narrow_smiles(good: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Expiries whose quotes leave a wing free: quoted only near the money (none outside ``NARROW_RANGE`` of the
    forward, e.g. a few days) or out of the money on one side of the forward only (a long expiry with puts only). The
    smile of the next complete expiry (the previous one for the last expiries), carried to the maturity
    (:func:`_carry_smile`; for a narrow expiry scaled to its at-the-money vol: the vols of the scenarios, and the
    spread of their forwards too when that spread alone gives more than its at-the-money vol), guides them: the mixture is fitted again
    to the expiry's own quotes (weight 1) plus that smile at ``NARROW_ANCHORS`` beyond the quotes (weight
    ``NARROW_ANCHOR_WEIGHT``), starting from it. ``smile_from`` is the expiry lent, ``fit_rms`` on the own quotes."""
    good = good.copy()
    good["smile_from"] = None
    lo, hi = np.log(NARROW_RANGE[0]), np.log(NARROW_RANGE[1])
    k_lo, k_hi = good["k_lo"].to_numpy(float), good["k_hi"].to_numpy(float)
    narrow = (k_lo >= lo - 1e-12) & (k_hi <= hi + 1e-12)
    one_sided = (k_hi < 0.0) | (k_lo > 0.0)               # out-of-the-money quotes on one side of the forward only
    guided = narrow | one_sided
    rows = list(good.index)
    for i, idx in enumerate(rows):
        if not guided[i]:
            continue
        donor = next((rows[j] for j in range(i + 1, len(rows)) if not guided[j]), None)
        if donor is None:                                  # the last expiries: the previous complete one, carried
            donor = next((rows[j] for j in range(i - 1, -1, -1) if not guided[j]), None)
        if donor is None:
            continue
        row, lend = good.loc[idx], good.loc[donor]
        t, forward = float(row["t"]), float(row["forward"])
        guide = _carry_smile(lend["mixture"], float(lend["t"]), t)
        if narrow[i]:                                      # quoted at the money: the guide at its at-the-money vol
            def scaled(beta, m=guide, spread=False):        # the vols of the scenarios (and their forwards' spread)
                means = np.exp(beta * np.log(m.means)) if spread else m.means
                return LognormalMixture(m.weights, means, beta * m.stdevs, m.default)

            def gap(beta, spread=False):
                return float(scaled(beta, spread=spread).implied_vols(1.0, [1.0], t)[0]) - float(row["atm_vol"])

            try:
                guide = scaled(brentq(gap, 0.2, 5.0))
            except ValueError:                             # the forwards' spread alone above the ATM vol: scale it too
                try:
                    guide = scaled(brentq(lambda b: gap(b, spread=True), 0.01, 5.0), spread=True)
                except ValueError:
                    continue
        qv = row["quote_vols"]
        used = qv["used"].to_numpy(bool)
        k, v = qv["strike"].to_numpy(float)[used], qv["iv_mid"].to_numpy(float)[used]
        w = np.where(qv["settlement"].to_numpy(bool)[used], 0.25, 1.0)
        x = np.array([m for m in NARROW_ANCHORS if np.log(m) < row["k_lo"] or np.log(m) > row["k_hi"]])
        kv = guide.implied_vols(1.0, x, t)
        ok = np.isfinite(kv)
        start = np.concatenate([np.log(np.maximum([guide.default, *guide.weights], 1e-12)), np.log(guide.means),
                                np.log(guide.stdevs)])
        try:
            mixture = LognormalMixture.fit(forward, t, np.concatenate([k, forward * x[ok]]),
                                           np.concatenate([v, kv[ok]]), n=n,
                                           weights=np.concatenate([w, np.full(ok.sum(), NARROW_ANCHOR_WEIGHT)]),
                                           x0=start)
        except ValueError:
            continue
        model = mixture.implied_vols(forward, qv["strike"].to_numpy(float), t)
        mixture.rms = float(np.sqrt(np.mean((w * (model[used] - v)) ** 2)))
        good.at[idx, "atm_vol"] = float(mixture.implied_vols(1.0, [1.0], t)[0])
        good.at[idx, "mixture"] = mixture
        good.at[idx, "fit_rms"] = mixture.rms
        good.at[idx, "smile_from"] = lend["expiry"]
        good.at[idx, "quote_vols"] = qv.assign(model_vol=model)
    return good


def _wing_spec(mixture: LognormalMixture, t: float, k_lo: float, k_hi: float, candidates: int = 25) -> dict:
    """The wings of one smile beyond its outermost quotes (log forward moneyness ``k_lo``, ``k_hi``).

    On each side the first junction ``k_j``, from the edge of the quotes outward, where a straight total variance
    ``w(k) = w(k_j) + s (k - k_j)`` keeps a density margin out to ``WING_RANGE``: ``s`` the slope of the mixture at
    ``k_j`` clipped to a total variance not decreasing away from the money and to Lee's ``WING_SLOPE_MAX`` (a clip that
    would bend the smile the wrong way is refused), and Gatheral's
    ``g(k) = (1 - k s / 2w)^2 - s^2 / 4 (1 / w + 1 / 4) >= WING_G_MIN`` on the wing (``w'' = 0``; ``g`` is the density
    over the Black density of the same total variance: near 0 the local vol explodes). Between the edge and ``k_j`` the
    mixture. Candidates: the edge, then inside the quotes up to ``WING_INWARD`` (the mixture is often steepest at its
    last quotes: inside, where the smile is less steep, the straight wing must stay within ``WING_QUOTE_TOL`` of the
    smile up to the edge), then beyond the edge. When none reaches the margin, the candidate with the largest density
    (``g >= 0``). No junction: the mixture's tail, unreliable.

    Returns:
        dict: ``{side: (k_j, w_j, s) or None}`` for ``"lo"`` / ``"hi"``, with the edges.
    """
    spec = {"k_lo": k_lo, "k_hi": k_hi}
    w_of = lambda kk: mixture.implied_vols(1.0, np.exp(np.atleast_1d(kk)), t) ** 2 * t
    for side, edge, far, sign in (("lo", k_lo, np.log(WING_RANGE[0]), -1.0), ("hi", k_hi, np.log(WING_RANGE[1]), 1.0)):
        spec[side] = None
        if sign * (far - edge) <= 0:
            continue
        h = 1e-3 * max(k_hi - k_lo, 1e-2)
        best, best_g = None, 0.0                                  # fallback: the largest density, g >= 0
        outward = edge + sign * np.linspace(0.0, 0.8 * abs(far - edge), candidates)
        inward = edge - sign * np.linspace(0.0, WING_INWARD, candidates)[1:]
        for kj in (outward[0], *inward, *outward[1:]):       # the edge, then inside the quotes, then beyond
            if sign * kj < 0.05:                                  # the junction stays out of the money
                continue
            wj, w_in = w_of([kj, kj - sign * h])
            if not (np.isfinite(wj) and np.isfinite(w_in)):
                continue
            s_in = (wj - w_in) / (sign * h)                      # dw/dk at the junction, from the inside
            s_out = float(np.clip(s_in, 0.0, WING_SLOPE_MAX) if sign > 0 else np.clip(s_in, -WING_SLOPE_MAX, 0.0))
            if sign * (s_out - s_in) < -1e-12:                    # a clip against the curvature: concave kink
                continue
            kw = np.linspace(kj, far, 200)
            ww = wj + s_out * (kw - kj)
            g = (1.0 - kw * s_out / (2.0 * ww)) ** 2 - s_out ** 2 / 4.0 * (1.0 / ww + 0.25)
            if not ((ww > 0).all() and g.min() >= 0.0):
                continue
            if sign * (kj - edge) < 0:                            # inside the quotes: close to the smile there
                kq = np.linspace(kj, edge, 20)
                gap = np.abs(np.sqrt(np.maximum(wj + s_out * (kq - kj), 0.0) / t) - np.sqrt(w_of(kq) / t))
                if not (np.isfinite(gap).all() and gap.max() <= WING_QUOTE_TOL):
                    continue
            if g.min() >= WING_G_MIN:
                best = (float(kj), float(wj), s_out)
                break
            if best is None or g.min() > best_g:
                best, best_g = (float(kj), float(wj), s_out), float(g.min())
        spec[side] = best
    return spec


def _smile_w(mixture: LognormalMixture, t: float, k: np.ndarray, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Total variance of a smile at log forward moneyness *k* with its wings (:func:`_wing_spec`), and whether each
    point is reliable (quoted, or the mixture up to a junction and the straight wing beyond it)."""
    k = np.asarray(k, dtype=float)
    w = mixture.implied_vols(1.0, np.exp(k), t) ** 2 * t
    reliable = (k >= spec["k_lo"]) & (k <= spec["k_hi"])
    for side, beyond in (("lo", k < spec["k_lo"]), ("hi", k > spec["k_hi"])):
        wing = spec[side]
        if wing is None:
            continue
        kj, wj, slope = wing
        straight = (k < kj) if side == "lo" else (k > kj)                # a junction may sit inside the quotes
        w = np.where(straight, wj + slope * (k - kj), w)
        reliable = reliable | beyond
    return w, reliable


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
                     settlement_weight: float = 0.25, min_quotes: int = MIN_QUOTES, min_repo_days: int = 30, tenors=None,
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
        tenors: Maturities of the output grid (years); default :func:`listed_tenors` of the calibrated expiries.
        moneyness: Strikes of the output grid as fractions of the spot.

    Returns:
        tuple: :class:`VolatilitySurface` (``tenors`` x ``moneyness`` x spot, ACT/365), repo :class:`RateCurve`
        (zero repo rates at the calibrated expiries from ``min_repo_days``) and the report (one row per expiry: ``expiry``, ``t``,
        ``forward``, ``repo``, ``forward_bid`` / ``forward_ask``, ``quotes``, ``source`` (market / settlement),
        ``k_lo`` / ``k_hi`` (log forward moneyness of the outermost quotes used: the mixture inside, the wings
        beyond), ``smile_from`` (an expiry with a wing free of quotes: the expiry whose smile guides it),
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
        if t < SHORT_DAYS / 365.0:                     # a few days: near the money only, the wings from the next expiry
            keep &= (k >= NARROW_RANGE[0] * forward - 1e-12) & (k <= NARROW_RANGE[1] * forward + 1e-12)
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
        k_used = np.log(k[keep] / forward)                 # log forward moneyness of the quotes used
        rows.append({"expiry": expiry, "t": t, **res, "quotes": int(keep.sum()), "source": source,
                     "k_lo": float(k_used.min()), "k_hi": float(k_used.max()),
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
    good = _borrow_narrow_smiles(good)
    report["smile_from"] = None
    for idx in good.index:                                # the report shows the smiles used
        for column in ("mixture", "fit_rms", "atm_vol", "smile_from", "quote_vols"):
            report.at[idx, column] = good.at[idx, column]
    times = good["t"].to_numpy(float)
    mixtures = list(good["mixture"])
    specs = [_wing_spec(m, tt, lo, hi) for m, tt, lo, hi in zip(mixtures, times, good["k_lo"], good["k_hi"])]
    sigma_h = _heston_atm(times, good["atm_vol"].to_numpy(float))
    m_grid = np.asarray(moneyness, dtype=float)
    tenors = listed_tenors(times) if tenors is None else list(tenors)
    matrix = np.empty((len(tenors), len(m_grid)))
    for i, t in enumerate(tenors):
        x = spot * m_grid / float(view.forward(t))                       # forward moneyness K / F(t)
        # every expiry at this forward moneyness, then the calendar step: raised to the earlier expiries where lower,
        # a reliable point only by earlier reliable ones (an unreliable tail never moves a quote or a reliable wing)
        pairs = [_smile_w(m, tt, np.log(x), sp) for m, tt, sp in zip(mixtures, times, specs)]
        w_exp = np.array([w for w, _ in pairs])
        rel = np.array([r for _, r in pairs])
        best_rel = np.full(len(x), np.nan)                               # max of the earlier reliable points
        for j in range(len(times)):
            if j:
                w_exp[j] = np.fmax(w_exp[j], np.where(rel[j], best_rel, w_exp[j - 1]))
            best_rel = np.fmax(best_rel, np.where(rel[j], w_exp[j], np.nan))
        if t <= times[0]:
            vols = np.sqrt(w_exp[0] / times[0])
        elif t >= times[-1]:
            ext = _extrapolate(mixtures[-1], times[-1], t) if t > times[-1] else mixtures[-1]
            target, atm = sigma_h(t), float(ext.implied_vols(1.0, [1.0], t)[0])
            if t > times[-1] and np.isfinite(atm) and atm < target:
                def gap(beta):
                    scaled = LognormalMixture(ext.weights, ext.means, beta * ext.stdevs, ext.default)
                    return float(scaled.implied_vols(1.0, [1.0], t)[0]) - target
                beta = brentq(gap, 1.0, 5.0)
                ext = LognormalMixture(ext.weights, ext.means, beta * ext.stdevs, ext.default)
            scale = np.sqrt(t / times[-1])                         # quote edges carried in ln(K/F) / sqrt(t)
            sp = specs[-1] if t == times[-1] else _wing_spec(ext, t, specs[-1]["k_lo"] * scale,
                                                             specs[-1]["k_hi"] * scale)
            w_ext, _ = _smile_w(ext, t, np.log(x), sp)
            vols = np.sqrt(np.fmax(w_ext, w_exp[-1]) / t)          # not below the last expiry
        else:
            w_fit = w_exp.copy()
            for j, tt in enumerate(times):                 # far strikes of a short expiry: no implied vol (price 0)
                if not np.isfinite(w_fit[j]).all():
                    w_fit[j] = _fill_wings(np.log(x), np.sqrt(w_fit[j] / tt)) ** 2 * tt
            vols = np.sqrt(PchipInterpolator(times, w_fit, axis=0)(t) / t)   # PCHIP keeps the monotonicity
        matrix[i] = _fill_wings(np.log(x), vols)
    surface = VolatilitySurface(reference_date=reference, calendar=None,
                                daycounter=DayCounter(DayCounterConvention.Actual365),
                                strike=[float(k) for k in spot * m_grid], maturity=[float(t) for t in tenors],
                                volatility_matrix=matrix.tolist())
    return surface, repo, report


def implied_spot(equity: Equity, quotes: pd.DataFrame, min_days: int = 3,
                 dividend_model: str = "continuous") -> tuple[float, date]:
    """Spot implied by the options, for an underlying whose price is delayed against its options (e.g. an index
    without real-time data): the implied forward ``F`` of the first expiry from *min_days* (the anchor, zero repo by
    construction), ``S = F P(0, T) + PV_div(T)``. The nearest expiry: the zero-repo assumption costs about
    ``repo T S`` (0.01% of the spot at one week), the error of the forward passes to the spot whatever ``T``. The forward does not depend on the spot (European options; American
    ones only through the early-exercise premia).

    Args:
        equity (Equity): Underlying (its spot only starts the iteration).
        quotes (pd.DataFrame): Option quotes (``expiry``, ``strike``, ``option_type``, ``bid``, ``ask``, ...).
        min_days (int): The anchor is the first expiry from this many days.
        dividend_model (str): As in :func:`implied_forward`.

    Returns:
        tuple: ``(spot, anchor expiry)``.

    Raises:
        ValueError: If no expiry with two-sided quotes is far enough.
    """
    q = _prepare_quotes(quotes)
    q = q[~q["settlement"]]
    reference = equity.discount_curve.reference_date
    expiries = sorted(e for e in set(q["expiry"]) if (e - reference).days >= min_days)
    for anchor in expiries:
        try:
            res = implied_forward(equity, anchor, q[q["expiry"] == anchor], dividend_model=dividend_model)
        except ValueError:
            continue
        _, df, _, s_net = _carry_inputs(equity, anchor)
        pv = float(equity.spot) - s_net
        return res["forward"] * df + pv, anchor
    raise ValueError(f"No expiry from {min_days} days with two-sided quotes to imply the spot of {equity.ticker}")


def _add_years(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year + n)
    except ValueError:                                     # 29 February
        return d.replace(year=d.year + n, day=28)


def extrapolate_dividends(equity: Equity, years: float = 10.0, repo: float = 0.0) -> pd.DataFrame:
    """Dividends beyond the last projected one, at a constant yield, to *years* (equity indices: the projections cover
    about 3 years, beyond them the repo implied by the options would absorb the missing dividends and the forward
    past the last listed expiry would grow as if none were paid).

    The last projected year (the 365 days to the last projected ex-date) is repeated year after year with its weekly
    pattern; each dividend keeps the yield of the one it repeats on the forward just before its ex-date,
    ``D(d + n years) = D(d) F(d + n years) / F(d)``, with ``F(t) = (S - PV_div(t)) e^{-repo t} / P(0, t)`` on all the
    dividends before ``t`` (the extrapolated ones included, in date order) and the repo held at *repo*: the dividends
    grow with the forward. Payment date = ex-date.

    Args:
        equity (Equity): Underlying with its projected dividends (spot, discount curve).
        years (float): Horizon from the reference date.
        repo (float): Repo zero rate held over the extrapolation (the level of the projected years).

    Returns:
        pd.DataFrame: ``ex_date, pay_date, amount`` of the extrapolated dividends (empty without projections or when
        they already reach the horizon); ``attrs``: ``last_projected``, ``yield`` (of the first extrapolated year on
        the forward at its middle), ``repo``.
    """
    columns = ["ex_date", "pay_date", "amount"]
    curve = equity.dividends
    reference = equity.discount_curve.reference_date
    if curve is None or not len(curve):
        return pd.DataFrame(columns=columns)
    pays = curve.payment_dates or curve.ex_dates
    divs = [(e, p, float(a)) for e, p, a in zip(curve.ex_dates, pays, curve.amounts) if e > reference]
    if not divs:
        return pd.DataFrame(columns=columns)
    last = max(e for e, _, _ in divs)
    horizon = _add_years(reference, int(years)) + timedelta(days=int(round((years - int(years)) * 365)))
    if last >= horizon:
        return pd.DataFrame(columns=columns)
    spot = float(equity.spot)
    disc = equity.discount_curve
    base = sorted((d for d in divs if d[0] > last - timedelta(days=365)), key=lambda d: d[0])
    targets = []                                           # the extrapolated ex-dates, increasing
    n = 1
    while _add_years(base[0][0], n) <= horizon:
        targets += [(_add_years(e, n), a, e) for e, _, a in base if _add_years(e, n) <= horizon]
        n += 1
    dates = sorted({p for _, p, _ in divs} | {e for e, _, _ in base} | {t for t, _, _ in targets}
                   | {last + timedelta(days=182)})
    dfs = dict(zip(dates, np.ravel(disc.discounts(dates).numpy()) if hasattr(disc, "discounts")
                   else [float(disc.discount(d)) for d in dates]))   # one evaluation of the curve

    def forward(t: date, pv_before: float) -> float:       # just before the ex-date t
        return (spot - pv_before) * np.exp(-repo * (t - reference).days / 365.0) / dfs[t]

    def pv_projected(t: date) -> float:
        return sum(a * dfs[p] for e, p, a in divs if e < t)

    base_forward = {e: forward(e, pv_projected(e)) for e, _, _ in base}
    pv_all, out = pv_projected(last + timedelta(days=1)), []       # every projection is before the targets
    for new, a, e in targets:
        amount = a * forward(new, pv_all) / base_forward[e]
        out.append({"ex_date": new, "pay_date": new, "amount": amount})
        pv_all += amount * dfs[new]
    frame = pd.DataFrame(out, columns=columns)
    frame.attrs.update(last_projected=last, repo=repo)
    if len(frame):
        middle = last + timedelta(days=182)
        first = frame[frame["ex_date"] <= _add_years(last, 1)]
        before = pv_projected(middle) + float(sum(r["amount"] * dfs[r["ex_date"]] for r in out if r["ex_date"] < middle))
        frame.attrs["yield"] = float(first["amount"].sum() / forward(middle, before))
    return frame


def _listed_forwards(equity: Equity, quotes: pd.DataFrame, after: date) -> list:
    """``[(expiry, forward)]`` implied by the options of the expiries after *after*, with the quotes of
    :func:`build_equity_vol` (two-sided, else the settlement prices); the expiries without a parity forward skipped."""
    q = _prepare_quotes(quotes)
    out = []
    for expiry in sorted(e for e in set(q["expiry"]) if e > after):
        grp = q[q["expiry"] == expiry]
        market = grp[~grp["settlement"]]
        try:
            out.append((expiry, float(implied_forward(equity, expiry, market if len(market) else grp)["forward"])))
        except ValueError:
            continue
    return out


def implied_index_dividends(equity: Equity, quotes: pd.DataFrame, keep_years: float = DIVIDEND_KEEP_YEARS,
                            years: float = DIVIDEND_YEARS, min_repo_days: int = 30) -> pd.DataFrame:
    """Dividends of an equity index consistent with its listed options, keeping the repo stable (the market practice:
    index dividends from the market, the repo a smooth financing / borrow level, not the residual of the carry).

    - The projected dividends of *equity* (Bloomberg) are kept to *keep_years* (the cutoff), where they are reliable.
    - The repo is held at the level implied by the options at the last listed expiry before the cutoff (from
      *min_repo_days*).
    - Beyond the cutoff, each listed expiry's parity forward gives the PV of the dividends to it with that repo,
      ``PV(T) = S - F(T) P(0, T) e^{repo T}``; the increase between two expiries is spread over the dividend dates of
      the period with the weekly pattern of the projections (their last year repeated: the May peak), scaled to it
      (a negative increase, from a noisy forward, gives no dividend).
    - Beyond the last listed expiry: constant yield to *years* (:func:`extrapolate_dividends`) from the implied ones.

    Args:
        equity (Equity): The index: spot, discount curve, projected dividends (the whole download).
        quotes (pd.DataFrame): Its option quotes.
        keep_years (float): Projections kept up to this horizon.
        years (float): Horizon of the dividends.
        min_repo_days (int): Shortest expiry for the repo level.

    Returns:
        pd.DataFrame: ``ex_date, pay_date, amount, source`` (``download``, ``implied``, ``extrapolated``) of the
        dividends to use; ``attrs``: ``cutoff``, ``repo``, ``implied`` (per listed expiry: ``expiry``, ``forward``,
        ``pv``, ``amount``), ``yield`` of the extrapolation. Without listed expiries beyond the cutoff: the projections
        to the cutoff and the constant yield from them.
    """
    columns = ["ex_date", "pay_date", "amount", "source"]
    curve = equity.dividends
    reference = equity.discount_curve.reference_date
    pays = curve.payment_dates or curve.ex_dates
    projected = sorted((e, p, float(a)) for e, p, a in zip(curve.ex_dates, pays, curve.amounts) if e > reference)
    if not projected:
        return pd.DataFrame(columns=columns)
    last_projected = projected[-1][0]
    cutoff = min(_add_years(reference, int(keep_years)) + timedelta(days=int(round((keep_years % 1) * 365))),
                 last_projected)
    kept = [d for d in projected if d[0] <= cutoff]
    template = [d for d in projected if d[0] > last_projected - timedelta(days=365)]      # one year of the pattern
    disc = equity.discount_curve
    spot = float(equity.spot)

    def with_dividends(items):
        e, p, a = zip(*items) if items else ((), (), ())
        dc = DividendCurve(reference, list(e), list(a), curve.currency, payment_dates=list(p),
                           daycounter_convention=DayCounterConvention.Actual365) if items else None
        return Equity(equity.ticker, equity.currency, equity.spot, disc, None, dc, None, None)

    view_kept = with_dividends(kept)
    repo = _projected_repo(view_kept, quotes, cutoff, min_repo_days)
    listed = _listed_forwards(view_kept, quotes, cutoff)
    rows = [{"ex_date": e, "pay_date": p, "amount": a, "source": "download"} for e, p, a in kept]
    implied_report = []
    items = list(kept)
    pv_prev = sum(a * float(disc.discount(p)) for _, p, a in kept)
    start = cutoff
    for expiry, forward in listed:
        t = equity.year_fraction(expiry)
        pv = spot - forward * float(disc.discount(expiry)) * np.exp(repo * t)
        dates = sorted({_add_years(e, n) for e, _, _ in template for n in range(0, 40)
                        if start < _add_years(e, n) <= expiry})
        shape = {_add_years(e, n): a for e, _, a in template for n in range(0, 40) if start < _add_years(e, n) <= expiry}
        weight = sum(shape[d] * float(disc.discount(d)) for d in dates)
        scale = max(pv - pv_prev, 0.0) / weight if weight > 0 else 0.0
        for d in dates:
            amount = scale * shape[d]
            if amount > 0:
                items.append((d, d, amount))
                rows.append({"ex_date": d, "pay_date": d, "amount": amount, "source": "implied"})
        implied_report.append({"expiry": expiry.isoformat(), "forward": forward, "pv": pv,
                               "amount": scale * sum(shape[d] for d in dates)})
        pv_prev = pv_prev + scale * weight
        start = expiry
    extra = extrapolate_dividends(with_dividends(items), years, repo)
    rows += [{"ex_date": r["ex_date"], "pay_date": r["pay_date"], "amount": float(r["amount"]),
              "source": "extrapolated"} for r in extra.to_dict("records")]
    out = pd.DataFrame(rows, columns=columns)
    out.attrs.update(cutoff=cutoff, repo=repo, implied=implied_report, last_projected=last_projected,
                     **{"yield": extra.attrs.get("yield")})
    return out


def _projected_repo(equity: Equity, quotes: pd.DataFrame, last: date, min_days: int = 30) -> float:
    """Repo implied by the options at the last listed expiry within the projected dividends (from *min_days*), with
    the quotes of :func:`build_equity_vol` (two-sided, else the settlement prices); 0 if none."""
    q = _prepare_quotes(quotes)
    reference = equity.discount_curve.reference_date
    expiries = sorted(e for e in set(q["expiry"]) if min_days <= (e - reference).days and e <= last)
    for expiry in reversed(expiries):
        grp = q[q["expiry"] == expiry]
        market = grp[~grp["settlement"]]
        try:
            return float(implied_forward(equity, expiry, market if len(market) else grp)["repo"])
        except ValueError:
            continue
    return 0.0


def _equity_indices(folder: Path) -> set:
    """Tickers of the equity indices of a snapshot (``Index`` tickers of ``snapshot_info.json`` with an equity
    spot; the rate fixings have none)."""
    info_path, spot_path = folder / "snapshot_info.json", folder / "eq_spot.csv"
    if not info_path.is_file() or not spot_path.is_file():
        return set()
    universe = json.loads(info_path.read_text(encoding="utf-8")).get("universe", [])
    codes = {"_".join(t.split()[:-1]) for t in universe if t.strip().endswith(" Index")}
    return codes & set(pd.read_csv(spot_path)["ticker"].astype(str))


def calibrate_snapshot_vols(folder) -> list:
    """Vol surfaces and repo curves of the equities of a snapshot folder (``<yyyymmdd>/<set>/`` with
    ``eq_option_quotes.csv``, curves already there): writes ``eq_vol.csv``, ``eq_repo.csv`` (read by
    ``MarketEnvironment.from_folder``), ``eq_vol_calibration.csv`` (one row per expiry with the mixture parameters,
    or the error of an underlying) and ``eq_vol_quotes.csv`` (bid / mid / ask and model vols of the quotes).

    Each underlying is calibrated on the reference spot of its quotes (``ref_spot``: the close of the quote session
    when the options are not of today), the surface strikes are that spot times ``GRID_MONEYNESS``. When
    ``spot_source`` starts with "implied" (a delayed price, e.g. SPX) the spot is :func:`implied_spot` and replaces the
    spot of ``eq_spot.csv``.

    Returns:
        list: The files written.
    """
    from ..markethandles.marketenvironment import MarketEnvironment
    from ..timehandles.utils import Settings

    folder = Path(folder)
    evaluation_date = _dt.datetime.strptime(folder.parent.name, "%Y%m%d").date()
    quotes = pd.read_csv(folder / "eq_option_quotes.csv")
    indices = _equity_indices(folder)
    # the downloaded dividends: eq_dividends_download.csv once written (the projections of the indices are replaced in
    # eq_dividends.csv beyond DIVIDEND_KEEP_YEARS), else the rows of eq_dividends.csv as downloaded
    download_path, div_path = folder / "eq_dividends_download.csv", folder / "eq_dividends.csv"
    div_table = None
    if download_path.is_file():
        div_table = pd.read_csv(download_path)
    elif div_path.is_file():
        div_table = pd.read_csv(div_path)
        if "source" in div_table:
            div_table = div_table[div_table["source"].isna() | (div_table["source"] == "download")]
        div_table = div_table[["ticker", "ccy", "ex_date", "pay_date", "amount"]].reset_index(drop=True)
    index_dividends = {}                                    # (ticker, ccy) -> the dividends used for an index
    previous = Settings.evaluation_date
    Settings.evaluation_date = evaluation_date
    vols, repos, reports, quote_rows, implied_spots = [], [], [], [], {}
    try:
        mkt = MarketEnvironment.from_folder(folder.parent.parent, evaluation_date, folder.name)
        for (ticker, ccy), grp in quotes.groupby(["ticker", "ccy"], sort=True):
            info = {"ticker": ticker, "ccy": ccy, "quote_date": grp["quote_date"].iloc[0],
                    "ref_spot": float(grp["ref_spot"].iloc[0]), "spot_source": grp["spot_source"].iloc[0]}
            try:
                eq = mkt.get_equity(ticker, ccy=Currency[ccy])
                dividends = eq.dividends
                index = ticker in indices and div_table is not None
                if index:                                   # the downloaded projections only
                    own = div_table[(div_table["ticker"] == ticker) & (div_table["ccy"] == ccy)]
                    dividends = (DividendCurve.from_dataframe(
                        reference_date=evaluation_date, df=own.sort_values("ex_date"), ex_date_col="ex_date",
                        amount_col="amount", declared_date_col=None, payment_date_col="pay_date",
                        currency=Currency(ccy), daycounter_convention=DayCounterConvention.Actual365)
                        if len(own) else None)
                view = Equity(ticker, eq.currency, tf.Variable(info["ref_spot"], dtype=DTYPE), eq.discount_curve,
                              None, dividends, eq.div_yield, None)
                if str(info["spot_source"]).startswith("implied"):      # delayed price: spot from the options
                    spot, anchor = implied_spot(view, grp)
                    info.update(ref_spot=spot, spot_source=f"implied from the options ({anchor.isoformat()})")
                    view.spot.assign(spot)
                    implied_spots[(ticker, ccy)] = spot
                if index and dividends is not None and len(dividends):
                    # projections to the cutoff, then implied by the listed forwards (repo held), then constant yield
                    used = implied_index_dividends(view, grp)
                    if len(used):
                        view = Equity(ticker, eq.currency, view.spot, eq.discount_curve, None, DividendCurve(
                            evaluation_date, list(used["ex_date"]), [float(a) for a in used["amount"]],
                            Currency(ccy), payment_dates=list(used["pay_date"]),
                            daycounter_convention=DayCounterConvention.Actual365), eq.div_yield, None)
                        index_dividends[(ticker, ccy)] = used
                        implied = used.attrs["implied"]
                        info.update(div_cutoff=used.attrs["cutoff"].isoformat(), div_repo_held=used.attrs["repo"],
                                    div_implied_to=implied[-1]["expiry"] if implied else None,
                                    div_extrapolated_yield=used.attrs.get("yield"))
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
    def write(name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(folder / name, index=False)

    write("eq_vol.csv", pd.DataFrame(vols, columns=["ticker", "ccy", "expiry", "strike", "vol"]))
    write("eq_repo.csv", pd.DataFrame(repos, columns=["ticker", "ccy", "pillar", "rate"]))
    write("eq_vol_calibration.csv", pd.DataFrame(reports))
    if implied_spots:                                     # the pricers use the same spot as the vols
        spots = pd.read_csv(folder / "eq_spot.csv")
        for (ticker, ccy), spot in implied_spots.items():
            spots.loc[(spots["ticker"] == ticker) & (spots["ccy"] == ccy), "spot"] = spot
        spots.to_csv(folder / "eq_spot.csv", index=False)
    write("eq_vol_quotes.csv", pd.DataFrame(quote_rows, columns=["ticker", "ccy", "expiry", "strike", "option_type",
                                                                 "iv_bid", "iv_mid", "iv_ask", "settlement", "used",
                                                                 "model_vol"]))
    written = ["eq_vol.csv", "eq_repo.csv", "eq_vol_calibration.csv", "eq_vol_quotes.csv"]
    if index_dividends:                                   # the indices' dividends replaced, the download kept apart
        if not download_path.is_file():
            div_table.to_csv(download_path, index=False)
            written.append("eq_dividends_download.csv")
        keep = div_table[[(t, c) not in index_dividends for t, c in zip(div_table["ticker"], div_table["ccy"])]]
        parts = [keep.assign(source="download")]
        for (ticker, ccy), used in index_dividends.items():
            parts.append(used.assign(ticker=ticker, ccy=ccy, ex_date=[d.isoformat() for d in used["ex_date"]],
                                     pay_date=[d.isoformat() for d in used["pay_date"]]))
        table = pd.concat(parts, ignore_index=True)[["ticker", "ccy", "ex_date", "pay_date", "amount", "source"]]
        table.sort_values(["ticker", "ccy", "ex_date"]).to_csv(div_path, index=False)
        written.append("eq_dividends.csv")
    return written

