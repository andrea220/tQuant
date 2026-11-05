from __future__ import annotations

from datetime import date
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf

from ..instruments.swap import Swap
from ..markethandles.marketenvironment import MarketEnvironment
from ..markethandles.utils import Currency, SwapType
from ..models.gaussianshortrate import GaussianShortRateModel
from ..numericalhandles.dtypes import DTYPE
from ..timehandles.utils import Settings


class SwapExposureGenerator:
    """Mark-to-market and exposure profile of a swap under a Gaussian short rate model.

    Paths of the model factors are simulated (exactly) on the exposure dates and the floating
    fixing dates; at each exposure date the swap is revalued on every path with the model's
    closed-form bonds ``P(t, T)``:

    - discounting with the model curve (it must be the swap's discount curve);
    - forwarding with the index curve of ``market_env`` through a deterministic multiplicative
      basis to the discount curve, ``P_f(t, T) = P_d(t, T) * [P_f(0,T)/P_f(0,t)] / [P_d(0,T)/P_d(0,t)]``
      (same forward formula as :class:`SwapPricer`, which it matches exactly at ``t = 0``);
    - a coupon already fixed at ``t`` uses the rate fixed on that path (historical index fixing
      if the fixing date is not after the evaluation date).

    Sign as :class:`SwapPricer` (payer: floating - fixed). The model curve and the index curve are
    market variables, so exposures are differentiable w.r.t. their pillars.

    Args:
        model (GaussianShortRateModel): Short rate model built on the swap's discount curve.
        exposure_dates (Sequence[date]): Exposure dates (>= evaluation date).
        n_paths (int): Number of paths.
        seed (int): Seed of the stateless Gaussian draws.

    After :meth:`simulate`: ``mtm`` ``[n_paths, n_dates]``, ``discount`` ``D(0, t)`` ``[n_paths, n_dates]``,
    ``expected_exposure`` (EE = E[max(V, 0)]), ``expected_negative_exposure`` (ENE = E[min(V, 0)]),
    ``discounted_expected_exposure`` (E[D max(V, 0)]), ``pfe(q)``.
    """

    def __init__(self, model: GaussianShortRateModel, exposure_dates: Sequence[date],
                 n_paths: int = 10_000, seed: int = 42) -> None:
        self._model = model
        self._dates = sorted(exposure_dates)
        self._n_paths = n_paths
        self._seed = seed
        self._mtm = None
        self._discount = None

    @property
    def model(self) -> GaussianShortRateModel:
        return self._model

    @property
    def exposure_dates(self) -> list:
        return self._dates

    def _require(self):
        if self._mtm is None:
            raise ValueError("Must call simulate()")

    @property
    def mtm(self) -> tf.Tensor:
        """Swap value ``[n_paths, n_dates]`` on each path and exposure date."""
        self._require()
        return self._mtm

    @property
    def discount(self) -> tf.Tensor:
        """Bank-account discount ``D(0, t)`` ``[n_paths, n_dates]``."""
        self._require()
        return self._discount

    @property
    def expected_exposure(self) -> tf.Tensor:
        self._require()
        return tf.reduce_mean(tf.maximum(self._mtm, 0.0), axis=0)

    @property
    def expected_negative_exposure(self) -> tf.Tensor:
        self._require()
        return tf.reduce_mean(tf.minimum(self._mtm, 0.0), axis=0)

    @property
    def discounted_expected_exposure(self) -> tf.Tensor:
        self._require()
        return tf.reduce_mean(self._discount * tf.maximum(self._mtm, 0.0), axis=0)

    def pfe(self, quantile: float = 0.975) -> np.ndarray:
        """Potential future exposure: *quantile* of ``max(V, 0)`` per date (numpy, no gradient)."""
        self._require()
        return np.quantile(np.maximum(self._mtm.numpy(), 0.0), quantile, axis=0)

    # ------------------------------------------------------------------

    def simulate(self, product: Swap, market_env: MarketEnvironment) -> None:
        """Simulate the swap value on the exposure dates.

        Args:
            product (Swap): The swap (index name ``CCY:TENOR``, as for :class:`SwapPricer`).
            market_env (MarketEnvironment): Market with the swap's index curve.

        Raises:
            ValueError: If an exposure date is before the evaluation date.
        """
        eval_date = Settings.evaluation_date
        if self._dates and self._dates[0] < eval_date:
            raise ValueError("exposure dates must not be before the evaluation date")
        ccy_str, tenor = product._index.name.split(":")
        fwd_curve = market_env.get_ir_curve(Currency[ccy_str], ticker=tenor)
        disc_curve = self._model.curve
        m = self._model

        floating = product.floating_leg.leg_flows
        fixed = product.fixed_leg.leg_flows
        future_fixings = sorted({c.fixing_date for c in floating if c.fixing_date > eval_date})
        sim_dates = sorted(set(d for d in self._dates if d > eval_date) | set(future_fixings))
        times = [m.time(d) for d in sim_dates]
        if times:
            xs, disc = m.simulate(times, self._n_paths, self._seed)
        col = {d: i for i, d in enumerate(sim_dates)}
        zero_state = tf.zeros([self._n_paths, m.size()], DTYPE)

        def state(d):
            return zero_state if d <= eval_date else xs[:, col[d]]

        def forward_bond_ratio(t, s, e, x):
            """``P_f(t, s) / P_f(t, e)`` per path (deterministic basis to the discount curve)."""
            basis = (tf.cast(fwd_curve.discount(s), DTYPE) / tf.cast(fwd_curve.discount(e), DTYPE)) * \
                    (tf.cast(disc_curve.discount(e), DTYPE) / tf.cast(disc_curve.discount(s), DTYPE))
            return basis * m.zero_bond(t, s, x) / m.zero_bond(t, e, x)

        # floating rate of every coupon: historical, or fixed on the path at its fixing date
        fixed_rate = {}
        for c in floating:
            if c.fixing_date <= eval_date:
                fixed_rate[id(c)] = tf.fill([self._n_paths], tf.constant(c.index.fixing(c.fixing_date), DTYPE))
            else:
                tau = c.index.daycounter.year_fraction(c.ref_period_start, c.ref_period_end)
                ratio = forward_bond_ratio(c.fixing_date, c.ref_period_start, c.ref_period_end, state(c.fixing_date))
                fixed_rate[id(c)] = (ratio - 1.0) / tau

        sign = 1.0 if product.swap_type == SwapType.Payer else -1.0
        mtm, discounts = [], []
        for d in self._dates:
            x = state(d)
            fix_pv = tf.zeros([self._n_paths], DTYPE)
            for c in fixed:
                if c.date > d:
                    fix_pv += c.amount * m.zero_bond(d, c.date, x)
            flt_pv = tf.zeros([self._n_paths], DTYPE)
            for c in floating:
                if c.date <= d:
                    continue
                if c.fixing_date <= d:
                    rate = fixed_rate[id(c)]
                else:
                    tau = c.index.daycounter.year_fraction(c.ref_period_start, c.ref_period_end)
                    rate = (forward_bond_ratio(d, c.ref_period_start, c.ref_period_end, x) - 1.0) / tau
                amount = c.nominal * (c._gearing * rate + c._spread) * c.accrual_period
                flt_pv += amount * m.zero_bond(d, c.date, x)
            mtm.append(sign * (flt_pv - fix_pv))
            discounts.append(tf.ones([self._n_paths], DTYPE) if d <= eval_date else disc[:, col[d]])
        self._mtm = tf.stack(mtm, axis=1)
        self._discount = tf.stack(discounts, axis=1)
