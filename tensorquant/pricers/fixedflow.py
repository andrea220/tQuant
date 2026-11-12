"""Discounting of fixed cash flows."""

import tensorflow as tf

from ..flows.fixedcoupon import FixedCoupon, FixedRateLeg
from ..markethandles.ircurve import RateCurve
from ..numericalhandles.dtypes import DTYPE
from ..timehandles.utils import Settings


class PrefetchedCurve:
    """A curve whose discount factors at the given dates are computed in one vectorised call
    (:meth:`RateCurve.discounts`) and served by :meth:`discount`; any other date or attribute goes to the curve.
    The leg pricers wrap their curves in it, the coupon pricers are unchanged."""

    def __init__(self, curve: RateCurve, dates) -> None:
        self._curve = curve
        dates = sorted(set(dates))
        self._dfs = dict(zip(dates, tf.unstack(curve.discounts(dates)))) if dates else {}

    def discount(self, term):
        df = self._dfs.get(term)
        return df if df is not None else self._curve.discount(term)

    def __getattr__(self, name):
        return getattr(self._curve, name)


def prefetched(curve, dates):
    """:class:`PrefetchedCurve` for a :class:`RateCurve` (vectorised discount factors), the curve itself otherwise."""
    return PrefetchedCurve(curve, dates) if hasattr(curve, "discounts") else curve


class FixedCouponDiscounting:
    """PV of a fixed coupon: ``amount * P(0, payment_date)``."""

    def __init__(self, coupon: FixedCoupon) -> None:
        self._coupon = coupon

    def calculate_price(self, discount_curve: RateCurve) -> tf.Tensor:
        """PV of the coupon (float64 scalar; zero once paid).

        The discount factor is read at the payment *date*, so the curve uses its own
        day count (not the coupon's accrual convention).
        """
        if self._coupon.has_occurred(Settings.evaluation_date):
            return tf.constant(0.0, dtype=DTYPE)
        return self._coupon.amount * tf.cast(discount_curve.discount(self._coupon.date), DTYPE)


class FixedLegDiscounting:
    """PV of a fixed leg: sum of the coupon PVs."""

    def __init__(self, leg: FixedRateLeg) -> None:
        self._leg = leg

    def calculate_price(self, discount_curve: RateCurve) -> tf.Tensor:
        """PV of the leg (float64 scalar)."""
        curve = prefetched(discount_curve, [cf.date for cf in self._leg.leg_flows])
        npv = tf.constant(0.0, dtype=DTYPE)
        for cf in self._leg.leg_flows:
            npv = npv + FixedCouponDiscounting(cf).calculate_price(curve)
        return npv
