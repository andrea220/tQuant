"""Discounting of floating (Ibor) and overnight (OIS) cash flows."""

from datetime import date

import tensorflow as tf

from ..flows.floatingcoupon import FloatingCoupon, FloatingRateLeg
from ..markethandles.ircurve import RateCurve
from ..numericalhandles.dtypes import DTYPE
from ..timehandles.utils import Settings


def _discount(curve: RateCurve, d: date) -> tf.Tensor:
    """Discount factor at a *date* (the curve applies its own day count)."""
    return tf.cast(curve.discount(d), DTYPE)


class OisCouponDiscounting:
    """Overnight coupon: compounded rate over the period, ``R = (P(s)/P(e) - 1) / tau``.

    ``tau`` is the coupon's accrual year fraction (its own day count), so
    ``N * R * tau = N (P(s)/P(e) - 1)``: a single-curve OIS floating leg telescopes to
    ``N (P(0, start) - P(0, end))`` when payment dates equal period ends.
    """

    def __init__(self, coupon: FloatingCoupon) -> None:
        self._coupon = coupon

    def floating_rate(self, start_date: date, end_date: date, term_structure: RateCurve) -> tf.Tensor:
        """Forecast compounded rate (period not started) or the historical fixing."""
        if start_date >= Settings.evaluation_date:
            tau = self._coupon.day_counter.year_fraction(start_date, end_date)
            return (_discount(term_structure, start_date) / _discount(term_structure, end_date) - 1.0) / tau
        new_date = self._coupon.index.fixing_date(self._coupon.fixing_date)
        return tf.constant(self._coupon.index.fixing(new_date), dtype=DTYPE)

    def amount(self, term_structure: RateCurve) -> tf.Tensor:
        """Coupon amount ``N (gearing * R + spread) * accrual``."""
        rate = self.floating_rate(self._coupon.ref_period_start, self._coupon.ref_period_end, term_structure)
        return self._coupon.nominal * (self._coupon._gearing * rate + self._coupon._spread) * self._coupon.accrual_period

    def calculate_price(self, term_structure: RateCurve) -> tf.Tensor:
        """PV of the coupon, discounted on the same curve (float64; zero once paid)."""
        if self._coupon.has_occurred(Settings.evaluation_date):
            return tf.constant(0.0, dtype=DTYPE)
        return self.amount(term_structure) * _discount(term_structure, self._coupon.date)


class FloatingCouponDiscounting:
    """Ibor coupon: forward from the estimation curve, discounted on the discount curve."""

    def __init__(self, coupon: FloatingCoupon) -> None:
        self._coupon = coupon
        self._discount_factor = None

    def calc_forward(self, ref_start: date, ref_end: date, term_structure: RateCurve) -> tf.Tensor:
        """Simple forward rate on the index day count: ``(P(s)/P(e) - 1) / tau_index``."""
        t = self._coupon.index.daycounter.year_fraction(ref_start, ref_end)
        return (_discount(term_structure, ref_start) / _discount(term_structure, ref_end) - 1.0) / t

    def floating_rate(self, start_date: date, end_date: date, term_structure: RateCurve) -> tf.Tensor:
        """Forecast forward (fixing in the future) or the historical fixing."""
        if self._coupon.fixing_date > Settings.evaluation_date:
            return self.calc_forward(start_date, end_date, term_structure)
        return tf.constant(self._coupon.index.fixing(self._coupon.fixing_date), dtype=DTYPE)

    def amount(self, term_structure: RateCurve) -> tf.Tensor:
        """Coupon amount ``N (gearing * rate + spread) * accrual`` (rate cached on the coupon)."""
        if self._coupon._rate is None:
            self._coupon._rate = self.floating_rate(
                self._coupon.ref_period_start, self._coupon.ref_period_end, term_structure
            )
        return (
            self._coupon.nominal
            * (self._coupon._gearing * self._coupon._rate + self._coupon._spread)
            * self._coupon.accrual_period
        )

    def calculate_price(self, disc_curve: RateCurve, est_curve: RateCurve) -> tf.Tensor:
        """PV of the coupon (float64; zero once paid)."""
        if self._coupon.has_occurred(Settings.evaluation_date):
            return tf.constant(0.0, dtype=DTYPE)
        if self._coupon._amount is None or self._discount_factor is None:
            self._calc(disc_curve, est_curve)
        return self._coupon._amount * self._discount_factor

    def _calc(self, disc_curve: RateCurve, est_curve: RateCurve) -> None:
        """Compute and cache rate, amount and discount factor."""
        self._coupon._rate = self.floating_rate(
            self._coupon.ref_period_start, self._coupon.ref_period_end, est_curve
        )
        self._coupon._amount = self.amount(est_curve)
        self._discount_factor = _discount(disc_curve, self._coupon.date)


class FloatingLegDiscounting:
    """PV of an Ibor leg: sum of the coupon PVs."""

    def __init__(self, leg: FloatingRateLeg) -> None:
        self._leg = leg

    def calculate_price(self, disc_curve: RateCurve, est_curve: RateCurve) -> tf.Tensor:
        """PV of the leg (float64 scalar)."""
        npv = tf.constant(0.0, dtype=DTYPE)
        for cf in self._leg.leg_flows:
            npv = npv + FloatingCouponDiscounting(cf).calculate_price(disc_curve, est_curve)
        return npv


class OisLegDiscounting:
    """PV of an overnight leg: sum of the coupon PVs (single curve)."""

    def __init__(self, leg: FloatingRateLeg) -> None:
        self._leg = leg

    def calculate_price(self, term_structure: RateCurve) -> tf.Tensor:
        """PV of the leg (float64 scalar)."""
        npv = tf.constant(0.0, dtype=DTYPE)
        for cf in self._leg.leg_flows:
            npv = npv + OisCouponDiscounting(cf).calculate_price(term_structure)
        return npv
