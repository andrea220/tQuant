"""
Pricing di cash flow fissi
"""

from ..flows.fixedcoupon import FixedCoupon, FixedRateLeg
from ..markethandles.ircurve import RateCurve
from ..timehandles.utils import Settings


class FixedCouponDiscounting:

    def __init__(self, coupon: FixedCoupon) -> None:
        self._coupon = coupon

    def calculate_price(self, discount_curve: RateCurve):
        if not self._coupon.has_occurred(Settings.evaluation_date):
            tau = self._coupon.day_counter.year_fraction(
                Settings.evaluation_date, self._coupon._payment_date
            )
            return self._coupon.amount * discount_curve.discount(tau)
        else:
            return 0


class FixedLegDiscounting:

    def __init__(self, leg: FixedRateLeg) -> None:
        self._leg = leg

    def calculate_price(self, discount_curve: RateCurve):
        if len(self._leg.leg_flows) == 0:
            return 0
        npv = 0
        for i in range(0, len(self._leg.leg_flows)):
            cf = self._leg.leg_flows[i]
            if not cf.has_occurred(Settings.evaluation_date):
                pricer = FixedCouponDiscounting(cf)
                npv += pricer.calculate_price(discount_curve)
        return npv
