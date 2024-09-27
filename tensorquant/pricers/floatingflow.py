# from .pricer import Pricer
from ..flows.floatingcoupon import FloatingCoupon, FloatingRateLeg
from ..markethandles.ircurve import RateCurve
from ..timehandles.utils import Settings
from datetime import date
from tensorflow import constant, float64


class OisCouponDiscounting:

    def __init__(self, coupon: FloatingCoupon) -> None:  # TODO non è oiscoupon??
        self._coupon = coupon

    def floating_rate(
        self, start_date: date, end_date: date, term_structure: RateCurve
    ):
        if start_date >= Settings.evaluation_date:  # forecast
            return term_structure.forward_rate(start_date, end_date)
        else:  # historical
            new_date = self._coupon.index.fixing_date(self._coupon.fixing_date)
            return self._coupon.index.fixing(new_date)

    def amount(self, term_structure: RateCurve) -> float:
        a = (
            self._coupon.nominal
            * (
                self._coupon._gearing
                * self.floating_rate(
                    self._coupon.ref_period_start,
                    self._coupon.ref_period_end,
                    term_structure,
                )
                + self._coupon._spread
            )
            * self._coupon.accrual_period
        )
        return a

    def calculate_price(self, term_structure: RateCurve):
        if not self._coupon.has_occurred(Settings.evaluation_date):
            payment_time = self._coupon.day_counter.year_fraction(
                Settings.evaluation_date, self._coupon._payment_date
            )
            return self.amount(term_structure) * term_structure.discount(payment_time)
        else:
            return 0

    # def price_aad(self, term_structure: RateCurve, evaluation_date: date):
    #     with tf.GradientTape() as tape:
    #         npv = self.calculate_price(term_structure, evaluation_date)
    #     return npv, tape


class FloatingCouponDiscounting:

    def __init__(self, coupon: FloatingCoupon) -> None:
        self._coupon = coupon
        self._discount_factor = None

    def calc_forward(self, ref_start, ref_end, term_structure):
        t = self._coupon.index.daycounter.year_fraction(ref_start, ref_end)
        disc1 = term_structure.discount(ref_start)
        disc2 = term_structure.discount(ref_end)
        return (disc1 / disc2 - 1) / t

    def floating_rate(
        self, start_date: date, end_date: date, term_structure: RateCurve
    ):
        if self._coupon.fixing_date > Settings.evaluation_date:
            # forecast forward rate
            return self.calc_forward(start_date, end_date, term_structure)
        else:
            # return historical fixing
            return constant(
                self._coupon.index.fixing(self._coupon.fixing_date), dtype=float64
            )

    def amount(self, term_structure) -> float:

        if self._coupon._rate == None:
            self._coupon._rate = self.floating_rate(
                self._coupon.ref_period_start,
                self._coupon.ref_period_end,
                term_structure,
            )
        return (
            self._coupon.nominal
            * (self._coupon._gearing * self._coupon._rate + self._coupon._spread)
            * self._coupon.accrual_period
        )

    def calculate_price(self, disc_curve: RateCurve, est_curve: RateCurve):
        if not self._coupon.has_occurred(Settings.evaluation_date):
            if self._coupon._amount == None or self._discount_factor == None:
                self._calc(disc_curve, est_curve)
            return self._coupon._amount * self._discount_factor
        else:
            return 0

    def _calc(self, disc_curve: RateCurve, est_curve: RateCurve):
        """cache results"""
        self._coupon._rate = self.floating_rate(
            self._coupon.ref_period_start, self._coupon.ref_period_end, est_curve
        )
        self._coupon._amount = self.amount(est_curve)
        payment_time = self._coupon.day_counter.year_fraction(
            Settings.evaluation_date, self._coupon._payment_date
        )
        self._discount_factor = disc_curve.discount(payment_time)


class FloatingLegDiscounting:

    def __init__(self, leg: FloatingRateLeg) -> None:
        self._leg = leg

    def calculate_price(self, disc_curve, est_curve):
        if len(self._leg.leg_flows) == 0:
            return 0
        npv = 0
        for i in range(0, len(self._leg.leg_flows)):
            cf = self._leg.leg_flows[i]
            if not cf.has_occurred(Settings.evaluation_date):
                pricer = FloatingCouponDiscounting(cf)
                npv += pricer.calculate_price(disc_curve, est_curve)
        return npv


class OisLegDiscounting:

    def __init__(self, leg: FloatingRateLeg) -> None:
        self._leg = leg

    def calculate_price(self, term_structure):
        if len(self._leg.leg_flows) == 0:
            return 0
        npv = 0
        for i in range(0, len(self._leg.leg_flows)):
            cf = self._leg.leg_flows[i]
            if not cf.has_occurred(Settings.evaluation_date):
                pricer = OisCouponDiscounting(cf)
                npv += pricer.calculate_price(term_structure)
        return npv
