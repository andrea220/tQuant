"""
Pricing di cash flow fissi
"""
from .pricer import Pricer 
from ..flows.fixedcoupon import FixedCoupon, FixedRateLeg
from ..markethandles.ircurve import RateCurve
from datetime import date
import tensorflow as tf


class FixedCouponDiscounting(Pricer):

    def __init__(self,
                 coupon: FixedCoupon) -> None:
        self._coupon = coupon

    def price(self, discount_curve: RateCurve, evaluation_date: date):
        if not self._coupon.has_occurred(evaluation_date):
            tau = self._coupon.day_counter.year_fraction(evaluation_date, self._coupon._payment_date)
            return self._coupon.amount * discount_curve.discount(tau)
        else:
            return 0
           
    def price_aad(self, discount_curve: RateCurve, evaluation_date: date): 
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, evaluation_date)
        return npv, tape
    

class FixedLegDiscounting(Pricer):

    def __init__(self,
                 leg: FixedRateLeg) -> None:
        self._leg = leg

    def price(self,
              discount_curve: RateCurve,
              evaluation_date: date):#, coupon_pricer: Pricer):
        if len(self._leg.leg_flows) == 0:
            return 0
        npv = 0
        for i in range(0, len(self._leg.leg_flows)):
            cf = self._leg.leg_flows[i]
            if not cf.has_occurred(evaluation_date):
                # pricer = coupon_pricer(cf)
                tau = cf.day_counter.year_fraction(evaluation_date, cf._payment_date)
                npv += cf.amount * discount_curve.discount(tau)
                # npv += pricer.price(discount_curve, evaluation_date)
        return npv

    def price_aad(self, discount_curve: RateCurve, evaluation_date: date, coupon_pricer: Pricer):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, evaluation_date, coupon_pricer)
        return npv, tape