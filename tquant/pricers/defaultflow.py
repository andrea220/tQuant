"""
Pricing di cash flow floating
"""
from .pricer import Pricer 
from ..flows.defaultcoupon import DefaultCoupon
from ..flows.defaultcoupon import DefaultLeg
from ..markethandles.ircurve import RateCurve
from ..markethandles.creditcurve import SurvivalProbabilityCurve

from datetime import date, timedelta
import tensorflow as tf



class DefaultCouponDiscounting(Pricer):

    def __init__(self,
                 coupon: DefaultCoupon,
                 convexity_adjustment: bool) -> None:
        self._coupon = coupon
        self._convexity_adj = convexity_adjustment

    def floating_rate(self, 
                      d1, 
                      term_structure: SurvivalProbabilityCurve,
                      evaluation_date):
        if self._convexity_adj:
            raise ValueError("Convexity Adjustment da implementare") #TODO
        else:
            return term_structure.survival_probability(d1, self._coupon.day_counter, evaluation_date)
    
    def amount(self, term_structure, evaluation_date)-> float: #TODO Ragionare sull'impletazione di coupon giornalieri
        d1 = self._coupon.accrual_start_date
        amount = 0
        while d1 < self._coupon.accrual_end_date:
            d2 = d1 + timedelta(days=1)
            #amount += self._coupon.nominal * (1 - self._coupon.recovery) * (self.floating_rate(d1, term_structure, evaluation_date) - self.floating_rate(d2, term_structure, evaluation_date))* self._coupon.accrual_period
            amount += self._coupon.nominal * (1 - self._coupon.recovery) * (self.floating_rate(d1, term_structure, evaluation_date) - self.floating_rate(d2, term_structure, evaluation_date))
            d1 = d2
        return amount
    
    def price(self, discount_curve: RateCurve, estimation_curve, evaluation_date: date):
        if not self._coupon.has_occurred(evaluation_date):
            tau = self._coupon.day_counter.year_fraction(evaluation_date, self._coupon._payment_date)
            return self.amount(estimation_curve, evaluation_date) * discount_curve.discount(tau)
        else:
            return 0
           
    def price_aad(self, discount_curve: RateCurve, estimation_curve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, estimation_curve, evaluation_date)
        return npv, tape


class DefaultLegDiscount(Pricer):

    def __init__(self,
                 leg: DefaultLeg) -> None:
        self._leg = leg

    def price(self, discount_curve, estimation_curve, evaluation_date: date, coupon_pricer: Pricer):
        if len(self._leg.leg_flows()) == 0:
            return 0
        npv = 0
        for i in range(0, len(self._leg.leg_flows())):
            cf = self._leg.leg_flows()[i]
            if not cf.has_occurred(evaluation_date):
                pricer = coupon_pricer(cf, False)
                npv += pricer.price(discount_curve, estimation_curve, evaluation_date)
        return npv

    def price_aad(self, discount_curve: RateCurve, estimation_curve, evaluation_date: date, coupon_pricer: Pricer):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, estimation_curve, evaluation_date, coupon_pricer)
        return npv, tape