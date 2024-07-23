from .pricer import Pricer 
from ..flows.premiumcoupon import PremiumCoupon
from ..flows.premiumcoupon import PremiumLeg
from ..markethandles.ircurve import RateCurve
from ..markethandles.creditcurve import SurvivalProbabilityCurve

from datetime import date
import tensorflow as tf



class PremiumCouponDiscounting(Pricer):

    def __init__(self,
                 coupon: PremiumCoupon,
                 convexity_adjustment: bool) -> None:
        self._coupon = coupon
        self._convexity_adj = convexity_adjustment

    def floating_rate(self, 
                      d1, 
                      term_structure: SurvivalProbabilityCurve,
                      evaluation_date
                      ):
        if self._convexity_adj:
            raise ValueError("Convexity Adjustment da implementare") #TODO
        else:
            return term_structure.survival_probability(d1, self._coupon.day_counter, evaluation_date)
    
    def amount(self, term_structure: SurvivalProbabilityCurve, evaluation_date)-> float:
        d1 = self._coupon.date
        amount = self._coupon.nominal * self._coupon.spread * self.floating_rate(d1, term_structure, evaluation_date) * self._coupon.accrual_period
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


class PremiumLegDiscount(Pricer):

    def __init__(self,
                 leg: PremiumLeg) -> None:
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