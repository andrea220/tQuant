"""
Pricing di cash flow inflation
"""
from ..interface.pricer import Pricer 
from ..flows.zeroinflationcoupon import ZeroInflationCoupon
from ..flows.inflationleg import InflationLeg
from ..markethandles.ircurve import RateCurve
from ..markethandles.inflationcurve import InflationCurveSimple
from ..utilities.utils import Settings
from datetime import date
import tensorflow as tf



class ZeroInflationCouponDiscounting(Pricer):

    def __init__(self,
                 coupon: ZeroInflationCoupon,
                 convexity_adjustment: bool) -> None:
        self._coupon = coupon
        self._convexity_adj = convexity_adjustment

    def floating_rate(self, 
                      fixing_date, 
                      term_structure: InflationCurveSimple, 
                      evaluation_date):
        if self._convexity_adj:
            raise ValueError("Convexity Adjustment da implementare") #TODO
        else:
            if fixing_date >= Settings.evaluation_date: # rate dell'inflazione
                return term_structure.inflation_value(fixing_date, 
                                                      self._coupon.day_counter, 
                                                      evaluation_date, 
                                                      self._coupon._calendar, 
                                                      self._coupon._payment_lag, 
                                                      self._coupon._payment_lag_period, 
                                                      self._coupon._bdc,
                                                      self._coupon._index._frequency)
            else: # historical
                return self._coupon.index.fixing(self._coupon.fixing_date)

    def amount(self, term_structure, evaluation_date)-> float: 
        ''' 
        cash flow futuro non scontato
        '''
        n = int(self._coupon.accrual_period) #Intero dalla formula
        fixing_part = (1.0 + self._coupon._strike)**n -1.0
        arg = self._coupon._w*((self.floating_rate(self._coupon.end_fixing_date,term_structure, evaluation_date)/self.floating_rate(self._coupon.fixing_date,term_structure, evaluation_date)-1.0) - fixing_part)
        a = self._coupon.nominal * arg 
        return a

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


""" class InflationLegDiscounting(Pricer):

    def __init__(self,
                 leg: InflationLeg) -> None:
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
        return npv, tape """