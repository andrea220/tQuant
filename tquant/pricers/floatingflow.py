"""
Pricing di cash flow floating
"""
from .pricer import Pricer 
from ..flows.floatingcoupon import FloatingCoupon, FloatingRateLeg
from ..markethandles.ircurve import RateCurve
from ..timehandles.utils import Settings
from datetime import date
import tensorflow as tf

class OisCouponDiscounting(Pricer):

    def __init__(self,
                 coupon: FloatingCoupon) -> None:
        self._coupon = coupon

    def floating_rate(self, 
                    start_date: date,
                    end_date: date,
                    term_structure: RateCurve,
                    evaluation_date: date):
        if start_date >= evaluation_date: # forecast
            return term_structure.forward_rate(start_date, end_date)#, self._coupon.day_counter, evaluation_date) 
        else: # historical
            new_date = self._coupon.index.fixing_date(self._coupon.fixing_date)
            return self._coupon.index.fixing(new_date)

    def amount(self, term_structure, evaluation_date)-> float: 
        ''' 
        cash flow futuro non scontato
        '''
        a = self._coupon.nominal * (self._coupon._gearing*self.floating_rate(self._coupon.ref_period_start, self._coupon.ref_period_end, term_structure,evaluation_date) + self._coupon._spread) * self._coupon.accrual_period
        return a

    def price(self, term_structure: RateCurve, evaluation_date: date):
        if not self._coupon.has_occurred(evaluation_date):
            payment_time = self._coupon.day_counter.year_fraction(evaluation_date, self._coupon._payment_date)
            return self.amount(term_structure, evaluation_date) * term_structure.discount(payment_time)
        else:
            return 0
        
    def price_aad(self, term_structure: RateCurve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(term_structure, evaluation_date)
        return npv, tape

class FloatingCouponDiscounting(Pricer):

    def __init__(self,
                 coupon: FloatingCoupon) -> None:
        self._coupon = coupon

    def floating_rate(self, 
                    start_date: date,
                    end_date: date,
                    term_structure: RateCurve,
                    evaluation_date: date):
        if start_date >= evaluation_date: # forecast
            return term_structure.forward_rate(start_date, end_date)#, self._coupon.day_counter, evaluation_date) 
        else: # historical
            new_date = self._coupon.index.fixing_date(self._coupon.fixing_date)
            return self._coupon.index.fixing(new_date)

    def amount(self, term_structure, evaluation_date)-> float: 
        ''' 
        cash flow futuro non scontato
        '''
        a = self._coupon.nominal * (self._coupon._gearing*self.floating_rate(self._coupon.ref_period_start, self._coupon.ref_period_end, term_structure,evaluation_date) + self._coupon._spread) * self._coupon.accrual_period
        return a

    def price(self, disc_curve: RateCurve, est_curve: RateCurve, evaluation_date: date):
        if not self._coupon.has_occurred(evaluation_date):
            payment_time = self._coupon.day_counter.year_fraction(evaluation_date, self._coupon._payment_date)
            return self.amount(est_curve, evaluation_date) * disc_curve.discount(payment_time)
        else:
            return 0
        
    def price_aad(self, disc_curve: RateCurve, est_curve: RateCurve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(disc_curve, est_curve, evaluation_date)
        return npv, tape
           


class FloatingCouponDiscountingOld(Pricer):

    def __init__(self,
                 coupon: FloatingCoupon,
                 convexity_adjustment: bool) -> None:
        self._coupon = coupon
        self._convexity_adj = convexity_adjustment

    def floating_rate(self, fixing_date, term_structure, evaluation_date):
        if self._convexity_adj:
            raise ValueError("Convexity Adjustment da implementare") #TODO Convexity Adjustment
        else:
            if fixing_date >= Settings.evaluation_date: # forecast
                d2 = self._coupon.index.fixing_maturity(fixing_date)
                return term_structure.forward_rate(fixing_date, d2, self._coupon.day_counter, evaluation_date) 
            else: # historical
                new_date = self._coupon.index.fixing_date(self._coupon.fixing_date)
                return self._coupon.index.fixing(new_date)

    def amount(self, term_structure, evaluation_date)-> float: 
        ''' 
        cash flow futuro non scontato
        '''
        a = self._coupon.nominal * (self._coupon._gearing*self.floating_rate(self._coupon.fixing_date,term_structure,evaluation_date) + self._coupon._spread) * self._coupon.accrual_period
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


class FloatingLegDiscountingOld(Pricer):

    def __init__(self,
                 leg: FloatingRateLeg) -> None:
        self._leg = leg

    def price(self, discount_curve, estimation_curve, evaluation_date: date, coupon_pricer: Pricer):
        if len(self._leg.leg_flows) == 0:
            return 0
        npv = 0
        for i in range(0, len(self._leg.leg_flows)):
            cf = self._leg.leg_flows[i]
            if not cf.has_occurred(evaluation_date):
                pricer = coupon_pricer(cf, False)
                npv += pricer.price(discount_curve, estimation_curve, evaluation_date)
        return npv

    def price_aad(self, discount_curve: RateCurve, estimation_curve, evaluation_date: date, coupon_pricer: Pricer):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, estimation_curve, evaluation_date, coupon_pricer)
        return npv, tape
    


class FloatingLegDiscounting(Pricer):

    def __init__(self,
                 leg: FloatingRateLeg) -> None:
        self._leg = leg

    def price(self, disc_curve, est_curve, evaluation_date: date):
        if len(self._leg.leg_flows) == 0:
            return 0
        npv = 0
        for i in range(0, len(self._leg.leg_flows)):
            cf = self._leg.leg_flows[i]
            if not cf.has_occurred(evaluation_date):
                pricer = FloatingCouponDiscounting(cf)
                npv += pricer.price(disc_curve, est_curve, evaluation_date)
        return npv
    
    def price_aad(self, disc_curve, est_curve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(disc_curve, est_curve, evaluation_date)
        return npv, tape
    
class OisLegDiscounting(Pricer):

    def __init__(self,
                 leg: FloatingRateLeg) -> None:
        self._leg = leg

    def price(self, term_structure, evaluation_date: date):
        if len(self._leg.leg_flows) == 0:
            return 0
        npv = 0
        for i in range(0, len(self._leg.leg_flows)):
            cf = self._leg.leg_flows[i]
            if not cf.has_occurred(evaluation_date):
                pricer = OisCouponDiscounting(cf)
                npv += pricer.price(term_structure, evaluation_date)
        return npv
    
    def price_aad(self, term_structure, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(term_structure, evaluation_date)
        return npv, tape
    