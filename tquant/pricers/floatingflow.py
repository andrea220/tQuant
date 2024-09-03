from .pricer import Pricer 
from ..flows.floatingcoupon import FloatingCoupon, FloatingRateLeg
from ..markethandles.ircurve import RateCurve
from ..timehandles.utils import Settings
from datetime import date
import tensorflow as tf

class OisCouponDiscounting:

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

    def calculate_price(self, term_structure: RateCurve, evaluation_date: date):
        if not self._coupon.has_occurred(evaluation_date):
            payment_time = self._coupon.day_counter.year_fraction(evaluation_date, self._coupon._payment_date)
            return self.amount(term_structure, evaluation_date) * term_structure.discount(payment_time)
        else:
            return 0
        
    def price_aad(self, term_structure: RateCurve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.calculate_price(term_structure, evaluation_date)
        return npv, tape

class FloatingCouponDiscounting:

    def __init__(self,
                 coupon: FloatingCoupon) -> None:
        self._coupon = coupon
        self._discount_factor = None

    def calc_forward(self,
                     ref_start,
                     ref_end,
                     term_structure):
        t = self._coupon.index.daycounter.year_fraction(ref_start, ref_end) 
        disc1 = term_structure.discount(ref_start)
        disc2 = term_structure.discount(ref_end)
        return (disc1/disc2 -1)/t

    def floating_rate(self, 
                    start_date: date,
                    end_date: date,
                    term_structure: RateCurve,
                    evaluation_date: date):
        if self._coupon.fixing_date > evaluation_date: 
            # forecast forward rate
            return self.calc_forward(start_date, end_date, term_structure) 
        else: 
            # return historical fixing 
            return tf.constant(self._coupon.index.fixing(self._coupon.fixing_date), dtype=tf.float64) 

    def amount(self, term_structure, evaluation_date)-> float: 
        ''' 
        cash flow futuro non scontato
        '''
        if self._coupon._rate == None:
            self._coupon._rate = self.floating_rate(self._coupon.ref_period_start, self._coupon.ref_period_end, term_structure,evaluation_date) 
        return self._coupon.nominal * (self._coupon._gearing*self._coupon._rate + self._coupon._spread) * self._coupon.accrual_period
    

    def calculate_price(self, disc_curve: RateCurve, est_curve: RateCurve, evaluation_date: date):
        if not self._coupon.has_occurred(evaluation_date):
            if self._coupon._amount == None or self._discount_factor == None:
                self._calc(disc_curve, est_curve, evaluation_date)
            # payment_time = self._coupon.day_counter.year_fraction(evaluation_date, self._coupon._payment_date)
            # return self.amount(est_curve, evaluation_date) * disc_curve.discount(payment_time)
            return self._coupon._amount * self._discount_factor
        else:
            return 0
    
    def _calc(self, disc_curve: RateCurve, est_curve: RateCurve, evaluation_date: date):
        """ cache results"""
        self._coupon._rate = self.floating_rate(self._coupon.ref_period_start, self._coupon.ref_period_end, est_curve, evaluation_date) 
        self._coupon._amount = self.amount(est_curve, evaluation_date)
        payment_time = self._coupon.day_counter.year_fraction(evaluation_date, self._coupon._payment_date)
        self._discount_factor = disc_curve.discount(payment_time)

           

class FloatingLegDiscounting:

    def __init__(self,
                 leg: FloatingRateLeg) -> None:
        self._leg = leg

    def calculate_price(self, disc_curve, est_curve, evaluation_date: date):
        if len(self._leg.leg_flows) == 0:
            return 0
        npv = 0
        for i in range(0, len(self._leg.leg_flows)):
            cf = self._leg.leg_flows[i]
            if not cf.has_occurred(evaluation_date):
                pricer = FloatingCouponDiscounting(cf)
                npv += pricer.calculate_price(disc_curve, est_curve, evaluation_date)
        return npv
    
    # def price_aad(self, disc_curve, est_curve, evaluation_date: date):
    #     with tf.GradientTape() as tape:
    #         npv = self.calculate_price(disc_curve, est_curve, evaluation_date)
    #     return npv, tape
    
class OisLegDiscounting:

    def __init__(self,
                 leg: FloatingRateLeg) -> None:
        self._leg = leg

    def calculate_price(self, term_structure, evaluation_date: date):
        if len(self._leg.leg_flows) == 0:
            return 0
        npv = 0
        for i in range(0, len(self._leg.leg_flows)):
            cf = self._leg.leg_flows[i]
            if not cf.has_occurred(evaluation_date):
                pricer = OisCouponDiscounting(cf)
                npv += pricer.calculate_price(term_structure, evaluation_date)
        return npv
    
    def price_aad(self, term_structure, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.calculate_price(term_structure, evaluation_date)
        return npv, tape
    