from ..interface.pricer import Pricer
# from ..instruments.fra import ForwardRateAgreement
from datetime import date 
import tensorflow as tf


class FraDiscountingEngine(Pricer):

    def __init__(self, fra) -> None:
        self.fra = fra

    def maturity_time(self, evaluation_date):
        return self.fra._day_counter.year_fraction(evaluation_date, self.fra._maturity_date)
    
    def price(self, discount_curve, estimation_curve, evaluation_date):
        return self.fra._notional * (estimation_curve.forward_rate(self.fra._start_date,
                                                                   self.fra._maturity_date,
                                                                    self.fra._day_counter,
                                                                    evaluation_date) - self.fra._rate) * self.fra.accrual * discount_curve.discount(self.maturity_time(evaluation_date))

    def price_aad(self, discount_curve, estimation_curve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, estimation_curve, evaluation_date)
        return npv, tape

    def implied_rate(self):
        pass
