from .pricer import Pricer
from ..instruments.deposit import Deposit
from ..markethandles.ircurve import RateCurve

from datetime import date 
import tensorflow as tf


class DepositEngine(Pricer):
    def __init__(self,
                 product: Deposit):
        
        if isinstance(product, Deposit):
            self.product = product
        else:
            raise TypeError("Wrong product type")

    def price(self, discount_curve: RateCurve, evaluation_date: date):
        day_counter = self.product.day_counter
        ts = day_counter.year_fraction(evaluation_date, self.product.start_date)
        te = day_counter.year_fraction(evaluation_date, self.product.end_date)
        df_s = discount_curve.discount(ts)
        df_e = discount_curve.discount(te)
        start_cashflow = 0.0
        if ts >= 0.0:
            start_cashflow = 1.0
        end_cashflow = 0.0
        if te > 0.0:
            yf = day_counter.year_fraction(self.product.start_date, self.product.end_date)
            end_cashflow = 1.0 + self.product.quote * yf
        start_cashflow *= self.product.notional
        end_cashflow *= self.product.notional
        return start_cashflow * df_s - end_cashflow * df_e
    
    def price_aad(self, discount_curve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, evaluation_date)
        return npv, tape
