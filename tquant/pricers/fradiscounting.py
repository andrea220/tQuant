from .pricer import Pricer
from ..instruments.forward import Fra
from ..markethandles.ircurve import RateCurve

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

class FraPricer(Pricer):
    def __init__(self, curve_map):
        super().__init__()
        self._curve_map = curve_map

    def price(self,
              product: Fra,
              as_of_date: date,
              curves: dict[RateCurve]):
        if isinstance(product, Fra):
            fra = product
            try:
                curve_usage = product.ccy.value + ":ON"
                curve_ccy, curve_tenor = curve_usage.split(":")
                dc = curves[self._curve_map[curve_ccy][curve_tenor]]

                curve_usage = product._index.name
                curve_ccy, curve_tenor = curve_usage.split(":")
                fc = curves[self._curve_map[curve_ccy][curve_tenor]]
            except:
                raise ValueError("Unknown Curve")

            pv = 0.0
            if fra.start_date > as_of_date:
                accrual = fra.day_counter.year_fraction(fra.start_date, fra.end_date)
                fwd = fc.forward_rate(fra.start_date, fra.end_date) #TODO quantlib check del forward
                pv += fra.notional * accrual * (fwd - fra.quote) * dc.discount(fra.day_counter.year_fraction(as_of_date, fra.end_date))
            return pv / (1 + fwd*accrual) 
        else:
            raise TypeError("Wrong product type")

    def price_aad(self, product,
              as_of_date: date,
              curves):
        with tf.GradientTape() as tape:
            npv = self.price(product, as_of_date, curves)
        return npv, tape
           
