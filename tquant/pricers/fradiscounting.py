from .pricer import Pricer
from ..instruments.forward import Fra
from ..markethandles.ircurve import RateCurve

from datetime import date 
import tensorflow as tf


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
           
