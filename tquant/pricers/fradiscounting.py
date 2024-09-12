from .pricer import Pricer
from ..instruments.forward import Fra
from ..markethandles.ircurve import RateCurve
from ..timehandles.utils import TimeUnit, BusinessDayConvention

from datetime import date
import tensorflow as tf


class FraPricer(Pricer):
    def __init__(self, curve_map):
        super().__init__()
        self._curve_map = curve_map

    def calculate_price(self, product: Fra, as_of_date: date, curves: dict[RateCurve]):
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
                fixing_d = product.fixing_date
                d1 = product._index.fixing_calendar.advance(
                    fixing_d, 2, TimeUnit.Days, BusinessDayConvention.ModifiedFollowing
                )  # valuedate-start date
                d2 = product._index.fixing_maturity(d1)
                t = product._index.daycounter.year_fraction(d1, d2)
                disc1 = fc.discount(d1)
                disc2 = fc.discount(d2)
                fwd = (disc1 / disc2 - 1) / t
                self.fwd = fwd
                pv += (
                    fra.notional
                    * accrual
                    * (fwd - fra.quote)
                    * dc.discount(
                        fra.day_counter.year_fraction(as_of_date, fra.end_date)
                    )
                )
            return pv / (1 + fwd * accrual)
        else:
            raise TypeError("Wrong product type")
