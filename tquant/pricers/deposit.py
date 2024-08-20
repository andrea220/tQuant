from .pricer import Pricer
from ..instruments.deposit import Deposit
from ..markethandles.ircurve import RateCurve
from ..timehandles.utils import DayCounterConvention
from ..timehandles.daycounter import DayCounter

from datetime import date 
import tensorflow as tf


class DepositPricer(Pricer):
    def __init__(self,
                 curve_map):
        super().__init__()
        self._curve_map = curve_map

    def price(self,
              product,
              as_of_date: date,
              curves):
        if isinstance(product, Deposit):
            deposit = product
            try:
                curve_usage = product.ccy.value + ":ON"
                curve_ccy, curve_tenor = curve_usage.split(":")
                curve = curves[self._curve_map[curve_ccy][curve_tenor]]
            except:
                raise ValueError("Unknown Curve")
            act365 = DayCounterConvention.Actual365
            day_counter = DayCounter(act365)
            ts = day_counter.year_fraction(as_of_date, deposit.start_date)
            te = day_counter.year_fraction(as_of_date, deposit.end_date)
            df_s = curve.discount(ts)
            df_e = curve.discount(te)
            start_cashflow = 0.0
            if ts >= 0.0:
                start_cashflow = 1.0
            end_cashflow = 0.0
            if te > 0.0:
                yf = deposit.day_counter.year_fraction(deposit.start_date, deposit.end_date)
                end_cashflow = 1.0 + deposit.quote * yf
            start_cashflow *= deposit.notional
            end_cashflow *= deposit.notional

            return start_cashflow * df_s - end_cashflow * df_e
        else:
            raise TypeError("Wrong product type")
        
    def price_aad(self, 
                    product,
                    as_of_date: date,
                    curves):
        with tf.GradientTape() as tape:
            npv = self.price(product,
                            as_of_date,
                            curves)
        return npv, tape