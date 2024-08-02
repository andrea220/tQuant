from .pricer import Pricer
from ..instruments.deposit import Deposit, DepositAP
from ..markethandles.ircurve import RateCurve
from ..timehandles.utils import DayCounterConvention
from ..timehandles.daycounter import DayCounter


from datetime import date 
import tensorflow as tf
from abc import abstractmethod


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

######################
class AbstractPricerAP:
    @abstractmethod
    def price(self,
              product,
              trade_date,
              curves):
        pass

class DepositPricerAP(AbstractPricerAP):
    def __init__(self,
                 curve_assignment):
        super().__init__()
        self.curve_assignment = curve_assignment

    def price(self,
              product,
              as_of_date: date,
              curves):
        if isinstance(product, DepositAP):
            deposit = product
            instance = {"CCY": deposit.ccy, "USAGE": "DISCOUNT"}
            curve_name = self.curve_assignment.get_curve_name(instance)
            curve = curves[curve_name]
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

