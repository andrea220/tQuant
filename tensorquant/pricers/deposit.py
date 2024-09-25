from .pricer import Pricer
from ..instruments.deposit import Deposit
from ..timehandles.utils import DayCounterConvention, Settings
from ..timehandles.daycounter import DayCounter



class DepositPricer(Pricer):
    def __init__(self, market_map):
        super().__init__()
        self._market_map = market_map

    def calculate_price(self, product, market):
        if isinstance(product, Deposit):
            try:
                curve_usage = product.ccy.value + ":ON"
                curve_ccy, curve_tenor = curve_usage.split(":")
                curve = market[self._market_map[f'IR:{curve_ccy}'][curve_tenor]]
            except:
                raise ValueError("Unknown Curve")
            day_counter = DayCounter(DayCounterConvention.Actual365)
            ts = day_counter.year_fraction(Settings.evaluation_date, product.start_date)
            te = day_counter.year_fraction(Settings.evaluation_date, product.end_date)
            df_s = curve.discount(ts)
            df_e = curve.discount(te)
            start_cashflow = 0.0
            if ts >= 0.0:
                start_cashflow = 1.0
            end_cashflow = 0.0
            if te > 0.0:
                yf = product.day_counter.year_fraction(
                    product.start_date, product.end_date
                )
                end_cashflow = 1.0 + product.rate * yf
            start_cashflow *= product.notional
            end_cashflow *= product.notional

            return start_cashflow * df_s - end_cashflow * df_e
        else:
            raise TypeError("Wrong product type")
