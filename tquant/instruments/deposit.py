from datetime import date

from .product import Product, ProductAP
from ..timehandles.utils import DayCounterConvention
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency


class Deposit(Product):
    def __init__(self,
                 currency: Currency,
                 issue_date: date,
                 start_date: date,
                 end_date: date,
                 notional: float,
                 day_count_convention: DayCounterConvention,
                 quote: float):
        super().__init__(currency, start_date, end_date)
        self.issue_date = issue_date
        self.notional = notional
        self.day_count_convention = day_count_convention
        self.day_counter = DayCounter(day_count_convention)
        self.quote = quote


    def __str__(self) -> str:
        return "{ currency: " + str(self.currency) + ",\n" \
                "trade_date: " + str(self.issue_date) + ",\n" \
                "start_date: " + str(self.start_date) + ",\n" \
                "end_date: " + str(self.end_date) + ",\n" \
                "notional: " + str(self.notional) + ",\n" \
                "day_count_convention: " + str(self.day_count_convention) + ",\n" \
                "quote: " + str(self.quote) + " }"



class DepositAP(ProductAP):
    def __init__(self,
                 ccy: str,
                 quote: float,
                 trade_date: date,
                 start_date: date,
                 end_date: date,
                 notional: float,
                 day_count_convention: DayCounterConvention):
        super().__init__(ccy, start_date, end_date, quote)
        self.quote = quote
        self.trade_date = trade_date
        self.start_date = start_date
        self.end_date = end_date
        self.notional = notional
        self.day_counter = DayCounter(day_count_convention)

if __name__ == "__main__":
    from dateutil.relativedelta import relativedelta
    from ..markethandles.ircurve import RateCurve
    from ..pricers.deposit import DepositEngine
    
    today = date(2024, 4, 30)
    spot_date = date(2024, 5, 2)
    maturity = spot_date + relativedelta(months=6)
    deposit = Deposit(Currency.EUR, today, spot_date, maturity, 100, DayCounterConvention.Actual360, 0.01)
    disc_curve = RateCurve([1.0/365, 1.0, 2.0, 4.0], [0.01, 0.02, 0.022, 0.03])
    pricer = DepositEngine(deposit)
    pv, tape = pricer.price_aad(disc_curve, today)
    sensitivities = tape.gradient(pv, [disc_curve.rates])
    print(pv)
    print(sensitivities)