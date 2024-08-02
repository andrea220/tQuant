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
