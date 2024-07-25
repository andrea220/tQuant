from datetime import date

from .product import Product
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
                 quote = None):
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
