from datetime import datetime

from dateutil.relativedelta import relativedelta

from .product import Product
from tquant import DayCounterConvention, DayCounter


class Deposit(Product):
    def __init__(self,
                 ccy: str,
                 quote: float,
                 trade_date: datetime,
                 start_date: datetime,
                 end_date: datetime,
                 notional: float,
                 day_count_convention: DayCounterConvention):
        super().__init__(ccy, start_date, end_date, quote)
        self.quote = quote
        self.trade_date = trade_date
        self.start_date = start_date
        self.end_date = end_date
        self.notional = notional
        self.day_counter = DayCounter(day_count_convention)

    def __str__(self) -> str:
        return "{ ccy: " + str(self.ccy) + ",\n" \
                "trade_date: " + str(self.trade_date) + ",\n" \
                "start_date: " + str(self.start_date) + ",\n" \
                "end_date: " + str(self.end_date) + ",\n" \
                "notional: " + str(self.notional) + ",\n" \
                "day_counter: " + str(self.day_counter) + " }"


if __name__ == "__main__":
    deposit = Deposit("EUR",
                      0.01,
                      datetime(2024, 4, 16),
                      datetime(2024, 4, 18),
                      datetime(2024, 4, 18) + relativedelta(months=3),
                      100,
                      DayCounterConvention.Actual365)

    print(deposit)
