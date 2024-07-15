import datetime

from tquant import Product, DayCounter


class Fra(Product):
    def __init__(self,
                 ccy: str,
                 start_date: datetime,
                 end_date: datetime,
                 notional: float,
                 quote: float,
                 period: str,
                 index_start_date: datetime,
                 index_end_date: datetime,
                 day_counter: DayCounter,
                 index_day_counter: DayCounter):
        super().__init__(ccy, start_date, end_date, quote)
        self.start_date = start_date
        self.end_date = end_date
        self.notional = notional
        self.period = period
        self.index_start_date = index_start_date
        self.index_end_date = index_end_date
        self.day_counter = day_counter
        self.index_day_counter = index_day_counter

    def __str__(self) -> str:
        return "{ ccy: " + str(self.ccy) + ",\n" \
               "start_date: " + str(self.start_date) + ",\n" \
                "end_date: " + str(self.end_date) + ",\n" \
                "notional: " + str(self.notional) + ",\n" \
                "quote: " + str(self.quote) + ",\n" \
                "index_start_date: " + str(self.index_start_date) + ",\n" \
                "index_end_date: " + str(self.index_end_date) + ",\n" \
                "day_counter: " + str(self.day_counter) + ",\n" \
                "index_end_date: " + str(self.index_day_counter) + " }"



