import datetime

from tquant import Product, DayCounterConvention, DayCounter


class Future(Product):
    def __init__(self,
                 ccy: str,
                 trade_date: datetime,
                 start_date: datetime,
                 end_date: datetime,
                 quote: float,
                 notional: float,
                 period: str,
                 day_counter_convention: DayCounterConvention):
        super().__init__(ccy, start_date, end_date, quote)
        self.trade_date = trade_date
        self.notional = notional
        self.period = period
        self.day_counter = DayCounter(day_counter_convention)

    def __str__(self):
        return "{ name: future" + ",\n" \
                 "ccy: " + str(self.ccy) + ",\n" \
                 "trade_date: " + str(self.trade_date) + ",\n" \
                 "start_date: " + str(self.start_date) + ",\n" \
                 "end_date: " + str(self.maturity) + ",\n" \
                 "notional: " + str(self.notional) + ",\n" \
                 "period: " + str(self.period) + ",\n" \
                 "day_counter_convention: " + str(self.day_counter)