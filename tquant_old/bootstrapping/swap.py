import datetime

from tquant import DayCounter, Product


class Swap(Product):
    def __init__(self,
                 ccy: str,
                 start: datetime,
                 end: datetime,
                 period_fix: str,
                 period_flt: str,
                 start_dates_fix: list[datetime],
                 end_dates_fix: list[datetime],
                 pay_dates_fix: list[datetime],
                 start_dates_float: list[datetime],
                 end_dates_float: list[datetime],
                 pay_dates_float: list[datetime],
                 fixing_dates: list[datetime],
                 fixing_rates: list[float],
                 notional: float,
                 quote: float,
                 margin: float,
                 day_counter_fix: DayCounter,
                 day_counter_float: DayCounter,
                 day_counter_index: DayCounter,
                 index_start_dates: list[datetime],
                 index_end_dates: list[datetime]):
        super().__init__(ccy, start, end, quote)
        self.period_fix = period_fix
        self.period_flt = period_flt
        self.start_dates_fix = start_dates_fix
        self.end_dates_fix = end_dates_fix
        self.pay_dates_fix = pay_dates_fix
        self.start_dates_float = start_dates_float
        self.end_dates_float = end_dates_float
        self.pay_dates_float = pay_dates_float
        self.fixing_dates = fixing_dates
        self.fixing_rates = fixing_rates
        self.notional = notional
        self.quote = quote
        self.margin = margin
        self.day_counter_fix = day_counter_fix
        self.day_counter_float = day_counter_float
        self.day_counter_index = day_counter_index
        self.index_start_dates = index_start_dates
        self.index_end_dates = index_end_dates

    def __str__(self) -> str:
        return "{ name: swap, \n" + \
                "ccy: " + str(self.ccy) + ",\n" \
                "start: " + str(self.start_date) + ",\n" \
                "end: " + str(self.maturity) + ",\n" \
                "start_dates_fix: " + str(self.start_dates_fix) + ",\n" \
                "end_dates_fix: " + str(self.end_dates_fix) + ",\n" \
                "pay_dates_fix: " + str(self.pay_dates_fix) + ",\n" \
                "start_dates_float: " + str(self.start_dates_float) + ",\n" \
                "end_dates_float: " + str(self.end_dates_float) + ",\n" \
                "pay_dates_float: " + str(self.pay_dates_float) + ",\n" \
                "fixing_dates: " + str(self.fixing_dates) + ",\n" \
                "fixing_rates: " + str(self.fixing_rates) + ",\n" \
                "notional: " + str(self.notional) + ",\n" \
                "quote: " + str(self.quote) + ",\n" \
                "margin: " + str(self.margin) + ",\n" \
                "day_counter_fix: " + str(self.day_counter_fix) + ",\n" \
                "day_counter_float: " + str(self.day_counter_float) + ",\n" \
                "day_counter_index: " + str(self.day_counter_index) + ",\n" \
                "index_start_dates: " + str(self.index_start_dates) + ",\n" \
                "index_end_dates: " + str(self.index_end_dates) + " }"




