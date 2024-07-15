import datetime

from tquant import Product, DayCounter


class Ois(Product):
    def __init__(self,
                 ccy: str,
                 start_date: datetime,
                 end_date: datetime,
                 start_dates_fix: list[datetime],
                 end_dates_fix: list[datetime],
                 pay_dates_fix: list[datetime],
                 start_dates_flt: list[datetime],
                 end_dates_flt: list[datetime],
                 pay_dates_flt: list[datetime],
                 fixing_dates: list[datetime],
                 fixing_rates: list[float],
                 quote: float,
                 notional: float,
                 day_counter_fix: DayCounter,
                 day_counter_flt: DayCounter):
        super().__init__(ccy, start_date, end_date, quote)
        self.start_dates_fix = start_dates_fix
        self.end_dates_fix = end_dates_fix
        self.pay_dates_fix = pay_dates_fix
        self.start_dates_flt = start_dates_flt
        self.end_dates_flt = end_dates_flt
        self.pay_dates_flt = pay_dates_flt
        self.fixing_dates = fixing_dates
        self.fixing_rates = fixing_rates
        self.notional = notional
        self.day_counter_fix = day_counter_fix
        self.day_counter_flt = day_counter_flt

    def __str__(self) -> str:
        return "{name: ois, \n" \
                "ccy: " + str(self.ccy) + ",\n" \
                "start: " + str(self.start_date) + ",\n" \
                "end: " + str(self.maturity) + ",\n" \
                "start_dates_fix: " + str(self.start_dates_fix) + ",\n" \
                "end_dates_fix: " + str(self.end_dates_fix) + ",\n" \
                "pay_dates_fix: " + str(self.pay_dates_fix) + ",\n" \
                "start_dates_flt: " + str(self.start_dates_flt) + ",\n" \
                "end_dates_flt: " + str(self.end_dates_flt) + ",\n" \
                "pay_dates_flt: " + str(self.pay_dates_flt) + ",\n" \
                "fixing_dates: " + str(self.fixing_dates) + ",\n" \
                "fixing_rates: " + str(self.fixing_rates) + ",\n" \
                "notional: " + str(self.notional) + ",\n" \
                "quote: " + str(self.quote) + ",\n" \
                "day_counter_fix " + str(self.day_counter_fix) + ",\n" \
                "day_counter_flt " + str(self.day_counter_flt) + "}"
