from datetime import date

from .product import Product
from ..timehandles.daycounter import DayCounter


class Ois(Product):
    def __init__(self,
                 ccy: str,
                 start_date: date,
                 end_date: date,
                 start_dates_fix: list[date],
                 end_dates_fix: list[date],
                 pay_dates_fix: list[date],
                 start_dates_flt: list[date],
                 end_dates_flt: list[date],
                 pay_dates_flt: list[date],
                 fixing_dates: list[date],
                 fixing_rates: list[float],
                 quote: float,
                 notional: float,
                 day_counter_fix: DayCounter,
                 day_counter_flt: DayCounter):
        super().__init__(ccy, start_date, end_date)
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
        self.quote = quote

    def __str__(self) -> str:
        return "{name: ois, \n" \
                "ccy: " + str(self.currency) + ",\n" \
                "start: " + str(self.start_date) + ",\n" \
                "end: " + str(self.end_date) + ",\n" \
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
