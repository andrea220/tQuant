import datetime

from tquant import Product, DayCounter


class BasisSwap(Product):
    def __init__(self,
                 ccy: str,
                 start_date: datetime,
                 maturity: datetime,
                 quote: float,
                 notional: float,
                 period1: str,
                 period2: str,
                 fixing_dates_leg1: list[datetime],
                 start_dates_leg1: list[datetime],
                 end_dates_leg1: list[datetime],
                 pay_dates_leg1: list[datetime],
                 fixing_rates_leg1: list[float],
                 index_start_dates_1: list[datetime],
                 index_end_dates_1: list[datetime],
                 fixing_dates_leg2: list[datetime],
                 start_dates_leg2: list[datetime],
                 end_dates_leg2: list[datetime],
                 pay_dates_leg2: list[datetime],
                 fixing_rates_leg2: list[float],
                 index_start_dates_2: list[datetime],
                 index_end_dates_2: list[datetime],
                 day_counter_1: DayCounter,
                 day_counter_2: DayCounter,
                 index_day_counter_1: DayCounter,
                 index_day_counter_2: DayCounter):
        super().__init__(ccy, start_date, maturity, quote)
        self.notional = notional
        self.period1 = period1
        self.period2 = period2
        self.fixing_dates_leg1 = fixing_dates_leg1
        self.start_dates_leg1 = start_dates_leg1
        self.end_dates_leg1 = end_dates_leg1
        self.pay_dates_leg1 = pay_dates_leg1
        self.fixing_rates_leg1 = fixing_rates_leg1
        self.index_start_dates_1 = index_start_dates_1
        self.index_end_dates_1 = index_end_dates_1
        self.fixing_dates_leg2 = fixing_dates_leg2
        self.start_dates_leg2 = start_dates_leg2
        self.end_dates_leg2 = end_dates_leg2
        self.pay_dates_leg2 = pay_dates_leg2
        self.fixing_rates_leg2 = fixing_rates_leg2
        self.index_start_dates_2 = index_start_dates_2
        self.index_end_dates_2 = index_end_dates_2
        self.day_counter_1 = day_counter_1
        self.day_counter_2 = day_counter_2
        self.index_day_counter_1 = index_day_counter_1
        self.index_day_counter_2 = index_day_counter_2

    def __str__(self) -> str:
        return "{ type: basis swap,\n " \
               "ccy: " + str(self.ccy) + ",\n" \
                "start: " + str(self.start_date) + ",\n" \
                "end: " + str(self.maturity) + ",\n" \
                "notional: " + str(self.notional) + ",\n" \
                "fixing_dates_leg1: " + str(self.fixing_dates_leg1) + ",\n" \
                "start_dates_leg1: " + str(self.start_dates_leg1) + ",\n" \
                "end_dates_leg1: " + str(self.end_dates_leg1) + ",\n" \
                "pay_dates_leg1: " + str(self.pay_dates_leg1) + ",\n" \
                "fixing_rates_leg1: " + str(self.fixing_rates_leg1) + ",\n" \
                "index_start_dates_1: " + str(self.index_start_dates_1) + ",\n" \
                "index_end_dates_1: " + str(self.index_end_dates_1) + ",\n" \
                "fixing_dates_leg2: " + str(self.fixing_dates_leg2) + ",\n" \
                "start_dates_leg2: " + str(self.start_dates_leg2) + ",\n" \
                "end_dates_leg2: " + str(self.end_dates_leg2) + ",\n" \
                "pay_dates_leg2: " + str(self.pay_dates_leg2) + ",\n" \
                "fixing_rates_leg2: " + str(self.fixing_rates_leg2) + ",\n" \
                "index_start_dates_2: " + str(self.index_start_dates_2) + ",\n" \
                "index_end_dates_2: " + str(self.index_end_dates_2) + ",\n" \
                "day_counter_1: " + str(self.day_counter_1) + ",\n" \
                "day_counter_2: " + str(self.day_counter_2) + ",\n" \
                "index_day_counter_1: " + str(self.index_day_counter_1) + ",\n" \
                "index_day_counter_2: " + str(self.index_day_counter_2) + " }"






