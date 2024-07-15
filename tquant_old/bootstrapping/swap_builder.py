import datetime
import math

from tquant import BusinessDayConvention, TARGET, TimeUnit, DayCounter, ProductBuilder
from tquant.bootstrapping.schedule_generator import ScheduleGenerator
from tquant.bootstrapping.swap import Swap
from tquant.bootstrapping.util import decode_term
from tquant.utilities.utils import DayCounterConvention, string_to_enum


class SwapBuilder(ProductBuilder):
    def __init__(self,
                 name: str,
                 ccy: str,
                 start_delay: int,
                 period_fixed: str,
                 period_float: str,
                 roll_convention: BusinessDayConvention,
                 notional: float,
                 margin: float,
                 day_count_fix: DayCounterConvention,
                 day_count_float: DayCounterConvention,
                 day_count_index: DayCounterConvention,
                 fixing_days: int):
        super().__init__(name, ccy, notional)
        self.ccy = ccy
        self.start_delay = start_delay
        self.period_fixed = period_fixed
        self.period_float = period_float
        self.roll_convention = roll_convention
        self.notional = notional
        self.margin = margin
        self.day_count_fix = day_count_fix
        self.day_count_float = day_count_float
        self.day_count_index = day_count_index
        self.fixing_days = fixing_days

    def build(self, trade_date: datetime, quote: float, term: str):

        calendar = TARGET()

        start_date = calendar.advance(
            trade_date, self.start_delay, TimeUnit.Days, self.roll_convention)

        period, time_unit = decode_term(term)

        maturity = calendar.advance(
            start_date, period, time_unit, self.roll_convention)

        period_fix, time_unit_fix = decode_term(self.period_fixed)
        period_float, time_unit_float = decode_term(self.period_float)

        schedule_generator = ScheduleGenerator()

        schedule_fix = schedule_generator.generate(
            start_date, maturity, period_fix, time_unit_fix, self.roll_convention)

        schedule_float = schedule_generator.generate(
            start_date, maturity, period_float, time_unit_float, self.roll_convention)

        start_fix = []
        end_fix = []
        pay_fix = []
        for i in range(len(schedule_fix) - 1):
            start_fix.append(schedule_fix[i])
            end_fix.append(schedule_fix[i + 1])
            pay_fix.append(calendar.adjust(schedule_fix[i + 1], self.roll_convention))

        day_count_fix = DayCounter(self.day_count_fix)

        start_flt = []
        end_flt = []
        pay_flt = []
        fixing_dates = []
        index_start_dates = []
        index_end_dates = []
        for i in range(len(schedule_float) - 1):
            start_flt.append(schedule_float[i])
            end_flt.append(schedule_float[i + 1])
            pay_flt.append(calendar.adjust(schedule_float[i + 1], self.roll_convention))
            fixing_dates.append(calendar.advance(start_flt[i], -self.fixing_days, TimeUnit.Days, self.roll_convention))

            index_start_date = calendar.advance(fixing_dates[i], self.fixing_days, TimeUnit.Days, self.roll_convention)
            index_end_date = calendar.advance(index_start_date, period_float, time_unit_float, self.roll_convention)

            index_start_dates.append(index_start_date)
            index_end_dates.append(index_end_date)

        day_count_float = DayCounter(self.day_count_float)
        day_count_index = DayCounter(self.day_count_index)

        fixing_rates = [0] * len(pay_flt)

        return Swap(self.ccy,
                    start_date,
                    maturity,
                    period_fix,
                    period_float,
                    start_fix,
                    end_fix,
                    pay_fix,
                    start_flt,
                    end_flt,
                    pay_flt,
                    fixing_dates,
                    fixing_rates,
                    self.notional,
                    quote,
                    self.margin,
                    day_count_fix,
                    day_count_float,
                    day_count_index,
                    index_start_dates,
                    index_end_dates)

    @classmethod
    def from_json(cls, item):
        return SwapBuilder(item["name"],
                           item["ccy"],
                           item["start_delay"],
                           item["period_fix"],
                           item["period_float"],
                           string_to_enum(BusinessDayConvention, item["roll_convention"]),
                           item["notional"],
                           item["margin"],
                           string_to_enum(DayCounterConvention, item["day_count_fix"]),
                           string_to_enum(DayCounterConvention, item["day_count_float"]),
                           string_to_enum(DayCounterConvention, item["day_count_index"]),
                           item["fixing_days"])


if __name__ == "__main__":
    builder = SwapBuilder(
        "swap", "EUR", 2, "1Y", "6M", BusinessDayConvention.ModifiedFollowing, 1.0, 0.0,
        DayCounterConvention.Thirty360, DayCounterConvention.Actual360, DayCounterConvention.Actual360, 2)
    swap = builder.build(datetime.datetime.now(), 0.01, "2Y")

    print(swap)
