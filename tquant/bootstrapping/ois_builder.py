import datetime

from tquant import BusinessDayConvention, DayCounterConvention, TARGET, TimeUnit, decode_term, ScheduleGenerator, \
    DayCounter, ProductBuilder, string_to_enum
from tquant.bootstrapping.ois import Ois


class OisBuilder(ProductBuilder):
    def __init__(self,
                 name: str,
                 ccy: str,
                 start_delay: int,
                 fixing_days: int,
                 period_fix: str,
                 period_flt: str,
                 roll_convention: BusinessDayConvention,
                 notional: float,
                 day_count_convention_fix: DayCounterConvention,
                 day_count_convention_flt: DayCounterConvention):
        super().__init__(name, ccy, notional)
        self.start_delay = start_delay
        self.fixing_days = fixing_days
        self.period_fix = period_fix
        self.period_flt = period_flt
        self.roll_convention = roll_convention
        self.notional = notional
        self.day_count_convention_fix = day_count_convention_fix
        self.day_count_convention_flt = day_count_convention_flt

    def build(self, trade_date: datetime, quote: float, term: str):
        calendar = TARGET()

        start_date = calendar.advance(
            trade_date, self.start_delay, TimeUnit.Days, self.roll_convention)

        period_maturity, time_unit = decode_term(term)
        maturity = calendar.advance(start_date, period_maturity, time_unit, self.roll_convention)

        period_fix, time_unit_fix = decode_term(self.period_fix)
        period_float, time_unit_float = decode_term(self.period_flt)

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

        day_count_fix = DayCounter(self.day_count_convention_fix)
        day_count_flt = DayCounter(self.day_count_convention_flt)

        start_flt = []
        end_flt = []
        pay_flt = []
        for i in range(len(schedule_float) - 1):
            start_flt.append(schedule_float[i])
            end_flt.append(schedule_float[i + 1])
            pay_flt.append(calendar.adjust(schedule_float[i + 1], self.roll_convention))

        fixing_dates = []
        fixing_rates = []

        return Ois(self.ccy,
                   start_date,
                   maturity,
                   start_fix,
                   end_fix,
                   pay_fix,
                   start_flt,
                   end_flt,
                   pay_flt,
                   fixing_dates,
                   fixing_rates,
                   quote,
                   self.notional,
                   day_count_fix,
                   day_count_flt)

    @classmethod
    def from_json(cls, item):
        return OisBuilder(item["name"],
                          item["ccy"],
                          item["start_delay"],
                          item["fixing_days"],
                          item["period_fix"],
                          item["period_float"],
                          string_to_enum(BusinessDayConvention, item["roll_convention"]),
                          item["notional"],
                          string_to_enum(DayCounterConvention, item["day_count_convention_fix"]),
                          string_to_enum(DayCounterConvention, item["day_count_convention_flt"]))


if __name__ == "__main__":
    builder = OisBuilder("ois",
                         "EUR",
                         2,
                         2,
                         "1Y",
                         "3M",
                         BusinessDayConvention.ModifiedFollowing,
                         1.0,
                         DayCounterConvention.Actual360,
                         DayCounterConvention.Actual360)

    ois = builder.build(datetime.datetime.now(), 0.01, "2Y")

    print(ois)
