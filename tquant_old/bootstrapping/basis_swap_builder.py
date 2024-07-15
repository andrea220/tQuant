import datetime

from tquant import ProductBuilder, BusinessDayConvention, DayCounterConvention, IborIndex, TARGET, TimeUnit, \
    decode_term, ScheduleGenerator, DayCounter, string_to_enum
from tquant.bootstrapping.basis_swap import BasisSwap


class BasisSwapBuilder(ProductBuilder):
    def __init__(self,
                 name: str,
                 ccy: str,
                 start_delay: int,
                 fixing_days: int,
                 period_leg_1: str,
                 period_leg_2: str,
                 roll_convention: BusinessDayConvention,
                 notional: float,
                 day_count_leg_1: DayCounterConvention,
                 day_count_leg_2: DayCounterConvention,
                 index_day_count_leg_1: DayCounterConvention,
                 index_day_count_leg_2: DayCounterConvention):
        super().__init__(name, ccy, notional)
        self.start_delay = start_delay
        self.fixing_days = fixing_days
        self.period_leg_1 = period_leg_1
        self.period_leg_2 = period_leg_2
        self.roll_convention = roll_convention
        self.notional = notional
        self.day_count_leg_1 = DayCounter(day_count_leg_1)
        self.day_count_leg_2 = DayCounter(day_count_leg_2)
        self.index_day_count_leg_1 = DayCounter(index_day_count_leg_1)
        self.index_day_count_leg_2 = DayCounter(index_day_count_leg_2)

    def build(self, trade_date: datetime, quote: float, term: str):
        calendar = TARGET()

        start_date = calendar.advance(
            trade_date, self.start_delay, TimeUnit.Days, self.roll_convention)

        period, time_unit = decode_term(term)

        maturity = calendar.advance(
            start_date, period, time_unit, self.roll_convention)

        period_1, time_unit_1 = decode_term(self.period_leg_1)
        period_2, time_unit_2 = decode_term(self.period_leg_2)

        schedule_generator = ScheduleGenerator()

        schedule_1 = schedule_generator.generate(
            start_date, maturity, period_1, time_unit_1, self.roll_convention)

        schedule_2 = schedule_generator.generate(
            start_date, maturity, period_2, time_unit_2, self.roll_convention)

        start_dates_leg_1 = []
        end_dates_leg_1 = []
        pay_dates_leg_1 = []
        fixing_dates_leg_1 = []
        index_start_dates_leg_1 = []
        index_end_dates_leg_1 = []
        for i in range(len(schedule_1) - 1):
            start_dates_leg_1.append(schedule_1[i])
            end_dates_leg_1.append(schedule_1[i + 1])
            pay_date = calendar.adjust(end_dates_leg_1[i], self.roll_convention)
            pay_dates_leg_1.append(pay_date)
            fixing_date = calendar.advance(start_dates_leg_1[i], -self.fixing_days, TimeUnit.Days, self.roll_convention)
            fixing_dates_leg_1.append(fixing_date)
            index_start = calendar.advance(
                fixing_date, self.start_delay, TimeUnit.Days, self.roll_convention)
            index_end = calendar.advance(index_start, period_1, time_unit_1, self.roll_convention)
            index_start_dates_leg_1.append(index_start)
            index_end_dates_leg_1.append(index_end)

        start_dates_leg_2 = []
        end_dates_leg_2 = []
        pay_dates_leg_2 = []
        fixing_dates_leg_2 = []
        index_start_dates_leg_2 = []
        index_end_dates_leg_2 = []
        for i in range(len(schedule_2) - 1):
            start_dates_leg_2.append(schedule_2[i])
            end_dates_leg_2.append(schedule_2[i + 1])
            pay_date = calendar.adjust(end_dates_leg_2[i], self.roll_convention)
            pay_dates_leg_2.append(pay_date)
            fixing_date = calendar.advance(start_dates_leg_2[i], -self.fixing_days, TimeUnit.Days, self.roll_convention)
            fixing_dates_leg_2.append(fixing_date)
            index_start = calendar.advance(
                fixing_date, self.start_delay, TimeUnit.Days, self.roll_convention)
            index_end = calendar.advance(index_start, period_2, time_unit_2, self.roll_convention)
            index_start_dates_leg_2.append(index_start)
            index_end_dates_leg_2.append(index_end)

        fixing_rates_leg1 = [0] * len(pay_dates_leg_1)
        fixing_rates_leg2 = [0] * len(pay_dates_leg_2)

        return BasisSwap(self.ccy,
                         start_date,
                         maturity,
                         quote,
                         self.notional,
                         self.period_leg_1,
                         self.period_leg_2,
                         fixing_dates_leg_1,
                         start_dates_leg_1,
                         end_dates_leg_1,
                         pay_dates_leg_1,
                         fixing_rates_leg1,
                         index_start_dates_leg_1,
                         index_end_dates_leg_1,
                         fixing_dates_leg_2,
                         start_dates_leg_2,
                         end_dates_leg_2,
                         pay_dates_leg_2,
                         fixing_rates_leg2,
                         index_start_dates_leg_2,
                         index_end_dates_leg_2,
                         self.day_count_leg_1,
                         self.day_count_leg_2,
                         self.index_day_count_leg_1,
                         self.index_day_count_leg_2)

    @classmethod
    def from_json(cls, item):
        return BasisSwapBuilder(item["name"],
                                item["ccy"],
                                item["start_delay"],
                                item["fixing_days"],
                                item["period_leg_1"],
                                item["period_leg_2"],
                                string_to_enum(BusinessDayConvention, item["roll_convention"]),
                                item["notional"],
                                string_to_enum(DayCounterConvention, item["day_count_leg_1"]),
                                string_to_enum(DayCounterConvention, item["day_count_leg_2"]),
                                string_to_enum(DayCounterConvention, item["index_day_count_leg_1"]),
                                string_to_enum(DayCounterConvention, item["index_day_count_leg_2"]))


if __name__ == "__main__":
    builder = BasisSwapBuilder("basis_swap",
                               "EUR",
                               2,
                               2,
                               "3M",
                               "6M",
                               BusinessDayConvention.ModifiedFollowing,
                               1.0,
                               DayCounterConvention.Actual360,
                               DayCounterConvention.Actual360,
                               DayCounterConvention.Actual360,
                               DayCounterConvention.Actual360)

    basis_swap = builder.build(datetime.datetime.now(), 0.01, "2Y")

    print(basis_swap)


