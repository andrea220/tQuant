import datetime

from tquant import BusinessDayConvention, DayCounterConvention, Index, TARGET, TimeUnit, DayCounter, ProductBuilder, \
    string_to_enum
from tquant.bootstrapping.fra import Fra
from tquant.bootstrapping.util import decode_term


class FraBuilder(ProductBuilder):
    def __init__(self,
                 name: str,
                 ccy: str,
                 start_delay: int,
                 fixing_days: int,
                 index_term: str,
                 roll_convention: BusinessDayConvention,
                 notional: float,
                 day_count_convention: DayCounterConvention,
                 index_day_count_convention: DayCounterConvention):
        super().__init__(name, ccy, notional)
        self.ccy = ccy
        self.start_delay = start_delay
        self.fixing_days = fixing_days
        self.index_term = index_term
        self.roll_convention = roll_convention
        self.notional = notional
        self.day_count_convention = day_count_convention
        self.index_day_count_convention = index_day_count_convention

    def build(self, trade_date: datetime, quote: float, term: str):
        # fra = 3M-6M
        shifters = term.split('-')
        if len(shifters) != 2:
            raise ValueError("Wrong term specified: " + term)

        calendar = TARGET()

        settlement_date = calendar.advance(
            trade_date, self.start_delay, TimeUnit.Days, self.roll_convention)

        start_period, start_time_unit = decode_term(shifters[0])
        start_date = calendar.advance(settlement_date, start_period, start_time_unit, self.roll_convention)

        end_period, end_time_unit = decode_term(shifters[1])
        end_date = calendar.advance(settlement_date, end_period, end_time_unit, self.roll_convention)

        day_counter = DayCounter(self.day_count_convention)
        index_day_counter = DayCounter(self.index_day_count_convention)

        fixing_date = calendar.advance(start_date, -self.fixing_days, TimeUnit.Days, self.roll_convention)
        index_start_date = calendar.advance(
            fixing_date, self.start_delay, TimeUnit.Days, self.roll_convention)

        index_period, index_time_unit = decode_term(self.index_term)
        index_end_date = calendar.advance(
            index_start_date, index_period, index_time_unit, self.roll_convention)

        return Fra(self.ccy,
                   start_date,
                   end_date,
                   self.notional,
                   quote,
                   index_period,
                   index_start_date,
                   index_end_date,
                   day_counter,
                   index_day_counter)

    @classmethod
    def from_json(cls, item):
        return FraBuilder(
            item["name"],
            item["ccy"],
            item["start_delay"],
            item["fixing_days"],
            item["index_term"],
            string_to_enum(BusinessDayConvention, item["roll_convention"]),
            item["notional"],
            string_to_enum(DayCounterConvention, item["day_count_convention"]),
            string_to_enum(DayCounterConvention, item["index_day_count_convention"]))


if __name__ == "__main__":
    builder = FraBuilder(
        "fra",
        "EUR",
        2,
        2,
        "3M",
        BusinessDayConvention.ModifiedFollowing,
        1.0, DayCounterConvention.Actual365,
        DayCounterConvention.Actual360)
    fra = builder.build(datetime.datetime.now(), 0.01, "3M-6M")

    print(fra)
