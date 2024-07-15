import datetime

from .product_builder import *
from tquant.bootstrapping.util import decode_term
from tquant.utilities.utils import *
from tquant.utilities.targetcalendar import *
from .deposit import *


class DepositBuilder(ProductBuilder):
    def __init__(self,
                 name: str,
                 ccy: str,
                 start_delay: int,
                 roll_convention: BusinessDayConvention,
                 day_count_convention: DayCounterConvention,
                 notional: float):
        super().__init__(name, ccy, notional)
        self.start_delay = start_delay
        self.roll_convention = roll_convention
        self.day_count_convention = day_count_convention

    def build(self, trade_date: datetime, quote: float, term: str):

        calendar = TARGET()

        if term == 'O/N' or term == 'ON':
            start_date = trade_date
            maturity = calendar.advance(start_date, 1, TimeUnit.Days, self.roll_convention)
        elif term == 'T/N' or term == 'TN':
            start_date = calendar.advance(trade_date, 1, TimeUnit.Days, self.roll_convention)
            maturity = calendar.advance(start_date, 1, TimeUnit.Days, self.roll_convention)
        elif term == 'S/N' or term == 'SN':
            start_date = calendar.advance(trade_date, 2, TimeUnit.Days, self.roll_convention)
            maturity = calendar.advance(start_date, 1, TimeUnit.Days, self.roll_convention)
        else:
            tenor, time_unit = decode_term(term)
            start_date = calendar.advance(trade_date, 2, TimeUnit.Days, self.roll_convention)
            maturity = calendar.advance(trade_date, tenor, time_unit, self.roll_convention)

        return Deposit(self.ccy,
                       quote,
                       trade_date,
                       start_date,
                       maturity,
                       self.notional,
                       self.day_count_convention)

    @classmethod
    def from_json(cls, item):
        return DepositBuilder(
            item["name"],
            item["ccy"],
            item["start_delay"],
            string_to_enum(BusinessDayConvention, item["roll_convention"]),
            string_to_enum(DayCounterConvention, item["day_count_convention"]),
            item["notional"])

    def __str__(self):
        return "{" + "name: " + str(self.name) + ",\n" \
            + "ccy: " + str(self.ccy) + ",\n" \
            + "start_delay: " + str(self.start_delay) + ",\n" \
            + "roll_convention: " + str(self.roll_convention) + ",\n" \
            + "day_count_convention: " + str(self.day_count_convention) + ",\n" \
            + "notional: " + str(self.notional) + "}"


if __name__ == "__main__":
    builder = DepositBuilder(
        "depo", "EUR", 2, BusinessDayConvention.ModifiedFollowing, DayCounterConvention.Actual360, 1.0)

    depoON = builder.build(datetime(2024, 4, 15), 0.01, "O/N")
    depoTN = builder.build(datetime(2024, 4, 15), 0.01, "T/N")
    depo3M = builder.build(datetime(2024, 4, 15), 0.01, "3M")

    print(depoON)
    print(depoTN)
    print(depo3M)
