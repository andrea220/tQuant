import datetime
import re

from tquant import ProductBuilder, TARGET, decode_term, BusinessDayConvention, DayCounterConvention, string_to_enum
from tquant.bootstrapping.future import Future


def map_month_label_to_index(label: str):
    month_labels = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    for i in range(12):
        if month_labels[i] == label:
            return i + 1
    raise ValueError("Wrong month label")


def nth_weekday(year, month, weekday, n):
    first_day = datetime.datetime(year, month, 1)
    first_weekday = first_day.weekday()
    days_to_first_weekday = (weekday - first_weekday + 7) % 7
    first_occurrence = first_day + datetime.timedelta(days=days_to_first_weekday)
    nth_occurrence = first_occurrence + datetime.timedelta(days=(n - 1) * 7)
    if nth_occurrence.month == month:
        return nth_occurrence
    else:
        raise ValueError("Wrong weekday")


class FutureBuilder(ProductBuilder):
    def __init__(self,
                 name: str,
                 ccy: str,
                 notional: float,
                 period: str,
                 day_count_convention: DayCounterConvention,
                 roll_convention: BusinessDayConvention):
        super().__init__(name, ccy, notional)
        self.period = period
        self.day_count_convention = day_count_convention
        self.roll_convention = roll_convention

    def build(self, trade_date: datetime, quote: float, term: str):
        match = re.match("(\\S{3})(\\s)(\\d{2})", term)
        if match:
            month = match.group(1)
            year = match.group(3)
        else:
            raise ValueError("Wrong term: " + term)
        month = map_month_label_to_index(month)
        year = 2000 + int(year)
        start_date = nth_weekday(year, month, 2, 3)
        # print(month)
        # print(year)
        # print(start_date)
        calendar = TARGET()
        period, time_unit = decode_term(self.period)
        end_date = calendar.advance(start_date, period, time_unit, self.roll_convention)
        # print(end_date)

        return Future(self.ccy,
                      trade_date,
                      start_date,
                      end_date,
                      quote,
                      self.notional,
                      self.period,
                      self.day_count_convention)

    @classmethod
    def from_json(cls, item):
        return FutureBuilder(item["name"],
                             item["ccy"],
                             item["notional"],
                             item["period"],
                             string_to_enum(DayCounterConvention, item["day_count_convention"]),
                             string_to_enum(BusinessDayConvention, item["roll_convention"]))


if __name__ == "__main__":
    builder = FutureBuilder("fut", "EUR", 1.0, "6M")
    builder.build(datetime.datetime.now(), 0.01, "FEB 24")
