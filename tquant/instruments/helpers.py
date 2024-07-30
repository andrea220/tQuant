from abc import ABC, abstractmethod

from datetime import date 
from ..markethandles.utils import Currency
from ..index.curverateindex import OvernightIndex 

from ..timehandles.utils import BusinessDayConvention, DayCounterConvention, TimeUnit
from ..timehandles.daycounter import DayCounter
from ..timehandles.schedule import ScheduleGenerator
from ..timehandles.targetcalendar import TARGET

from ..instruments.deposit import Deposit
from ..instruments.ois import Ois

import re


def string_to_enum(enum_class, enum_str):
    try:
        return enum_class.__members__[enum_str]
    except KeyError:
        raise ValueError(f"'{enum_str}' is not a valid member of {enum_class.__name__}")
    
def decode_term(term: str):
    match = re.match(r'(\d+)([A-Za-z]+)', term)
    if match:
        period = int(match.group(1))
        unit = match.group(2)
    else:
        raise ValueError("Wrong term: " + term)
    if unit == 'd' or unit == 'D':
        time_unit = TimeUnit.Days
    elif unit == 'w' or unit == 'W':
        time_unit = TimeUnit.Weeks
    elif unit == 'm' or unit == 'M':
        time_unit = TimeUnit.Months
    elif unit == 'y' or unit == 'Y':
        time_unit = TimeUnit.Years
    else:
        raise ValueError
    return period, time_unit


class Helper(ABC):
    def __init__(self,
                 name: str,
                 currency: Currency):
        self.name = name
        self.currency = currency

    @abstractmethod
    def build(self):
        pass


class DepositHelper(Helper):
    def __init__(self,
                 currency: Currency,
                 start_delay: int,
                 roll_convention: BusinessDayConvention,
                 day_count_convention: DayCounterConvention,
                 notional: float):
        super().__init__(name='depo', currency=currency)
        self.notional = notional
        self.start_delay = start_delay
        self.roll_convention = roll_convention
        self.day_count_convention = day_count_convention

    def build(self,
              trade_date: date,
              quote: float,
              term: str):
        if self.currency == Currency.EUR:
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

            return Deposit(self.currency,
                        trade_date,
                        start_date,
                        maturity,
                        self.notional,
                        self.day_count_convention,
                        quote)

    @classmethod
    def from_json(cls, item):
        return DepositHelper(
            item["name"],
            item["ccy"],
            item["start_delay"],
            string_to_enum(BusinessDayConvention, item["roll_convention"]),
            string_to_enum(DayCounterConvention, item["day_count_convention"]),
            item["notional"])

    def __str__(self):
        return "{" + "ccy: " + str(self.currency) + ",\n" \
            + "start_delay: " + str(self.start_delay) + ",\n" \
            + "roll_convention: " + str(self.roll_convention) + ",\n" \
            + "day_count_convention: " + str(self.day_count_convention) + ",\n" \
            + "notional: " + str(self.notional) + "}"

class OisHelper(Helper):
    def __init__(self,
                 currency: Currency,
                 start_delay: int,
                 fixing_days: int,
                 period_fix: int,
                 period_flt: int,
                 roll_convention: BusinessDayConvention,
                 notional: float,
                 day_count_convention_fix: DayCounterConvention,
                 day_count_convention_flt: DayCounterConvention):
        super().__init__(name = 'ois', currency=currency)
        self.start_delay = start_delay
        self.fixing_days = fixing_days
        self.period_fix = period_fix
        self.period_flt = period_flt
        self.roll_convention = roll_convention
        self.notional = notional
        self.day_count_convention_fix = day_count_convention_fix
        self.day_count_convention_flt = day_count_convention_flt

    def build(self, trade_date: date, quote: float, term: str):
        if self.currency == Currency.EUR:
            calendar = TARGET()
            index = OvernightIndex('ESTR', calendar)
        else:
            raise TypeError("Not present for currency")

        start_date = calendar.advance(
            trade_date, self.start_delay, TimeUnit.Days, self.roll_convention)

        period_maturity, time_unit = decode_term(term)
        maturity = calendar.advance(start_date, period_maturity, time_unit, self.roll_convention)

        period_fix, time_unit_fix = decode_term(self.period_fix)
        period_float, time_unit_float = decode_term(self.period_flt)

        schedule_generator = ScheduleGenerator(calendar, self.roll_convention)

        schedule_fix = schedule_generator.generate(
            start_date, maturity, period_fix, time_unit_fix)

        schedule_float = schedule_generator.generate(
            start_date, maturity, period_float, time_unit_float)

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

        return Ois(schedule_float, 
                            schedule_fix,
                            self.notional,
                            index,
                            quote,
                            day_count_fix,
                            day_count_flt,
                            Currency.EUR)

class OisHelperAP(Helper):
    def __init__(self,
                 currency: Currency,
                 start_delay: int,
                 fixing_days: int,
                 period_fix: str,
                 period_flt: str,
                 roll_convention: BusinessDayConvention,
                 notional: float,
                 day_count_convention_fix: DayCounterConvention,
                 day_count_convention_flt: DayCounterConvention):
        super().__init__(name = 'ois', currency=currency)
        self.start_delay = start_delay
        self.fixing_days = fixing_days
        self.period_fix = period_fix
        self.period_flt = period_flt
        self.roll_convention = roll_convention
        self.notional = notional
        self.day_count_convention_fix = day_count_convention_fix
        self.day_count_convention_flt = day_count_convention_flt

    def build(self, trade_date: date, quote: float, term: str):
        calendar = TARGET()

        start_date = calendar.advance(
            trade_date, self.start_delay, TimeUnit.Days, self.roll_convention)

        period_maturity, time_unit = decode_term(term)
        maturity = calendar.advance(start_date, period_maturity, time_unit, self.roll_convention)

        period_fix, time_unit_fix = decode_term(self.period_fix)
        period_float, time_unit_float = decode_term(self.period_flt)

        schedule_generator = ScheduleGenerator()

        schedule_fix = schedule_generator.generate(
            start_date, maturity, period_fix, time_unit_fix, calendar, self.roll_convention)

        schedule_float = schedule_generator.generate(
            start_date, maturity, period_float, time_unit_float, calendar, self.roll_convention)

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

        return Ois(self.currency,
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
        return OisHelper(item["name"],
                          item["ccy"],
                          item["start_delay"],
                          item["fixing_days"],
                          item["period_fix"],
                          item["period_float"],
                          string_to_enum(BusinessDayConvention, item["roll_convention"]),
                          item["notional"],
                          string_to_enum(DayCounterConvention, item["day_count_convention_fix"]),
                          string_to_enum(DayCounterConvention, item["day_count_convention_flt"]))

