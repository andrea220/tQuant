from abc import ABC, abstractmethod

from datetime import date 
from ..markethandles.utils import Currency
from ..index.curverateindex import OvernightIndex 

from ..timehandles.utils import BusinessDayConvention, DayCounterConvention, TimeUnit
from ..timehandles.daycounter import DayCounter
from ..timehandles.schedule import ScheduleGenerator, ScheduleGeneratorAP
from ..timehandles.targetcalendar import TARGET, Calendar
from ..index.curverateindex import OvernightIndex
from ..instruments.deposit import Deposit#, DepositAP
from ..instruments.ois import Ois#, OisAP, OisTest
from ..instruments.forward import Fra
from ..instruments.swap import Swap
from ..index.curverateindex import IborIndex, OvernightIndex, Index

import re

from abc import ABC, abstractmethod


# class ProductBuilderAP(ABC):
#     def __init__(self, name: str, ccy: str, notional: float):
#         self.name = name
#         self.ccy = ccy
#         self.notional = notional

#     @abstractmethod
#     def build(self, trade_date, quote: float, term: str):
#         pass

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


# class DepositBuilderAP(ProductBuilderAP):
#     def __init__(self,
#                  name: str,
#                  ccy: str,
#                  start_delay: int,
#                  roll_convention: BusinessDayConvention,
#                  day_count_convention: DayCounterConvention,
#                  notional: float):
#         super().__init__(name, ccy, notional)
#         self.start_delay = start_delay
#         self.roll_convention = roll_convention
#         self.day_count_convention = day_count_convention

#     def build(self, trade_date: date, quote: float, term: str):

#         calendar = TARGET()

#         if term == 'O/N' or term == 'ON':
#             start_date = trade_date
#             maturity = calendar.advance(start_date, 1, TimeUnit.Days, self.roll_convention)
#         elif term == 'T/N' or term == 'TN':
#             start_date = calendar.advance(trade_date, 1, TimeUnit.Days, self.roll_convention)
#             maturity = calendar.advance(start_date, 1, TimeUnit.Days, self.roll_convention)
#         elif term == 'S/N' or term == 'SN':
#             start_date = calendar.advance(trade_date, 2, TimeUnit.Days, self.roll_convention)
#             maturity = calendar.advance(start_date, 1, TimeUnit.Days, self.roll_convention)
#         else:
#             tenor, time_unit = decode_term(term)
#             start_date = calendar.advance(trade_date, 2, TimeUnit.Days, self.roll_convention)
#             maturity = calendar.advance(trade_date, tenor, time_unit, self.roll_convention)

#         return DepositAP(self.ccy,
#                        quote,
#                        trade_date,
#                        start_date,
#                        maturity,
#                        self.notional,
#                        self.day_count_convention)

# class OisBuilderAP(ProductBuilderAP):
#     def __init__(self,
#                  name: str,
#                  ccy: str,
#                  start_delay: int,
#                  fixing_days: int,
#                  period_fix: str,
#                  period_flt: str,
#                  roll_convention: BusinessDayConvention,
#                  notional: float,
#                  day_count_convention_fix: DayCounterConvention,
#                  day_count_convention_flt: DayCounterConvention):
#         super().__init__(name, ccy, notional)
#         self.start_delay = start_delay
#         self.fixing_days = fixing_days
#         self.period_fix = period_fix
#         self.period_flt = period_flt
#         self.roll_convention = roll_convention
#         self.notional = notional
#         self.day_count_convention_fix = day_count_convention_fix
#         self.day_count_convention_flt = day_count_convention_flt

#     def build(self, trade_date: date, quote: float, term: str):
#         calendar = TARGET()

#         start_date = calendar.advance(
#             trade_date, self.start_delay, TimeUnit.Days, self.roll_convention)

#         period_maturity, time_unit = decode_term(term)
#         maturity = calendar.advance(start_date, period_maturity, time_unit, self.roll_convention)

#         period_fix, time_unit_fix = decode_term(self.period_fix)
#         period_float, time_unit_float = decode_term(self.period_flt)

#         schedule_generator = ScheduleGeneratorAP()

#         schedule_fix = schedule_generator.generate(
#             start_date, maturity, period_fix, time_unit_fix, self.roll_convention)

#         schedule_float = schedule_generator.generate(
#             start_date, maturity, period_float, time_unit_float, self.roll_convention)

#         start_fix = []
#         end_fix = []
#         pay_fix = []
#         for i in range(len(schedule_fix) - 1):
#             start_fix.append(schedule_fix[i])
#             end_fix.append(schedule_fix[i + 1])
#             pay_fix.append(calendar.adjust(schedule_fix[i + 1], self.roll_convention))

#         day_count_fix = DayCounter(self.day_count_convention_fix)
#         day_count_flt = DayCounter(self.day_count_convention_flt)

#         start_flt = []
#         end_flt = []
#         pay_flt = []
#         for i in range(len(schedule_float) - 1):
#             start_flt.append(schedule_float[i])
#             end_flt.append(schedule_float[i + 1])
#             pay_flt.append(calendar.adjust(schedule_float[i + 1], self.roll_convention))

#         fixing_dates = []
#         fixing_rates = []

#         return OisAP(self.ccy,
#                    start_date,
#                    maturity,
#                    start_fix,
#                    end_fix,
#                    pay_fix,
#                    start_flt,
#                    end_flt,
#                    pay_flt,
#                    fixing_dates,
#                    fixing_rates,
#                    quote,
#                    self.notional,
#                    day_count_fix,
#                    day_count_flt)

# class OisBuilderTest(ProductBuilderAP):
#     def __init__(self,
#                  name: str,
#                  ccy: str,
#                  start_delay: int,
#                  fixing_days: int,
#                  period_fix: str,
#                  period_flt: str,
#                  roll_convention: BusinessDayConvention,
#                  notional: float,
#                  day_count_convention_fix: DayCounterConvention,
#                  day_count_convention_flt: DayCounterConvention):
#         super().__init__(name, ccy, notional)
#         self.start_delay = start_delay
#         self.fixing_days = fixing_days
#         self.period_fix = period_fix
#         self.period_flt = period_flt
#         self.roll_convention = roll_convention
#         self.notional = notional
#         self.day_count_convention_fix = day_count_convention_fix
#         self.day_count_convention_flt = day_count_convention_flt

#     def build(self, trade_date: date, quote: float, term: str):
#         calendar = TARGET()
#         index = OvernightIndex("ESTR", calendar)

#         start_date = calendar.advance(
#             trade_date, self.start_delay, TimeUnit.Days, self.roll_convention)

#         period_maturity, time_unit = decode_term(term)
#         maturity = calendar.advance(start_date, period_maturity, time_unit, self.roll_convention)

#         period_fix, time_unit_fix = decode_term(self.period_fix)
#         period_float, time_unit_float = decode_term(self.period_flt)

#         schedule_generator = ScheduleGeneratorAP()

#         schedule_fix = schedule_generator.generate(
#             start_date, maturity, period_fix, time_unit_fix, self.roll_convention)

#         schedule_float = schedule_generator.generate(
#             start_date, maturity, period_float, time_unit_float, self.roll_convention)

#         start_fix = []
#         end_fix = []
#         pay_fix = []
#         for i in range(len(schedule_fix) - 1):
#             start_fix.append(schedule_fix[i])
#             end_fix.append(schedule_fix[i + 1])
#             pay_fix.append(calendar.adjust(schedule_fix[i + 1], self.roll_convention))

#         day_count_fix = DayCounter(self.day_count_convention_fix)
#         day_count_flt = DayCounter(self.day_count_convention_flt)

#         start_flt = []
#         end_flt = []
#         pay_flt = []
#         for i in range(len(schedule_float) - 1):
#             start_flt.append(schedule_float[i])
#             end_flt.append(schedule_float[i + 1])
#             pay_flt.append(calendar.adjust(schedule_float[i + 1], self.roll_convention))

#         fixing_dates = []
#         fixing_rates = []

#         return OisTest(self.ccy,
#                    start_date,
#                    maturity,
#                    start_fix,
#                    end_fix,
#                    pay_fix,
#                    start_flt,
#                    end_flt,
#                    pay_flt,
#                    fixing_dates,
#                    fixing_rates,
#                    quote,
#                    self.notional,
#                    day_count_fix,
#                    day_count_flt,
#                    index)







##### new 

class ProductGenerator(ABC):
    def __init__(self, name: str, ccy: str, notional: float):
        self.name = name
        self.ccy = ccy
        self.notional = notional

    @abstractmethod
    def build(self, trade_date, quote: float, term: str):
        pass

class DepositGenerator(ProductGenerator):
    def __init__(self,
                 ccy: Currency,
                 start_delay: int,
                 roll_convention: BusinessDayConvention,
                 day_count_convention: DayCounterConvention,
                 notional: float,
                 calendar: Calendar):
        super().__init__('depo', ccy, notional)
        self.start_delay = start_delay
        self.roll_convention = roll_convention
        self.day_count_convention = day_count_convention
        self.calendar = calendar

    def build(self, trade_date: date, quote: float, term: str):
        if term == 'O/N' or term == 'ON':
            start_date = trade_date
            maturity = self.calendar.advance(start_date, 1, TimeUnit.Days, self.roll_convention)
        elif term == 'T/N' or term == 'TN':
            start_date = self.calendar.advance(trade_date, 1, TimeUnit.Days, self.roll_convention)
            maturity = self.calendar.advance(start_date, 1, TimeUnit.Days, self.roll_convention)
        elif term == 'S/N' or term == 'SN':
            start_date = self.calendar.advance(trade_date, 2, TimeUnit.Days, self.roll_convention)
            maturity = self.calendar.advance(start_date, 1, TimeUnit.Days, self.roll_convention)
        else:
            tenor, time_unit = decode_term(term)
            start_date = self.calendar.advance(trade_date, 2, TimeUnit.Days, self.roll_convention)
            maturity = self.calendar.advance(trade_date, tenor, time_unit, self.roll_convention)

        return Deposit(self.ccy,
                       quote,
                       trade_date,
                       start_date,
                       maturity,
                       self.notional,
                       self.day_count_convention)



class OisGenerator(ProductGenerator):
    def __init__(self,
                 ccy: str,
                 start_delay: int,
                 fixing_days: int,
                 period_fix: str,
                 period_flt: str,
                 roll_convention: BusinessDayConvention,
                 notional: float,
                 day_count_convention_fix: DayCounterConvention,
                 day_count_convention_flt: DayCounterConvention,
                 calendar: Calendar,
                 index: Index):
        super().__init__('ois', ccy, notional)
        self.start_delay = start_delay
        self.fixing_days = fixing_days
        self.period_fix = period_fix
        self.period_flt = period_flt
        self.roll_convention = roll_convention
        self.notional = notional
        self.day_count_convention_fix = day_count_convention_fix
        self.day_count_convention_flt = day_count_convention_flt
        self.calendar = calendar
        self.index = index

    def build(self, trade_date: date, quote: float, term: str):

        start_date = self.calendar.advance(
            trade_date, self.start_delay, TimeUnit.Days, self.roll_convention)

        period_maturity, time_unit = decode_term(term)
        maturity = self.calendar.advance(start_date, period_maturity, time_unit, self.roll_convention)

        period_fix, time_unit_fix = decode_term(self.period_fix)
        period_float, time_unit_float = decode_term(self.period_flt)

        # schedule_generator = ScheduleGeneratorAP()
        schedule_generator = ScheduleGenerator(self.calendar, self.roll_convention)
        schedule_fix = schedule_generator.generate(start_date, maturity, period_fix, time_unit_fix)
        # schedule_fix = schedule_generator.generate(
        #     start_date, maturity, period_fix, time_unit_fix, self.roll_convention)

        schedule_float = schedule_generator.generate(start_date, maturity, period_float, time_unit_float)
        # schedule_float = schedule_generator.generate(
        #     start_date, maturity, period_float, time_unit_float, self.roll_convention)

        start_fix = []
        end_fix = []
        pay_fix = []
        for i in range(len(schedule_fix) - 1):
            start_fix.append(schedule_fix[i])
            end_fix.append(schedule_fix[i + 1])
            pay_fix.append(self.calendar.adjust(schedule_fix[i + 1], self.roll_convention))

        day_count_fix = DayCounter(self.day_count_convention_fix)
        day_count_flt = DayCounter(self.day_count_convention_flt)

        start_flt = []
        end_flt = []
        pay_flt = []
        for i in range(len(schedule_float) - 1):
            start_flt.append(schedule_float[i])
            end_flt.append(schedule_float[i + 1])
            pay_flt.append(self.calendar.adjust(schedule_float[i + 1], self.roll_convention))

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
                   day_count_flt,
                   self.index)
    

class FraGenerator(ProductGenerator):
    def __init__(self,
                 ccy: str,
                 start_delay: int,
                 fixing_days: int,
                 index_term: str,
                 roll_convention: BusinessDayConvention,
                 notional: float,
                 day_count_convention: DayCounterConvention,
                 calendar: Calendar,
                 index: Index):
        super().__init__('fra', ccy, notional)
        self.ccy = ccy
        self.start_delay = start_delay
        self.fixing_days = fixing_days
        self.index_term = index_term
        self.roll_convention = roll_convention
        self.notional = notional
        self.day_count_convention = day_count_convention
        # self.index_day_count_convention = index_day_count_convention
        self.calendar = calendar
        self.index = index

    def build(self, trade_date: date, quote: float, term: str):

        shifters = term.split('-')
        if len(shifters) != 2:
            raise ValueError("Wrong term specified: " + term)

        settlement_date = self.calendar.advance(
            trade_date, self.start_delay, TimeUnit.Days, self.roll_convention)

        start_period, start_time_unit = decode_term(shifters[0])
        start_date = self.calendar.advance(settlement_date, start_period, start_time_unit, self.roll_convention)

        end_period, end_time_unit = decode_term(shifters[1])
        end_date = self.calendar.advance(settlement_date, end_period, end_time_unit, self.roll_convention)

        day_counter = DayCounter(self.day_count_convention)
        # index_day_counter = DayCounter(self.index_day_count_convention)

        # fixing_date = calendar.advance(start_date, -self.fixing_days, TimeUnit.Days, self.roll_convention)
        # index_start_date = calendar.advance(
        #     fixing_date, self.start_delay, TimeUnit.Days, self.roll_convention)

        # index_period, index_time_unit = decode_term(self.index_term)
        # index_end_date = calendar.advance(
        #     index_start_date, index_period, index_time_unit, self.roll_convention)

        return Fra(self.ccy,
                   start_date,
                   end_date,
                   self.notional,
                   quote,
                   day_counter,
                   self.index)
    

class SwapGenerator(ProductGenerator):
    def __init__(self,
                 ccy: Currency,
                 start_delay: int,
                 period_fix: str,
                 period_flt: str,
                 roll_convention: BusinessDayConvention,
                 notional: float,
                 day_count_convention_fix: DayCounterConvention,
                 day_count_convention_flt: DayCounterConvention,
                 calendar: Calendar,
                 index: Index):
        super().__init__('swap', ccy, notional)
        self.start_delay = start_delay
        # self.fixing_days = fixing_days
        self.period_fix = period_fix
        self.period_flt = period_flt
        self.roll_convention = roll_convention
        self.notional = notional
        self.day_count_convention_fix = day_count_convention_fix
        self.day_count_convention_flt = day_count_convention_flt
        self.calendar = calendar
        self.index = index

    def build(self, trade_date: date, quote: float, term: str):

        start_date = self.calendar.advance(
            trade_date, self.start_delay, TimeUnit.Days, self.roll_convention)

        period_maturity, time_unit = decode_term(term)
        maturity = self.calendar.advance(start_date, period_maturity, time_unit, self.roll_convention)

        period_fix, time_unit_fix = decode_term(self.period_fix)
        period_float, time_unit_float = decode_term(self.period_flt)

        # schedule_generator = ScheduleGeneratorAP()
        schedule_generator = ScheduleGenerator(self.calendar, self.roll_convention)
        schedule_fix = schedule_generator.generate(start_date, maturity, period_fix, time_unit_fix)
        # schedule_fix = schedule_generator.generate(
        #     start_date, maturity, period_fix, time_unit_fix, self.roll_convention)

        schedule_float = schedule_generator.generate(start_date, maturity, period_float, time_unit_float)
        # schedule_float = schedule_generator.generate(
        #     start_date, maturity, period_float, time_unit_float, self.roll_convention)

        start_fix = []
        end_fix = []
        pay_fix = []
        for i in range(len(schedule_fix) - 1):
            start_fix.append(schedule_fix[i])
            end_fix.append(schedule_fix[i + 1])
            pay_fix.append(self.calendar.adjust(schedule_fix[i + 1], self.roll_convention))

        day_count_fix = DayCounter(self.day_count_convention_fix)
        day_count_flt = DayCounter(self.day_count_convention_flt)

        start_flt = []
        end_flt = []
        pay_flt = []
        for i in range(len(schedule_float) - 1):
            start_flt.append(schedule_float[i])
            end_flt.append(schedule_float[i + 1])
            pay_flt.append(self.calendar.adjust(schedule_float[i + 1], self.roll_convention))

        return Swap(self.ccy,
                   start_date,
                   maturity,
                   start_fix,
                   end_fix,
                   pay_fix,
                   start_flt,
                   end_flt,
                   pay_flt,
                   quote,
                   self.notional,
                   day_count_fix,
                   day_count_flt,
                   self.index)


