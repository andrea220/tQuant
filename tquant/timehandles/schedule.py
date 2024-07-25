from datetime import date

from .tqcalendar import Calendar
from .utils import TimeUnit, BusinessDayConvention


class ScheduleGenerator: #TODO review schedule generator
    def generate(self,
                 start: date,
                 end: date,
                 tenor: int,
                 time_unit: TimeUnit,
                 calendar: Calendar,
                 business_day_convention: BusinessDayConvention): #TODO da valutare se togliere la bd e recuperarla altrove(calendario?)
        schedule = []
        current_date = start
        schedule.append(current_date)
        while True:
            current_date = calendar.advance(current_date, tenor, time_unit, business_day_convention)
            if current_date < end:
                schedule.append(current_date)
            else:
                schedule.append(end)
                break
        return schedule

