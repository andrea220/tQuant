from datetime import date

from .tqcalendar import Calendar
from .targetcalendar import TARGET
from .utils import TimeUnit, BusinessDayConvention


class ScheduleGenerator:
    def __init__(
        self, calendar: Calendar, business_day_convention: BusinessDayConvention
    ):
        self.calendar = calendar
        self.business_day_convention = business_day_convention

    def generate(self, start: date, end: date, tenor: int, time_unit: TimeUnit):
        schedule = []
        current_date = start
        schedule.append(current_date)
        while True:
            current_date = self.calendar.advance(
                current_date, tenor, time_unit, self.business_day_convention
            )
            if current_date < end:
                schedule.append(current_date)
            else:
                schedule.append(end)
                break
        return schedule
