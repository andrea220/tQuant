from .utils import DayCounterConvention
from datetime import date


class DayCounter:

    def __init__(self,
                day_counter_convention: DayCounterConvention,
                include_last_day: bool = False
                ):
        self.day_counter_convention = day_counter_convention
        self.include_last_day = 1 if include_last_day else 0


    def year_days(self,
                  year: int):
        if year % 4 == 0:
            return 366
        else:
            return 365
    
    def year_fraction(self,
                  d1:date,
                  d2:date):
        if self.day_counter_convention == DayCounterConvention.Actual360:
            return self.day_count(d1,d2) / 360.0
        elif self.day_counter_convention == DayCounterConvention.Actual365:
            return self.day_count(d1,d2) / 365.0
        elif self.day_counter_convention == DayCounterConvention.Thirty360:
            return self.day_count(d1,d2) / 360.0
        elif self.day_counter_convention == DayCounterConvention.Thirty360E:
            return self.day_count(d1,d2) / 360.0
        elif self.day_counter_convention == DayCounterConvention.ActualActual:
            
            if d1.year == d2.year:
                return self.day_count(d1,d2)/self.year_days(d1.year)
            else:
                yearfraction = (date(d1.year,12,31) - d1).days/self.year_days(d1.year)
                date_year = d1.year
                while date_year < d2.year:
                    yearfraction = yearfraction + 1
                    date_year = date_year + 1
                yearfraction = (d2 - date(d2.year,1,1))/self.year_days(d2.year)
                return yearfraction

    def day_count(self,
                  d1:date,
                  d2:date):
        if d1 == d2:
            return 0.0
        if self.day_counter_convention == DayCounterConvention.Actual360:
            return (d2-d1).days + self.include_last_day
        elif self.day_counter_convention == DayCounterConvention.Actual365 or self.day_counter_convention == DayCounterConvention.ActualActual:
            return (d2-d1).days
        elif self.day_counter_convention == DayCounterConvention.Thirty360:
            dd1 = d1.day
            dd2 = d2.day
            if dd1 == 31:
                dd1 = 30
            if dd2 == 31 and dd1 == 30:
                dd2 = 30
            return 360.0*(d2.year - d1.year) + 30.0*(d2.month - d1.month) + dd2 - dd1
        elif self.day_counter_convention == DayCounterConvention.Thirty360E:
            dd1 = d1.day
            dd2 = d2.day
            if dd1 == 31:
                dd1 = 30
            if dd2 == 31:
                dd2 = 30
            return 360.0*(d2.year - d1.year) + 30.0*(d2.month - d1.month) + dd2 - dd1

    def __str__(self) -> str:
        return self.day_counter_convention.name




