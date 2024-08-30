from datetime import date 
from .daycounter import DayCounterConvention, DayCounter

class DateGrid:
    def __init__(self,
                 dates: list[date],
                 daycounter_convention: DayCounterConvention) -> None:
        self._dates = dates 
        self._daycounter_convention = daycounter_convention
        self._daycounter = DayCounter(daycounter_convention)
        self._times = [self._daycounter.year_fraction(dates[0], d) for d in dates]
    
    @property
    def dates(self):
        return self._dates

    @property
    def daycounter_convention(self):
        return self._daycounter_convention

    @property
    def daycounter(self):
        return self._daycounter
    
    @property
    def times(self):
        return self._times