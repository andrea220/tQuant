from abc import ABC, abstractmethod
from typing import Optional
from ..utilities.utils import DayCounterConvention
from ..utilities.daycounter import DayCounter
from .curve import AbstractCurve
from .tqcalendar import Calendar

class AbstractHelper(ABC):
    
    def __init__(self,
                 rate: float,
                 calendar: Calendar, 
                 settlement_days: Optional[int] = None,
                 day_counter: Optional[DayCounter] = None):
      self.rate = rate
      self.calendar = calendar
      self.settlement_days = settlement_days if settlement_days is not None else 0 # In giorni, partenza ritardata dell'opzione rispetto a data di riferimento
      self.day_counter = day_counter if day_counter is not None else DayCounter(DayCounterConvention.Actual360)
    
    @abstractmethod
    def pillar(self):
        pass
    
    @abstractmethod
    def bootstrap(self, curve: AbstractCurve):
        pass
    