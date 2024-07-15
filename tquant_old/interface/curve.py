from abc import ABC, abstractmethod
from typing import Optional, Union, List
from datetime import date
from .tqcalendar import Calendar
from ..utilities.daycounter import DayCounter, DayCounterConvention
from ..utilities.interpolation import InterpolationType
from ..utilities.utils import CompoundingType
from ..utilities.utils import TimeUnit, BusinessDayConvention

class AbstractCurve(ABC):

    def __init__(self,
                 calendar: Calendar,
                 day_counter: Optional[DayCounter] = None,
                 pillars: Optional[Union[date, List[date]]] = None,
                 dfs: Optional[Union[float, List[float]]] = None,
                 T1: Optional[Union[int, float, List[int], List[float]]] = None):
       
        self.calendar = calendar
        self.day_counter = day_counter if day_counter is not None else DayCounter(DayCounterConvention.Actual360)
        self.pillars = pillars if pillars is not None else []
        self.dfs = dfs if dfs is not None else []
        
        #T1 indicano i settlement days, nel caso i fattori di sconto non siano riferiti ad oggi
        if T1 is not None:
            self.T1 =  T1
        else:
            if pillars is not None:
                self.T1 = [0]*len(pillars)
            else:
                self.T1 = []
    
    @abstractmethod
    def discount(self,
                 date: date,
                 interpolation: Optional[InterpolationType] = None):
        pass
    
    @abstractmethod
    def zero_rate(self,
                  date: date,
                  interpolation: Optional[InterpolationType] = None,
                  compounding: Optional[CompoundingType] = None):
        pass

    @abstractmethod
    def forward_rate(self,
                    start_date: date,
                    end_date: date,
                    interpolation: Optional[InterpolationType] = None):
        pass

    @abstractmethod
    def annuity(self,
                start_date: date,
                end_date: date,
                schedule: int,
                time_unit: TimeUnit,
                convention: BusinessDayConvention,
                fixingDays: Optional[int] = None,
                interpolation: Optional[InterpolationType] = None):
        pass

    def insert_pillar(self,
                      pillar: date,
                      rate: float,
                      T1: Optional[Union[int,float]] = None):

        if T1 is None:
            T1 = 0
        
        self.pillars.append(pillar)
        self.dfs.append(rate)
        self.T1.append(T1)  