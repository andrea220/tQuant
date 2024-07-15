from abc import ABC, abstractmethod
from datetime import date 
from utilities_old.time import Settings



class Index(ABC):
    ''' 
    Abstract index object
    '''
    @abstractmethod
    def __init__(self,
                 name: str,
                 fixing_calendar,
                 fixing_time_series: dict 
                 ) -> None:
        self._name = name
        self._fixing_calendar = fixing_calendar
        self._fixing_time_series = fixing_time_series

        
    @abstractmethod
    def forecast_fixing(self)-> float:
        print("nooo")
        return  

    @property
    def name(self) -> str:
        ''' 
        Name of the index
        '''
        return self._name
    
    @property
    def fixing_time_series(self)-> dict:
        return self._fixing_time_series  
    
    @fixing_time_series.setter
    def fixing_time_series(self, input_fixings)-> None:
        self._fixing_time_series = input_fixings
 
    @property
    def fixing_calendar(self) -> str:
        ''' 
        Fixing calendar of the index
        '''
        return self._fixing_calendar

    def is_valid_fixing_date(self, date: date)-> bool:
        return True 
    
    def add_fixing(self, date: date, value: float)-> None:
        fixing_point = {
            self.name: {
                date: value
            }
        }
        if self.fixing_time_series is None:
            self.fixing_time_series = fixing_point
        else:
            #overwrite it
            self.fixing_time_series[self.name][date] = value 

    def past_fixing(self, date: date)-> bool:
        past_fixings = self.fixing_time_series
        if self.is_valid_fixing_date(date):
            return past_fixings[self.name][date]
        else:
            raise ValueError("Not a valid fixing date!")

    def fixing(self, date)-> float:
        if not self.is_valid_fixing_date:
            raise ValueError("Not a valid date")
        
        if date >= Settings.evaluation_date:
            # time series empty, try to forecast the fixing 
            return self.forecast_fixing(date)
        
        elif date < Settings.evaluation_date:
            if self.fixing_time_series is None:
                raise ValueError(f"Missing {self.name} fixing for {date}") 
            
            if self.name in list(self.fixing_time_series.keys()):
                if date in list(self.fixing_time_series[self.name].keys()):
                    # return historical fixing for index/date
                    return self.past_fixing(date)
                else:
                    raise ValueError(f"{self.name} fixing time series is not complete, missing {date}")
            else:
                raise ValueError(f"Missing {self.name} fixing for {date}")
                

class Index_dev(ABC):
    ''' 
    Abstract index object
    '''
    @abstractmethod
    def __init__(self,
                 name: str,
                 fixing_calendar,
                 fixing_time_series: dict 
                 ) -> None:
        self._name = name
        self._fixing_calendar = fixing_calendar
        self._fixing_time_series = fixing_time_series

        
    @abstractmethod
    def forecast_fixing(self)-> float:
        return  

    @property
    def name(self) -> str:
        ''' 
        Name of the index
        '''
        return self._name
    
    @property
    def fixing_time_series(self)-> dict:
        return self._fixing_time_series  
    
    @fixing_time_series.setter
    def fixing_time_series(self, input_fixings)-> None:
        self._fixing_time_series = input_fixings
 
    @property
    def fixing_calendar(self) -> str:
        ''' 
        Fixing calendar of the index
        '''
        return self._fixing_calendar

    def is_valid_fixing_date(self, date: date)-> bool:
        return True 
    
    def add_fixing(self, date: date, value: float)-> None:
        fixing_point = {
            self.name: {
                date: value
            }
        }
        if self.fixing_time_series is None:
            self.fixing_time_series = fixing_point
        else:
            #overwrite it
            self.fixing_time_series[self.name][date] = value 

    def past_fixing(self, date: date)-> bool:
        past_fixings = self.fixing_time_series
        if self.is_valid_fixing_date(date):
            return past_fixings[self.name][date]
        else:
            raise ValueError("Not a valid fixing date!")

    def fixing(self,
               date,
               term_structure
               )-> float:
        if not self.is_valid_fixing_date:
            raise ValueError("Not a valid date")
        
        if date >= Settings.evaluation_date:
            # time series empty, try to forecast the fixing 
            return self.forecast_fixing(date, term_structure)
        
        elif date < Settings.evaluation_date:
            if self.fixing_time_series is None:
                raise ValueError(f"Missing {self.name} fixing for {date}") 
            
            if self.name in list(self.fixing_time_series.keys()):
                if date in list(self.fixing_time_series[self.name].keys()):
                    # return historical fixing for index/date
                    return self.past_fixing(date)
                else:
                    raise ValueError(f"{self.name} fixing time series is not complete, missing {date}")
            else:
                raise ValueError(f"Missing {self.name} fixing for {date}")
          

