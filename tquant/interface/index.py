from abc import ABC, abstractmethod
from datetime import date

from calendar import Calendar
from ..utilities.utils import Settings



class Index(ABC):
    """
    Abstract class representing an index.

    Attributes:
    -----------
        _name: str
            The name of the index.
        _fixing_calendar Calendar
            The calendar used for determining fixing dates.
        _fixing_time_series: dict
            A dictionary containing fixing time series data.

    """
    @abstractmethod
    def __init__(self,
                 name: str,
                 fixing_calendar: Calendar,
                 fixing_time_series: dict 
                 ) -> None:
        self._name = name
        self._fixing_calendar = fixing_calendar
        self._fixing_time_series = fixing_time_series

    @property
    def name(self) -> str:
        """
        Get the name of the index.

        Returns:
        -----------
            str: The name of the index.

        """
        return self._name
    
    @property
    def fixing_time_series(self)-> dict:
        """
        Get the fixing time series data.

        Returns:
        -----------
            dict: A dictionary containing fixing time series data.

        """
        return self._fixing_time_series  
    
    @fixing_time_series.setter
    def fixing_time_series(self, input_fixings)-> None:
        """
        Set the fixing time series data.

        Parameters:
        -----------
            input_fixings: The fixing time series data to be set.
        """
        self._fixing_time_series = input_fixings
 
    @property
    def fixing_calendar(self) -> Calendar:
        """
        Get the fixing calendar of the index.

        Returns:
        -----------
            str: The fixing calendar of the index.

        """
        return self._fixing_calendar

    def is_valid_fixing_date(self, date: date) -> bool:
        """
        Check if the given date is a valid fixing date.

        Parameters:
        -----------
            date: date
                The date to be checked.

        Returns:
        -----------
            bool: True if the date is a valid fixing date, False otherwise.

        """
        # TODO da implementare la funzione in modo che valuti se la data in input sia valida dato un calendario
        return True 
    
    def add_fixing(self, date: date, value: float)-> None:
        """
        Add a fixing value for a specific date to the fixing time series data.

        Parameters:
        -----------
            date: date
                The date of the fixing.
            value: float
                The fixing value.
        """
        fixing_point = {
            self.name: {
                date: value
            }
        }
        if self.fixing_time_series is None:
            # create the dict
            self.fixing_time_series = fixing_point
        else:
            # write into it
            self.fixing_time_series[self.name][date] = value 

    def past_fixing(self, date: date)-> bool:
        """
        Get the past fixing value for a specific date.

        Parameters:
        -----------
            date: date
                The date for which past fixing is requested.

        Returns:
        -----------
            bool: The past fixing value for the given date.

        Raises:
            ValueError: If the given date is not a valid fixing date.

        """
        past_fixings = self.fixing_time_series
        if self.is_valid_fixing_date(date):
            return past_fixings[self.name][date]
        else:
            raise ValueError("Not a valid fixing date!")

    def fixing(self,
               date
               )-> float:
        """
        Get the fixing value for a specific date.

        Parameters:
        -----------
            date: date
                The date for which fixing is requested.

        Returns:
        -----------
            float: The fixing value for the given date.

        Raises:
        -----------
            ValueError: If the given date is not a valid date.

        """
        if not self.is_valid_fixing_date:
            raise ValueError("Not a valid date")
        
        if date >= Settings.evaluation_date:
            # time series empty, try to forecast the fixing 
            raise ValueError("Fixing are only available for historical dates.")
            # return self.forecast_fixing(date, term_structure)
        
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
          

