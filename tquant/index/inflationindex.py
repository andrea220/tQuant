
from datetime import date
from ..utilities.utils import BusinessDayConvention, TimeUnit, Settings, Frequency
from ..interface.index import Index
from ..interface.tqcalendar import Calendar

#TODO parlare dell'implementazione come in Quantlib e di cosa ci serve.



class InflationIndex(Index):
    """
    Represents an Inflation index.

    Attributes:
    -----------
        name: str
            The name of the index.
        observation_lag: int
        observation_lag_period: TimeUnit
            The obseration lag linked to the index
        fixing_calendar: Calendar
            The calendar used for determining fixing dates.
        fixing_days: int, optional
            The number of days for fixing. Defaults to None.
        time_series: dict, optional
            A dictionary containing time series data. Defaults to None.

    Note:
    -----------
        Inherits from Index abstract class.

    """
    def __init__(self,
                 name: str,
                 tenor: int,
                 time_unit: TimeUnit,
                 fixing_calendar: Calendar,
                 frequency: Frequency = Frequency.Monthly,
                 fixing_days: int = 0,
                 time_series: dict = None,
                 revised: bool = False) -> None:
        super().__init__(name,
                         fixing_calendar,
                         time_series)      
        self._fixing_days = fixing_days
        self._tenor = tenor
        self._time_unit = time_unit 
        self._revised = revised  
        self._frequency = frequency    
    
    @property
    def fixing_days(self) -> int:
        """
        Get the number of fixing days for the index.

        Returns:
        -----------
            int: The number of fixing days if set, otherwise 0.

        """
        if self._fixing_days == None:
            return 0
        else:
            return self._fixing_day
        
    def fixing_maturity(self, fixing_date: date) -> date:
        """
        Calculate the fixing maturity date based on the fixing date and index conventions.

        Parameters:
        -----------
            fixing_date: date
                The fixing date.

        Returns:
        -----------
            date: The maturity date for the fixing helper.

        """
        return self.fixing_calendar.advance(fixing_date,
                                            self._tenor, 
                                            self._time_unit,
                                            BusinessDayConvention.ModifiedFollowing 
                                            )

    def fixing_date(self, value_date: date) -> date:
        return self.fixing_calendar.advance(value_date, self.fixing_days, TimeUnit.Days, BusinessDayConvention.ModifiedFollowing)