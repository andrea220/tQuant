
from datetime import date
from ..timehandles.utils import BusinessDayConvention, TimeUnit
from .index import Index
from ..timehandles.tqcalendar import Calendar
from ..markethandles.utils import Currency
from ..timehandles.daycounter import DayCounter, DayCounterConvention

class OvernightIndex(Index):
    """
    Represents an Overnight index, typically used for short-term interest rates.

    This class inherits from the abstract `Index` class and includes additional 
    attributes and methods specific to overnight indices, such as the number of fixing 
    days and currency.
    """
    def __init__(self,
                 fixing_calendar: Calendar,
                 currency: Currency = None,
                 fixing_days: int = None,
                 time_series: dict = None) -> None:
        """
        Initializes an OvernightIndex instance with the specified attributes.

        Parameters:
        -----------
        fixing_calendar: Calendar
            The calendar used for determining fixing dates.
        currency: Currency, optional
            The currency associated with the index (default is None).
        fixing_days: int, optional
            The number of days before the fixing date (default is None).
        time_series: dict, optional
            A dictionary containing the fixing time series data (default is None).
        """
        super().__init__(currency.value + ":ON",
                         fixing_calendar,
                         time_series)      
        self._fixing_days = fixing_days
        self._currency = currency
    
    @property
    def fixing_days(self) -> int:
        """
        Gets the number of fixing days for the index.

        Returns:
        -----------
        int: The number of fixing days if set; otherwise, 0.
        """
        if self._fixing_days == None:
            return 0
        else:
            return self._fixing_days
    
    def fixing_maturity(self, fixing_date: date) -> date:
        """
        Calculates the fixing maturity date based on the fixing date and index conventions.

        The maturity date is typically the date on which the funds are returned for an 
        overnight index.

        Parameters:
        -----------
        fixing_date: date
            The fixing date.

        Returns:
        -----------
        date: The maturity date for the fixing.
        """
        return self.fixing_calendar.advance(fixing_date,
                                            1, 
                                            TimeUnit.Days,
                                            BusinessDayConvention.ModifiedFollowing 
                                            )

    def fixing_date(self, value_date: date) -> date:
        """
        Calculates the fixing date based on the value date and the number of fixing days.

        Parameters:
        -----------
        value_date: date
            The value date from which the fixing date is calculated.

        Returns:
        -----------
        date: The calculated fixing date.
        """
        return self.fixing_calendar.advance(value_date, -self.fixing_days, TimeUnit.Days, BusinessDayConvention.Preceding)
   

class IborIndex(Index):
    """
    Represents an Interbank Offered Rate (IBOR) index.

    This class models an IBOR index, which is a key interest rate at which banks 
    lend to and borrow from one another in the interbank market. The class inherits 
    from the abstract `Index` class and includes additional attributes such as tenor, 
    time unit, and currency.
    """
    def __init__(self,
                 fixing_calendar: Calendar,
                 tenor: int,
                 time_unit: TimeUnit,
                 currency: Currency,
                 fixing_days: int = None,
                 time_series: dict = None) -> None:
        """
        Initializes an IborIndex instance with the specified attributes.

        Parameters:
        -----------
        fixing_calendar: Calendar
            The calendar used for determining fixing dates.
        tenor: int
            The tenor (duration) of the IBOR rate.
        time_unit: TimeUnit
            The unit of time for the tenor (e.g., days, months).
        currency: Currency
            The currency associated with the index.
        fixing_days: int, optional
            The number of days before the fixing date (default is None).
        time_series: dict, optional
            A dictionary containing the fixing time series data (default is None).
        """
        super().__init__(currency.value + ":" + str(tenor) +  time_unit.value[0],
                         fixing_calendar,
                         time_series)      
        self._fixing_days = fixing_days
        self._tenor = tenor 
        self._time_unit = time_unit
        self._currency = currency  
        self._daycounter = DayCounter(DayCounterConvention.Actual360)      
    
    @property
    def fixing_days(self) -> int:
        """
        Gets the number of fixing days for the index.

        Returns:
        -----------
        int: The number of fixing days if set; otherwise, 0.
        """
        if self._fixing_days == None:
            return 0
        else:
            return self._fixing_days

    @property
    def daycounter(self):
        return self._daycounter

    def fixing_maturity(self, fixing_date: date) -> date:
        """
        Calculates the fixing maturity date based on the fixing date and index conventions.

        The maturity date is typically the date on which the IBOR rate applies.

        Parameters:
        -----------
        fixing_date: date
            The fixing date.

        Returns:
        -----------
        date: The maturity date for the fixing.
        """
        return self.fixing_calendar.advance(fixing_date,
                                            self._tenor, 
                                            self._time_unit,
                                            BusinessDayConvention.ModifiedFollowing 
                                            )

    def fixing_date(self, value_date: date) -> date:
        """
        Calculates the fixing date based on the value date and the number of fixing days.

        Parameters:
        -----------
        value_date: date
            The value date from which the fixing date is calculated.

        Returns:
        -----------
        date: The calculated fixing date.
        """
        return self.fixing_calendar.advance(value_date, -self.fixing_days, TimeUnit.Days, BusinessDayConvention.Preceding)
   



