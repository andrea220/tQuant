import datetime

from ..timehandles.utils import BusinessDayConvention, TimeUnit, Frequency
from .index import Index
from ..timehandles.tqcalendar import Calendar


class InflationIndex(Index):
    """
    Represents an Inflation index.

    Attributes:
        name (str): The name of the index.
        observation_lag (int): The observation lag linked to the index.
        observation_lag_period (TimeUnit): The time unit associated with the observation lag.
        fixing_calendar (Calendar): The calendar used for determining fixing dates.
        fixing_days (int, optional): The number of days for fixing. Defaults to 0.
        time_series (dict, optional): A dictionary containing time series data. Defaults to None.
        revised (bool): Indicates whether the index is revised.
        frequency (Frequency): The frequency at which the index is observed. Defaults to Monthly.

    Note:
        Inherits from Index abstract class.
    """

    def __init__(
        self,
        name: str,
        tenor: int,
        time_unit: TimeUnit,
        fixing_calendar: Calendar,
        frequency: Frequency = Frequency.Monthly,
        fixing_days: int = 0,
        time_series: dict = None,
        revised: bool = False,
    ) -> None:
        """
        Initializes the InflationIndex.

        Args:
            name (str): The name of the index.
            tenor (int): The time period (in units) for which the index is defined.
            time_unit (TimeUnit): The time unit for the tenor (e.g., days, months).
            fixing_calendar (Calendar): The calendar used for determining fixing dates.
            frequency (Frequency, optional): The frequency of the index. Defaults to Frequency.Monthly.
            fixing_days (int, optional): The number of fixing days. Defaults to 0.
            time_series (dict, optional): Time series data for the index. Defaults to None.
            revised (bool, optional): Whether the index is revised. Defaults to False.
        """
        super().__init__(name, fixing_calendar, time_series)
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
            int: The number of fixing days if set, otherwise 0.
        """
        if self._fixing_days is None:
            return 0
        return self._fixing_days

    def fixing_maturity(self, fixing_date: datetime.date) -> datetime.date:
        """
        Calculate the fixing maturity date based on the fixing date and index conventions.

        Args:
            fixing_date (datetime.date): The fixing date.

        Returns:
            datetime.date: The maturity date for the fixing helper, calculated by advancing the fixing date by the tenor and time unit, based on the business day convention.
        """
        return self.fixing_calendar.advance(
            fixing_date,
            self._tenor,
            self._time_unit,
            BusinessDayConvention.ModifiedFollowing,
        )

    def fixing_date(self, value_date: datetime.date) -> datetime.date:
        """
        Calculate the fixing date for the given value date.

        Args:
            value_date (datetime.date): The date for which the fixing date is required.

        Returns:
            datetime.date: The fixing date, calculated by advancing the value date backwards by the number of fixing days, based on the business day convention.
        """
        return self.fixing_calendar.advance(
            value_date,
            -self.fixing_days,
            TimeUnit.Days,
            BusinessDayConvention.Preceding,
        )
