from abc import ABC, abstractmethod
from datetime import date

from ..timehandles.tqcalendar import Calendar
from ..timehandles.utils import Settings


class Index(ABC):
    """
    An abstract base class representing a financial index.

    This class serves as a blueprint for specific index implementations, providing
    core attributes and methods for handling index fixings and related data.

    """

    @abstractmethod
    def __init__(
        self, name: str, fixing_calendar: Calendar, fixing_time_series: dict
    ) -> None:
        """
        Initializes an Index instance with the specified attributes.

        Parameters:
        -----------
        name: str
            The name of the index.
        fixing_calendar: Calendar
            The calendar used for determining fixing dates.
        fixing_time_series: dict
            A dictionary containing the fixing time series data.
        """
        self._name = name
        self._fixing_calendar = fixing_calendar
        self._fixing_time_series = fixing_time_series

    @property
    def name(self) -> str:
        """
        Gets the name of the index.

        Returns:
        -----------
        str: The name of the index.
        """
        return self._name

    @property
    def fixing_time_series(self) -> dict:
        """
        Gets the fixing time series data.

        Returns:
        -----------
        dict: A dictionary containing the fixing time series data.
        """
        return self._fixing_time_series

    @fixing_time_series.setter
    def fixing_time_series(self, input_fixings: dict) -> None:
        """
        Sets the fixing time series data.

        Parameters:
        -----------
        input_fixings: dict
            A dictionary containing the fixing time series data to be set.
        """
        self._fixing_time_series = input_fixings

    @property
    def fixing_calendar(self) -> Calendar:
        """
        Gets the fixing calendar of the index.

        Returns:
        -----------
        Calendar: The fixing calendar of the index.
        """
        return self._fixing_calendar

    def is_valid_fixing_date(self, fixing_date: date) -> bool:
        """
        Checks if the given date is a valid fixing date based on the index's calendar.

        Parameters:
        -----------
        date: date
            The date to be checked.

        Returns:
        -----------
        bool: True if the date is a valid fixing date, False otherwise.

        Note:
        -----------
        This method is currently a placeholder and always returns True.
        """
        return self._fixing_calendar.is_business_day(fixing_date)

    def add_fixing(self, fixing_date: date, value: float) -> None:
        """
        Adds a fixing value for a specific date to the fixing time series data.

        Parameters:
        -----------
        date: date
            The date of the fixing.
        value: float
            The fixing value.
        """
        fixing_point = {self.name: {fixing_date: value}}
        if self.fixing_time_series is None:
            # create the dict
            self.fixing_time_series = fixing_point
        else:
            # write into it
            self.fixing_time_series[self.name][fixing_date] = value

    def past_fixing(self, fixing_date: date) -> bool:
        """
        Retrieves the past fixing value for a specific date.

        Parameters:
        -----------
        date: date
            The date for which the past fixing is requested.

        Returns:
        -----------
        float: The past fixing value for the given date.

        Raises:
        -----------
        ValueError: If the given date is not a valid fixing date or the fixing is missing.
        """
        past_fixings = self.fixing_time_series
        if self.is_valid_fixing_date(fixing_date):
            return past_fixings[self.name][fixing_date]
        raise ValueError("Not a valid fixing date!")

    def fixing(self, fixing_date: date) -> float:
        """
        Retrieves the fixing value for a specific date.

        Parameters:
        -----------
        date: date
            The date for which the fixing is requested.

        Returns:
        -----------
        float: The fixing value for the given date.

        Raises:
        -----------
        ValueError: If the given date is not a valid date, if fixings are missing, or
                    if the requested date is a future date.
        """
        if not self.is_valid_fixing_date:
            raise ValueError("Not a valid date")

        if fixing_date > Settings.evaluation_date:
            raise ValueError("Fixing are only available for historical dates.")

        elif fixing_date <= Settings.evaluation_date:
            if self.fixing_time_series is None:
                raise ValueError(f"Missing {self.name} fixing for {fixing_date}")

            if self.name in list(self.fixing_time_series.keys()):
                if fixing_date in list(self.fixing_time_series[self.name].keys()):
                    # return historical fixing for index/date
                    return self.past_fixing(fixing_date)
                raise ValueError(
                    f"{self.name} fixing time series is not complete, missing {fixing_date}"
                )
            raise ValueError(f"Missing {self.name} fixing for {fixing_date}")
