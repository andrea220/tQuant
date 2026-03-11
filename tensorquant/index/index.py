from abc import ABC, abstractmethod
import datetime
import warnings

from ..timehandles.tqcalendar import Calendar
from ..timehandles.utils import Settings


class Index(ABC):
    """
    An abstract base class representing a financial index.

    This class defines a generic framework for financial indices, providing core attributes
    such as the index name, fixing calendar, and time series data. Specific indices (e.g.,
    overnight or IBOR indices) should inherit from this class and implement additional logic
    as required.
    """

    @abstractmethod
    def __init__(
        self, name: str, fixing_calendar: Calendar, fixing_time_series: dict = None
    ) -> None:
        """
        Initializes an Index instance with the specified attributes.

        Args:
            name (str): The name of the index.
            fixing_calendar (Calendar): The calendar used to determine valid fixing dates.
            fixing_time_series (dict): A dictionary containing the time series of index fixings.
        """
        self._name = name
        self._fixing_calendar = fixing_calendar
        self._fixing_time_series = fixing_time_series

    @property
    def name(self) -> str:
        """
        Gets the name of the index.

        Returns:
            str: The name of the index.
        """
        return self._name

    @property
    def fixing_time_series(self) -> dict:
        """
        Gets the fixing time series data.

        Returns:
            dict: A dictionary containing the time series of fixings.
        """
        return self._fixing_time_series

    @fixing_time_series.setter
    def fixing_time_series(self, input_fixings: dict) -> None:
        """
        Sets the fixing time series data.

        Args:
            input_fixings (dict): A dictionary containing the new time series of fixings.
                                  Expected shape: {date: value}
        """
        if input_fixings is None:
            self._fixing_time_series = None
            return

        # Basic type check on the container
        if not isinstance(input_fixings, dict):
            raise TypeError(
                f"input_fixings must be a dict mapping datetime.date to float, got {type(input_fixings)}"
            )

        # Early duplicate check on the input keys (fixing_date)
        input_dates = list(input_fixings.keys())
        if len(set(input_dates)) != len(input_dates):
            raise ValueError("Duplicate fixing_date detected in input_fixings")

        # Type and duplicate checks
        # Existing dates in the current time series (if any)
        existing_dates = set()
        if (
            self._fixing_time_series is not None
            and self.name in self._fixing_time_series
        ):
            existing_dates = set(self._fixing_time_series[self.name].keys())

        seen_dates = set()
        cleaned_fixings = {}
        for fixing_date, value in input_fixings.items():
            if not isinstance(fixing_date, datetime.date):
                raise TypeError(
                    f"Fixing date keys must be datetime.date, got {type(fixing_date)}"
                )

            # Duplicate inside the new input
            if fixing_date in seen_dates:
                raise ValueError(f"Duplicate fixing date in input: {fixing_date}")
            seen_dates.add(fixing_date)

            # Duplicate versus existing time series
            if fixing_date in existing_dates:
                raise ValueError(
                    f"Duplicate fixing date versus existing time series: {fixing_date}"
                )

            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Fixing values must be float (or int), got {type(value)}"
                )

            cleaned_fixings[fixing_date] = float(value)

        # Build the internal structure directly as {index_name: {date: value}}
        self._fixing_time_series = {self.name: cleaned_fixings}

    @property
    def fixing_calendar(self) -> Calendar:
        """
        Gets the fixing calendar for the index.

        Returns:
            Calendar: The calendar used to determine valid fixing dates.
        """
        return self._fixing_calendar

    def is_valid_fixing_date(self, fixing_date: datetime.date) -> bool:
        """
        Checks whether the given date is a valid fixing date based on the index's calendar.

        Args:
            fixing_date (datetime.date): The date to check.

        Returns:
            bool: True if the date is a valid fixing date, False otherwise.

        Note:
            This method relies on the calendar's business day determination logic.
        """
        return self._fixing_calendar.is_business_day(fixing_date)

    def add_fixing(self, fixing_date: datetime.date, value: float) -> None:
        """
        Adds a fixing for a specific date to the fixing time series.

        Args:
            fixing_date (datetime.date): The date of the fixing.
            value (float): The fixing value for the given date.
        """
        # Type checks
        if not isinstance(fixing_date, datetime.date):
            raise TypeError(
                f"Fixing date must be datetime.date, got {type(fixing_date)}"
            )

        if not isinstance(value, (int, float)):
            raise TypeError(f"Fixing value must be float (or int), got {type(value)}")


        # Initialize the internal structure if needed
        if self._fixing_time_series is None:
            self._fixing_time_series = {self.name: {fixing_date: value}}
            return

        # Ensure the index key exists
        if self.name not in self._fixing_time_series:
            self._fixing_time_series[self.name] = {}

        # Overwrite if duplicate date and emit a warning
        if fixing_date in self._fixing_time_series[self.name]:
            warnings.warn(
                f"Overwriting existing fixing for {self.name} on {fixing_date}",
                UserWarning,
            )

        # Write the fixing
        self._fixing_time_series[self.name][fixing_date] = value

    def past_fixing(self, fixing_date: datetime.date) -> float:
        """
        Retrieves the past fixing value for a specific date.

        Args:
            fixing_date (datetime.date): The date for which the past fixing is requested.

        Returns:
            float: The fixing value for the given date.

        Raises:
            ValueError: If the fixing date is not valid or if the fixing is missing for the given date.
        """
        past_fixings = self.fixing_time_series
        if self.is_valid_fixing_date(fixing_date):
            return past_fixings[self.name][fixing_date]
        raise ValueError("Not a valid fixing date!")

    def fixing(self, fixing_date: datetime.date) -> float:
        """
        Retrieves the fixing value for a specific date, ensuring it is not a future date.

        Args:
            fixing_date (datetime.date): The date for which the fixing is requested.

        Returns:
            float: The fixing value for the given date.

        Raises:
            ValueError: If the fixing date is invalid, missing, or in the future.
        """
        if not self.is_valid_fixing_date(fixing_date):
            raise ValueError("Not a valid date")

        if fixing_date > Settings.evaluation_date:
            raise ValueError("Fixing are only available for historical dates.")

        # if fixing_date <= Settings.evaluation_date:
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
