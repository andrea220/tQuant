import datetime

from ..timehandles.utils import BusinessDayConvention, TimeUnit
from .index import Index
from ..timehandles.tqcalendar import Calendar
from ..markethandles.utils import Currency
from ..timehandles.daycounter import DayCounter, DayCounterConvention


class OvernightIndex(Index):
    """
    Represents an Overnight index, typically used for short-term interest rates.

    This class inherits from the abstract `Index` class and models an overnight index,
    commonly used for financial instruments with very short tenors, such as overnight
    lending rates.
    """

    def __init__(
        self,
        fixing_calendar: Calendar,
        currency: Currency = None,
        fixing_days: int = None,
        time_series: dict = None,
    ) -> None:
        """
        Returns the number of fixing days for the index.

        If the number of fixing days is not set, returns 0.

        Returns:
            int: The number of fixing days or 0 if not set.
        """
        super().__init__(currency.value + ":ON", fixing_calendar, time_series)
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
        if self._fixing_days is None:
            return 0
        return self._fixing_days

    def fixing_maturity(self, fixing_date: datetime.date) -> datetime.date:
        """
        Calculates the maturity date for an overnight index based on the fixing date.

        The maturity date is typically one business day after the fixing date.

        Args:
            fixing_date (datetime.date): The date of the fixing.

        Returns:
            datetime.date: The calculated maturity date.
        """
        return self.fixing_calendar.advance(
            fixing_date, 1, TimeUnit.Days, BusinessDayConvention.ModifiedFollowing
        )

    def fixing_date(self, value_date: datetime.date) -> datetime.date:
        """
        Calculates the fixing date based on the value date and the number of fixing days.

        Args:
            value_date (datetime.date): The value date to calculate the fixing date from.

        Returns:
            datetime.date: The calculated fixing date.
        """
        return self.fixing_calendar.advance(
            value_date,
            -self.fixing_days,
            TimeUnit.Days,
            BusinessDayConvention.Preceding,
        )


class IborIndex(Index):
    """
    Represents an Interbank Offered Rate (IBOR) index.

    This class models an IBOR index, which reflects the interest rates at which banks
    lend to each other in the interbank market. The class extends the `Index` class and
    includes additional attributes specific to IBOR indices, such as tenor and currency.
    """

    def __init__(
        self,
        fixing_calendar: Calendar,
        tenor: int,
        time_unit: TimeUnit,
        currency: Currency,
        fixing_days: int = None,
        time_series: dict = None,
    ) -> None:
        """
        Initializes an IborIndex instance with the given attributes.

        Args:
            fixing_calendar (Calendar): The calendar used to calculate fixing dates.
            tenor (int): The duration of the IBOR rate (e.g., 1, 3, 6 months).
            time_unit (TimeUnit): The time unit for the tenor (e.g., days, months).
            currency (Currency): The currency associated with the IBOR index.
            fixing_days (int, optional): Number of days before the fixing date (default is None).
            time_series (dict, optional): A dictionary containing time series data for past fixings (default is None).
        """
        super().__init__(
            currency.value + ":" + str(tenor) + time_unit.value[0],
            fixing_calendar,
            time_series,
        )
        self._fixing_days = fixing_days
        self._tenor = tenor
        self._time_unit = time_unit
        self._currency = currency
        self._daycounter = DayCounter(DayCounterConvention.Actual360)

    @property
    def fixing_days(self) -> int:
        """
        Returns the number of fixing days for the IBOR index.

        If the number of fixing days is not set, returns 0.

        Returns:
            int: The number of fixing days or 0 if not set.
        """
        if self._fixing_days is None:
            return 0
        return self._fixing_days

    @property
    def daycounter(self) -> DayCounter:
        """
        Returns the day count convention used for the IBOR index.

        Returns:
            DayCounter: The day count convention for calculating interest accruals (e.g., Actual/360).
        """
        return self._daycounter

    def fixing_maturity(self, fixing_date: datetime.date) -> datetime.date:
        """
        Calculates the maturity date for the IBOR index based on the fixing date.

        The maturity date is determined by advancing the fixing date by the tenor
        according to the calendar and business day conventions.

        Args:
            fixing_date (datetime.date): The fixing date of the IBOR rate.

        Returns:
            datetime.date: The calculated maturity date.
        """
        return self.fixing_calendar.advance(
            fixing_date,
            self._tenor,
            self._time_unit,
            BusinessDayConvention.ModifiedFollowing,
        )

    def fixing_date(self, value_date: datetime.date) -> datetime.date:
        """
        Calculates the fixing date based on the value date and the number of fixing days.

        Args:
            value_date (datetime.date): The value date to calculate the fixing date from.

        Returns:
            datetime.date: The calculated fixing date.
        """
        return self.fixing_calendar.advance(
            value_date,
            -self.fixing_days,
            TimeUnit.Days,
            BusinessDayConvention.Preceding,
        )
