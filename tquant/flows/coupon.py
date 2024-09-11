import datetime
from abc import ABC, abstractmethod

from .cashflow import CashFlow
from ..timehandles.daycounter import DayCounter


class Coupon(CashFlow, ABC):
    """
    Abstract class representing a coupon payment.

    This class inherits from `CashFlow` and provides a foundation for different types of
    coupon payments. It defines necessary attributes and methods that subclasses must implement.
    """

    @abstractmethod
    def __init__(
        self,
        payment_date: datetime.date,
        nominal: float,
        daycounter: DayCounter,
        accrual_start_date: datetime.date,
        accrual_end_date: datetime.date,
        ref_period_start: datetime.date,
        ref_period_end: datetime.date,
    ):
        """
        Initialize a Coupon instance with the specified attributes.

        Args:
            payment_date (datetime.date): The date on which the coupon payment is made.
            nominal (float): The nominal (face value) amount of the coupon.
            daycounter (DayCounter): The day count convention used to calculate accrued interest.
            accrual_start_date (datetime.date): The start date of the accrual period.
            accrual_end_date (datetime.date): The end date of the accrual period.
            ref_period_start (datetime.date): The start date of the reference period.
            ref_period_end (datetime.date): The end date of the reference period.
        """

        self._payment_date = payment_date
        self._nominal = nominal
        self._daycounter = daycounter
        self._accrual_start_date = accrual_start_date
        self._accrual_end_date = accrual_end_date
        self._ref_period_start = ref_period_start
        self._ref_period_end = ref_period_end

    @property
    @abstractmethod
    def rate(self) -> float:
        """
        Get the interest rate associated with the coupon.

        Subclasses must implement this method to return the specific interest rate for the coupon.

        Returns:
            float: The interest rate of the coupon.
        """
        return

    @abstractmethod
    def accrued_amount(self) -> float:
        """
        Get the accrued amount of the coupon up to the current date.

        Subclasses must implement this property to calculate the accrued interest for the coupon.

        Returns:
            float: The accrued amount of the coupon.
        """
        return

    @property
    def date(self) -> datetime.date:
        """
        Get the payment date of the coupon.

        Returns:
            datetime.date: The payment date of the coupon.
        """
        return self._payment_date

    @property
    def nominal(self) -> float:
        """
        Get the nominal (face value) amount of the coupon.

        Returns:
            float: The nominal amount of the coupon.
        """
        return self._nominal

    @property
    def daycounter(self) -> DayCounter:
        """
        Get the day counter convention used by the coupon.

        Returns:
            DayCounter: The day counter convention used for calculating accrued interest.
        """
        return self._daycounter

    @property
    def accrual_start_date(self) -> datetime.date:
        """
        Get the start date of the accrual period.

        Returns:
            datetime.date: The start date of the accrual period.
        """
        return self._accrual_start_date

    @property
    def accrual_end_date(self) -> datetime.date:
        """
        Get the end date of the accrual period.

        Returns:
            datetime.date: The end date of the accrual period.
        """
        return self._accrual_end_date

    @property
    def ref_period_start(self) -> datetime.date:
        """
        Get the start date of the reference period.

        Returns:
            datetime.date: The start date of the reference period.
        """
        return self._ref_period_start

    @property
    def ref_period_end(self) -> datetime.date:
        """
        Get the end date of the reference period.

        Returns:
            datetime.date: The end date of the reference period.
        """
        return self._ref_period_end

    @property
    def accrual_period(self) -> float:
        """
        Get the fraction of the year that represents the accrual period.

        Uses the `DayCounter` to calculate the year fraction between the accrual start and end dates.

        Returns:
            float: The fraction of the year that represents the accrual period.
        """
        return self._daycounter.year_fraction(
            self.accrual_start_date, self.accrual_end_date
        )

    @property
    def accrual_days(self) -> int:
        """
        Get the number of days in the accrual period.

        Uses the `DayCounter` to calculate the number of days between the accrual start and end dates.

        Returns:
            int: The number of days in the accrual period.
        """
        return self._daycounter.day_count(
            self.accrual_start_date, self.accrual_end_date
        )

    @property
    def accrued_period(self, d: datetime.date) -> float:
        """
        Get the fraction of the year that has accrued up to the specified date.

        Uses the `DayCounter` to calculate the year fraction between the accrual start date
        and the minimum of the specified date and the accrual end date.

        Args:
            d (datetime.date): The date up to which to calculate the accrued period.

        Returns:
            float: The fraction of the year that has accrued up to the specified date.
        """
        return self._daycounter.year_fraction(
            self.accrual_start_date, min(d, self.accrual_end_date)
        )

    @property
    def accrued_days(self, d: datetime.date) -> int:
        """
        Get the number of days that have accrued up to the specified date.

        Uses the `DayCounter` to calculate the number of days between the accrual start date
        and the minimum of the specified date and the accrual end date.

        Args:
            d (datetime.date): The date up to which to calculate the accrued days.

        Returns:
            int: The number of days that have accrued up to the specified date.
        """
        return self._daycounter.day_count(
            self.accrual_start_date, min(d, self.accrual_end_date)
        )
