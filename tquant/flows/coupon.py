from .cashflow import CashFlow
from abc import ABC, abstractmethod
from ..timehandles.daycounter import DayCounter
from datetime import date

class Coupon(CashFlow, ABC):
    """
    An abstract class representing a coupon payment, inheriting from the `CashFlow` class.

    This class provides the foundation for different types of coupon payments by defining
    the necessary attributes and methods that must be implemented by subclasses.
    """

    @abstractmethod
    def __init__(self,
                 payment_date: date,
                 nominal: float,
                 daycounter: DayCounter,
                 accrual_start_date: date,
                 accrual_end_date: date,
                 ref_period_start: date,
                 ref_period_end: date):
        """
        Initializes a Coupon instance with the specified attributes.

        Parameters:
        -------
        payment_date: date
            The date on which the coupon payment is made.
        nominal: float
            The nominal (face value) amount of the coupon.
        daycounter: DayCounter
            The day count convention used to calculate accrued interest.
        accrual_start_date: date
            The start date of the accrual period.
        accrual_end_date: date
            The end date of the accrual period.
        ref_period_start: date
            The start date of the reference period.
        ref_period_end: date
            The end date of the reference period.
        """
        
        self._payment_date = payment_date
        self._nominal = nominal
        self._daycounter = daycounter
        self._accrual_start_date = accrual_start_date
        self._accrual_end_date = accrual_end_date
        self._ref_period_start = ref_period_start
        self._ref_period_end = ref_period_end
        # self._ex_coupon_date = ex_coupon_date
    
    @abstractmethod
    def rate(self) -> float:
        """
        Returns the interest rate associated with the coupon.

        This method must be implemented by any subclass to provide the specific interest
        rate for the coupon.

        Returns:
        -------
        float
            The interest rate of the coupon.
        """
        return 
    
    @property
    @abstractmethod
    def accrued_amount(self) -> float:
        """
        Returns the accrued amount of the coupon up to the current date.

        This method must be implemented by any subclass to calculate the accrued interest
        for the coupon.

        Returns:
        -------
        float
            The accrued amount of the coupon.
        """
        return
    
    @property
    def date(self) -> date:
        """
        Returns the payment date of the coupon.

        Returns:
        -------
        date
            The payment date of the coupon.
        """
        return self._payment_date

    @property
    def nominal(self) -> float:
        """
        Returns the nominal (face value) amount of the coupon.

        Returns:
        -------
        float
            The nominal amount of the coupon.
        """
        return self._nominal
    
    @property
    def daycounter(self) -> DayCounter:
        """
        Returns the day counter convention used by the coupon.

        Returns:
        -------
        DayCounter
            The day counter convention used for calculating accrued interest.
        """
        return self._daycounter
    
    @property
    def accrual_start_date(self) -> date:
        """
        Returns the start date of the accrual period.

        Returns:
        -------
        date
            The start date of the accrual period.
        """
        return self._accrual_start_date

    @property
    def accrual_end_date(self) -> date:
        """
        Returns the end date of the accrual period.

        Returns:
        -------
        date
            The end date of the accrual period.
        """
        return self._accrual_end_date

    @property
    def ref_period_start(self) -> date:
        """
        Returns the start date of the reference period.

        Returns:
        -------
        date
            The start date of the reference period.
        """
        return self._ref_period_start

    @property
    def ref_period_end(self) -> date:
        """
        Returns the end date of the reference period.

        Returns:
        -------
        date
            The end date of the reference period.
        """
        return self._ref_period_end

    @property
    def accrual_period(self) -> float:
        """
        Returns the fraction of the year that represents the accrual period.

        Uses the `DayCounter` to calculate the year fraction between the accrual start and 
        end dates.

        Returns:
        -------
        float
            The fraction of the year that represents the accrual period.
        """
        return self._daycounter.year_fraction(self.accrual_start_date,
                                                 self.accrual_end_date
                                            )  
    
    @property
    def accrual_days(self) -> int:
        """
        Returns the number of days in the accrual period.

        Uses the `DayCounter` to calculate the number of days between the accrual start and 
        end dates.

        Returns:
        -------
        int
            The number of days in the accrual period.
        """
        return self._daycounter.day_count(self.accrual_start_date,
                                            self.accrual_end_date
                                            )
        
    @property
    def accrued_period(self, d: date) -> float:
        """
        Returns the fraction of the year that has accrued up to the specified date.

        Uses the `DayCounter` to calculate the year fraction between the accrual start date 
        and the minimum of the specified date and the accrual end date.

        Parameters:
        -------
        d: date
            The date up to which to calculate the accrued period.

        Returns:
        -------
        float
            The fraction of the year that has accrued up to the specified date.
        """
        return self._daycounter.year_fraction(self.accrual_start_date,
                                            min(d, self.accrual_end_date)
                                            ) 

    @property
    def accrued_days(self, d: date) -> int:
        """
        Returns the number of days that have accrued up to the specified date.

        Uses the `DayCounter` to calculate the number of days between the accrual start date 
        and the minimum of the specified date and the accrual end date.

        Parameters:
        -------
        d: date
            The date up to which to calculate the accrued days.

        Returns:
        -------
        int
            The number of days that have accrued up to the specified date.
        """
        return self._daycounter.day_count(self.accrual_start_date,
                                            min(d, self.accrual_end_date)
                                            )

