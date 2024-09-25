import datetime

from .product import Product
from ..timehandles.utils import DayCounterConvention
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency


class Deposit(Product):
    """
    Represents a deposit financial product.

    Attributes:
        ccy (Currency): The currency of the deposit.
        rate (float): The interest rate for the deposit.
        trade_date (datetime.date): The trade date of the deposit.
        start_date (datetime.date): The start date of the deposit.
        end_date (datetime.date): The end date or maturity date of the deposit.
        notional (float or int): The notional amount of the deposit.
        day_counter (DayCounter): Day count convention used to calculate day fractions.
    """

    def __init__(
        self,
        ccy: Currency,
        rate: float,
        trade_date: datetime.date,
        start_date: datetime.date,
        end_date: datetime.date,
        notional: int | float,
        day_count_convention: DayCounterConvention,
    ):
        """
        Initializes the Deposit instance.

        Args:
            ccy (Currency): The currency in which the deposit is denominated.
            rate (float): The interest rate for the deposit.
            trade_date (datetime.date): The trade date of the deposit.
            start_date (datetime.date): The date on which the deposit starts.
            end_date (datetime.date): The maturity or end date of the deposit.
            notional (int or float): The notional amount for the deposit.
            day_count_convention (DayCounterConvention): The day count convention for calculating interest.
        """
        super().__init__(ccy, start_date, end_date)
        self._rate = rate
        self._day_count_convention = day_count_convention
        self._trade_date = trade_date
        self._notional = notional
        self._day_counter = DayCounter(day_count_convention)

    @property
    def rate(self) -> float:
        return self._rate 
    
    @property
    def day_count_convention(self) -> DayCounterConvention:
        """
        Get the day count convention used for interest calculations.

        Returns:
            DayCounterConvention: The day count convention used for this deposit.
        """
        return self._day_count_convention

    @property
    def trade_date(self) -> datetime.date:
        """
        Get the trade date of the deposit.

        Returns:
            datetime.date: The trade date.
        """
        return self._trade_date

    @property
    def notional(self) -> float:
        """
        Get the notional amount of the deposit.

        Returns:
            float: The notional amount.
        """
        return self._notional

    @property
    def day_counter(self) -> DayCounter:
        """
        Get the day count convention used for interest calculations.

        Returns:
            DayCounter: The day count convention object.
        """
        return self._day_counter
