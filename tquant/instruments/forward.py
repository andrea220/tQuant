from datetime import date
from .product import Product
from ..index.curverateindex import IborIndex
from ..timehandles.utils import DayCounterConvention
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency


class Fra(Product):
    """
    Represents a Forward Rate Agreement (FRA).

    A Forward Rate Agreement (FRA) is a financial contract that allows parties to lock in an interest rate 
    for a future period. This class models the FRA including its notional amount, start and end dates, 
    the day count convention used for interest calculations, and the associated Ibor index.

    Attributes:
        ccy (Currency): The currency of the FRA.
        start_date (date): The start date of the FRA.
        end_date (date): The end date of the FRA.
        notional (float): The notional amount of the FRA.
        quote (float): The FRA quote or rate.
        day_count_convention (DayCounterConvention): The day count convention used for interest calculations.
        index (IborIndex): The Ibor index used for the FRA.
    """
    def __init__(
        self,
        ccy: Currency,
        start_date: date,
        end_date: date,
        notional: float,
        quote: float,
        day_count_convention: DayCounterConvention,
        index: IborIndex,
    ):
        """
        Initialize a Forward Rate Agreement (FRA) instance.

        Args:
            ccy (Currency): The currency of the FRA.
            start_date (date): The start date of the FRA.
            end_date (date): The end date of the FRA.
            notional (float): The notional amount of the FRA.
            quote (float): The FRA quote or rate.
            day_count_convention (DayCounterConvention): The day count convention used for interest calculations.
            index (IborIndex): The Ibor index used for the FRA.
        """
        super().__init__(ccy, start_date, end_date, quote)
        self._day_count_convention = day_count_convention
        self._notional = notional
        self._day_counter = DayCounter(day_count_convention)
        self._index = index

    @property
    def day_count_convention(self) -> DayCounterConvention:
        """
        Get the day count convention used for interest calculations.

        Returns:
            DayCounterConvention: The day count convention.
        """
        return self._day_count_convention

    @property
    def notional(self) -> float:
        """
        Get the notional amount of the FRA.

        Returns:
            float: The notional amount.
        """
        return self._notional

    @property
    def day_counter(self) -> DayCounter:
        """
        Get the day count convention object used for interest calculations.

        Returns:
            DayCounter: The day count convention object.
        """
        return self._day_counter

    @property
    def fixing_date(self):
        """
        Get the fixing date based on the start date and index.

        Returns:
            date: The fixing date.
        """
        return self._index.fixing_date(self.start_date)
