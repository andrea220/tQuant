from datetime import date
from .product import Product
from ..index.curverateindex import IborIndex
from ..timehandles.utils import DayCounterConvention
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency


class Fra(Product):

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
        super().__init__(ccy, start_date, end_date, quote)
        self._day_count_convention = day_count_convention
        self._notional = notional
        self._day_counter = DayCounter(day_count_convention)
        self._index = index

    @property
    def day_count_convention(self) -> DayCounterConvention:
        return self._day_count_convention

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

    @property
    def fixing_date(self):
        return self._index.fixing_date(self.start_date)
