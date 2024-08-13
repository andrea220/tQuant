from datetime import date
from .product import Product
from ..index.index import Index
from ..timehandles.utils import DayCounterConvention
from ..timehandles.tqcalendar import Calendar
from ..markethandles.utils import Currency

class Fra(Product):
    """
    Represents a Forward Rate Agreement (FRA).

    An FRA is a financial contract between two parties to exchange interest payments on 
    a notional amount, based on a fixed interest rate and a floating rate index, over a 
    specified period in the future. This class inherits from the `Product` class and 
    includes additional attributes specific to FRAs, such as notional, day count 
    convention, and the interest rate index.
    """ 
    def __init__(self,
                 ccy: Currency,
                 start_date: date,
                 end_date: date,
                 notional: float,
                 quote: float,
                 day_counter: DayCounterConvention,
                 index: Index):
        """
        Initializes an Fra instance with the specified attributes.

        Parameters:
        -----------
        ccy: Currency
            The currency in which the FRA is denominated.
        start_date: date
            The date on which the FRA starts, which is when the floating rate is set.
        end_date: date
            The date on which the FRA ends, which is when the interest payments are exchanged.
        notional: float
            The principal or face value on which the interest payments are calculated.
        quote: float
            The fixed interest rate agreed upon in the FRA contract.
        day_counter: DayCounterConvention
            The day count convention used to calculate accrued interest.
        index: Index
            The floating rate index used in the FRA, such as LIBOR or EURIBOR.
        """
        super().__init__(ccy, start_date, end_date, quote)
        self.start_date = start_date
        self.end_date = end_date
        self.notional = notional
        self.day_counter = day_counter
        self._index = index
