from datetime import date
from .product import Product
from ..timehandles.utils import DayCounterConvention
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency


class Deposit(Product):
    """ 
    Represents a deposit product.

    This class models a deposit with specific attributes such as currency, trade date, 
    start date, end date, notional amount, and day count convention. It inherits from 
    the `Product` class and adds additional attributes specific to deposits.

    """
    def __init__(self,
                 ccy: Currency,
                 quote: float,
                 trade_date: date, 
                 start_date: date,
                 end_date: date,
                 notional: float,
                 day_count_convention: DayCounterConvention):
        """
        Initializes a Deposit instance with the specified attributes.

        Parameters:
        -----------
        ccy: Currency
            The currency in which the deposit is denominated.
        quote: float
            The interest rate or yield quoted for the deposit.
        trade_date: date
            The date on which the deposit was traded.
        start_date: date
            The date on which the deposit starts accruing interest.
        end_date: date
            The date on which the deposit matures.
        notional: float
            The principal or face value of the deposit.
        day_count_convention: DayCounterConvention
            The day count convention used to calculate accrued interest.
        """
        super().__init__(ccy, start_date, end_date, quote)
        self.quote = quote
        self.trade_date = trade_date
        self.start_date = start_date
        self.end_date = end_date
        self.notional = notional
        self.day_counter = DayCounter(day_count_convention)

