from datetime import date

from .product import Product
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency
from ..index.curverateindex import OvernightIndex
from ..flows.fixedcoupon import FixedRateLeg
from ..flows.floatingcoupon import FloatingRateLeg


class Ois(Product):
    """
    Represents an Overnight Indexed Swap (OIS).

    An OIS is a type of interest rate swap where one party pays a fixed rate of interest 
    and the other party pays a floating rate that is typically linked to an overnight index 
    (e.g., ESTR, Fed Funds). This class models the OIS with both fixed and floating legs, 
    including attributes for fixing dates, rates, and relevant day count conventions.
    """ 
    def __init__(self,
                 ccy: Currency,
                 start_date: date,
                 end_date: date,
                 start_dates_fix: list[date],
                 end_dates_fix: list[date],
                 pay_dates_fix: list[date],
                 start_dates_flt: list[date],
                 end_dates_flt: list[date],
                 pay_dates_flt: list[date],
                 fixing_dates: list[date],
                 fixing_rates: list[float],
                 quote: float,
                 notional: float,
                 day_counter_fix: DayCounter,
                 day_counter_flt: DayCounter,
                 index: OvernightIndex): #TODO add swaptype
        """
        Initializes an Ois instance with the specified attributes.

        Parameters:
        -----------
        ccy: Currency
            The currency in which the OIS is denominated.
        start_date: date
            The date on which the OIS starts.
        end_date: date
            The date on which the OIS ends.
        start_dates_fix: list[date]
            The start dates for the fixed leg periods.
        end_dates_fix: list[date]
            The end dates for the fixed leg periods.
        pay_dates_fix: list[date]
            The payment dates for the fixed leg.
        start_dates_flt: list[date]
            The start dates for the floating leg periods.
        end_dates_flt: list[date]
            The end dates for the floating leg periods.
        pay_dates_flt: list[date]
            The payment dates for the floating leg.
        fixing_dates: list[date]
            The dates on which the floating rate fixings are observed.
        fixing_rates: list[float]
            The observed floating rates on the fixing dates.
        quote: float
            The fixed interest rate agreed upon in the OIS contract.
        notional: float
            The principal or face value on which the interest payments are calculated.
        day_counter_fix: DayCounter
            The day count convention used for the fixed leg.
        day_counter_flt: DayCounter
            The day count convention used for the floating leg.
        index: Index
            The floating rate index used in the OIS, such as EONIA or Fed Funds.
        """
        super().__init__(ccy, start_date, end_date, quote)
        self.start_dates_fix = start_dates_fix
        self.end_dates_fix = end_dates_fix
        self.pay_dates_fix = pay_dates_fix
        self.start_dates_flt = start_dates_flt
        self.end_dates_flt = end_dates_flt
        self.pay_dates_flt = pay_dates_flt
        self.fixing_dates = fixing_dates
        self.fixing_rates = fixing_rates
        self.notional = notional
        self.day_counter_fix = day_counter_fix
        self.day_counter_flt = day_counter_flt

        # self.swap_type = swap_type
        self._notionals = [notional]*len(pay_dates_fix)
        self._rates = [quote]*len(pay_dates_fix)
        self._gearings = [1]*len(pay_dates_flt)
        self._margins = [0]*len(pay_dates_flt)
        self._index = index

        self.fixed_leg = FixedRateLeg(pay_dates_fix, start_dates_fix, end_dates_fix,
                                    self._notionals, self._rates, day_counter_fix)         
        self.floating_leg = FloatingRateLeg(pay_dates_flt, start_dates_flt, end_dates_flt,
                                    self._notionals, self._gearings, self._margins, index, day_counter_flt)      