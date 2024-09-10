from datetime import date
from .product import Product
from ..index.index import Index
from ..flows.fixedcoupon import FixedRateLeg 
from ..flows.floatingcoupon import FloatingRateLeg
from ..timehandles.utils import DayCounterConvention
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency, SwapType

class Swap(Product): 
    """
    Represents an interest rate swap.

    An interest rate swap is a financial derivative contract in which two parties agree 
    to exchange interest rate payments on a notional amount over a specified period. 
    Typically, one party pays a fixed rate while the other pays a floating rate. This 
    class models the swap with both fixed and floating legs, including attributes for 
    payment dates, notional amounts, and relevant day count conventions.
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
                 quote: float,
                 notional: float,
                 day_counter_fix: DayCounter,
                 day_counter_flt: DayCounter,
                 index: Index) -> None: #TODO inserire payer/receiver 
        """
        Initializes a Swap instance with the specified attributes.

        Parameters:
        -----------
        ccy: Currency
            The currency in which the swap is denominated.
        start_date: date
            The date on which the swap starts.
        end_date: date
            The date on which the swap ends.
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
        quote: float
            The fixed interest rate agreed upon in the swap contract.
        notional: float
            The principal or face value on which the interest payments are calculated.
        day_counter_fix: DayCounter
            The day count convention used for the fixed leg.
        day_counter_flt: DayCounter
            The day count convention used for the floating leg.
        index: Index
            The floating rate index used in the swap, such as LIBOR or EURIBOR.
        """
        super().__init__(ccy, start_date, end_date, quote)
        self.start_dates_fix = start_dates_fix
        self.end_dates_fix = end_dates_fix
        self.pay_dates_fix = pay_dates_fix
        self.start_dates_flt = start_dates_flt
        self.end_dates_flt = end_dates_flt
        self.pay_dates_flt = pay_dates_flt
        self.notional = notional
        self.day_counter_fix = day_counter_fix
        self.day_counter_flt = day_counter_flt

        # self.swap_type = swap_type
        self._fix_notionals = [notional]*len(pay_dates_fix)
        self._rates = [quote]*len(pay_dates_fix)

        self._float_notionals = [notional]*len(pay_dates_flt)
        self._gearings = [1]*len(pay_dates_flt)
        self._margins = [0]*len(pay_dates_flt)
        self._index = index

        self.fixed_leg = FixedRateLeg(pay_dates_fix, start_dates_fix, end_dates_fix,
                                    self._fix_notionals, self._rates, day_counter_fix)         
        self.floating_leg = FloatingRateLeg(pay_dates_flt, start_dates_flt, end_dates_flt,
                                    self._float_notionals, self._gearings, self._margins, index, day_counter_flt)      


# DEPRECATED
class InterestRateSwap(Product): 
    ''' 
    classe custom che gestisce schedule in input
    '''
    def __init__(self, 
                float_schedule: list[date],
                fix_schedule: list[date],
                float_notionals: list[float],
                fix_notionals: list[float],
                gearings: list[float],
                spreads: list[float],
                index: Index,
                fixed_coupons,
                fixed_daycounter: DayCounterConvention,
                floating_daycounter: DayCounterConvention,
                swap_type: SwapType = SwapType.Payer, 
                currency: Currency = None
                ) -> None:
        super().__init__(currency, fix_schedule[0], fix_schedule[-1])
        self.swap_type = swap_type
        self.fixed_leg = FixedRateLeg(fix_schedule, fix_notionals, fixed_coupons, fixed_daycounter)
        self.floating_leg = FloatingRateLeg(float_schedule, float_notionals, gearings, spreads, index, floating_daycounter)
        
    @property
    def price(self):
        if self._price is not None:
            return self._price
        else:
            raise ValueError("price not assigned")
    @price.setter
    def price(self, value):
        self._price = value

# DEPRECATED
class InterestRateSwap2(Product):
    def __init__(self, 
                leg1,
                leg2,
                currency: Currency = None
                ) -> None:
        super().__init__()
        self.fixed_leg = leg1
        self.floating_leg = leg2
        self._currency = currency
        
