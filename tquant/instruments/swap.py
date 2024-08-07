from datetime import date
from .product import Product
from ..index.index import Index
from ..flows.fixedcoupon import FixedRateLeg 
from ..flows.floatingcoupon import FloatingRateLeg
from ..timehandles.utils import DayCounterConvention
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency, SwapType

class Swap(Product):
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
                 index: Index):
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

class InterestRateSwap(Product): #TODO creare classe generica che gestisca tutte le gambe
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
        
