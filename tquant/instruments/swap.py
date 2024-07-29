from datetime import date
from .product import Product
from ..index.index import Index
from ..flows.fixedcoupon import FixedRateLeg 
from ..flows.floatingcoupon import FloatingRateLeg
from ..timehandles.utils import DayCounterConvention
from ..markethandles.utils import Currency, SwapType



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
        
