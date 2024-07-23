from datetime import date
from .trade import Trade
from ..index.index import Index
from ..flows.fixedcoupon import FixedRateLeg 
from ..flows.floatingcoupon import FloatingRateLeg
from ..timehandles.utils import DayCounterConvention
from ..markethandles.utils import Currency



class InterestRateSwap(Trade):
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
                currency: Currency = None
                ) -> None:
        super().__init__()
        self.fixed_leg = FixedRateLeg(fix_schedule, fix_notionals, fixed_coupons, fixed_daycounter)
        self.floating_leg = FloatingRateLeg(float_schedule, float_notionals, gearings, spreads, index, floating_daycounter)
        self._currency = currency

class InterestRateSwap2(Trade):
    def __init__(self, 
                leg1,
                leg2,
                currency: Currency = None
                ) -> None:
        super().__init__()
        self.fixed_leg = leg1
        self.floating_leg = leg2
        self._currency = currency
        
