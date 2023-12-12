from index.curverateindex import CurveRateIndex, Schedule
from interface.floatingleg import *
from interface.fixedleg import *

class SwapDirection:
    PAYER = 1
    RCEIVER = -1

class SwapFixedFloating:
    def __init__(self, 
                float_schedule: Schedule,
                fix_schedule: Schedule,
                float_notionals: list[float],
                fix_notionals: list[float],
                gearings: list[float],
                spreads: list[float],
                index: CurveRateIndex,
                fix_coupons
                ) -> None:
        self.floating_leg = FloatingRateLeg(float_schedule, float_notionals, gearings, spreads, index)
        self.fixed_leg = FixedRateLeg(fix_schedule, fix_notionals, fix_coupons)

    def price(self, discount_curve, evaluation_date: datetime):
        return self.floating_leg.npv(discount_curve, evaluation_date) + self.fixed_leg.npv(discount_curve, evaluation_date)   
        
    