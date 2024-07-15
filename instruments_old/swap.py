from index.curverateindex import IborIndex
from interface_old.floatingleg import *
from interface_old.fixedleg import *
from interface_old.trade import Trade

class SwapDirection:
    PAYER = 1
    RECEIVER = -1

# class SwapFixedFloating(Trade):
#     def __init__(self, 
#                 float_schedule: Schedule,
#                 fix_schedule: Schedule,
#                 float_notionals: list[float],
#                 fix_notionals: list[float],
#                 gearings: list[float],
#                 spreads: list[float],
#                 index: CurveRateIndex,
#                 fix_coupons
#                 ) -> None:
#         super().__init__()
#         self.floating_leg = FloatingRateLeg(float_schedule, float_notionals, gearings, spreads, index)
#         self.fixed_leg = FixedRateLeg(fix_schedule, fix_notionals, fix_coupons)

#     def price(self, discount_curve, evaluation_date: datetime):
#         return self.floating_leg.npv(discount_curve, evaluation_date) + self.fixed_leg.npv(discount_curve, evaluation_date)   
        
    

class SwapFixedFloating(Trade):
    def __init__(self, 
                float_schedule: list[date],
                fix_schedule: list[date],
                float_notionals: list[float],
                fix_notionals: list[float],
                gearings: list[float],
                spreads: list[float],
                index: IborIndex,
                fix_coupons
                ) -> None:
        super().__init__()
        self.floating_leg = FloatingRateLeg(float_schedule, float_notionals, gearings, spreads, index)
        self.fixed_leg = FixedRateLeg(fix_schedule, fix_notionals, fix_coupons)

    def price(self, discount_curve, term_structure, evaluation_date: date):
        return self.floating_leg.npv(discount_curve,
                                     term_structure,
                                     evaluation_date) + self.fixed_leg.npv(discount_curve, 
                                                                           evaluation_date)   
        
    