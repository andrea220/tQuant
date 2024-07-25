from datetime import date
from .product import Product
from ..index.index import Index
from ..flows.premiumcoupon import PremiumLeg 
from ..flows.defaultcoupon import DefaultLeg
from ..timehandles.utils import CompoundingType, Frequency
from ..timehandles.daycounter import DayCounter


class CreditDefaultSwap(Product):
    def __init__(self,
                swap_type,
                fixed_schedule,
                fixed_notional,
                fixed_spread,
                fixed_day_counter: DayCounter,
                default_schedule,
                default_notional,
                recovery_rate,
                default_day_counter,
                fixed_compounding_type: CompoundingType = CompoundingType.Simple,
                fixed_frequency: Frequency = Frequency.Quarterly,
                default_compounding_type: CompoundingType = CompoundingType.Simple,
                default_frequency: Frequency = Frequency.Quarterly
                ) -> None:
        super().__init__()
        self.swap_type = swap_type
        self.sign = -1 if swap_type.value=="Payer" else 1

        self.fixed_schedule = fixed_schedule
        self.fixed_notional = fixed_notional
        self.fixed_day_counter = fixed_day_counter
        self.fixed_compounding_type = fixed_compounding_type
        self.fixed_frequency = fixed_frequency

        self.default_schedule = default_schedule
        self.default_notional = default_notional
        self.recovery_rate = recovery_rate
        self.default_day_counter = default_day_counter
        self.default_compounding_type = default_compounding_type
        self.default_frequency = default_frequency

        self.premium_leg = PremiumLeg(fixed_schedule, 
                            fixed_notional, 
                            fixed_spread, 
                            fixed_day_counter,
                            fixed_compounding_type, 
                            fixed_frequency
                            )
        self.default_leg = DefaultLeg(default_schedule, 
                            default_notional, 
                            recovery_rate, 
                            default_day_counter,
                            default_compounding_type, 
                            default_frequency
                            )
        
    @classmethod
    def from_legs(cls, 
                  swap_type, 
                  leg1: PremiumLeg, 
                  leg2: DefaultLeg):

        return cls(swap_type,
                   leg1._schedule,
                   leg1._notionals,
                   leg1._spreads,
                   leg1._daycounter,
                   leg2._schedule,
                   leg2._notionals,
                   leg2._recovery,
                   leg2._daycounter,
                   leg1._compounding,
                   leg1._frequency,
                   leg2._compounding,
                   leg2._frequency)
    