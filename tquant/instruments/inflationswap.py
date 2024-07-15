from datetime import date
from ..interface.trade import Trade
from ..interface.index import Index
from ..flows.fixedleg import FixedRateLeg 
from ..flows.inflationleg import InflationLeg
from ..utilities.utils import DayCounterConvention, SwapType, CompoundingType, Frequency, TimeUnit
from typing import Optional, Union


class InflationSwap(Trade):
    def __init__(self, 
                swap_type: SwapType,
                fixed_schedule,
                nominal,
                fixed_rate,
                fixed_day_counter,
                yoy_schedule,
                inflation_index,
                yoy_day_counter,
                spread = 0.0,
                gearing = 1.0,
                observation_lag: int = 0.0,
                observation_lag_period: TimeUnit = TimeUnit.Months,
                compounding_fixed: CompoundingType = CompoundingType.Simple,
                frequency_fixed: Frequency = Frequency.Annual
                ) -> None:
        super().__init__()
        self.swap_type: swap_type
        self.sign = -1 if swap_type.value=="Payer" else 1
        self.fixed_leg = FixedRateLeg(fixed_schedule,nominal,fixed_rate, fixed_day_counter, compounding_fixed, frequency_fixed)
        self.inflation_leg =  InflationLeg(yoy_schedule,nominal,inflation_index,yoy_day_counter, spread, gearing, observation_lag, observation_lag_period)

    @classmethod
    def from_legs(cls, 
                  swap_type: SwapType, 
                  leg1: FixedRateLeg, 
                  leg2: InflationLeg):

        return cls(swap_type,
                   leg1._schedule,
                   leg1._notionals,
                   leg1._rates,
                   leg1._daycounter,
                   leg2._schedule,
                   leg2._index,
                   leg2._daycounter,
                   leg2._spread,
                   leg2._gearing,
                   leg2._observation_lag,
                   leg2._observation_lag_period,
                   leg1._compounding,
                   leg1._frequency)