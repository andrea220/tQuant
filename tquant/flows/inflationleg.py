from datetime import date 

# from ..markethandles.interestrate import InterestRate
from .inflationcoupon import InflationCoupon
from ..index.inflationindex import InflationIndex
from ..utilities.utils import Settings, CompoundingType, Frequency, SwapType, TimeUnit
from ..utilities.daycounter import DayCounter
import pandas as pd


class InflationLeg:
    def __init__(self,
                 schedule: list[date],
                 notionals: list[float],
                 index: InflationIndex,
                 daycounter: DayCounter,
                 spread: float = 0.0,
                 gearing: float = 1.0,
                 observation_lag: int = 0,
                 observation_lag_period: TimeUnit = TimeUnit.Months
                 ) -> None:
        self._schedule = schedule
        self._notionals = notionals
        self._index = index
        self._daycounter = daycounter
        self._observation_lag = observation_lag
        self._observation_lag_period = observation_lag_period
        self._spread = spread
        self._gearing = gearing
  
    @property
    def schedule(self):
        return self._schedule
    
    @property
    def notionals(self):
        return self._notionals
    
    @property
    def index(self):
        return self._index
    
    @property
    def daycounter(self):
        return self._daycounter
    
    def leg_flows(self):
        ''' 
        Define the leg as a list of FixedCoupon objects
        TBD: definire bene tutti gli accrual 
        '''
        leg = []
        for i in range(1, len(self._schedule)):
            period_start_date = self._schedule[i-1]
            payment_date = self.schedule[i]
            nom = self._notionals[i-1]

            if isinstance(self._spread,(float,int)):
                spread = self._spread
            else:
                spread = self._spread[i-1]

            if isinstance(self._gearing,(float,int)):
                gearing = self._gearing
            else:
                gearing = self._gearing[i-1]

            leg.append(InflationCoupon(payment_date,
                                    nom,
                                    period_start_date,
                                    payment_date,
                                    self._index,
                                    period_start_date,
                                    payment_date,
                                    self._daycounter,
                                    spread=spread,
                                    gearing=gearing,
                                    payment_lag=self._observation_lag,
                                    payment_lag_period=self._observation_lag_period
                                    )
                            )
        return leg
    
    def display_flows(self):
        flows = self.leg_flows()
        leg_display = pd.DataFrame()
        for i in range(len(flows)):
            coupon_flow = flows[i].display()
            leg_display = pd.concat([leg_display, coupon_flow], axis = 0)
        return leg_display