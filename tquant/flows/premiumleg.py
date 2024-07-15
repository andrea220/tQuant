from datetime import date 

from .premiumcoupon import PremiumCoupon
from ..markethandles.interestrate import InterestRate
from ..utilities.utils import CompoundingType, Frequency, Settings
from ..utilities.daycounter import DayCounter
import pandas as pd



class PremiumLeg:
    def __init__(self,
                 schedule: list[date],
                 notionals: list[float],
                 spreads: list[float], 
                 daycounter: DayCounter,
                 compounding: CompoundingType = CompoundingType.Simple,
                 frequency: Frequency = Frequency.Annual) -> None:
        self._schedule = schedule
        self._notionals = notionals
        self._spreads = spreads
        self._daycounter = daycounter
        self._compounding = compounding
        self._frequency = frequency
  
    @property
    def schedule(self):
        return self._schedule
    
    @property
    def notionals(self):
        return self._notionals
    
    @property
    def frequency(self):
        return self._frequency
    
    @property
    def compounding(self):
        return self._compounding

    @property
    def spreads(self):
        return [InterestRate(r, self._daycounter, self._compounding, self._frequency) for r in self._spreads]   

    @property
    def payment_adjustment(self):
        return self._payment_adjustment 
    
    @payment_adjustment.setter
    def payment_adjustment(self, payment_adjustment: float):
        self._payment_adjustment = payment_adjustment


    def leg_flows(self):
        ''' 
        Define the leg as a list of PremiumCoupon objects
        TBD: definire bene tutti gli accrual 
        '''
        leg = []
        for i in range(1, len(self._schedule)):
            period_start_date = self._schedule[i-1]
            payment_date = self.schedule[i]
            nom = self._notionals[i-1]
            s = self._spreads[i-1]

            leg.append(PremiumCoupon(payment_date,
                                    nom,
                                    period_start_date,
                                    payment_date,
                                    period_start_date,
                                    payment_date,
                                    s,
                                    self._daycounter)
                            )
        return leg
    
    def display_flows(self):
        flows = self.leg_flows()
        leg_display = pd.DataFrame()
        for i in range(len(flows)):
            coupon_flow = flows[i].display()
            leg_display = pd.concat([leg_display, coupon_flow], axis = 0)
        return leg_display
