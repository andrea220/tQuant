from datetime import date 

# from ..markethandles.interestrate import InterestRate
from .floatingcoupon import FloatingCoupon
from ..index.curverateindex import IborIndex
from ..utilities.daycounter import DayCounter
import pandas as pd


class FloatingRateLeg:
    def __init__(self,
                 schedule: list[date],
                 notionals: list[float],
                 gearings: list[float],
                 spreads: list[float],
                 index: IborIndex,
                 daycounter: DayCounter
                 ) -> None:
        self._schedule = schedule
        self._notionals = notionals
        self._gearings = gearings
        self._spreads = spreads 
        self._index = index
        self._daycounter = daycounter
  
    @property
    def schedule(self):
        return self._schedule
    
    @property
    def notionals(self):
        return self._notionals
    
    @property
    def gearings(self):
        return self._gearings
    
    @property
    def spreads(self):
        return self._spreads
    
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

            leg.append(FloatingCoupon(payment_date,
                                    nom,
                                    period_start_date,
                                    payment_date,
                                    self._index,
                                    self._gearings[i-1],
                                    self._spreads[i-1],
                                    period_start_date,
                                    payment_date,
                                    self._daycounter
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

        
