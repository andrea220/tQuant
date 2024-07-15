from datetime import date 

from .defaultcoupon import DefaultCoupon
from ..markethandles.interestrate import InterestRate
from ..utilities.utils import CompoundingType, Frequency, Settings
from ..utilities.daycounter import DayCounter
import tensorflow as tf
import pandas as pd
from datetime import timedelta



class DefaultLeg:
    def __init__(self,
                 schedule: list[date],
                 notionals: list[float],
                 recovery: list[float], 
                 daycounter: DayCounter,
                 compounding: CompoundingType = CompoundingType.Simple,
                 frequency: Frequency = Frequency.Annual) -> None:
        self._schedule = schedule
        self._notionals = notionals
        self._recovery = recovery
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
    def recovery(self):
        return self._recovery     

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
            intermediate_date = period_start_date + timedelta(days=1)
            nom = self._notionals[i-1]
            recovery = self._recovery
            while intermediate_date <= self.schedule[i]:
                payment_date = intermediate_date           
                leg.append(DefaultCoupon(payment_date,
                                        nom,
                                        recovery,
                                        period_start_date,
                                        payment_date,
                                        period_start_date,
                                        payment_date,
                                        self._daycounter)
                                        )
                period_start_date = intermediate_date
                intermediate_date = period_start_date + timedelta(days=1)
        return leg
    
    def display_flows(self):
        flows = self.leg_flows()
        leg_display = pd.DataFrame()
        for i in range(len(flows)):
            coupon_flow = flows[i].display()
            leg_display = pd.concat([leg_display, coupon_flow], axis = 0)
        return leg_display