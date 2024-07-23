from datetime import date, timedelta 
from .coupon import Coupon
from ..timehandles.utils import CompoundingType, Frequency, Settings
# from ..numericalhandles.interpolation import interpolate
from ..timehandles.daycounter import DayCounter
import pandas as pd

class DefaultCoupon(Coupon):
    ''' 
    Concrete Fixed Coupon object. 
    
    '''
    def __init__(self,
                 payment_date: date,
                 nominal: float,
                 recovery: float,
                 accrual_start_date: date,
                 accrual_end_date: date,
                 ref_period_start: date,
                 ref_period_end: date,
                 daycounter: DayCounter
                 ):
        super().__init__(payment_date, nominal, daycounter, accrual_start_date, accrual_end_date, ref_period_start, ref_period_end)
        self._daycounter = daycounter
        self._recovery = recovery


    @property
    def recovery(self):
        return self._recovery
    
    @property
    def day_counter(self):
        return self._daycounter

               
    def display(self):
        coupon_display = pd.DataFrame([self.ref_period_start,
                                        self.ref_period_end,
                                        self.date,
                                        self._nominal,
                                        self._recovery,
                                        self.accrual_period,
                                        self._daycounter.day_counter_convention.name#,
                                        #self.amount
                                        ]).T

        coupon_display.columns = ['start_period',
                                'end_period',
                                'payment_date',
                                'notional',
                                'recovery',
                                'accrual',
                                'day_counter'#,
                                #'amount'
                                ]
        return coupon_display
    
    @property
    def amount(self)-> float:
        d1 = self.accrual_start_date
        amount = 0
        while d1 <= self._accrual_end_date:
            d2 = d1 + timedelta(days=1)
            amount += self.nominal * (1 - self.recovery) * (self.survival_prob_interp(d1) - self.survival_prob_interp(d2)) * self.daycounter.year_fraction(d1, d2)
            d1 = d2
        self._amount = amount
        return self._amount
    
    @property
    def accrual_period(self):
        return self._daycounter.year_fraction(self.accrual_start_date, self.accrual_end_date)
        
    def rate(self, coupon_pricer): #TODO
        self._rate = coupon_pricer.forecasted_rate
        return coupon_pricer.forecasted_rate
    
    def accrued_amount(self, d: date): 
        if d <= self.accrual_start_date or d > self._payment_date:
            return 0
        else:
            d1 = self.accrual_start_date
            amount = 0
            while d1 <= min(self._accrual_end_date, d):
                d2 = d1 + timedelta(days=1)
                amount += self.nominal * (1 - self.recovery) * (self.survival_prob_interp(d1) - self.survival_prob_interp(d2)) * self.daycounter.year_fraction(d1, d2)
                d1 = d2
            return amount
        

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