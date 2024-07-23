from datetime import date 
from .coupon import Coupon
from ..markethandles.interestrate import InterestRate
from ..timehandles.utils import CompoundingType, Frequency, Settings
from ..timehandles.daycounter import DayCounter
import pandas as pd

class PremiumCoupon(Coupon):

    def __init__(self,
                 payment_date: date,
                 nominal: float,
                 accrual_start_date: date,
                 accrual_end_date: date,
                 ref_period_start: date,
                 ref_period_end: date,
                 spread: float,
                 daycounter: DayCounter
                 ):
        super().__init__(payment_date, nominal, daycounter, accrual_start_date, accrual_end_date, ref_period_start, ref_period_end)
#        self._spread = InterestRate(spread, daycounter, CompoundingType.Simple, Frequency.Annual)
        self._spread = spread
        self._daycounter = daycounter
        
    @property
    def spread(self):
        return self._spread
    
    @property
    def day_counter(self):
        return self._daycounter
    
    def display(self):
        coupon_display = pd.DataFrame([self.ref_period_start,
                                        self.ref_period_end,
                                        self.date,
                                        self._nominal,
                                        self.accrual_period,
                                        self._daycounter.day_counter_convention.name,
                                        self._spread.rate,
                                        self.amount
                                        ]).T

        coupon_display.columns = ['start_period',
                                'end_period',
                                'payment_date',
                                'notional',
                                'accrual',
                                'day_counter',
                                'spread',
                                'amount'
                                ]
        return coupon_display
    
    @property
    def amount(self)-> float:
        self._amount = self.nominal * self.survival_probability * (self._spread.compound_factor(self.accrual_start_date,
                                                                    self.accrual_end_date,
                                                                    self.ref_period_start,
                                                                    self.ref_period_end,
                                                                    ) - 1)
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
            return self.nominal * self.survival_probability * (self._spread.compound_factor(self.accrual_start_date,
                                                            min(d, self.accrual_end_date),
                                                            self.ref_period_start,
                                                            self.ref_period_end
                                                            ) - 1)

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
