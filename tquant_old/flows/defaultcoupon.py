from datetime import date, timedelta 
from ..interface.coupon import Coupon
from ..markethandles.interestrate import InterestRate
from ..utilities.utils import CompoundingType, Frequency, Settings
from ..utilities.interpolation import interpolate
from ..utilities.daycounter import DayCounter
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