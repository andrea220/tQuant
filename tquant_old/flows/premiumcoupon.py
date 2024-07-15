from datetime import date 
from ..interface.coupon import Coupon
from ..markethandles.interestrate import InterestRate
from ..utilities.utils import CompoundingType, Frequency, Settings
from ..utilities.daycounter import DayCounter
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
