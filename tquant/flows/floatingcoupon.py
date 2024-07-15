from datetime import date, timedelta
from ..interface.coupon import Coupon
# from ..markethandles.interestrate import InterestRate
from ..utilities.utils import Settings
from ..utilities.daycounter import DayCounter
from ..index.curverateindex import IborIndex
import pandas as pd

        
class FloatingCoupon(Coupon):
    ''' 
    Concrete Floating Coupon object. 
    '''
    def __init__(self,
                 payment_date: date,
                 nominal: float,
                 accrual_start_date: date,
                 accrual_end_date: date,
                 index: IborIndex, 
                 gearing: float, # multiplicative coefficient of the index fixing
                 spread: float, # fixed spread
                 ref_period_start: date,
                 ref_period_end: date,
                 daycounter: DayCounter,
                 is_in_arrears: bool = False, #TODO da implementare convexity adjustments
                 fixing_days: int = None,
                 ): 
        super().__init__(payment_date, nominal, daycounter, accrual_start_date, accrual_end_date, ref_period_start, ref_period_end)
        self._day_counter = daycounter
        self._fixing_days = fixing_days 
        self._index = index
        self._gearing = gearing
        self._spread = spread
        self._is_in_arrears = is_in_arrears

    @property
    def day_counter(self):
        return self._day_counter
    
    @property
    def fixing_days(self):
        if self._fixing_days is None:
            if self.index is not None:
                return self.index.fixing_days
            else:
                return 0
    
    @property
    def index(self):
        return self._index
            
    @property
    def is_in_arrears(self):
        return self._is_in_arrears
    
    @property
    def fixing_date(self):
        if self.is_in_arrears:
            ref_date = self.accrual_end_date 
        else:
            ref_date = self.accrual_start_date
        return ref_date + timedelta(self.fixing_days)
        
    @property
    def accrual_period(self): 
        return self._daycounter.year_fraction(self.accrual_start_date, self.accrual_end_date)

    
    def display(self):

        coupon_display = pd.DataFrame([self.ref_period_start,
                                        self.ref_period_end,
                                        self.date,
                                        self._nominal,
                                        self.fixing_date,
                                        self._fixing_days,
                                        self._index.name,
                                        self.accrual_period,
                                        self.is_in_arrears,
                                        self._gearing,
                                        self._spread,
                                        self._daycounter.day_counter_convention.name
                                        ]).T

        coupon_display.columns = ['start_period',
                                'end_period',
                                'payment_date',
                                'notional',
                                'fixing_date',
                                'fixing_days',
                                'index',
                                'accrual',
                                'in_arrears',
                                'gearing',
                                'spread',
                                'day_counter'
                                ]
        return coupon_display
    
    def amount(self, coupon_pricer)-> float: #TODO creare il pricer
        ''' 
        Da testare funzione _rate.compoundFactor (versione overloaded su file interestrate.hpp)
        '''
        a = self.nominal * (self._gearing * self.rate(coupon_pricer) + self._spread) * self.accrual_period
        return a
    
    def rate(self, coupon_pricer): #TODO
        self._rate = coupon_pricer.forecasted_rate
        return coupon_pricer.forecasted_rate
       
    def accrued_amount(self, d: date):
        if d <= self.accrual_start_date or d > self._payment_date:
            return 0
        else:
            return self.nominal * (self._rate.compound_factor(self.accrual_start_date,
                                                            min(d, self.accrual_end_date),
                                                            self.ref_period_start,
                                                            self.ref_period_end
                                                            ) - 1)
        
    