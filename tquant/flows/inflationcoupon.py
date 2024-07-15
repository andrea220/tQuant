from datetime import date, timedelta
from ..interface.coupon import Coupon
# from ..markethandles.interestrate import InterestRate
from ..utilities.utils import Settings, TimeUnit, BusinessDayConvention, SwapType
from ..utilities.daycounter import DayCounter
from ..index.inflationindex import InflationIndex
from ..utilities.targetcalendar import Calendar, TARGET
import pandas as pd

#TODO mettere il gearing opzionale

class InflationCoupon(Coupon):
    ''' 
    Concrete Floating Coupon object. 
    '''
    def __init__(self,
                 payment_date: date,
                 nominal: float,
                 accrual_start_date: date,
                 accrual_end_date: date,
                 index: InflationIndex,
                 ref_period_start: date,
                 ref_period_end: date,
                 daycounter: DayCounter,
                 spread: float = 0.0,
                 gearing: float = 1.0,
                 payment_lag: int = 0.0,
                 payment_lag_period: TimeUnit = TimeUnit.Days,
                 calendar: Calendar = TARGET(),
                 bdc: BusinessDayConvention = BusinessDayConvention.Following,
                 is_in_arrears: bool = False, #TODO da implementare convexity adjustments
                 fixing_days: int = None,
                 ): 
        super().__init__(payment_date, nominal, daycounter, accrual_start_date, accrual_end_date, ref_period_start, ref_period_end)
        self._day_counter = daycounter
        self._fixing_days = fixing_days 
        self._index = index
        self._is_in_arrears = is_in_arrears
        self._payment_lag = payment_lag
        self._payment_lag_period = payment_lag_period
        self._calendar = calendar
        self._bdc = bdc
        self._spread = spread
        self._gearing = gearing

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
    def end_fixing_date(self):
        if self.is_in_arrears:
            ref_date = self.accrual_start_date 
        else:
            ref_date = self.accrual_end_date
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
                                        self._spread,
                                        self._gearing,
                                        self.accrual_period,
                                        self.is_in_arrears,
                                        self._daycounter.day_counter_convention.name
                                        ]).T

        coupon_display.columns = ['start_period',
                                'end_period',
                                'payment_date',
                                'notional',
                                'fixing_date',
                                'fixing_days',
                                'index',
                                'spread',
                                'gearing',
                                'accrual',
                                'in_arrears',
                                'day_counter'
                                ]
        return coupon_display
    
    def amount(self, coupon_pricer)-> float: #TODO creare il pricer
        ''' 
        Da testare funzione _rate.compoundFactor (versione overloaded su file interestrate.hpp)
        '''
        if self.rate(coupon_pricer) > 0:
            a = self.nominal * (self.rate(coupon_pricer)) * self.accrual_period
        else:
            a = 0
        return a
    
    def rate(self, coupon_pricer): #TODO
        self._rate = coupon_pricer.forecasted_rate
        return coupon_pricer.forecasted_rate
       
    def accrued_amount(self, coupon_pricer, d: date): #TODO
        if d <= self.accrual_start_date or d > self._payment_date:
            return 0
        else:
            if self.rate(coupon_pricer) > 0:
                return self.nominal * (self._rate.compound_factor(self.accrual_start_date,
                                                            min(d, self.accrual_end_date),
                                                            self.ref_period_start,
                                                            self.ref_period_end
                                                            ) - 1)
            else:
                return 0
        
    