from .cashflow import CashFlow
from abc import ABC, abstractmethod
from ..timehandles.daycounter import DayCounter
from datetime import date

class Coupon(CashFlow, ABC):

    @abstractmethod
    def __init__(self,
                 payment_date: date,
                 nominal: float,
                 daycounter: DayCounter,
                 accrual_start_date: date,
                 accrual_end_date: date,
                 ref_period_start: date,
                 ref_period_end: date):
        
        self._payment_date = payment_date
        self._nominal = nominal
        self._daycounter = daycounter
        self._accrual_start_date = accrual_start_date
        self._accrual_end_date = accrual_end_date
        self._ref_period_start = ref_period_start
        self._ref_period_end = ref_period_end
        # self._ex_coupon_date = ex_coupon_date
    
    @abstractmethod
    def rate(self):
        return 
    
    @property
    @abstractmethod
    def accrued_amount(self):
        #TODO da capire come gestire l'accrued amount
        return
    
    @property
    def date(self):
        ''' 
        payment date
        '''
        return self._payment_date

    @property
    def nominal(self):
        ''' 
        nominal
        '''
        return self._nominal
    
    @property
    def daycounter(self):
        return self._daycounter
    
    @property
    def accrual_start_date(self):
        ''' 
        period accrual start
        '''
        return self._accrual_start_date

    @property
    def accrual_end_date(self):
        ''' 
        period accrual end
        '''
        return self._accrual_end_date

    @property
    def ref_period_start(self):
        return self._ref_period_start

    @property
    def ref_period_end(self):
        return self._ref_period_end

    @property
    def accrual_period(self):
        #TODO da validare -- vedi riga 44 coupon.cpp
        return self._daycounter.year_fraction(self.accrual_start_date,
                                                 self.accrual_end_date
                                            )  
    
    @property
    def accrual_days(self):
        #TODO da validare -- da implementare dayCount(accrualstart, accrualend) vedi r52 coupon.cpp
        return self._daycounter.day_count(self.accrual_start_date,
                                            self.accrual_end_date
                                            )
        
    @property
    def accrued_period(self, d: date):
        #TODO da validare -- riga 57 coupon.cpp
        return self._daycounter.year_fraction(self.accrual_start_date,
                                            min(d, self.accrual_end_date)
                                            ) 

    @property
    def accrued_days(self, d: date):
        #TODO da validare -- riga 71 coupon.cpp
        return self._daycounter.day_count(self.accrual_start_date,
                                            min(d, self.accrual_end_date)
                                            )

