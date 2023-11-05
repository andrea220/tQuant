from interface.cashflow import CashFlow
from abc import ABC, abstractmethod



class Coupon(CashFlow, ABC):

    @abstractmethod
    def __init__(self,
                 payment_date,
                 nominal,
                 accrual_start_date,
                 accrual_end_date,
                 ref_period_start,
                 ref_period_end,
                 ex_coupon_date):
        
        self._payment_date = payment_date
        self._nominal = nominal
        self._accrual_start_date = accrual_start_date
        self._accrual_end_date = accrual_end_date
        self._ref_period_start = ref_period_start
        self._ref_period_end = ref_period_end
        self._ex_coupon_date = ex_coupon_date
    
    @property
    def date(self):
        return self._payment_date

    @property
    def nominal(self):
        return self._nominal

    @property
    def accrual_start_date(self):
        return self._accrual_start_date

    @property
    def accrual_end_date(self):
        return self._accrual_end_date

    @property
    def ref_period_start(self):
        return self._ref_period_start

    @property
    def ref_period_end(self):
        return self._ref_period_end

    @property
    def ex_coupon_date(self):
        return self._ex_coupon_date

    @property
    def accrual_period(self):
        if self._accrual_period == None:
            return 0 #da implementare yearFraction(accrualstart, accrualend) vedi r44 coupon.cpp
        return self._accrual_period
    
    @property
    def accrual_days(self):
        return 0 #da implementare dayCount(accrualstart, accrualend) vedi r52 coupon.cpp
        
    @property
    def accrued_period(self):
        #da scrivere 
        pass 

    @property
    def accrued_days(self):
        #da scrivere 
        pass 

