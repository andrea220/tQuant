import datetime 
from interface.coupon import Coupon
from structures.compounding import *
from structures.interestrate import InterestRate
from utilities.time import *
import pandas as pd


class FixedCoupon(Coupon):
    ''' 
    Concrete Fixed Coupon object. 
    
    '''
    def __init__(self,
                 payment_date: datetime.date,
                 nominal: float,
                 accrual_start_date: datetime.date,
                 accrual_end_date: datetime.date,
                 ref_period_start: datetime.date,
                 ref_period_end: datetime.date,
                 ex_coupon_date: datetime.date,
                 r,
                 day_counter = None
                 ):
        super().__init__(payment_date, nominal, accrual_start_date, accrual_end_date, ref_period_start, ref_period_end, ex_coupon_date)
        self._rate = InterestRate(r, Compounding.Simple, Frequency.Annual)
        self._day_counter = day_counter
        
    @property
    def rate(self):
        return self._rate
    
    @property
    def day_counter(self):
        return self._day_counter
    
    @property
    def amount(self)-> float:
        ''' 
        Da testare funzione _rate.compoundFactor (versione overloaded su file interestrate.hpp)
        '''
        self._amount = self.nominal * (self._rate.compoundFactor(self.accrual_start_date,
                                                                 self.accrual_end_date,
                                                                 self.ref_period_start,
                                                                 self.ref_period_end
                                                                 ) - 1)
        return self._amount
    
    @property
    def accruedAmount(self, d: datetime):
        if d <= self.accrual_start_date or d > self._payment_date:
            return 0
        else:
            return self.nominal * (self._rate.compoundFactor(self.accrual_start_date,
                                                            min(d, self.accrual_end_date),
                                                            self.ref_period_start,
                                                            self.ref_period_end
                                                            ) - 1)


class FixedRateLeg:
    def __init__(self,
                 schedule: Schedule,
                 notionals: list[float],
                 coupon_rates: list[float], # oppure singolo tasso da implementare
                 compounding: Compounding = Compounding.Simple,
                 frequency: Frequency = Frequency.Annual) -> None:
        self._schedule = schedule
        self._notionals = notionals
        self._rates = coupon_rates
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
    def coupon_rates(self):
        return [InterestRate(r,  self._compounding, self._frequency) for r in self._rates]

    @property
    def payment_adjustment(self):
        return self._payment_adjustment 
    
    @payment_adjustment.setter
    def payment_adjustment(self, payment_adjustment: float):
        self._payment_adjustment = payment_adjustment


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
            r = self._rates[i-1]

            leg.append(FixedCoupon(payment_date,
                                    nom,
                                    period_start_date,
                                    payment_date,
                                    period_start_date,
                                    payment_date,
                                    period_start_date,
                                    r,
                                    None)
                            )
        return leg
    
    def display_flows(self):
        flows = self.leg_flows()
        period_start_date = []
        period_end_date = []
        for i in range(len(flows)):
            period_start_date.append(self.schedule[i])
            period_end_date.append(self.schedule[i+1])
        amounts = [flows[t].amount for t in range(len(flows))]
        leg_overview = {}
        leg_overview['period_start_date'] = period_start_date
        leg_overview['period_end_date'] = period_end_date
        leg_overview['notional'] = self.notionals
        leg_overview['coupon_rate'] = self._rates
        leg_overview['amounts'] = amounts
        return pd.DataFrame(leg_overview)

    def npv(self, discount_curve, evaluation_date: datetime):
        ''' 
        Calcola l'npv data una gamba qualsiasi
        Si potrebbe eliminare la data di valutazione in quanto recuperabile come prima data delle curva in input
        '''
        if len(self.leg_flows()) == 0:
            return 0
        npv = 0
        for i in range(0, len(self.leg_flows())):
            cf = self.leg_flows()[i]
            if not cf.has_occurred(evaluation_date):
                dt = cf.date - evaluation_date
                dt_year_fraction = dt.days/365
                npv += cf.amount * discount_curve.discount(dt_year_fraction)
        return npv