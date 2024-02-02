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
    
    def display(self, 
                disc_curve,
                evaluation_date):
        Settings.evaluation_date = evaluation_date

        if not self.has_occurred(evaluation_date):
            dt = self.date - evaluation_date
            dt_year_fraction = dt.days/365
            disc_factor = (disc_curve.discount(dt_year_fraction)).numpy()
            amount = self.amount
            coupon_npv = amount * disc_factor
            is_expired = False
        else:
            coupon_npv = 0
            disc_factor = 0
            amount = 0
            is_expired = True

        coupon_display = pd.DataFrame([self.ref_period_start,
                                        self.ref_period_end,
                                        self.date,
                                        self._nominal,
                                        self.accrual_period,
                                        self._day_counter,
                                        amount,
                                        disc_factor,
                                        coupon_npv,
                                        is_expired
                                        ]).T

        coupon_display.columns = ['start_period',
                                'end_period',
                                'payment_date',
                                'notional',
                                'accrual',
                                'day_counter',
                                'amount',
                                'discount_factor',
                                'coupon_pv',
                                'is_expired'
                                ]
        return coupon_display
    
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
    def accrual_period(self):
        return (self.accrual_end_date - self.accrual_start_date).days / 365
    
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
    
    def display_flows(self,
                      disc_curve,
                      evaluation_date):
        flows = self.leg_flows()
        leg_display = pd.DataFrame()
        for i in range(len(flows)):
            coupon_flow = flows[i].display(disc_curve,
                                           evaluation_date)
            leg_display = pd.concat([leg_display, coupon_flow], axis = 0)
        return leg_display

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