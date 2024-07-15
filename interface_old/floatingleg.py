from datetime import datetime, timedelta
from interface_old.coupon import *
from index.curverateindex import * 
from utilities_old.time import Settings
import pandas as pd
 

        
class FloatingCoupon(Coupon):
    ''' 
    Concrete Floating Coupon object. 
    '''
    def __init__(self,
                 payment_date: datetime,
                 nominal: float,
                 accrual_start_date: datetime,
                 accrual_end_date: datetime,
                 index: IborIndex, 
                 gearing: float, # multiplicative coefficient of the index fixing
                 spread: float, # fixed spread
                 ref_period_start: datetime,
                 ref_period_end: datetime,
                 daycounter: DayCounter,
                 ex_coupon_date: datetime,
                 is_in_arrears: bool = False, # da implementare convexity adjustments
                 fixing_days: int = None,
                 ):
        super().__init__(payment_date, nominal, accrual_start_date, accrual_end_date, ref_period_start, ref_period_end, ex_coupon_date)
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
        return (self.accrual_end_date - self.accrual_start_date).days / 365

    
    def display(self, 
                disc_curve,
                term_structure,
                evaluation_date):
        Settings.evaluation_date = evaluation_date

        if not self.has_occurred(evaluation_date):
            dt = self.date - evaluation_date
            dt_year_fraction = dt.days/365
            disc_factor = (disc_curve.discount(dt_year_fraction)).numpy()
            amount = self.amount(term_structure)
            coupon_npv = amount * disc_factor
            fixing = self.index.fixing(self.fixing_date,
                                        term_structure)
            is_expired = False
        else:
            coupon_npv = 0
            disc_factor = 0
            fixing = 0
            amount = 0
            is_expired = True

        coupon_display = pd.DataFrame([self.ref_period_start,
                                        self.ref_period_end,
                                        self.date,
                                        self._nominal,
                                        self.fixing_date,
                                        self._fixing_days,
                                        self._index.name,
                                        fixing,
                                        self.accrual_period,
                                        self.is_in_arrears,
                                        self._gearing,
                                        self._spread,
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
                                'fixing_date',
                                'fixing_days',
                                'index',
                                'fixing',
                                'accrual',
                                'in_arrears',
                                'gearing',
                                'spread',
                                'day_counter',
                                'amount',
                                'discount_factor',
                                'coupon_pv',
                                'is_expired'
                                ]
        return coupon_display
    
    def amount(self, term_structure)-> float:
        ''' 
        Da testare funzione _rate.compoundFactor (versione overloaded su file interestrate.hpp)
        '''
        a = self.nominal * (self._gearing * self.index.fixing(self.fixing_date,
                                                                 term_structure) + self._spread) * self.accrual_period
        return a
       
    def accruedAmount(self, d: datetime):
        if d <= self.accrual_start_date or d > self._payment_date:
            return 0
        else:
            return self.nominal * (self._rate.compoundFactor(self.accrual_start_date,
                                                            min(d, self.accrual_end_date),
                                                            self.ref_period_start,
                                                            self.ref_period_end
                                                            ) - 1)
        
class FloatingRateLeg:
    def __init__(self,
                 schedule: list[date],
                 notionals: list[float],
                 gearings: list[float],
                 spreads: list[float],
                 index: IborIndex
                 ) -> None:
        self._schedule = schedule
        self._notionals = notionals
        self._gearings = gearings
        self._spreads = spreads 
        self._index = index
  
    @property
    def schedule(self):
        return self._schedule
    
    @property
    def notionals(self):
        return self._notionals


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

            leg.append(FloatingCoupon(payment_date,
                                    nom,
                                    period_start_date,
                                    payment_date,
                                    self._index,
                                    self._gearings[i-1],
                                    self._spreads[i-1],
                                    period_start_date,
                                    payment_date,
                                    None,
                                    period_start_date
                                    )
                            )
        return leg
    
    def display_flows(self,
                      disc_curve,
                      term_structure,
                      evaluation_date):
        flows = self.leg_flows()
        leg_display = pd.DataFrame()
        for i in range(len(flows)):
            coupon_flow = flows[i].display(disc_curve,
                                           term_structure,
                                           evaluation_date)
            leg_display = pd.concat([leg_display, coupon_flow], axis = 0)
        return leg_display

    def npv(self,
            discount_curve,
            term_structure,
            evaluation_date: datetime):
        ''' 
        Calcola l'npv data una gamba qualsiasi
        Si potrebbe eliminare la data di valutazione in quanto recuperabile come prima data delle curva in input
        '''
        Settings.evaluation_date = evaluation_date
        if len(self.leg_flows()) == 0:
            return 0
        npv = 0
        for i in range(0, len(self.leg_flows())):
            cf = self.leg_flows()[i]
            if not cf.has_occurred(evaluation_date):
                dt = cf.date - evaluation_date
                dt_year_fraction = dt.days/365
                npv += cf.amount(term_structure) * discount_curve.discount(dt_year_fraction)
        return npv
        
