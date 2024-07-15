from datetime import date 

from .fixedcoupon import FixedCoupon
from ..markethandles.interestrate import InterestRate
from ..utilities.utils import CompoundingType, Frequency, Settings
from ..utilities.daycounter import DayCounter
import pandas as pd



class FixedRateLeg:
    def __init__(self,
                 schedule: list[date],
                 notionals: list[float],
                 coupon_rates: list[float], 
                 daycounter: DayCounter,
                 compounding: CompoundingType = CompoundingType.Simple,
                 frequency: Frequency = Frequency.Annual) -> None:
        self._schedule = schedule
        self._notionals = notionals
        self._rates = coupon_rates
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
    def coupon_rates(self):
        return [InterestRate(r, self._daycounter, self._compounding, self._frequency) for r in self._rates]

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
# d1,
#                               nominal,
#                               accrual_start_date= evaluation_date,
#                               accrual_end_date= d1,
#                               ref_period_start= evaluation_date,
#                               ref_period_end= d1,
#                               r= 0.02,
#                               daycounter = daycounter
            leg.append(FixedCoupon(payment_date,
                                    nom,
                                    period_start_date,
                                    payment_date,
                                    period_start_date,
                                    payment_date,
                                    r,
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

    # def npv(self, discount_curve, evaluation_date: date):
    #     ''' 
    #     Calcola l'npv data una gamba qualsiasi
    #     Si potrebbe eliminare la data di valutazione in quanto recuperabile come prima data delle curva in input
    #     '''
    #     # TODO da spostare in pricingengine
    #     if len(self.leg_flows()) == 0:
    #         return 0
    #     npv = 0
    #     for i in range(0, len(self.leg_flows())):
    #         cf = self.leg_flows()[i]
    #         if not cf.has_occurred(evaluation_date):
    #             dt = cf.date - evaluation_date
    #             dt_year_fraction = dt.days/365
    #             npv += cf.amount * discount_curve.discount(dt_year_fraction)
    #     return npv