import tensorflow as tf
from datetime import datetime 
from structures_old.ratecurves import *


class SwapSimple:
    def __init__(self, notional: float, fixed_rate: float, payer: bool, schedule: list[datetime]):
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.payer = payer
        self.schedule = schedule

    def price(self, aod: datetime, rc: RateCurveSimple):
        deltas = [date - aod for date in self.schedule]
        terms = [delta.days / 365.0 for delta in deltas]
        floating_leg_pv = rc.discount(terms[0]) - rc.discount(terms[-1])
        fixed_leg_pv = tf.zeros_like(floating_leg_pv)
        for i in range(1, len(self.schedule)):
            fixed_leg_pv = fixed_leg_pv + self.fixed_rate * rc.discount(terms[i]) * (terms[i] - terms[i - 1])
        if self.payer:
            return self.notional * (floating_leg_pv - fixed_leg_pv)
        else:
            return self.notional * (fixed_leg_pv - floating_leg_pv)

