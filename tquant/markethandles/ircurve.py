import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np

from .interestrate import InterestRate
from .interpolation import NaturalCubicSpline, LinearInterp
from ..timehandles.utils import CompoundingType, Frequency


class RateCurve:
    def __init__(self, pillars, rates, interp="LINEAR"):
        ''' 
        classe dummy per le curve
        '''
        self.pillars = pillars  # list
        self.rates = [tf.Variable(r, dtype=dtypes.float64) for r in rates]  # tensor list
        if interp == "LINEAR":
            self.interp = LinearInterp(self.pillars, np.array(self.rates))
        elif interp == "CUBIC_SPLINE":
            self.interp = NaturalCubicSpline(self.pillars, np.array(self.rates))
        else:
            raise ValueError("Wrong interpolation type")

    def get_rate_at_pillar(self, i: int):
        return self.rates[i]

    def set_rate(self, i: int, rate: float):
        self.rates[i] = rate
        self.interp.y[i] = rate

    def set_rates(self, rates: list[float]):
        self.rates = rates
        self.interp.y = rates

    def get_rate(self, term: float):
        r = self.interp.interpolate(term)
        return r

    def discount(self, term: float):
        r = self.get_rate(term)
        return tf.exp(-r * term)

    def inst_fwd(self, t: float):
        # time-step needed for differentiation
        dt = 0.01
        expr = - (tf.math.log(self.discount(t + dt)) - tf.math.log(self.discount(t - dt))) / (2 * dt)
        return expr

    def forward_rate(self,
                     d1,
                     d2,
                     daycounter,
                     evaluation_date):
        ''' 
        Calcola il tasso forward.
        '''
        tau = daycounter.year_fraction(d1, d2)
        df1 = self.discount(daycounter.year_fraction(evaluation_date, d1))
        df2 = self.discount(daycounter.year_fraction(evaluation_date, d2))
        return (df1 / df2 - 1) / tau


class DiscountCurveSimple:
    def __init__(self, pillars: list[float], rates: list[float]):
        self.pillars = pillars
        self.discount_factors = tf.Variable(rates, dtype=dtypes.float64)  # tensor list

    def discount(self, term: float):
        if term <= self.pillars[0]:
            first_df = self.discount_factors[0]
            return first_df
        if term >= self.pillars[-1]:
            last_df = self.discount_factors[-1]
            return last_df
        for i in range(0, len(self.pillars) - 1):
            if term < self.pillars[i + 1]:
                dtr = 1 / (self.pillars[i + 1] - self.pillars[i])
                w1 = (self.pillars[i + 1] - term) * dtr
                w2 = (term - self.pillars[i]) * dtr
                r1 = w1 * self.discount_factors[i]
                r2 = w2 * self.discount_factors[i + 1]
                rm = r1 + r2
                return rm

    def zero_rate(self, term: float):
        compound = 1.0 / self.discount(term)
        return InterestRate.implied_rate(compound, CompoundingType.Simple, Frequency.Annual, term).rate

    def inst_fwd(self, t: float):
        # time-step needed for differentiation
        dt = 0.01
        expr = - (tf.math.log(self.discount(t + dt)) - tf.math.log(self.discount(t - dt))) / (2 * dt)
        return expr

    def forward_rate(self,
                     d1,
                     d2,
                     daycounter,
                     evaluation_date):
        ''' 
        Calcola il tasso forward.
        '''
        tau = daycounter.year_fraction(d1, d2)
        df1 = self.discount(daycounter.year_fraction(evaluation_date, d1))
        df2 = self.discount(daycounter.year_fraction(evaluation_date, d2))
        return (df1 / df2 - 1) / tau
