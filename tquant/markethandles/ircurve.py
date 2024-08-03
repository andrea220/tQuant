import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np

from .interestrate import InterestRate
from ..numericalhandles.interpolation import NaturalCubicSpline, LinearInterp
from ..timehandles.utils import CompoundingType, Frequency


# class RateCurve:
#     def __init__(self, pillars, rates):
#         self.pillars = pillars # list
#         self.rates = [tf.Variable(r, dtype=dtypes.float64) for r in rates]     # tensor list

#     def discount(self, term):
#         if term <= self.pillars[0]:
#             nrt = -term * self.rates[0]
#             df = tf.exp(nrt)
#             return df
#         if term >= self.pillars[-1]:
#             nrt = -term * self.rates[-1]
#             df = tf.exp(nrt)
#             return df
#         for i in range(0, len(self.pillars) - 1):
#             if term < self.pillars[i + 1]:
#                 dtr = 1 / (self.pillars[i + 1] - self.pillars[i])
#                 w1 = (self.pillars[i + 1] - term) * dtr
#                 w2 = (term - self.pillars[i]) * dtr
#                 r1 = w1 * self.rates[i]
#                 r2 = w2 * self.rates[i + 1]
#                 rm = r1 + r2
#                 nrt = -term * rm
#                 df = tf.exp(nrt)
#                 return df
    
#     def forward_rate(self,
#                      d1,
#                      d2,
#                      daycounter,
#                      evaluation_date):
#         ''' 
#         Calcola il tasso forward.
#         '''
#         tau = daycounter.year_fraction(d1, d2)
#         df1 = self.discount(daycounter.year_fraction(evaluation_date, d1))
#         df2 = self.discount(daycounter.year_fraction(evaluation_date, d2))
#         return (df1 / df2 - 1) / tau
    
#     def inst_fwd(self, t: float):
#         # time-step needed for differentiation
#         dt = 0.01
#         expr = - (tf.math.log(self.discount(t + dt)) - tf.math.log(self.discount(t - dt))) / (2 * dt)
#         return expr

class RateCurve:
    def __init__(self, pillars, rates, interp: str):
        self.pillars = pillars # list
        self.rates = [tf.Variable(r, dtype=dtypes.float64) for r in rates]     # tensor list
        self.interpolation_type = interp
        if self.interpolation_type == "LINEAR":
            self.interp = LinearInterp(self.pillars, self.rates)
        else:
            raise ValueError("Wrong interpolation type")
            
    def discount(self, term):
        if term <= self.pillars[0]:
            nrt = -term * self.rates[0]
            df = tf.exp(nrt)
            return df
        if term >= self.pillars[-1]:
            nrt = -term * self.rates[-1]
            df = tf.exp(nrt)
            return df
        else:
            nrt = -term * self.interp.interpolate(term)
            df = tf.exp(nrt)
            return df
    
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
    
    def inst_fwd(self, t: float):
        # time-step needed for differentiation
        dt = 0.01
        expr = - (tf.math.log(self.discount(t + dt)) - tf.math.log(self.discount(t - dt))) / (2 * dt)
        return expr
    
    def set_rates(self, rates): 
        self.rates = [tf.Variable(r, dtype=dtypes.float64) for r in rates]
        self.interp.y = self.rates
        # if self.interpolation_type == "LINEAR":
        #     self.interp = LinearInterp(self.pillars, self.rates)
        # else:
        #     raise ValueError("Wrong interpolation type")

class LinearInterpAP:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def interpolate(self, x_new):
        if x_new <= self.x[0]:
            return self.y[0]
        elif x_new >= self.x[-1]:
            return self.y[-1]
        else:
            i = np.searchsorted(self.x, x_new) - 1
            w1 = (self.x[i + 1] - x_new) / (self.x[i + 1] - self.x[i])
            w2 = 1.0 - w1
            r1 = self.y[i]
            r2 = self.y[i + 1]
            return w1 * r1 + w2 * r2

def compute_coefficients(x: np.ndarray, y: np.ndarray):
    n = len(x)
    h = np.diff(x)
    alpha = [3 * (y[i + 1] - y[i]) / h[i] - 3 * (y[i] - y[i - 1]) / h[i - 1] for i in range(1, n - 1)]

    A = np.zeros((n, n))
    b = np.zeros(n)

    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = alpha[i - 1]

    c = np.linalg.solve(A, b)

    a = y[:-1]
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c[:-1], d


class NaturalCubicSplineAP:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.a, self.b, self.c, self.d = compute_coefficients(x, y)

    def interpolate(self, x_new: float):
        i = np.searchsorted(self.x, x_new) - 1
        if i < 0:
            i = 0
        if i >= len(self.x) - 1:
            i = len(self.x) - 2

        dx = x_new - self.x[i]
        yi = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2 + self.d[i] * dx ** 3

        return yi

class RateCurveAP:
    def __init__(self, pillars, rates, interp="LINEAR"):
        ''' 
        classe dummy per le curve
        '''
        self.pillars = pillars  # list
        self.rates = [tf.Variable(r, dtype=dtypes.float64) for r in rates]  # tensor list
        if interp == "LINEAR":
            self.interp = LinearInterpAP(self.pillars, np.array(self.rates))
        elif interp == "CUBIC_SPLINE":
            self.interp = NaturalCubicSplineAP(self.pillars, np.array(self.rates))
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
