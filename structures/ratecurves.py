import tensorflow as tf
from structures.interestrate import InterestRate
from structures.compounding import Compounding, Frequency


class RateCurveSimple:
    def __init__(self, pillars: list[float], rates: list[float]):
        self.pillars = pillars 
        self.rates = tf.Variable(rates, dtype=tf.float64)     # tensor list

    def discount(self, term: float):
        if term <= self.pillars[0]:
            nrt = -term * self.rates[0]
            df = tf.exp(nrt)
            return df
        if term >= self.pillars[-1]:
            nrt = -term * self.rates[-1]
            df = tf.exp(nrt)
            return df
        for i in range(0, len(self.pillars) - 1):
            if term < self.pillars[i + 1]:
                dtr = 1 / (self.pillars[i + 1] - self.pillars[i])
                w1 = (self.pillars[i + 1] - term) * dtr
                w2 = (term - self.pillars[i]) * dtr
                r1 = w1 * self.rates[i]
                r2 = w2 * self.rates[i + 1]
                rm = r1 + r2
                nrt = -term * rm
                df = tf.exp(nrt)
                return df
    
    def inst_fwd(self, t: float):
        # time-step needed for differentiation
        dt = 0.01    
        expr = - (tf.math.log(self.discount(t+dt))- tf.math.log(self.discount(t-dt)))/(2*dt)
        return expr
    

class DiscountCurveSimple:
    def __init__(self, pillars: list[float], rates: list[float]):
        self.pillars = pillars 
        self.discount_factors = tf.Variable(rates, dtype=tf.float64)     # tensor list

    def discount(self, term: float):
        if term <= self.pillars[0]:
            first_df = self.discount_factors[0]
            return first_df
        if term >= self.pillars[-1]:
            last_df =  self.discount_factors[-1]
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
        return InterestRate.impliedRate(compound, Compounding.Simple, Frequency.Annual, term).rate
    

