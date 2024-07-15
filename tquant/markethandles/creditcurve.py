import tensorflow as tf
from .interestrate import InterestRate
from ..utilities.utils import CompoundingType, Frequency
from ..utilities.daycounter import DayCounter
from ..utilities.interpolation import interpolate
from tensorflow.python.framework import dtypes
from datetime import date, timedelta

#ndp = non default probabilities
class SurvivalProbabilityCurve:
    def __init__(self, 
                 pillars, 
                 survival_probabilities):
        self.pillars = pillars # list
        self.survival_probabilities = tf.Variable(survival_probabilities, dtype=dtypes.float64)

    def survival_probability(self, 
                             d1, 
                             day_counter: DayCounter,
                             evaluation_date
                             ):
        d2 = day_counter.year_fraction(evaluation_date, d1)
        return interpolate(self.pillars, self.survival_probabilities, d2, logaritm = True)
