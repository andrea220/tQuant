import tensorflow as tf
from .interestrate import InterestRate
from ..utilities.utils import CompoundingType, Frequency, TimeUnit, BusinessDayConvention
from ..utilities.daycounter import DayCounter
from tensorflow.python.framework import dtypes
from datetime import date


class YoYInflationCurveSimple:
    def __init__(self, pillars: list[float], rates: list[float]):
        self.pillars = pillars 
        self.rates = tf.Variable(rates, dtype=dtypes.float64)     # tensor list

    def adjust_date(self, 
                    value_date: date, 
                    frequency: Frequency) -> date:

        if frequency == Frequency.Daily:
            diff_date = value_date
        elif frequency == Frequency.Monthly:
            diff_date = date(value_date.year,value_date.month,1)
        elif frequency == Frequency.Bimonthly:
            month = round(value_date.month/2) * 2 - 2
            diff_date = date(value_date.year,month,1)
        elif frequency == Frequency.Quarterly:
            month = round(value_date.month/3) * 3 - 2
            diff_date = date(value_date.year,month,1)
        elif frequency == Frequency.EveryFourthMonth:
            month = round(value_date.month/4) * 4 - 3
            diff_date = date(value_date.year,month,1)
        elif frequency == Frequency.Semiannual:
            month = round(value_date.month/6) * 6 - 5
            diff_date = date(value_date.year,month,1)
        elif frequency == Frequency.Annual:
            year = value_date.year - 1
            diff_date = date(year,value_date.month,1)
        return diff_date

    def rate(self, 
             d1, 
             dc: DayCounter, 
             evaluation_date,
             calendar,
             observation_lag: int = 0,
             observation_lag_period: TimeUnit = TimeUnit.Months,
             bdc: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
             frequency: Frequency = Frequency.Monthly
             ):
        if observation_lag != 0:
            d1 = calendar.advance(d1, -observation_lag,observation_lag_period,bdc)

        d1 = self.adjust_date(d1,frequency)
        
        term = dc.day_count(evaluation_date,d1)
        if term <= self.pillars[0]:
            first_df = self.rates[0]
            return first_df
        if term >= self.pillars[-1]:
            last_df =  self.rates[-1]
            return last_df
        for i in range(0, len(self.pillars) - 1):
            if term < self.pillars[i + 1]:
                dtr = 1 / (self.pillars[i + 1] - self.pillars[i])
                w1 = (self.pillars[i + 1] - term) * dtr
                w2 = (term - self.pillars[i]) * dtr
                r1 = w1 * self.rates[i]
                r2 = w2 * self.rates[i + 1]
                rm = r1 + r2
                return rm

    #TODO - Stabilire le convenzioni mensili
