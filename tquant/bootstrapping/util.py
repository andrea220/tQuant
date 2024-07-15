import datetime
import re

from tquant import TimeUnit, DayCounterConvention, DayCounter, RateCurve

import numpy as np


def decode_term(term: str) -> (int, TimeUnit):
    match = re.match(r'(\d+)([A-Za-z]+)', term)
    if match:
        period = int(match.group(1))
        unit = match.group(2)
    else:
        raise ValueError("Wrong term: " + term)
    if unit == 'd' or unit == 'D':
        time_unit = TimeUnit.Days
    elif unit == 'w' or unit == 'W':
        time_unit = TimeUnit.Weeks
    elif unit == 'm' or unit == 'M':
        time_unit = TimeUnit.Months
    elif unit == 'y' or unit == 'Y':
        time_unit = TimeUnit.Years
    else:
        raise ValueError
    return period, time_unit


def calculate_forward(as_of_date: datetime,
                      fc: RateCurve,
                      index_start: datetime,
                      index_end: datetime,
                      yf: float):
    act365 = DayCounterConvention.Actual365
    day_counter = DayCounter(act365)
    t1 = day_counter.year_fraction(as_of_date, index_start)
    t2 = day_counter.year_fraction(as_of_date, index_end)
    df1 = fc.discount(t1)
    df2 = fc.discount(t2)
    forward = (df1 / df2 - 1.0) / yf
    return np.float64(forward)
