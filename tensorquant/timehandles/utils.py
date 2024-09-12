from enum import Enum
from datetime import date
import re


class Settings:
    """
    A class to represent general settings
    """

    evaluation_date = date.today()


class TimeUnit(Enum):
    """
    Enumeration of time-units.
    """

    Days = "Days"
    Weeks = "Weeks"
    Months = "Months"
    Years = "Years"

    def __str__(self):
        return self.value


def decode_term(term: str):
    match = re.match(r"(\d+)([A-Za-z]+)", term)
    if match:
        period = int(match.group(1))
        unit = match.group(2)
    else:
        raise ValueError("Wrong term: " + term)
    if unit == "d" or unit == "D":
        time_unit = TimeUnit.Days
    elif unit == "w" or unit == "W":
        time_unit = TimeUnit.Weeks
    elif unit == "m" or unit == "M":
        time_unit = TimeUnit.Months
    elif unit == "y" or unit == "Y":
        time_unit = TimeUnit.Years
    else:
        raise ValueError
    return period, time_unit


class DayCounterConvention(Enum):
    Actual360 = "Actual360"
    Actual365 = "Actual365"
    Thirty360 = "Thirty360"
    Thirty360E = "Thirty360E"
    ActualActual = "ActualActual"

    def __str__(self):
        return self.value


class BusinessDayConvention(Enum):
    """
    Enumeration of business-day conventions.
    """

    Following = "Following"
    ModifiedFollowing = "Modified Following"
    HalfMonthModifiedFollowing = "Half-Month Modified Following"
    Preceding = "Preceding"
    ModifiedPreceding = "Modified Preceding"
    Unadjusted = "Unadjusted"
    Nearest = "Nearest"

    def __str__(self):
        return self.value


class CompoundingType(Enum):
    Compounded = "compounded"
    Simple = "simple"
    Continuous = "continuous"

    def __str__(self):
        return self.value


class Frequency(Enum):
    NoFrequency = -1
    Once = 0
    Annual = 1
    Semiannual = 2
    EveryFourthMonth = 3
    Quarterly = 4
    Bimonthly = 6
    Monthly = 12
    EveryFourthWeek = 13
    Biweekly = 26
    Weekly = 52
    Daily = 365
    OtherFrequency = 999
