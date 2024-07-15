from enum import Enum
from datetime import date


class Settings:
    """ 
    A class to represent general settings
    """
    evaluation_date = date.today()


# class DayCounter:
#     def __init__(self, name):
#         self.name = name

class Position(Enum):
    Long = 1
    Short = -1


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
    Continuous = "continuos"

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


class InterpolationType(Enum):
    Linear = 'linear'
    Quadratic = "quadratic"
    Cubic = "cubic"

    def __str__(self):
        return self.value


class SwapType(Enum):
    """
    Enumeration of time-units.
    """
    Payer = "Payer"
    Receiver = "Receiver"

    def __str__(self):
        return self.value


def string_to_enum(enum_class, enum_str):
    try:
        return enum_class.__members__[enum_str]
    except KeyError:
        raise ValueError(f"'{enum_str}' is not a valid member of {enum_class.__name__}")
    
class Currency(Enum):
    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    CHF = "CHF"
    AUD = "AUD"

    def __str__(self):
        return self.value
