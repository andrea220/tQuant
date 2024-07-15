from enum import Enum
from datetime import date

class Settings:
    evaluation_date = date.today()


class DayCounter:
    def __init__(self, name):
        self.name = name



class TimeUnit(Enum):
    Days = "Days"
    Weeks = "Weeks"
    Months = "Months"
    Years = "Years"

    def __str__(self):
        return self.value


class BusinessDayConvention(Enum):
    Following = "Following"
    ModifiedFollowing = "Modified Following"
    HalfMonthModifiedFollowing = "Half-Month Modified Following"
    Preceding = "Preceding"
    ModifiedPreceding = "Modified Preceding"
    Unadjusted = "Unadjusted"
    Nearest = "Nearest"

    def __str__(self):
        return self.value