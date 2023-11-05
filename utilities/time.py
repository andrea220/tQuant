from enum import Enum
from datetime import date

class Settings:
    evaluation_date = date.today()


class DayCounter:
    def __init__(self, name):
        self.name = name

class Schedule:
    def __init__(self):
        pass

class Calendar:
    def __init__(self):
        pass
   
class BusinessDayConvention(Enum):
    Following = 0


    