from interface.calendar import Calendar
from datetime import date, timedelta
from dateutil.easter import *


class TARGET(Calendar):
    def __init__(self) -> None:
        pass 

    def is_business_day(self, d: date):
        ''' 
        check se la data in input è un bd oppure no
        '''
        ny = d.day == 1 and d.month == 1
        em = d == easter(d.year) + timedelta(1)
        gf = d == easter(d.year) - timedelta(2)
        ld = d.day == 1 and d.month == 5
        c = d.day == 25 and d.month == 12
        cg = d.day == 26 and d.month == 12

        if self.is_weekend(d) or ny or gf or em or ld or c or cg:
            return False
        else:
            return True