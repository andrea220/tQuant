from abc import ABC, abstractmethod
from datetime import date 
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from utilities_old.time import TimeUnit, BusinessDayConvention
from calendar import monthrange

class Calendar(ABC):
    ''' 
    abstract base class for calendar implementations
    '''

    def advance(self,
                start_date: date,
                period: int,
                time_unit: TimeUnit,
                convention: BusinessDayConvention,
                end_of_month: bool = False):
        ''' 
        Advances the given date as specified by the given period and
            returns the result.
        '''
        
        if start_date is None:
            raise ValueError("null date")

        if period == 0:
            return self.adjust(start_date, convention)

        d1 = start_date

        if time_unit == TimeUnit.Days:
            while period > 0:
                d1 += timedelta(days=1)
                while self.is_holiday(d1):
                    d1 += timedelta(days=1)
                period -= 1

            while period < 0:
                d1 -= timedelta(days=1)
                while self.is_holiday(d1):
                    d1 -= timedelta(days=1)
                period += 1

        elif time_unit == TimeUnit.Weeks:
            d1 += relativedelta(weeks = period) 
            d1 = self.adjust(d1, convention)

        elif time_unit == TimeUnit.Months:
            d1 += relativedelta(months = period) 
            d1 = self.adjust(d1, convention)

            if end_of_month and self.is_end_of_month(start_date):
                return date(d1.year, d1.month, self.end_of_month(d1))
            
        else:  # Months or Years
            d1 += relativedelta(years = period) 
            d1 = self.adjust(d1, convention)
            if end_of_month and self.is_end_of_month(start_date):
                return date(d1.year, d1.month, self.end_of_month(d1))

        return d1

    def adjust(self,
               d: date,
               c: BusinessDayConvention):
        ''' 
        Adjust date based on the business day convention
        '''
        if d is None:
            raise ValueError("null date")

        if c == BusinessDayConvention.Unadjusted:
            return d

        d1 = d

        if c in [BusinessDayConvention.Following, BusinessDayConvention.ModifiedFollowing, BusinessDayConvention.HalfMonthModifiedFollowing]:
            while self.is_holiday(d1):
                d1 += timedelta(days=1)

            if c in [BusinessDayConvention.ModifiedFollowing, BusinessDayConvention.HalfMonthModifiedFollowing]:
                if d1.month != d.month:
                    return self.adjust(d, BusinessDayConvention.Preceding)

                if c == BusinessDayConvention.HalfMonthModifiedFollowing and d.day <= 15 and d1.day > 15:
                    return self.adjust(d, BusinessDayConvention.Preceding)

        elif c in [BusinessDayConvention.Preceding, BusinessDayConvention.ModifiedPreceding]:
            while self.is_holiday(d1):
                d1 -= timedelta(days=1)

            if c == BusinessDayConvention.ModifiedPreceding and d1.month != d.month:
                return self.adjust(d, BusinessDayConvention.Following)

        elif c == BusinessDayConvention.Nearest:
            d2 = d
            while self.is_holiday(d1) and self.is_holiday(d2):
                d1 += timedelta(days=1)
                d2 -= timedelta(days=1)

            if self.is_holiday(d1):
                return d2
            else:
                return d1

        else:
            raise ValueError("unknown business-day convention")

        return d1
    
    def is_end_of_month(self, d: date):
        return d.month != self.adjust((d + timedelta(1)), 
                                        BusinessDayConvention.ModifiedFollowing).month
    
    def is_weekend(self, d: date):
        return d.weekday() in [5,6]

    def end_of_month(self, d: date):
        _, last_day = monthrange(d.year, d.month)
        return last_day

    def is_holiday(self, d: date):
        return not self.is_business_day(d)

    @abstractmethod
    def is_business_day(self, d: date):
        ''' 
        abstract method depending on different calendars
        '''
        pass


