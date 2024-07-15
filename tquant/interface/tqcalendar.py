from abc import ABC, abstractmethod
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from calendar import monthrange

from ..utilities.utils import TimeUnit, BusinessDayConvention


class Calendar(ABC):
    ''' 
    The abstract class for calendar implementations.
    '''

    def advance(self,
                start_date: date,
                period: int,
                time_unit: TimeUnit,
                convention: BusinessDayConvention,
                end_of_month: bool = False):
        """
        Advance a given start date by a specified period using the given time unit and business day convention.

        Parameters:
        -------
            start_date (date): The starting date.
            period (int): The number of time units to advance the start date by.
            time_unit (TimeUnit): The unit of time to use for advancing the start date.
            convention (BusinessDayConvention): The business day convention to use for adjusting the dates.
            end_of_month (bool, optional): Whether to adjust to the end of the month if the original start date is at the end of the month. Defaults to False.

        Returns:
        -------
            date: The advanced date.
        """        
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
        Adjust date based on the business day convention.

        Parameters:
        -------
            d (date): The date to adjust.
            c (BusinessDayConvention): The business day convention.

        Returns:
        -------
            date: The adjusted date.
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
    
    def is_end_of_month(self, d: date) -> bool:
        ''' 
        Check if a given date is the end of the month.

        Parameters:
        -------
            d (date): The date to check.

        Returns:
        -------
            bool: True if the given date is the end of the month, False otherwise.
        '''
        return d.month != self.adjust((d + timedelta(1)), 
                                        BusinessDayConvention.ModifiedFollowing).month
    
    def is_weekend(self, d: date) -> bool:
        ''' 
        Check if a given date is a weekend.

        Parameters:
        -------
            d (date): The date to check.

        Returns:
        -------
            bool: True if the given date is a weekend, False otherwise.
        '''
        return d.weekday() in [5,6]

    def end_of_month(self, d: date) -> int:
        ''' 
        Get the last day of the month for a given date.

        Parameters:
        -------
            d (date): The date for which to find the last day of the month.

        Returns:
        -------
            int: The last day of the month.
        '''
        _, last_day = monthrange(d.year, d.month)
        return last_day

    def is_holiday(self, d: date) -> bool:
        ''' 
        Check if a given date is a holiday.

        Parameters:
        -------
            d (date): The date to check.

        Returns:
        -------
            bool: True if the given date is a holiday, False otherwise.
        '''
        return not self.is_business_day(d)

    @abstractmethod
    def is_business_day(self, d: date):
        ''' 
        Check if a given date is a business day.

        Parameters:
        -------
            d (date): The date to check.

        Returns:
        -------
            bool: True if the given date is a business day, False otherwise.
        '''
        pass
