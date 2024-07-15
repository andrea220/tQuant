import math
from gregorian_date import *
 
class SimpleShifter:
    def __init__(self, period: str) -> None:
        duration = period[0: len(period) - 1]
        self.time_unit = period[len(period) - 1:]
        self.duration = int(duration)
    
    def add_period(self, date: int, duration: int, time_unit: str):
        if time_unit.lower() == 'y':
            date = self.add_months(date, duration * 12)
        elif time_unit.lower() == 'm':
            date = self.add_months(date, duration)
        elif time_unit.lower() == "w":
            date += duration * 7
        elif time_unit.lower() == "d":
            date += duration
        return date
            
    def modify(self, date: int):
        return self.add_period(date, self.duration, self.time_unit)

    def normalize(self, date):
        if date.month < 1:
                dy = int(12 - date.month) / 12
                date.month += 12 * dy
                date.year -= dy
        elif date.month > 12:
            dy = (date.month - 1) / 12
            date.month -= 12 * dy
            date.year += dy

    def add_months(self, date: int, duration: int):
        gd = self.excel_serial_to_gregorian(date)
        gd.month += duration
        self.normalize(gd)
        return self.gregorian_to_excel_serial(gd)
        
    def mjd_to_gregorian(self, mjd: int):          
        Z = int(mjd + 678882)
        G = Z - 0.25
        A = math.floor(G / 36524.25)
        B = A - math.floor(A / 4.0)
        year = int(math.floor((B + G) / 365.25))
        C = B + Z - math.floor(year * 365.25)
        month = int(math.floor((5.0 * C + 456.0) / 153.0))
        day = int(C - math.floor((month * 153.0 - 457.0) / 5.0))
        
        if month > 12:
            year = year + 1
            month = month - 12
        gd = GregorianDate(year, month, day)
        return gd

    def excel_serial_to_gregorian(self, excel):
        MJD_TO_EXCEL = -15018
        gd = self.mjd_to_gregorian(excel - MJD_TO_EXCEL)
        return gd

    def toMjd(self, Y: int, M: int, D: int):
            if M < 3:
                M = M + 12
                Y = Y - 1
            intJD = D + int((153.0 * M - 457.0) / 5.0) + 365 * Y + int(math.floor(Y / 4.0)) - int(math.floor(Y / 100.0)) + int(math.floor(Y / 400.0));
            mjd = intJD - 678882;
            return mjd  

    def gregorian_to_excel_serial(self, gd: GregorianDate):
        mjd = int(self.toMjd(gd.year, gd.month, gd.day));
        MJD_TO_EXCEL = -15018;                
        return mjd + MJD_TO_EXCEL;

    def tMdj(self, Y: int, M: int, D: int):
            intJD
            mjd

            if M < 3:
                M = M + 12
                Y = Y - 1
            intJD = D + int((153.0 * M - 457.0) / 5.0) + 365 * Y + int(math.floor(Y / 4.0)) - int(math.floor(Y / 100.0)) + int(math.floor(Y / 400.0));
            mjd = intJD - 678882;
            return mjd         

    