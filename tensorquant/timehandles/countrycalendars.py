"""National calendars for the overnight rate curves: :class:`UnitedStates` (Federal Reserve holidays, the SOFR
calendar) and :class:`UnitedKingdom` (England and Wales bank holidays, the SONIA calendar)."""

from datetime import date, timedelta

from dateutil.easter import easter

from .tqcalendar import Calendar

MON, THU = 0, 3


class UnitedStates(Calendar):
    """Federal Reserve calendar (SOFR fixings and USD OIS).

    New Year, Martin Luther King (3rd Monday of January, from 1983), Washington's birthday (3rd Monday of February),
    Memorial Day (last Monday of May), Juneteenth (June 19, from 2022), Independence Day, Labor Day (1st Monday of
    September), Columbus Day (2nd Monday of October), Veterans Day (November 11), Thanksgiving (4th Thursday of
    November), Christmas. A holiday on a Sunday moves to the Monday; one on a Saturday is not moved.
    """

    def __init__(self) -> None:
        pass

    def is_business_day(self, d: date) -> bool:
        if self.is_weekend(d):
            return False
        y, m, day, w = d.year, d.month, d.day, d.weekday()

        def fixed(month: int, dd: int) -> bool:          # on the day, or the Monday after a Sunday
            return m == month and (day == dd or (day == dd + 1 and w == MON))

        holiday = (
            fixed(1, 1)
            or (m == 1 and w == MON and 15 <= day <= 21 and y >= 1983)
            or (m == 2 and w == MON and 15 <= day <= 21)
            or (m == 5 and w == MON and day >= 25)
            or (fixed(6, 19) and y >= 2022)
            or fixed(7, 4)
            or (m == 9 and w == MON and day <= 7)
            or (m == 10 and w == MON and 8 <= day <= 14)
            or fixed(11, 11)
            or (m == 11 and w == THU and 22 <= day <= 28)
            or fixed(12, 25)
        )
        return not holiday


class UnitedKingdom(Calendar):
    """England and Wales bank holidays (SONIA fixings and GBP OIS).

    New Year (or the next Monday), Good Friday, Easter Monday, early May (1st Monday), spring (last Monday of May),
    summer (last Monday of August), Christmas and Boxing Day (moved to Monday / Tuesday when on a weekend), and the
    one-off days (moved May holidays, jubilees, royal events).
    """

    SPECIAL = {date(1999, 12, 31), date(2002, 6, 3), date(2002, 6, 4), date(2011, 4, 29), date(2012, 6, 4),
               date(2012, 6, 5), date(2020, 5, 8), date(2022, 6, 2), date(2022, 6, 3), date(2022, 9, 19),
               date(2023, 5, 8), date(1995, 5, 8)}
    # regular May Mondays replaced in those years by a one-off day
    MOVED = {date(1995, 5, 1), date(2002, 5, 27), date(2012, 5, 28), date(2020, 5, 4), date(2022, 5, 30)}

    def __init__(self) -> None:
        pass

    def is_business_day(self, d: date) -> bool:
        if self.is_weekend(d):
            return False
        if d in self.SPECIAL:
            return False
        m, day, w = d.month, d.day, d.weekday()
        e = easter(d.year)
        regular = (
            (m == 1 and (day == 1 or (day in (2, 3) and w == MON)))
            or d == e - timedelta(days=2)
            or d == e + timedelta(days=1)
            or (m == 5 and w == MON and day <= 7)
            or (m == 5 and w == MON and day >= 25)
            or (m == 8 and w == MON and day >= 25)
            or (m == 12 and (day == 25 or (day == 27 and w in (MON, 1))))
            or (m == 12 and (day == 26 or (day == 28 and w in (MON, 1))))
        )
        return not (regular and d not in self.MOVED)
