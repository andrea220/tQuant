"""Calendars: TARGET, UnitedStates (Federal Reserve: SOFR), UnitedKingdom (England bank holidays: SONIA)."""

import unittest
from datetime import date, timedelta

import tensorquant as tq


def holidays(calendar, year):
    """Weekdays that are not business days in *year*."""
    d, out = date(year, 1, 1), []
    while d.year == year:
        if d.weekday() < 5 and not calendar.is_business_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


class TestCalendars(unittest.TestCase):
    def test_target_2026(self):
        self.assertEqual(holidays(tq.TARGET(), 2026),
                         [date(2026, 1, 1), date(2026, 4, 3), date(2026, 4, 6), date(2026, 5, 1), date(2026, 12, 25)])

    def test_united_states_2026(self):
        """Federal Reserve 2026: July 4 is a Saturday, not moved to the Friday."""
        self.assertEqual(holidays(tq.UnitedStates(), 2026),
                         [date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 5, 25), date(2026, 6, 19),
                          date(2026, 9, 7), date(2026, 10, 12), date(2026, 11, 11), date(2026, 11, 26),
                          date(2026, 12, 25)])
        self.assertTrue(tq.UnitedStates().is_business_day(date(2026, 7, 3)))

    def test_united_states_sunday_rule_and_juneteenth(self):
        us = tq.UnitedStates()
        self.assertFalse(us.is_business_day(date(2023, 1, 2)))       # New Year on a Sunday -> Monday
        self.assertFalse(us.is_business_day(date(2027, 7, 5)))       # July 4 on a Sunday -> Monday
        self.assertTrue(us.is_business_day(date(2021, 6, 18)))       # Juneteenth only from 2022
        self.assertFalse(us.is_business_day(date(2022, 6, 20)))      # June 19, 2022 a Sunday -> Monday

    def test_united_kingdom_2026(self):
        """England 2026: Boxing Day on a Saturday -> Monday 28."""
        self.assertEqual(holidays(tq.UnitedKingdom(), 2026),
                         [date(2026, 1, 1), date(2026, 4, 3), date(2026, 4, 6), date(2026, 5, 4), date(2026, 5, 25),
                          date(2026, 8, 31), date(2026, 12, 25), date(2026, 12, 28)])

    def test_united_kingdom_special_days(self):
        uk = tq.UnitedKingdom()
        for d in (date(2020, 5, 8), date(2022, 6, 2), date(2022, 6, 3), date(2022, 9, 19), date(2023, 5, 8),
                  date(2021, 12, 27), date(2021, 12, 28)):
            with self.subTest(day=d):
                self.assertFalse(uk.is_business_day(d))
        for d in (date(2020, 5, 4), date(2022, 5, 30)):          # the regular Mondays moved those years
            with self.subTest(day=d):
                self.assertTrue(uk.is_business_day(d))

    def test_advance_skips_the_holidays(self):
        mf = tq.BusinessDayConvention.ModifiedFollowing
        self.assertEqual(tq.UnitedStates().advance(date(2026, 11, 25), 1, tq.TimeUnit.Days, mf), date(2026, 11, 27))
        self.assertEqual(tq.UnitedKingdom().advance(date(2026, 12, 24), 1, tq.TimeUnit.Days, mf), date(2026, 12, 29))


if __name__ == "__main__":
    unittest.main()
