from datetime import date, datetime

from tquant.utilities import TARGET
from tquant.utilities import TimeUnit, BusinessDayConvention


class ScheduleGenerator:
    def generate(self, start: date, end: date, tenor: int, time_unit: TimeUnit,
                 business_day_convention: BusinessDayConvention):
        calendar = TARGET()
        schedule = []
        current_date = start
        schedule.append(current_date)
        while True:
            current_date = calendar.advance(current_date, tenor, time_unit, business_day_convention)
            if current_date < end:
                schedule.append(current_date)
            else:
                schedule.append(end)
                break
        return schedule


if __name__ == "__main__":
    generator = ScheduleGenerator()
    schedule = generator.generate(datetime(2024, 4, 15),
                                  datetime(2024, 12, 31),
                                  3,
                                  TimeUnit.Months,
                                  BusinessDayConvention.ModifiedFollowing)

    for date in schedule:
        print(date.strftime("%Y-%m-%d"))
