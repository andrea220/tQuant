from datetime import date

from .product import Product
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency, SwapType
from ..index.index import Index
from ..timehandles.utils import DayCounterConvention
from ..flows.fixedcoupon import FixedRateLeg 
from ..flows.floatingcoupon import FloatingRateLeg

class Ois(Product):
    def __init__(self,
                 ccy: str,
                 start_date: date,
                 end_date: date,
                 start_dates_fix: list[date],
                 end_dates_fix: list[date],
                 pay_dates_fix: list[date],
                 start_dates_flt: list[date],
                 end_dates_flt: list[date],
                 pay_dates_flt: list[date],
                 fixing_dates: list[date],
                 fixing_rates: list[float],
                 quote: float,
                 notional: float,
                 day_counter_fix: DayCounter,
                 day_counter_flt: DayCounter):
        super().__init__(ccy, start_date, end_date)
        self.start_dates_fix = start_dates_fix
        self.end_dates_fix = end_dates_fix
        self.pay_dates_fix = pay_dates_fix
        self.start_dates_flt = start_dates_flt
        self.end_dates_flt = end_dates_flt
        self.pay_dates_flt = pay_dates_flt
        self.fixing_dates = fixing_dates
        self.fixing_rates = fixing_rates
        self.notional = notional
        self.day_counter_fix = day_counter_fix
        self.day_counter_flt = day_counter_flt
        self.quote = quote

    def __str__(self) -> str:
        return "{name: ois, \n" \
                "ccy: " + str(self.currency) + ",\n" \
                "start: " + str(self.start_date) + ",\n" \
                "end: " + str(self.end_date) + ",\n" \
                "start_dates_fix: " + str(self.start_dates_fix) + ",\n" \
                "end_dates_fix: " + str(self.end_dates_fix) + ",\n" \
                "pay_dates_fix: " + str(self.pay_dates_fix) + ",\n" \
                "start_dates_flt: " + str(self.start_dates_flt) + ",\n" \
                "end_dates_flt: " + str(self.end_dates_flt) + ",\n" \
                "pay_dates_flt: " + str(self.pay_dates_flt) + ",\n" \
                "fixing_dates: " + str(self.fixing_dates) + ",\n" \
                "fixing_rates: " + str(self.fixing_rates) + ",\n" \
                "notional: " + str(self.notional) + ",\n" \
                "quote: " + str(self.quote) + ",\n" \
                "day_counter_fix " + str(self.day_counter_fix) + ",\n" \
                "day_counter_flt " + str(self.day_counter_flt) + "}"

class OisTest(Product):
    def __init__(self,
                float_schedule: list[date],
                fix_schedule: list[date],
                notional: float,
                index: Index,
                fixed_rate: float,
                fixed_daycounter: DayCounterConvention,
                floating_daycounter: DayCounterConvention,
                swap_type: SwapType = SwapType.Payer, 
                currency: Currency = None):
        super().__init__(currency, fix_schedule[0], fix_schedule[-1])
        self.swap_type = swap_type
        fix_notionals = [notional]*(len(fix_schedule) - 1)
        fix_rates = [fixed_rate]*(len(fix_schedule) - 1)
        
        flt_notionals = [notional]*(len(float_schedule) - 1)
        gearings = [1.0]*(len(float_schedule) - 1)
        spreads = [0.0]*(len(float_schedule) - 1)

        self.fixed_leg = FixedRateLeg(fix_schedule, fix_notionals, fix_rates, fixed_daycounter)
        self.floating_leg = FloatingRateLeg(float_schedule, flt_notionals, gearings, spreads, index, floating_daycounter)

        self._price = None

    @property
    def price(self):
        if self._price is not None:
            return self._price
        else:
            raise ValueError("price not assigned")
    @price.setter
    def price(self, value):
        self._price = value
