from datetime import date

from .product import Product, ProductAP
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency, SwapType
from ..index.index import Index
from ..timehandles.utils import DayCounterConvention
from ..flows.fixedcoupon import FixedRateLeg 
from ..flows.floatingcoupon import FloatingRateLeg

class OisAP(ProductAP):
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
        super().__init__(ccy, start_date, end_date, quote)
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

class Ois(Product):
    def __init__(self,
                schedule: list[date],
                notional: float,
                index: Index,
                fixed_rate: float,
                fixed_daycounter: DayCounterConvention,
                swap_type: SwapType = SwapType.Payer, 
                currency: Currency = None):
        super().__init__(currency, schedule[0], schedule[-1])
        self.swap_type = swap_type
        notionals = [notional]*(len(schedule) - 1)
        fix_rates = [fixed_rate]*(len(schedule) - 1)

        gearings = [1.0]*(len(schedule) - 1)
        spreads = [0.0]*(len(schedule) - 1)

        self.fixed_leg = FixedRateLeg(schedule, notionals, fix_rates, fixed_daycounter)
        self.floating_leg = FloatingRateLeg(schedule, notionals, gearings, spreads, index, fixed_daycounter) #TODO correggere il floating dc

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


class OisTest(ProductAP):
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
                 day_counter_flt: DayCounter,
                 index):
        super().__init__(ccy, start_date, end_date, quote)
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

        # self.swap_type = swap_type
        fix_notionals = [notional]*len(pay_dates_fix) 
        fix_rates = [quote]*len(pay_dates_fix) 

        flt_notionals = [notional]*len(pay_dates_flt) 
        gearings = [1.0]*len(pay_dates_flt) 
        spreads = [0.0]*len(pay_dates_flt) 

        self.fixed_leg = FixedRateLeg(pay_dates_fix, fix_notionals, fix_rates, day_counter_fix)
        self.floating_leg = FloatingRateLeg(pay_dates_flt, flt_notionals, gearings, spreads, index, day_counter_flt) 

