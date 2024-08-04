from datetime import date

from .product import Product, ProductAP
from ..timehandles.daycounter import DayCounter
# from ..markethandles.utils import Currency, SwapType
from ..index.index import Index
# from ..timehandles.utils import DayCounterConvention
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
        self._notionals = [notional]*len(pay_dates_fix)
        self._rates = [quote]*len(pay_dates_fix)
        self._gearings = [1]*len(pay_dates_flt)
        self._margins = [0]*len(pay_dates_flt)
        self._index = index

        self.fixed_leg = FixedRateLeg(pay_dates_fix, start_dates_fix, end_dates_fix,
                                    self._notionals, self._rates, day_counter_fix)         
        self.floating_leg = FloatingRateLeg(pay_dates_flt, start_dates_flt, end_dates_flt,
                                    self._notionals, self._gearings, self._margins, index, day_counter_flt)      



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
        self._notionals = [notional]*len(pay_dates_fix)
        self._rates = [quote]*len(pay_dates_fix)
        self._gearings = [1]*len(pay_dates_flt)
        self._margins = [0]*len(pay_dates_flt)
        self._index = index

        self.fixed_leg = FixedRateLeg(pay_dates_fix, start_dates_fix, end_dates_fix,
                                    self._notionals, self._rates, day_counter_fix)         
        self.floating_leg = FloatingRateLeg(pay_dates_flt, start_dates_flt, end_dates_flt,
                                    self._notionals, self._gearings, self._margins, index, day_counter_flt)      