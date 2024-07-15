import datetime
from typing import Dict
import numpy as np

from tquant import RateCurve, BusinessDayConvention, DayCounterConvention, DayCounter, IborIndex, TARGET, TimeUnit, \
    Product
from tquant.bootstrapping.abstract_pricer import AbstractPricer
from tquant.bootstrapping.curve_assignment import CurveAssignment
from tquant.bootstrapping.swap import Swap
from tquant.bootstrapping.swap_builder import SwapBuilder
from tquant.bootstrapping.util import calculate_forward


class SwapPricer(AbstractPricer):

    def __init__(self,
                 curve_assignment: CurveAssignment):
        super().__init__()
        self.curve_assignment = curve_assignment

    def price(self,
              product: Product,
              as_of_date: datetime,
              curves: Dict[str, RateCurve]):
        if isinstance(product, Swap):
            swap = product
            discount = {"CCY": swap.ccy, "USAGE": "DISCOUNT"}
            forecast = {"CCY": swap.ccy, "USAGE": "FORECAST", "INDEX_TENOR": swap.period_flt}
            fc_name = self.curve_assignment.get_curve_name(forecast)
            dc_name = self.curve_assignment.get_curve_name(discount)
            fc = curves[fc_name]
            dc = curves[dc_name]
            act365 = DayCounterConvention.Actual365
            day_counter = DayCounter(act365)
            pv_fix = 0.0
            for i in range(len(swap.pay_dates_fix)):
                pay_date = swap.pay_dates_fix[i]
                if pay_date > as_of_date:
                    yf = swap.day_counter_fix.year_fraction(swap.start_dates_fix[i], swap.end_dates_fix[i])
                    cashflow = swap.notional * swap.quote * yf
                    pv_fix += cashflow * dc.discount(day_counter.year_fraction(as_of_date, pay_date))

            pv_flt = 0.0
            for i in range(len(swap.pay_dates_float)):
                pay_date = swap.pay_dates_float[i]
                if pay_date > as_of_date:
                    yf = swap.day_counter_float.year_fraction(swap.start_dates_float[i], swap.end_dates_float[i])
                    if swap.fixing_dates[i] < as_of_date:
                        rate = swap.fixing_rates[i]
                    elif swap.fixing_dates[i] == as_of_date and swap.fixing_rates[i] != 0.0:
                        rate = swap.fixing_rates[i]
                    else:
                        index_yf = swap.day_counter_index.year_fraction(
                            swap.index_start_dates[i], swap.index_end_dates[i])
                        rate = calculate_forward(
                            as_of_date, fc, swap.index_start_dates[i], swap.index_end_dates[i], index_yf)
                    cashflow = swap.notional * (rate + swap.margin) * yf
                    pv_flt += cashflow * dc.discount(day_counter.year_fraction(as_of_date, pay_date))

            return pv_flt - pv_fix
        else:
            raise TypeError("Wrong product type")


if __name__ == "__main__":
    today = datetime.datetime.now()
    builder = SwapBuilder("SWAP",
                          "EUR",
                          2,
                          "1Y",
                          "3M",
                          BusinessDayConvention.ModifiedFollowing,
                          1.0,
                          0.0,
                          DayCounterConvention.Thirty360,
                          DayCounterConvention.Actual360,
                          DayCounterConvention.Actual360,
                          2)
    market_data = {"EONIA": RateCurve([1.0 / 365.0], [0.01]),
                   "EURIBOR 3M": RateCurve([1.0 / 365.0], [0.01])}
    swap = builder.build(datetime.datetime.now(), 0.01, "2Y")
    data = [["EUR", "DISCOUNT", "", "EONIA"],
            ["EUR", "FORECAST", "3M", "EURIBOR 3M"]]
    attributes = ["CCY", "USAGE", "INDEX_TENOR"]
    curve_assignment = CurveAssignment(data, attributes)
    pricer = SwapPricer(curve_assignment)
    pv = pricer.price(swap, today, market_data)
    print(pv.numpy())
