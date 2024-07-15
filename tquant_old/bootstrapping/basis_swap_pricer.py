import datetime
from typing import Dict

from tquant import RateCurve, IborIndex, TARGET, TimeUnit, BusinessDayConvention, DayCounterConvention, DayCounter, \
    Product
from tquant.bootstrapping.abstract_pricer import AbstractPricer
from tquant.bootstrapping.basis_swap import BasisSwap
from tquant.bootstrapping.basis_swap_builder import BasisSwapBuilder
from tquant.bootstrapping.curve_assignment import CurveAssignment
from tquant.bootstrapping.util import calculate_forward

import scipy.optimize as opt


class BasisSwapPricer(AbstractPricer):
    def __init__(self, curve_assignment: CurveAssignment):
        super().__init__()
        self.curve_assignment = curve_assignment

    def price(self,
              product: Product,
              as_of_date: datetime,
              curves: Dict[str, RateCurve]):
        if isinstance(product, BasisSwap):
            basis_swap = product
            discount = {"CCY": basis_swap.ccy, "USAGE": "DISCOUNT"}
            forecast1 = {"CCY": basis_swap.ccy, "USAGE": "FORECAST", "INDEX_TENOR": basis_swap.period1}
            forecast2 = {"CCY": basis_swap.ccy, "USAGE": "FORECAST", "INDEX_TENOR": basis_swap.period2}
            fc_name_1 = self.curve_assignment.get_curve_name(forecast1)
            fc_name_2 = self.curve_assignment.get_curve_name(forecast2)
            dc_name = self.curve_assignment.get_curve_name(discount)
            fc1 = curves[fc_name_1]
            fc2 = curves[fc_name_2]
            dc = curves[dc_name]
            act365 = DayCounterConvention.Actual365
            day_counter = DayCounter(act365)
            pv_flt_1 = 0.0
            for i in range(len(basis_swap.pay_dates_leg1)):
                pay_date = basis_swap.pay_dates_leg1[i]
                if pay_date > as_of_date:
                    yf = basis_swap.day_counter_1.year_fraction(
                        basis_swap.start_dates_leg1[i], basis_swap.end_dates_leg1[i])
                    if basis_swap.fixing_dates_leg1[i] < as_of_date:
                        rate = basis_swap.fixing_rates_leg1[i]
                    elif basis_swap.fixing_dates_leg1[i] == as_of_date and basis_swap.fixing_rates_leg1[i] != 0.0:
                        rate = basis_swap.fixing_rates_leg1[i]
                    else:
                        index_yf = basis_swap.index_day_counter_1.year_fraction(
                            basis_swap.index_start_dates_1[i], basis_swap.index_end_dates_1[i])
                        rate = calculate_forward(
                            as_of_date,
                            fc1,
                            basis_swap.index_start_dates_1[i],
                            basis_swap.index_end_dates_1[i],
                            index_yf)
                    cashflow = basis_swap.notional * (rate + basis_swap.quote) * yf
                    pv_flt_1 += cashflow * dc.discount(day_counter.year_fraction(as_of_date, pay_date))

            pv_flt_2 = 0.0
            for i in range(len(basis_swap.pay_dates_leg2)):
                pay_date = basis_swap.pay_dates_leg2[i]
                if pay_date > as_of_date:
                    yf = basis_swap.day_counter_2.year_fraction(
                        basis_swap.start_dates_leg2[i], basis_swap.end_dates_leg2[i])
                    if basis_swap.fixing_dates_leg2[i] < as_of_date:
                        rate = basis_swap.fixing_rates_leg2[i]
                    elif basis_swap.fixing_dates_leg2[i] == as_of_date and basis_swap.fixing_rates_leg2[i] != 0.0:
                        rate = basis_swap.fixing_rates_leg2[i]
                    else:
                        index_yf = basis_swap.index_day_counter_2.year_fraction(
                            basis_swap.index_start_dates_2[i], basis_swap.index_end_dates_2[i])
                        rate = calculate_forward(
                            as_of_date,
                            fc2,
                            basis_swap.index_start_dates_2[i],
                            basis_swap.index_end_dates_2[i],
                            index_yf)
                    cashflow = basis_swap.notional * rate * yf
                    pv_flt_2 += cashflow * dc.discount(day_counter.year_fraction(as_of_date, pay_date))

            return pv_flt_1 - pv_flt_2
        else:
            raise TypeError("Wrong product type")


class ObjectiveFunction:
    def __init__(self,
                 swap: Product,
                 today: datetime,
                 market_data: Dict[str, RateCurve],
                 pricer: BasisSwapPricer,
                 curve: RateCurve,
                 i: int):
        self.swap = swap
        self.today = today
        self.market_data = market_data
        self.pricer = pricer
        self.curve = curve
        self.i = i

    def __call__(self, x):
        self.curve.set_rate(self.i, x)
        return pricer.price(swap, today, market_data)


if __name__ == "__main__":
    today = datetime.datetime.now()
    builder = BasisSwapBuilder("BASIS_SWAP",
                               "EUR",
                               2,
                               2,
                               "3M",
                               "3M",
                               BusinessDayConvention.ModifiedFollowing,
                               1.0,
                               DayCounterConvention.Thirty360,
                               DayCounterConvention.Actual360,
                               DayCounterConvention.Actual360,
                               DayCounterConvention.Actual360)
    market_data = {"EONIA": RateCurve([1.0 / 365.0], [0.01]),
                   "EURIBOR 3M": RateCurve([1.0 / 365.0], [0.01]),
                   "EURIBOR 6M": RateCurve([1.0 / 365.0], [0.01])}
    swap = builder.build(datetime.datetime.now(), 0.01, "2Y")
    data = [["EUR", "DISCOUNT", "", "EONIA"],
            ["EUR", "FORECAST", "3M", "EURIBOR 3M"],
            ["EUR", "FORECAST", "6M", "EURIBOR 6M"]]
    attributes = ["CCY", "USAGE", "INDEX_TENOR"]
    curve_assignment = CurveAssignment(data, attributes)
    pricer = BasisSwapPricer(curve_assignment)
    pv = pricer.price(swap, today, market_data)

    curve = market_data["EURIBOR 3M"]
    f = ObjectiveFunction(swap, today, market_data, pricer, curve, 0)
    root = opt.newton(f, 0.0)

    print(pv.numpy())
    print(f(root).numpy())
    print(root.numpy())






