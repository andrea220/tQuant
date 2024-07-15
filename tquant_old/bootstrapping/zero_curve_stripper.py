import datetime
import time
from typing import Dict
from tensorflow.python.framework import dtypes

from dateutil.relativedelta import relativedelta
from tensorflow.python.framework import dtypes

import numpy as np

import tensorflow as tf

from tquant import RateCurve, ProductBuilder, DepositBuilder, BusinessDayConvention, DayCounterConvention, DayCounter, \
    Product
from tquant.bootstrapping.abstract_pricer import AbstractPricer
from tquant.bootstrapping.basis_swap_builder import BasisSwapBuilder
from tquant.bootstrapping.curve_assignment import CurveAssignment
from tquant.bootstrapping.fra_builder import FraBuilder
from tquant.bootstrapping.future_builder import FutureBuilder
from tquant.bootstrapping.multidimensional_newton import newton_multidimensional
from tquant.bootstrapping.ois_builder import OisBuilder
from tquant.bootstrapping.pricer_factory import PricerFactory
from tquant.bootstrapping.spread_curve import SpreadCurve
from tquant.bootstrapping.swap_builder import SwapBuilder


class ZeroCurveStripper:

    def strip(self,
              bootstrapping_curve_name: str,
              trade_date: datetime,
              generators: list[str],
              maturities: list[str],
              quotes: list[float],
              curve_assignment: CurveAssignment,
              generator_map: dict[str, ProductBuilder],
              market_data: dict[str, RateCurve],
              solver_type: str = "LOCAL",
              is_spread_curve: bool = False,
              base_curve_name: str = ""):
        print(bootstrapping_curve_name)
        products = []
        pricers = []
        for i, generator in enumerate(generators):
            builder = generator_map[generator]
            product = builder.build(trade_date, quotes[i], maturities[i])
            products.append(product)
            #print(maturities[i])
            #print(product)
            pricer = PricerFactory.create(product, curve_assignment)
            pricers.append(pricer)
        pillars = []
        zero_rates = []
        act365 = DayCounterConvention.Actual365
        day_counter = DayCounter(act365)
        for i in range(len(maturities)):
            pillars.append(day_counter.year_fraction(trade_date, products[i].maturity))
            zero_rates.append(0.01)
        if is_spread_curve:
            base_curve = market_data[base_curve_name]
            bootstrapping_curve = SpreadCurve(pillars, zero_rates, base_curve)
        else:
            bootstrapping_curve = RateCurve(pillars, zero_rates)
        market_data[bootstrapping_curve_name] = bootstrapping_curve
        if solver_type == "LOCAL":
            for i in range(len(products)):
                pricer = pricers[i]
                rate = bootstrapping_curve.get_rate_at_pillar(i)
                step = 0.01
                rate1 = rate - step
                rate2 = rate + step
                bootstrapping_curve.set_rate(i, rate1)
                pv1 = pricer.price(products[i], trade_date, market_data)
                bootstrapping_curve.set_rate(i, rate2)
                pv2 = pricer.price(products[i], trade_date, market_data)
                max_iterations = 10
                while pv1 * pv2 >= 0.0:
                    step *= 2.0
                    if abs(pv1) > abs(pv2):
                        rate1 = rate2
                        pv1 = pv2
                        rate2 += step
                        bootstrapping_curve.set_rate(i, rate2)
                        pv2 = pricer.price(products[i], trade_date, market_data)
                    else:
                        rate2 = rate1
                        pv2 = pv1
                        rate1 -= step
                        bootstrapping_curve.set_rate(i, rate1)
                        pv1 = pricer.price(products[i], trade_date, market_data)
                    max_iterations -= 1
                    if max_iterations == 0:
                        break
                max_iterations = 50
                while rate2 - rate1 > 1.0e-16:
                    rate = (rate1 + rate2) * 0.5
                    bootstrapping_curve.set_rate(i, rate)
                    pv = pricer.price(products[i], trade_date, market_data)
                    if pv > 0.0:
                        rate2 = rate
                    else:
                        rate1 = rate
                    max_iterations -= 1
                    if max_iterations == 0:
                        break
                pv = pricer.price(products[i], trade_date, market_data)
                print("PV[" + str(i) + "] = " + str(pv.numpy()))

            for i, rate in enumerate(bootstrapping_curve.rates):
                print("Rate[" + str(i) + "] = " + str(rate.numpy()))
        elif solver_type == "GLOBAL":
            func = ObjectiveFunction(trade_date,
                                     bootstrapping_curve,
                                     products,
                                     pricers,
                                     market_data)
            x = np.array(zero_rates).astype(np.float64)
            solution = newton_multidimensional(func, x)
            bootstrapping_curve.set_rates(solution)

            ret = func(solution)
            for i in range(len(products)):
                print("PV[" + str(i) + "] = " + str(ret[i]))
            for i in range(len(products)):
                print("Rate[" + str(i) + "] = " + str(bootstrapping_curve.rates[i]))

        return bootstrapping_curve


class ObjectiveFunction:
    def __init__(self,
                 trade_date: datetime,
                 rate_curve: RateCurve,
                 products: list[Product],
                 pricers: list[AbstractPricer],
                 curves: Dict[str, RateCurve]):
        self.trade_date = trade_date
        self.rate_curve = rate_curve
        self.products = products
        self.pricers = pricers
        self.curves = curves

    def __call__(self, x):
        self.rate_curve.set_rates(x)
        res = np.zeros(len(self.pricers))
        for i, (pricer, product) in enumerate(zip(self.pricers, self.products)):
            res[i] = pricer.price(product, self.trade_date, self.curves)
        return res


if __name__ == "__main__":
    trade_date = datetime.datetime.now()
    generators = ["depo", "ois", "ois"]
    maturities = ["6M", "1Y", "2Y"]
    quotes = [0.01, 0.01, 0.01]
    data = [["EUR", "DISCOUNT", "EUR EONIA"]]
    attributes = ["CCY", "USAGE"]
    curve_assignment = CurveAssignment(data, attributes)
    depo_builder = DepositBuilder("depo",
                                  "EUR",
                                  2,
                                  BusinessDayConvention.ModifiedFollowing,
                                  DayCounterConvention.Actual360,
                                  1.0)
    ois_builder = OisBuilder("ois",
                             "EUR",
                             2,
                             2,
                             "6M",
                             "1Y",
                             BusinessDayConvention.ModifiedFollowing,
                             1.0,
                             DayCounterConvention.Actual360,
                             DayCounterConvention.Actual360)
    generator_map = {"depo": depo_builder, "ois": ois_builder}
    stripper = ZeroCurveStripper()
    curve = stripper.strip(
        "EUR EONIA", trade_date, generators, maturities, quotes, curve_assignment, generator_map, {}, "GLOBAL")

    trade_date = datetime.datetime.now()
    generators = ["fra", "future", "swap", "swap", "swap", "swap"]
    maturities = ["3M-9M", "JUN 25", "1Y", "2Y", "3Y", "4Y"]
    quotes = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    data = [["EUR", "DISCOUNT", "EUR EONIA"],
            ["EUR", "FORECAST", "EUR EURIBOR 6M"]]
    attributes = ["CCY", "USAGE"]
    curve_assignment = CurveAssignment(data, attributes)
    fra_builder = FraBuilder("fra",
                             "EUR",
                             2,
                             2,
                             "6M",
                             BusinessDayConvention.ModifiedFollowing,
                             1.0,
                             DayCounterConvention.Actual360,
                             DayCounterConvention.Actual360)
    future_builder = FutureBuilder("future",
                                   "EUR",
                                   1.0,
                                   "6M",
                                   DayCounterConvention.Actual360,
                                   BusinessDayConvention.ModifiedFollowing)
    swap_builder = SwapBuilder("swap",
                               "EUR",
                               2,
                               "1Y",
                               "6M",
                               BusinessDayConvention.ModifiedFollowing,
                               1.0,
                               0.0,
                               DayCounterConvention.Actual360,
                               DayCounterConvention.Actual360,
                               DayCounterConvention.Actual360,
                               2)
    generator_map = {"fra": fra_builder, "future": future_builder, "swap": swap_builder}
    eur_eonia = RateCurve([1.0], [0.01])
    market_data = {"EUR EONIA": eur_eonia}
    stripper = ZeroCurveStripper()
    curve = stripper.strip("EUR EURIBOR 6M", trade_date, generators, maturities, quotes, curve_assignment,
                           generator_map, market_data, "GLOBAL")

    """
    bootstrapping basis swaps
    """
    trade_date = datetime.datetime.now()
    generators = ["basis_swap", "basis_swap", "basis_swap", "basis_swap"]
    maturities = ["1Y", "2Y", "3Y", "4Y"]
    quotes = [0.01, 0.01, 0.01, 0.01]
    data = [["EUR", "DISCOUNT", "", "EUR EONIA"],
            ["EUR", "FORECAST", "3M", "EUR EURIBOR 3M"],
            ["EUR", "FORECAST", "6M", "EUR EURIBOR 6M"]]

    attributes = ["CCY", "USAGE", "INDEX_TENOR"]
    curve_assignment = CurveAssignment(data, attributes)
    basis_swap_builder = BasisSwapBuilder(generators[0],
                                          "EUR",
                                          2,
                                          2,
                                          "3M",
                                          "6M",
                                          BusinessDayConvention.ModifiedFollowing,
                                          1.0,
                                          DayCounterConvention.Actual360,
                                          DayCounterConvention.Actual360,
                                          DayCounterConvention.Actual360,
                                          DayCounterConvention.Actual360)
    generator_map = {"basis_swap": basis_swap_builder}
    eur_eonia = RateCurve([1.0], [0.01])
    eur_euribor_6m = RateCurve([1.0], [0.01])
    market_data = {"EUR EONIA": eur_eonia, "EUR EURIBOR 6M": eur_euribor_6m}
    stripper = ZeroCurveStripper()
    curve = stripper.strip("EUR EURIBOR 3M", trade_date, generators, maturities, quotes, curve_assignment,
                           generator_map, market_data, "GLOBAL")
