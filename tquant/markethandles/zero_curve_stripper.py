from datetime import date
import numpy as np

from .curve_assignment import CurveAssignment
from ..instruments.helpers import Helper
from .ircurve import RateCurve

class ZeroCurveStripper:

    def strip(self,
              bootstrapping_curve_name: str,
              trade_date: date,
              generators: list[str],
              maturities: list[str],
              quotes: list[float],
              curve_assignment: CurveAssignment,
              generator_map: dict[str, Helper],
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
                max_iterations = 10
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


