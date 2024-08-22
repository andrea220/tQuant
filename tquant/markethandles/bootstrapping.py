from datetime import date 
from .ircurve import RateCurve
from ..instruments.product import Product
from ..pricers.pricer import Pricer
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..timehandles.utils import BusinessDayConvention, TimeUnit
from ..timehandles.targetcalendar import TARGET
from ..instruments.helpers import DepositGenerator, OisGenerator, FraGenerator,SwapGenerator
from ..pricers.factory import PricerAssignment
from ..numericalhandles.newton import newton
from ..markethandles.utils import Currency
from ..index.curverateindex import OvernightIndex, IborIndex
import numpy as np

class CurveBootstrap:
    def __init__(self,
                 evaluation_date: date,
                 daycount_convention: DayCounterConvention,
                 curve_map: dict) -> None:
        self.evaluation_date = evaluation_date
        self.day_counter = DayCounter(daycount_convention)
        self._curve_map = curve_map
        target_calendar = TARGET()

        eur_depo_builder = DepositGenerator(Currency.EUR,
                                            2,
                                            BusinessDayConvention.ModifiedFollowing,
                                            DayCounterConvention.Actual360,
                                            1.0,
                                            target_calendar)
        eur_ois_builder = OisGenerator(Currency.EUR,
                                        2,
                                        2,
                                        "1Y",
                                        "1Y",
                                        BusinessDayConvention.ModifiedFollowing,
                                        1.0,
                                        DayCounterConvention.Actual360,
                                        DayCounterConvention.Actual360,
                                        target_calendar,
                                        OvernightIndex(target_calendar, Currency.EUR))
        eur_fra_builder = FraGenerator(Currency.EUR,
                                        2,
                                        2,
                                        "6M",
                                        BusinessDayConvention.ModifiedFollowing,
                                        1.0,
                                        DayCounterConvention.Actual360, 
                                        target_calendar,
                                        IborIndex(target_calendar, 6, TimeUnit.Months, Currency.EUR))      
        eur_swap6m_builder = SwapGenerator(Currency.EUR,
                                           2,
                                           "1Y",
                                           "6M",
                                           BusinessDayConvention.ModifiedFollowing,
                                           1.0,
                                           DayCounterConvention.Actual360,
                                           DayCounterConvention.Actual360,
                                           target_calendar,
                                           IborIndex(target_calendar, 6, TimeUnit.Months, Currency.EUR)
                                           )
        
        self.eur_generator_map = {"depo": eur_depo_builder, 
                                  "ois": eur_ois_builder,
                                  "fra": eur_fra_builder,
                                  "swap": eur_swap6m_builder}

    def strip(self,
              generators: list[str],
              maturities: list[str],
              quotes: list[float],
              curve_name: str,
              currency: Currency,
              interpolation: str = 'LINEAR',
              market_data: dict = {},
              is_spread_curve: bool = False):
        if currency == Currency.EUR:
            generator_map = self.eur_generator_map
        products = []
        pricers = []
        for i, generator in enumerate(generators):
            builder = generator_map[generator]
            product = builder.build(self.evaluation_date, quotes[i], maturities[i])
            products.append(product)
            pricer = PricerAssignment.create(product, self._curve_map)
            pricers.append(pricer)
        pillars = []
        zero_rates = []
        for i in range(len(maturities)):
            pillars.append(self.day_counter.year_fraction(self.evaluation_date, products[i].maturity))
            zero_rates.append(0.01)
        if is_spread_curve:
            # base_curve = market_data[base_curve_name]
            # bootstrapping_curve = SpreadCurve(pillars, zero_rates, base_curve)
            pass
        else:
            bootstrapping_curve = RateCurve(self.evaluation_date, pillars, zero_rates, interpolation)
        market_data[curve_name] = bootstrapping_curve
        func = ObjectiveFunction(self.evaluation_date,
                                bootstrapping_curve,
                                products,
                                pricers,
                                market_data)
        x = np.array(zero_rates).astype(np.float64) # initial guess
        bootstrapped_rates, rates_jac = newton(func, x)
        bootstrapping_curve._set_rates(bootstrapped_rates)
        bootstrapping_curve.jacobian = rates_jac
        return bootstrapping_curve


class ObjectiveFunction:
    def __init__(self,
                 trade_date: date,
                 rate_curve: RateCurve,
                 products: list[Product],
                 pricers: list[Pricer],
                 curves: dict[str, RateCurve]):
        self.trade_date = trade_date
        self.rate_curve = rate_curve
        self.products = products
        self.pricers = pricers
        self.curves = curves

    def __call__(self, x):
        self.rate_curve._set_rates(x)
        res = np.zeros(len(self.pricers))
        jac = np.zeros((len(x), len(x)))
        for i, (pricer, product) in enumerate(zip(self.pricers, self.products)):
            pv, tape = pricer.price(product, self.trade_date, self.curves, True)
            gradients = tape.gradient(pv, [self.rate_curve._rates])
            res[i] = pv
            jac[i,:] = np.array(gradients[0])
        return res, np.nan_to_num(jac, nan=0.0)