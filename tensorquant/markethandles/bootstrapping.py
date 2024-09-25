import datetime
import numpy

from .ircurve import RateCurve
from ..instruments.product import Product
from ..pricers.pricer import Pricer
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..timehandles.utils import BusinessDayConvention, TimeUnit, Settings
from ..timehandles.targetcalendar import TARGET
from ..instruments.helpers import (
    DepositGenerator,
    OisGenerator,
    FraGenerator,
    SwapGenerator,
)
from ..pricers.factory import PricerAssignment
from ..numericalhandles.newton import newton
from ..markethandles.utils import Currency
from ..index.curverateindex import OvernightIndex, IborIndex


class CurveBootstrap:
    """
    Class to bootstrap interest rate curves from market quotes and instrument generators.

    Attributes:
        evaluation_date (datetime.date): The date of curve evaluation.
        day_counter (DayCounter): The day count convention used for the curve.
        _curve_map (dict): A map of existing curves to be used during the bootstrap process.
        eur_generator_map (dict): A map of instrument generators for EUR instruments.
    """

    def __init__(
        self,
        evaluation_date: datetime.date,
        daycount_convention: DayCounterConvention,
        curve_map: dict,
    ) -> None:
        """
        Initializes the CurveBootstrap instance with evaluation date, day count convention, and curve map.

        Args:
            evaluation_date (datetime.date): The curve evaluation date.
            daycount_convention (DayCounterConvention): The day count convention to use.
            curve_map (dict): A dictionary mapping curve names to RateCurve objects.

        """
        # self.evaluation_date = evaluation_date
        Settings.evaluation_date = evaluation_date
        self.day_counter = DayCounter(daycount_convention)
        self._curve_map = curve_map
        target_calendar = TARGET()

        eur_depo_builder = DepositGenerator(
            Currency.EUR,
            2,
            BusinessDayConvention.ModifiedFollowing,
            DayCounterConvention.Actual360,
            1.0,
            target_calendar,
        )
        eur_ois_builder = OisGenerator(
            Currency.EUR,
            2,
            2,
            "1Y",
            "1Y",
            BusinessDayConvention.ModifiedFollowing,
            1.0,
            DayCounterConvention.Actual360,
            DayCounterConvention.Actual360,
            target_calendar,
            OvernightIndex(target_calendar, Currency.EUR),
        )
        eur_fra_builder = FraGenerator(
            Currency.EUR,
            2,
            2,
            "6M",
            BusinessDayConvention.ModifiedFollowing,
            1.0,
            DayCounterConvention.Actual360,
            target_calendar,
            IborIndex(target_calendar, 6, TimeUnit.Months, Currency.EUR),
        )
        eur_swap6m_builder = SwapGenerator(
            Currency.EUR,
            2,
            "1Y",
            "6M",
            BusinessDayConvention.ModifiedFollowing,
            1.0,
            DayCounterConvention.Actual360,
            DayCounterConvention.Actual360,
            target_calendar,
            IborIndex(target_calendar, 6, TimeUnit.Months, Currency.EUR),
        )

        self.eur_generator_map = {
            "Dp": eur_depo_builder,
            "Os": eur_ois_builder,
            "Fr": eur_fra_builder,
            "Sw": eur_swap6m_builder,
        }

    def strip(
        self,
        generators: list[str],
        maturities: list[str],
        quotes: list[float],
        curve_name: str,
        currency: Currency,
        interpolation: str = "LINEAR",
        market_data: dict = {},
        is_spread_curve: bool = False,
        daycounter_convention=DayCounterConvention.ActualActual,
    ):
        """
        Bootstraps an interest rate curve from market quotes and instrument generators.

        Args:
            generators (list[str]): A list of instrument generators (e.g., "Dp" for deposit, "Os" for OIS).
            maturities (list[str]): A list of instrument maturities (e.g., "1M", "6M", "1Y").
            quotes (list[float]): A list of market quotes corresponding to the instruments.
            curve_name (str): The name of the resulting bootstrapped curve.
            currency (Currency): The currency of the curve (e.g., EUR).
            interpolation (str, optional): The type of interpolation (default is "LINEAR").
            market_data (dict, optional): A dictionary containing existing market data (default is empty).
            is_spread_curve (bool, optional): Whether the curve is a spread curve (default is False).
            daycounter_convention (DayCounterConvention, optional): The day count convention for the curve (default is ActualActual).

        Returns:
            RateCurve: The bootstrapped interest rate curve.

        Raises:
            KeyError: If the generator key does not exist in the generator map.
        """
        if currency == Currency.EUR:
            generator_map = self.eur_generator_map
        products = []
        pricers = []
        for i, generator in enumerate(generators):
            builder = generator_map[generator]
            product = builder.build(Settings.evaluation_date, quotes[i], maturities[i])
            products.append(product)
            pricer = PricerAssignment.create(product, self._curve_map)
            pricers.append(pricer)
        pillars = []
        zero_rates = []
        for i in range(len(maturities)):
            pillars.append(
                self.day_counter.year_fraction(
                    Settings.evaluation_date, products[i].end_date
                )
            )
            zero_rates.append(0.01)
        if is_spread_curve:  # TODO basis curve bootstrapping
            # base_curve = market_data[base_curve_name]
            # bootstrapping_curve = SpreadCurve(pillars, zero_rates, base_curve)
            pass
        else:
            bootstrapping_curve = RateCurve(
                Settings.evaluation_date,
                pillars,
                zero_rates,
                interpolation,
                daycounter_convention,
            )
        market_data[curve_name] = bootstrapping_curve
        func = ObjectiveFunction(
            bootstrapping_curve, products, pricers, market_data
        )
        x = numpy.array(zero_rates).astype(numpy.float64)  # initial guess
        bootstrapped_rates, rates_jac = newton(func, x)
        bootstrapping_curve._set_rates(bootstrapped_rates)
        bootstrapping_curve.jacobian = rates_jac
        return bootstrapping_curve


class ObjectiveFunction:
    """
    Represents the objective function used during the curve bootstrapping process.

    Attributes:
        trade_date (datetime.date): The date of the trade or evaluation.
        rate_curve (RateCurve): The rate curve being bootstrapped.
        products (list[Product]): A list of products to price during the bootstrap.
        pricers (list[Pricer]): A list of pricers for the given products.
        curves (dict[str, RateCurve]): A dictionary of existing market curves.
    """

    def __init__(
        self,
        # trade_date: datetime.date,
        rate_curve: RateCurve,
        products: list[Product],
        pricers: list[Pricer],
        curves: dict[str, RateCurve],
    ):
        """
        Initializes the ObjectiveFunction with trade date, rate curve, products, pricers, and market curves.

        Args:
            trade_date (datetime.date): The date of the trade or evaluation.
            rate_curve (RateCurve): The rate curve being bootstrapped.
            products (list[Product]): A list of products to price.
            pricers (list[Pricer]): A list of pricers corresponding to the products.
            curves (dict[str, RateCurve]): A dictionary of existing market curves.

        """
        # self.trade_date = trade_date
        self.rate_curve = rate_curve
        self.products = products
        self.pricers = pricers
        self.curves = curves

    def __call__(self, x):
        """
        Evaluates the objective function for the given set of rates and calculates Jacobian matrix.

        Args:
            x (numpy.ndarray): The current set of rates for bootstrapping.

        Returns:
            tuple: A tuple containing:
                - res (numpy.ndarray): The pricing results for the products.
                - jac (numpy.ndarray): The Jacobian matrix representing rate sensitivities.
        """
        self.rate_curve._set_rates(x)
        res = numpy.zeros(len(self.pricers))
        jac = numpy.zeros((len(x), len(x)))
        for i, (pricer, product) in enumerate(zip(self.pricers, self.products)):
            # pv, tape = pricer.price(product, self.trade_date, self.curves, True)
            pricer.price(product, self.curves, True)
            gradients = pricer.tape.gradient(product.price, [self.rate_curve._rates])
            res[i] = product.price
            jac[i, :] = numpy.array(gradients[0])
        return res, numpy.nan_to_num(jac, nan=0.0)
