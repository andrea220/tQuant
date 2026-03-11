import datetime
import numpy
import tensorflow as tf
import pandas as pd

from .ircurve import RateCurve
from .marketenvironment import MarketEnvironment
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
        market_env (MarketEnvironment): The market environment used for pricing during bootstrap.
        eur_generator_map (dict): A map of instrument generators for EUR instruments.
    """

    def __init__(
        self,
        evaluation_date: datetime.date,
        daycount_convention: DayCounterConvention,
        market_env: MarketEnvironment,
    ) -> None:
        """
        Initializes the CurveBootstrap instance with evaluation date, day count convention,
        and market environment.

        Args:
            evaluation_date (datetime.date): The curve evaluation date.
            daycount_convention (DayCounterConvention): The day count convention to use.
            market_env (MarketEnvironment): The market environment providing access to
                existing market data (curves, spots, volatilities). The bootstrapped curve
                will be added to its internal market dict during strip().

        """
        Settings.evaluation_date = evaluation_date
        self.day_counter = DayCounter(daycount_convention)
        self.market_env = market_env
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
        is_spread_curve: bool = False,
        daycounter_convention=DayCounterConvention.ActualActual,
    ):
        """
        Bootstraps an interest rate curve from market quotes and instrument generators.

        The bootstrapped curve is registered in the market environment under ``curve_name``
        so that subsequent ``strip()`` calls can reference it as a discount / forward curve.

        Args:
            generators (list[str]): A list of instrument generators (e.g., "Dp" for deposit, "Os" for OIS).
            maturities (list[str]): A list of instrument maturities (e.g., "1M", "6M", "1Y").
            quotes (list[float]): A list of market quotes corresponding to the instruments.
            curve_name (str): The market key for the bootstrapped curve (e.g., "IR:EUR:ESTR:SPOT").
            currency (Currency): The currency of the curve (e.g., EUR).
            interpolation (str, optional): The type of interpolation (default is "LINEAR").
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
            pricer = PricerAssignment.create(product)
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
        self.market_env._market[curve_name] = bootstrapping_curve
        func = ObjectiveFunction(bootstrapping_curve, products, pricers, self.market_env)
        x = numpy.array(zero_rates, dtype=numpy.float64)
        bootstrapped_rates, jac = newton(func, x)
        bootstrapping_curve._set_rates(bootstrapped_rates)
        instrument_labels = [f"{gen}_{mat}" for gen, mat in zip(generators, maturities)]
        bootstrapping_curve.jacobian = pd.DataFrame(
            jac,
            index=instrument_labels,    # righe  : dNPV_i/dr_j  →  strumento i-esimo
            columns=instrument_labels,  # colonne: dNPV_i/dr_j  →  pillar/tasso j-esimo
        )
        # Calcola dNPV_k/dq_k per ogni strumento di calibrazione (normalizzazione par sensitivity).
        # Necessario perché J è costruita con nozionale=1 e la sensibilità al par rate non è
        # unitaria ma scala con l'annuity dello strumento (~T per brevi, ~N*avg_df per N anni).
        bump = 1e-4
        calib_quote_dv01 = numpy.zeros(len(products))
        for k in range(len(products)):
            builder = generator_map[generators[k]]
            product_up = builder.build(
                Settings.evaluation_date, float(quotes[k]) + bump, maturities[k]
            )
            npv_up = float(pricers[k].calculate_price(product_up, self.market_env))
            npv_base = float(pricers[k].calculate_price(products[k], self.market_env))
            calib_quote_dv01[k] = (npv_up - npv_base) / bump
        bootstrapping_curve.calib_quote_dv01 = pd.Series(
            calib_quote_dv01, index=instrument_labels
        )
        return bootstrapping_curve


class ObjectiveFunction:
    """
    Global objective function for curve bootstrapping.

    Each call prices all N instruments at the current rate vector and returns
    the NPV vector together with the NxN Jacobian computed via TensorFlow
    autodiff (GradientTape).  None gradients (rates not used by a given
    instrument) are treated as zero.

    Attributes:
        rate_curve (RateCurve): The rate curve being bootstrapped.
        products (list[Product]): A list of products to price during the bootstrap.
        pricers (list[Pricer]): A list of pricers for the given products.
        market_env (MarketEnvironment): The market environment used for pricing.
    """

    def __init__(
        self,
        rate_curve: RateCurve,
        products: list[Product],
        pricers: list[Pricer],
        market_env: MarketEnvironment,
    ):
        """
        Initializes the ObjectiveFunction with rate curve, products, pricers, and market environment.

        Args:
            rate_curve (RateCurve): The rate curve being bootstrapped.
            products (list[Product]): A list of products to price.
            pricers (list[Pricer]): A list of pricers corresponding to the products.
            market_env (MarketEnvironment): The market environment providing access
                to market data (curves, spots, volatilities).
        """
        self.rate_curve = rate_curve
        self.products = products
        self.pricers = pricers
        self.market_env = market_env

    def __call__(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Price all instruments at rates *x* and return NPVs and their Jacobian.

        Args:
            x (numpy.ndarray): Current rate vector (one entry per pillar).

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]:
                - ``res``: NPV vector of shape ``(N,)``.
                - ``jac``: Jacobian matrix of shape ``(N, N)`` where
                  ``jac[i, j] = dNPV_i / dr_j``.  Entries corresponding to
                  unused rates (None autodiff gradients) are set to zero.
        """
        self.rate_curve._set_rates(x)
        n = len(self.pricers)
        res = numpy.zeros(n)
        jac = numpy.zeros((n, n))
        for i, (pricer, product) in enumerate(zip(self.pricers, self.products)):
            with tf.GradientTape() as tape:
                npv = pricer.calculate_price(product, self.market_env)
            # tape.gradient returns one gradient per watched variable;
            # None means that variable did not contribute to NPV_i → treat as 0
            gradients = tape.gradient(npv, self.rate_curve._rates)
            res[i] = float(npv)
            jac[i, :] = [
                float(g.numpy()) if g is not None else 0.0 for g in gradients
            ]
        return res, jac
