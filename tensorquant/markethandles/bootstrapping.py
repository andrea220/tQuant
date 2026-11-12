import datetime
from pathlib import Path

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
from ..timehandles.countrycalendars import UnitedKingdom, UnitedStates
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


A360, A365, T360 = DayCounterConvention.Actual360, DayCounterConvention.Actual365, DayCounterConvention.Thirty360

# Market conventions of the curves bootstrapped from quotes (``ir_quotes.csv``): OIS curves (overnight fixing as an
# O/N deposit, then OIS with annual fixed and compounded overnight legs) and Ibor curves (the fixing as a spot FRA on
# the index tenor, FRAs, swaps against the index, discounted on the currency's OIS curve), each on its calendar
# (TARGET, Federal Reserve, England).
QUOTE_CONVENTIONS = {
    ("EUR", "ESTR"): {"kind": "ois", "spot": 2, "day_count": A360, "calendar": TARGET},
    ("USD", "SOFR"): {"kind": "ois", "spot": 2, "day_count": A360, "calendar": UnitedStates},
    ("GBP", "SONIA"): {"kind": "ois", "spot": 0, "day_count": A365, "calendar": UnitedKingdom},
    ("EUR", "6M"): {"kind": "ibor", "spot": 2, "index_months": 6, "fixed_period": "1Y", "fixed_day_count": T360,
                    "float_day_count": A360, "discount": "ESTR", "calendar": TARGET},
}


def _quote_instrument(ccy: Currency, conv: dict, instrument: str, tenor: str, quote: float, today: datetime.date):
    """The calibration instrument of one quote (notional 1)."""
    cal, mf = conv["calendar"](), BusinessDayConvention.ModifiedFollowing
    if conv["kind"] == "ois":
        if instrument == "fixing":                               # overnight fixing: O/N deposit from today
            return DepositGenerator(ccy, 0, mf, conv["day_count"], 1.0, cal).build(today, quote, "ON")
        if instrument != "ois":
            raise ValueError(f"OIS curve: unexpected instrument '{instrument}'")
        return OisGenerator(ccy, conv["spot"], "1Y", "1Y", mf, 1.0, conv["day_count"], conv["day_count"], cal,
                            OvernightIndex(cal, ccy)).build(today, quote, tenor)
    months = conv["index_months"]
    index = IborIndex(cal, months, TimeUnit.Months, ccy)
    if instrument in ("fixing", "fra"):
        fra = FraGenerator(ccy, conv["spot"], conv["spot"], f"{months}M", mf, 1.0, conv["float_day_count"], cal, index)
        end = months if instrument == "fixing" else _tenor_months(tenor)
        return fra.build(today, quote, f"{end - months}M-{end}M")    # the fixing: a spot FRA on the index tenor
    if instrument == "swap":
        return SwapGenerator(ccy, conv["spot"], conv["fixed_period"], f"{months}M", mf, 1.0, conv["fixed_day_count"],
                             conv["float_day_count"], cal, index).build(today, quote, tenor)
    raise ValueError(f"Ibor curve: unexpected instrument '{instrument}'")


def _tenor_months(tenor: str) -> int:
    n, unit = int(tenor[:-1]), tenor[-1].upper()
    if unit not in "MY":
        raise ValueError(f"tenor '{tenor}': months or years expected")
    return n * (12 if unit == "Y" else 1)


def bootstrap_from_quotes(quotes: pd.DataFrame, evaluation_date: datetime.date,
                          market_env: MarketEnvironment, interpolation: str = "MONOTONE_CONVEX") -> dict:
    """Bootstrap the curves of a quotes table (``ir_quotes.csv``: ``ccy, curve, tenor, instrument, quote``) and register
    them in *market_env* (OIS curves first: the Ibor curves are discounted on them).

    Each curve: one pillar per instrument end date (zero rates, Actual365; *interpolation* ``"MONOTONE_CONVEX"``,
    Hagan-West, continuous forwards, or ``"LINEAR"``), all the instruments repriced to
    zero at once (Newton, Jacobian by AAD, as :class:`CurveBootstrap`). Kept on the curve for the par
    sensitivities: ``calibration_instruments`` (``[(label, product)]``), ``jacobian`` (``dNPV_i / dr_j``, instruments x
    pillars, labelled by instrument) and ``calib_quote_dv01`` (``dNPV_i / dq_i``, 1bp bump of the quote).

    Args:
        quotes: The quotes (decimal rates).
        evaluation_date: Curve reference date (``Settings.evaluation_date`` during the bootstrap, restored after).
        market_env: Where the curves are registered (``IR:<CCY>:<curve>:SPOT``).

    Returns:
        dict: ``{market key: RateCurve}``.

    Raises:
        ValueError: A curve without conventions (``QUOTE_CONVENTIONS``), an unknown instrument, or an Ibor curve whose
            discount curve is neither in the quotes nor in *market_env*.
    """
    previous = Settings.evaluation_date
    Settings.evaluation_date = evaluation_date
    try:
        groups = list(quotes.groupby(["ccy", "curve"], sort=False))
        for (ccy, name), _ in groups:
            if (ccy, name) not in QUOTE_CONVENTIONS:
                raise ValueError(f"no conventions for the curve {ccy} {name} (QUOTE_CONVENTIONS)")
        groups.sort(key=lambda g: QUOTE_CONVENTIONS[g[0]]["kind"] != "ois")      # OIS first
        day_counter = DayCounter(A365)
        out = {}
        for (ccy, name), rows in groups:
            conv = QUOTE_CONVENTIONS[(ccy, name)]
            currency = Currency[ccy]
            if conv["kind"] == "ibor":
                discount_key = f"IR:{ccy}:{conv['discount']}:SPOT"
                if discount_key not in out:
                    try:
                        market_env.get_ir_curve(currency, conv["discount"])
                    except (ValueError, KeyError):
                        raise ValueError(f"curve {ccy} {name}: discount curve {conv['discount']} missing") from None
            instruments = []
            for r in rows.itertuples(index=False):
                product = _quote_instrument(currency, conv, r.instrument, r.tenor, float(r.quote), evaluation_date)
                instruments.append((f"{r.instrument} {r.tenor}", product, r))
            instruments.sort(key=lambda x: x[1].end_date)
            seen, unique, source_rows = set(), [], []
            for label, product, row in instruments:               # one pillar per end date
                if product.end_date not in seen:
                    seen.add(product.end_date)
                    unique.append((label, product))
                    source_rows.append(row)
            products = [p for _, p in unique]
            pricers = [PricerAssignment.create(p) for p in products]
            pillars = [day_counter.year_fraction(evaluation_date, p.end_date) for p in products]
            curve = RateCurve(evaluation_date, pillars, [0.02] * len(pillars), interpolation, A365)
            key = f"IR:{ccy}:{name}:SPOT"
            market_env.set_ir_curve(key, curve)
            func = ObjectiveFunction(curve, products, pricers, market_env)
            rates, jac = newton(func, numpy.full(len(pillars), 0.02))
            curve._set_rates(rates)
            labels = [label for label, _ in unique]
            curve.jacobian = pd.DataFrame(jac, index=labels, columns=labels)
            curve.calibration_instruments = unique
            bump, dv01 = 1e-4, []                         # dNPV_i / dq_i, the curve coefficients computed once
            with curve.pinned():
                for (label, product), pricer, row in zip(unique, pricers, source_rows):
                    bumped = _quote_instrument(currency, conv, row.instrument, row.tenor, float(row.quote) + bump,
                                               evaluation_date)
                    dv01.append((float(pricer.calculate_price(bumped, market_env))
                                 - float(pricer.calculate_price(product, market_env))) / bump)
            curve.calib_quote_dv01 = pd.Series(dv01, index=labels)
            curve.calibration_quotes = pd.Series([float(r.quote) for r in source_rows], index=labels)
            out[key] = curve
        return out
    finally:
        Settings.evaluation_date = previous


def calibration_frames(curves: dict) -> tuple:
    """Jacobian and calibration table of bootstrapped curves, long format.

    Returns:
        tuple: ``ir_curves_jacobian.csv`` (``ccy, curve, instrument, pillar, value``: ``dNPV_instrument / d zero
        rate at the pillar of instrument pillar``) and ``ir_calibration.csv`` (``ccy, curve, instrument, end_date, quote,
        quote_dv01``) frames.
    """
    jac_rows, cal_rows = [], []
    for key, curve in curves.items():
        _, ccy, name, _ = key.split(":")
        for label in curve.jacobian.index:
            for pillar in curve.jacobian.columns:
                jac_rows.append({"ccy": ccy, "curve": name, "instrument": label, "pillar": pillar,
                                 "value": float(curve.jacobian.loc[label, pillar])})
        for (label, product) in curve.calibration_instruments:
            cal_rows.append({"ccy": ccy, "curve": name, "instrument": label, "end_date": product.end_date.isoformat(),
                             "quote": float(curve.calibration_quotes[label]),
                             "quote_dv01": float(curve.calib_quote_dv01[label])})
    return (pd.DataFrame(jac_rows, columns=["ccy", "curve", "instrument", "pillar", "value"]),
            pd.DataFrame(cal_rows, columns=["ccy", "curve", "instrument", "end_date", "quote", "quote_dv01"]))


def bootstrap_snapshot(folder) -> list:
    """Bootstrap the curves of a snapshot folder (``<yyyymmdd>/<set>/`` with ``ir_quotes.csv``) and write
    ``ir_curves.csv`` (zero rates at the pillar dates), ``ir_curves_jacobian.csv`` and ``ir_calibration.csv`` next to
    the quotes: a later ``MarketEnvironment.from_folder`` reads the curves without bootstrapping again.

    Returns:
        list: The files written.
    """
    folder = Path(folder)
    quotes = pd.read_csv(folder / "ir_quotes.csv")
    evaluation_date = datetime.datetime.strptime(folder.parent.name, "%Y%m%d").date()
    env = MarketEnvironment({})
    curves = bootstrap_from_quotes(quotes, evaluation_date, env)
    previous = Settings.evaluation_date
    Settings.evaluation_date = evaluation_date
    try:
        env.evaluation_date = evaluation_date
        tables = env.to_frames()
    finally:
        Settings.evaluation_date = previous
    jacobian, calibration = calibration_frames(curves)
    tables["ir_curves.csv"].to_csv(folder / "ir_curves.csv", index=False)
    jacobian.to_csv(folder / "ir_curves_jacobian.csv", index=False)
    calibration.to_csv(folder / "ir_calibration.csv", index=False)
    return ["ir_curves.csv", "ir_curves_jacobian.csv", "ir_calibration.csv"]


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
        self.market_env.set_ir_curve(curve_name, bootstrapping_curve)
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
        # all the instruments priced under one tape, the curve's interpolation coefficients computed once
        with tf.GradientTape(persistent=True) as tape:
            with self.rate_curve.pinned():
                npvs = [pricer.calculate_price(product, self.market_env)
                        for pricer, product in zip(self.pricers, self.products)]
        for i, npv in enumerate(npvs):
            # None means that variable did not contribute to NPV_i -> 0
            gradients = tape.gradient(npv, self.rate_curve.variables)
            res[i] = float(npv)
            jac[i, :] = [float(g.numpy()) if g is not None else 0.0 for g in gradients]
        del tape
        return res, jac
