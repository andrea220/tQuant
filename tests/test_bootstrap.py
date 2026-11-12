"""CurveBootstrap: strip ESTR (deposit + OIS) then EUR 6M (FRA + swaps) and reprice the instruments to zero."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorquant as tq
from tensorquant.markethandles.bootstrapping import CurveBootstrap, bootstrap_from_quotes, bootstrap_snapshot
from tensorquant.markethandles.marketenvironment import MarketEnvironment

EVAL_DATE = date(2026, 5, 15)
ESTR = (["Dp", "Os", "Os", "Os", "Os"], ["6M", "1Y", "2Y", "5Y", "10Y"], [0.0200, 0.0210, 0.0225, 0.0245, 0.0265])
EUR6M = (["Fr", "Sw", "Sw", "Sw"], ["6M-12M", "2Y", "5Y", "10Y"], [0.0240, 0.0250, 0.0270, 0.0290])


class TestBootstrap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tq.Settings.evaluation_date = EVAL_DATE
        placeholder = tq.RateCurve(EVAL_DATE, [1.0], [0.0], "LINEAR", tq.DayCounterConvention.Actual365)
        cls.mkt = MarketEnvironment({"IR:EUR:ESTR:SPOT": placeholder, "IR:EUR:6M:SPOT": placeholder})
        cls.boot = CurveBootstrap(EVAL_DATE, tq.DayCounterConvention.Actual365, cls.mkt)
        cls.estr = cls.boot.strip(*ESTR, "IR:EUR:ESTR:SPOT", tq.Currency.EUR,
                                  daycounter_convention=tq.DayCounterConvention.Actual365)
        cls.eur6m = cls.boot.strip(*EUR6M, "IR:EUR:6M:SPOT", tq.Currency.EUR,
                                   daycounter_convention=tq.DayCounterConvention.Actual365)

    def test_curves_registered_in_market(self):
        self.assertIs(self.mkt.get_ir_curve(tq.Currency.EUR), self.estr)
        self.assertIs(self.mkt.get_ir_curve(tq.Currency.EUR, "6M"), self.eur6m)

    def test_instruments_reprice_to_zero(self):
        tq.Settings.evaluation_date = EVAL_DATE
        for generators, maturities, quotes in (ESTR, EUR6M):
            for gen, mat, quote in zip(generators, maturities, quotes):
                product = self.boot.eur_generator_map[gen].build(EVAL_DATE, quote, mat)
                tq.PricerAssignment.create(product).price(product, self.mkt)
                with self.subTest(instrument=f"{gen}_{mat}"):
                    self.assertAlmostEqual(float(product.price), 0.0, delta=1e-9)

    def test_rates_are_float64_variables(self):
        for curve in (self.estr, self.eur6m):
            for var in curve.variables:
                self.assertIsInstance(var, tf.Variable)
                self.assertEqual(var.dtype, tf.float64)

    def test_set_ir_curve_validates(self):
        with self.assertRaises(ValueError):
            self.mkt.set_ir_curve("EQ:EUR:X:SPOT", self.estr)
        with self.assertRaises(TypeError):
            self.mkt.set_ir_curve("IR:EUR:ESTR:SPOT", 1.0)


def market_quotes():
    """Quotes in the ``ir_quotes.csv`` layout (as ``marketdata`` writes them from the Bloomberg ICVS curves): fixings,
    FRAs, OIS and swaps of EUR ESTR, EUR 6M, USD SOFR, GBP SONIA (mid, decimal)."""
    rows = []
    for ccy, curve, fixing, ois in (("EUR", "ESTR", None, 0.0245), ("USD", "SOFR", None, 0.039),
                                    ("GBP", "SONIA", 0.0398, 0.040)):
        if fixing is not None:
            rows.append((ccy, curve, "1D", "fixing", fixing))
        for k, tenor in enumerate(["1W", "1M", "3M", "6M", "1Y", "18M", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y",
                                   "30Y"]):
            rows.append((ccy, curve, tenor, "ois", ois + 0.0005 * np.sqrt(k)))
    rows.append(("EUR", "6M", "6M", "fixing", 0.0302))
    for k, tenor in enumerate(["7M", "9M", "12M", "15M", "18M"]):
        rows.append(("EUR", "6M", tenor, "fra", 0.0313 + 0.0006 * k))
    for k, tenor in enumerate(["2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]):
        rows.append(("EUR", "6M", tenor, "swap", 0.0336 + 0.0003 * np.sqrt(k)))
    return pd.DataFrame(rows, columns=["ccy", "curve", "tenor", "instrument", "quote"])


class TestBootstrapFromQuotes(unittest.TestCase):
    """``bootstrap_from_quotes``: market conventions per curve (EUR / USD OIS annual ACT/360 T+2, SONIA ACT/365 T+0,
    EUR 6M swaps annual 30/360 against 6M ACT/360, discounted on ESTR), every instrument repriced to zero."""

    @classmethod
    def setUpClass(cls):
        tq.Settings.evaluation_date = EVAL_DATE
        cls.mkt = MarketEnvironment({})
        cls.curves = bootstrap_from_quotes(market_quotes(), EVAL_DATE, cls.mkt)

    def test_curves_registered(self):
        self.assertEqual(sorted(self.curves), ["IR:EUR:6M:SPOT", "IR:EUR:ESTR:SPOT", "IR:GBP:SONIA:SPOT",
                                               "IR:USD:SOFR:SPOT"])
        self.assertIs(self.mkt.get_ir_curve(tq.Currency.EUR), self.curves["IR:EUR:ESTR:SPOT"])
        self.assertIs(self.mkt.get_ir_curve(tq.Currency.EUR, "6M"), self.curves["IR:EUR:6M:SPOT"])
        self.assertIs(self.mkt.get_ir_curve(tq.Currency.GBP), self.curves["IR:GBP:SONIA:SPOT"])

    def test_every_instrument_reprices_to_zero(self):
        tq.Settings.evaluation_date = EVAL_DATE
        for key, curve in self.curves.items():
            for label, product in curve.calibration_instruments:
                tq.PricerAssignment.create(product).price(product, self.mkt)
                with self.subTest(curve=key, instrument=label):
                    self.assertAlmostEqual(float(product.price), 0.0, delta=1e-9)

    def test_conventions(self):
        instruments = dict(self.curves["IR:EUR:6M:SPOT"].calibration_instruments)
        swap = instruments["swap 5Y"]
        self.assertEqual(swap._day_counter_fix.day_counter_convention, tq.DayCounterConvention.Thirty360)
        sonia = dict(self.curves["IR:GBP:SONIA:SPOT"].calibration_instruments)["ois 1Y"]
        self.assertEqual(sonia.start_date, EVAL_DATE)                                  # T+0
        estr = dict(self.curves["IR:EUR:ESTR:SPOT"].calibration_instruments)["ois 1Y"]
        self.assertEqual(estr.start_date, tq.TARGET().advance(EVAL_DATE, 2, tq.TimeUnit.Days,
                                                              tq.BusinessDayConvention.ModifiedFollowing))

    def test_market_from_frames_with_quotes(self):
        frames = {"ir_quotes.csv": market_quotes()}
        mkt = MarketEnvironment.from_frames(frames, EVAL_DATE)
        for key, curve in self.curves.items():
            built = mkt.get_ir_curve(tq.Currency[key.split(":")[1]], key.split(":")[2])
            np.testing.assert_allclose([float(v) for v in built.variables], [float(v) for v in curve.variables],
                                       atol=1e-12)

    def test_calendars_per_currency(self):
        sofr = dict(self.curves["IR:USD:SOFR:SPOT"].calibration_instruments)["ois 1Y"]
        self.assertEqual(sofr.start_date, tq.UnitedStates().advance(EVAL_DATE, 2, tq.TimeUnit.Days,
                                                                    tq.BusinessDayConvention.ModifiedFollowing))
        # 2026-05-25 is Memorial Day: a USD date rolls past it, a EUR one does not
        self.assertFalse(tq.UnitedStates().is_business_day(date(2026, 5, 25)))
        self.assertTrue(tq.TARGET().is_business_day(date(2026, 5, 25)))

    def test_quote_dv01(self):
        """dNPV / dq of a 5Y payer OIS (notional 1) is minus its annuity: between -5 and -4."""
        dv01 = self.curves["IR:EUR:ESTR:SPOT"].calib_quote_dv01["ois 5Y"]
        self.assertTrue(-5.0 < dv01 < -4.0)

    def test_stored_bootstrap_round_trip(self):
        """bootstrap_snapshot writes the curves, the Jacobian and the quote DV01s; from_folder reads them back without
        bootstrapping again (same rates, same Jacobian)."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / EVAL_DATE.strftime("%Y%m%d") / "1200"
            folder.mkdir(parents=True)
            market_quotes().to_csv(folder / "ir_quotes.csv", index=False)
            written = bootstrap_snapshot(folder)
            self.assertEqual(written, ["ir_curves.csv", "ir_curves_jacobian.csv", "ir_calibration.csv"])
            mkt = MarketEnvironment.from_folder(tmp)
        for key, curve in self.curves.items():
            _, ccy, name, _ = key.split(":")
            loaded = mkt.get_ir_curve(tq.Currency[ccy], name)
            with self.subTest(curve=key):
                self.assertFalse(hasattr(loaded, "calibration_instruments"))        # read, not bootstrapped
                np.testing.assert_allclose([float(v) for v in loaded.variables], [float(v) for v in curve.variables],
                                           atol=1e-12)
                np.testing.assert_allclose(loaded.jacobian.to_numpy(), curve.jacobian.to_numpy(), rtol=1e-12)
                self.assertEqual(list(loaded.jacobian.index), list(curve.jacobian.index))
                np.testing.assert_allclose(loaded.calib_quote_dv01.to_numpy(), curve.calib_quote_dv01.to_numpy(),
                                           rtol=1e-8)                           # 1bp bumps of two separate runs

    def test_invalid_quotes(self):
        quotes = market_quotes()
        with self.assertRaises(ValueError):                        # a curve without conventions
            bootstrap_from_quotes(quotes.assign(curve=quotes["curve"].replace("SOFR", "FEDFUNDS")), EVAL_DATE,
                                  MarketEnvironment({}))
        with self.assertRaises(ValueError):                        # EUR 6M without its discount curve
            bootstrap_from_quotes(quotes[quotes["curve"] == "6M"], EVAL_DATE, MarketEnvironment({}))


class TestPricerAssignment(unittest.TestCase):
    def test_vanilla_option_gets_black_scholes(self):
        opt = tq.VanillaOption(tq.Currency.EUR, start_date=EVAL_DATE, end_date=date(2027, 5, 15),
                               option_type=tq.OptionType.Call, strike=100.0, underlying="X",
                               exercise_type=tq.ExerciseType.European)
        self.assertIsInstance(tq.PricerAssignment.create(opt), tq.BlackScholesPricer)

    def test_unknown_product_raises(self):
        with self.assertRaises(TypeError):
            tq.PricerAssignment.create(object())


if __name__ == "__main__":
    unittest.main()
