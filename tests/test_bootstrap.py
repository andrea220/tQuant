"""CurveBootstrap: strip ESTR (deposit + OIS) then EUR 6M (FRA + swaps) and reprice the instruments to zero."""

import unittest
from datetime import date

import tensorflow as tf

import tensorquant as tq
from tensorquant.markethandles.bootstrapping import CurveBootstrap
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
