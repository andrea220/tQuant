"""Risk factors live in market objects as float64 tf.Variables (refactor phase 1)."""

import unittest
from datetime import date

import numpy as np
import tensorflow as tf

from tensorquant.markethandles.dividendcurve import DividendCurve
from tensorquant.markethandles.ircurve import FlatCurve, RateCurve
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.utils import Currency
from tensorquant.markethandles.volatilitysurface import (
    BlackConstantVolatility,
    VolatilitySurface,
)
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

EVAL_DATE = date(2026, 1, 2)
STRIKES = [80.0, 100.0, 120.0]
MATURITIES = [0.5, 1.0, 2.0]
VOLS = [
    [0.24, 0.22, 0.23],
    [0.23, 0.21, 0.22],
    [0.22, 0.20, 0.21],
]


def make_curve() -> RateCurve:
    return RateCurve(
        reference_date=EVAL_DATE,
        pillars=[0.25, 1.0, 2.0, 5.0],
        rates=[0.02, 0.022, 0.023, 0.025],
        interp="LINEAR",
        daycounter_convention=DayCounterConvention.Actual365,
    )


def make_surface() -> VolatilitySurface:
    return VolatilitySurface(
        reference_date=EVAL_DATE,
        calendar=None,
        daycounter=DayCounter(DayCounterConvention.Actual365),
        strike=STRIKES,
        maturity=MATURITIES,
        volatility_matrix=VOLS,
    )


def make_market_dict() -> dict:
    return {
        "IR:EUR:ESTR:SPOT": make_curve(),
        "EQ:EUR:SX5E:SPOT": 5000.0,
        "EQ:EUR:SX5E:VOL": make_surface(),
        "EQ:EUR:SX5E:REPO": 0.01,
        "EQ:EUR:SX5E:DIVYIELD": 0.02,
        "EQ:EUR:SX5E:DIV": DividendCurve(
            reference_date=EVAL_DATE,
            ex_dates=[date(2026, 6, 15)],
            amounts=[1.2],
            currency=Currency.EUR,
            daycounter_convention=DayCounterConvention.Actual365,
        ),
    }


class TestEquityScalars(unittest.TestCase):
    def setUp(self):
        self.raw = make_market_dict()
        self.mkt = MarketEnvironment(market=self.raw)

    def test_scalars_are_float64_variables(self):
        for getter in (self.mkt.get_eq_spot, self.mkt.get_eq_repo, self.mkt.get_eq_div_yield):
            value = getter("SX5E", Currency.EUR)
            self.assertIsInstance(value, tf.Variable)
            self.assertEqual(value.dtype, tf.float64)
        self.assertEqual(float(self.mkt.get_eq_spot("SX5E", Currency.EUR)), 5000.0)

    def test_same_variable_on_every_call(self):
        self.assertIs(
            self.mkt.get_eq_spot("SX5E", Currency.EUR),
            self.mkt.get_eq_spot("SX5E", Currency.EUR),
        )

    def test_input_dict_not_mutated(self):
        self.assertIsInstance(self.raw["EQ:EUR:SX5E:SPOT"], float)

    def test_existing_float64_variable_is_reused(self):
        spot = tf.Variable(4000.0, dtype=tf.float64)
        market = make_market_dict()
        market["EQ:EUR:SX5E:SPOT"] = spot
        mkt = MarketEnvironment(market=market)
        self.assertIs(mkt.get_eq_spot("SX5E", Currency.EUR), spot)


class TestRiskFactors(unittest.TestCase):
    def setUp(self):
        self.mkt = MarketEnvironment(market=make_market_dict())

    def test_keys_and_shapes(self):
        rf = self.mkt.risk_factors()
        self.assertEqual(
            sorted(rf),
            [
                "EQ:EUR:SX5E:DIV",
                "EQ:EUR:SX5E:DIVYIELD",
                "EQ:EUR:SX5E:REPO",
                "EQ:EUR:SX5E:SPOT",
                "EQ:EUR:SX5E:VOL",
                "IR:EUR:ESTR:SPOT",
            ],
        )
        self.assertEqual(len(rf["IR:EUR:ESTR:SPOT"]), 4)
        self.assertEqual(rf["EQ:EUR:SX5E:VOL"][0].shape, (3, 3))
        for variables in rf.values():
            for v in variables:
                self.assertIsInstance(v, tf.Variable)
                self.assertEqual(v.dtype, tf.float64)

    def test_risk_factors_are_the_market_objects(self):
        rf = self.mkt.risk_factors()
        self.assertIs(rf["EQ:EUR:SX5E:SPOT"][0], self.mkt.get_eq_spot("SX5E", Currency.EUR))
        surface = self.mkt.get_eq_vol_surface("SX5E", Currency.EUR)
        self.assertIs(rf["EQ:EUR:SX5E:VOL"][0], surface.volatility_matrix)
        curve = self.mkt.get_ir_curve(Currency.EUR)
        for a, b in zip(rf["IR:EUR:ESTR:SPOT"], curve.variables):
            self.assertIs(a, b)

    def test_one_gradient_call_over_all_factors(self):
        spot = self.mkt.get_eq_spot("SX5E", Currency.EUR)
        curve = self.mkt.get_ir_curve(Currency.EUR)
        surface = self.mkt.get_eq_vol_surface("SX5E", Currency.EUR)
        with tf.GradientTape() as tape:
            value = spot * surface.volatility(100.0, 1.0) * curve.discount(1.5)
        grads = tape.gradient(value, self.mkt.risk_factors())
        self.assertGreater(float(grads["EQ:EUR:SX5E:SPOT"][0]), 0.0)
        self.assertIsNone(grads["EQ:EUR:SX5E:REPO"][0])
        self.assertEqual(grads["EQ:EUR:SX5E:VOL"][0].shape, (3, 3))


class TestRateCurveVariables(unittest.TestCase):
    def test_set_rates_keeps_identity(self):
        curve = make_curve()
        before = curve.variables
        curve._set_rates([0.03, 0.031, 0.032, 0.033])
        for a, b in zip(before, curve.variables):
            self.assertIs(a, b)
        self.assertEqual(curve.rates, [0.03, 0.031, 0.032, 0.033])

    def test_set_rates_wrong_length(self):
        with self.assertRaises(ValueError):
            make_curve()._set_rates([0.03])

    def test_rates_follow_assign(self):
        curve = make_curve()
        curve.variables[0].assign(0.05)
        self.assertEqual(curve.rates[0], 0.05)
        self.assertEqual(curve.nodes[0][1], 0.05)

    def test_pillar_gradient_matches_finite_difference(self):
        curve = make_curve()
        T, h = 1.5, 1e-6
        with tf.GradientTape() as tape:
            df = curve.discount(T)
        grads = [0.0 if g is None else float(g) for g in tape.gradient(df, curve.variables)]
        for i, var in enumerate(curve.variables):
            base = float(var)
            var.assign(base + h)
            up = float(curve.discount(T))
            var.assign(base - h)
            dn = float(curve.discount(T))
            var.assign(base)
            self.assertAlmostEqual(grads[i], (up - dn) / (2 * h), places=6)
        # Linear interpolation on zero rates: bucket sum = parallel sensitivity -T * DF
        self.assertAlmostEqual(sum(grads), -T * float(df), places=10)

    def test_flat_curve_single_variable(self):
        curve = FlatCurve(EVAL_DATE, 0.03, DayCounterConvention.Actual365)
        self.assertEqual(len(curve.variables), 1)
        with tf.GradientTape() as tape:
            df = curve.discount(2.0)
        (grad,) = tape.gradient(df, curve.variables)
        self.assertAlmostEqual(float(grad), -2.0 * float(df), places=10)
        curve.rate = 0.04
        self.assertEqual(curve.rate, 0.04)
        self.assertAlmostEqual(float(curve.discount(2.0)), np.exp(-0.08), places=12)


class TestVolatilitySurfaceVariables(unittest.TestCase):
    def setUp(self):
        self.surface = make_surface()

    def test_matrix_is_float64_variable(self):
        m = self.surface.volatility_matrix
        self.assertIsInstance(m, tf.Variable)
        self.assertEqual(m.dtype, tf.float64)
        np.testing.assert_array_equal(m.numpy(), np.array(VOLS))

    def test_values(self):
        self.assertAlmostEqual(float(self.surface.volatility(100.0, 1.0)), 0.21, places=15)
        # bilinear between (K=100,120) x (T=1,2)
        K, T = 110.0, 1.5
        expected = 0.25 * (0.21 + 0.22 + 0.20 + 0.21)
        self.assertAlmostEqual(float(self.surface.volatility(K, T)), expected, places=15)
        # flat extrapolation
        self.assertAlmostEqual(float(self.surface.volatility(200.0, 5.0)), 0.21, places=15)

    def test_float32_strike_input(self):
        k = np.float32(100.0)
        self.assertAlmostEqual(float(self.surface.volatility(k, 1.0)), 0.21, places=15)

    def test_gradient_is_bilinear_weights(self):
        K, T = 90.0, 0.75  # weights 0.25 on each of the 4 surrounding nodes
        with tf.GradientTape() as tape:
            vol = self.surface.volatility(K, T)
        grad = tape.gradient(vol, self.surface.volatility_matrix).numpy()
        expected = np.zeros((3, 3))
        expected[0:2, 0:2] = 0.25
        np.testing.assert_allclose(grad, expected, atol=1e-15)
        self.assertAlmostEqual(grad.sum(), 1.0, places=15)

    def test_black_constant_volatility(self):
        flat = BlackConstantVolatility(EVAL_DATE, 0.2)
        self.assertIs(flat.volatility(100.0, 1.0), flat.volatility_matrix)
        self.assertIs(flat.flat_vol, flat.volatility_matrix)
        self.assertEqual(flat.volatility_matrix.dtype, tf.float64)
        self.assertEqual(flat.variables, [flat.volatility_matrix])


if __name__ == "__main__":
    unittest.main()
