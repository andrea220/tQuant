"""RateCurve / FlatCurve / DividendCurve: values and dtypes (pinned before the dtype refactor)."""

import unittest
from datetime import date, timedelta

import numpy as np
import tensorflow as tf

import tensorquant as tq

from tensorquant.markethandles.dividendcurve import DividendCurve
from tensorquant.markethandles.ircurve import FlatCurve, RateCurve
from tensorquant.markethandles.utils import Currency
from tensorquant.timehandles.daycounter import DayCounterConvention

EVAL_DATE = date(2026, 1, 2)
PILLARS = [0.25, 1.0, 2.0, 5.0]
RATES = [0.02, 0.022, 0.023, 0.025]


def make_curve() -> RateCurve:
    return RateCurve(
        reference_date=EVAL_DATE,
        pillars=PILLARS,
        rates=RATES,
        interp="LINEAR",
        daycounter_convention=DayCounterConvention.Actual365,
    )


def ref_discount(t: float) -> float:
    """Independent reference: linear interpolation of zero rates, flat extrapolation."""
    r = np.interp(t, PILLARS, RATES)  # np.interp extrapolates flat
    return float(np.exp(-r * t))


class TestRateCurveValues(unittest.TestCase):
    def setUp(self):
        self.curve = make_curve()

    def test_discount_matches_reference(self):
        for t in [0.1, 0.25, 0.6, 1.0, 1.5, 3.0, 5.0, 9.0]:
            with self.subTest(t=t):
                self.assertAlmostEqual(float(self.curve.discount(t)), ref_discount(t), places=14)

    def test_discount_at_zero(self):
        self.assertEqual(float(self.curve.discount(0.0)), 1.0)

    def test_discount_date_equals_year_fraction(self):
        d = date(2027, 7, 1)
        t = (d - EVAL_DATE).days / 365.0
        self.assertAlmostEqual(float(self.curve.discount(d)), float(self.curve.discount(t)), places=15)

    def test_discount_rejects_int(self):
        with self.assertRaises(TypeError):
            self.curve.discount(1)

    def test_zero_rate_is_continuously_compounded(self):
        """r(T) = -ln P(0,T) / T: the pillar rates on the pillars, linear in between, flat outside, r(0) = r_1."""
        for t in [0.1, 0.5, 1.5, 7.0]:
            with self.subTest(t=t):
                self.assertAlmostEqual(float(self.curve.zero_rate(t)), -np.log(ref_discount(t)) / t, places=12)
        for t, r in zip(PILLARS, RATES):
            self.assertAlmostEqual(float(self.curve.zero_rate(t)), r, places=12)
        self.assertEqual(float(self.curve.zero_rate(0.0)), RATES[0])

    def test_forward_rate(self):
        t1, t2 = 1.0, 2.5
        expected = (ref_discount(t1) / ref_discount(t2) - 1) / (t2 - t1)
        self.assertAlmostEqual(float(self.curve.forward_rate(t1, t2)), expected, places=12)
        d1, d2 = EVAL_DATE + timedelta(days=365), EVAL_DATE + timedelta(days=730)
        expected_d = (ref_discount(1.0) / ref_discount(2.0) - 1) / 1.0
        self.assertAlmostEqual(float(self.curve.forward_rate(d1, d2)), expected_d, places=12)

    def test_from_zcb_round_trip(self):
        dfs = [ref_discount(t) for t in PILLARS]
        dates = [EVAL_DATE + timedelta(days=int(round(t * 365))) for t in PILLARS]
        curve = RateCurve.from_zcb(EVAL_DATE, dates, dfs, "LINEAR", DayCounterConvention.Actual365)
        for d, df in zip(dates, dfs):
            self.assertAlmostEqual(float(curve.discount(d)), df, places=14)

    def test_flat_curve(self):
        flat = FlatCurve(EVAL_DATE, 0.03, DayCounterConvention.Actual365)
        for t in [0.5, 2.0, 30.0]:
            self.assertAlmostEqual(float(flat.discount(t)), np.exp(-0.03 * t), places=14)


class TestRateCurveDtypes(unittest.TestCase):
    """Policy: market objects return float64 tensors."""

    def setUp(self):
        self.curve = make_curve()

    def assertFloat64Tensor(self, x):
        self.assertTrue(tf.is_tensor(x), f"expected a tensor, got {type(x).__name__}")
        self.assertEqual(x.dtype, tf.float64)

    def test_discount_dtype(self):
        for t in [0.0, 0.1, 1.5, 9.0]:
            with self.subTest(t=t):
                self.assertFloat64Tensor(self.curve.discount(t))
        self.assertFloat64Tensor(self.curve.discount(date(2027, 7, 1)))

    def test_rates_dtype(self):
        self.assertFloat64Tensor(self.curve.zero_rate(1.5))
        self.assertFloat64Tensor(self.curve.forward_rate(1.0, 2.0))

    def test_discount_at_zero_is_differentiable_constant(self):
        with tf.GradientTape() as tape:
            df = self.curve.discount(0.0)
        grads = tape.gradient(df, self.curve.variables)
        self.assertTrue(all(g is None for g in grads))


class TestInstFwd(unittest.TestCase):
    """inst_fwd: exact f(t) = r(t) + t r'(t), right derivative at a pillar, flat zero rate outside the pillars."""

    def setUp(self):
        self.curve = make_curve()

    def test_exact_inside_segments(self):
        # Inside a segment -log P(t) = r(t)·t is quadratic: f(t) = r(t) + t·r'(t).
        cases = {
            0.5: 0.020 + (0.002 / 0.75) * 0.25 + 0.5 * (0.002 / 0.75),
            1.5: 0.0225 + 1.5 * 0.001,
            3.0: 0.023 + (0.002 / 3.0) * 1.0 + 3.0 * (0.002 / 3.0),
            7.0: 0.025,  # flat extrapolation
            0.0: 0.020,  # flat extrapolation on the left
        }
        for t, expected in cases.items():
            with self.subTest(t=t):
                self.assertAlmostEqual(float(self.curve.inst_fwd(t)), expected, places=12)

    def test_pillars_and_short_end(self):
        """At a pillar the right derivative (the next segment's slope); before the first pillar r_1; away from the
        pillars = central difference of -ln P."""
        self.assertAlmostEqual(float(self.curve.inst_fwd(1.0)), 0.022 + 1.0 * 0.001, places=14)          # right
        self.assertAlmostEqual(float(self.curve.inst_fwd(0.25)), 0.020 + 0.25 * 0.002 / 0.75, places=14)
        self.assertAlmostEqual(float(self.curve.inst_fwd(0.004)), 0.020, places=14)
        h = 1e-5
        for t in (0.6, 1.4, 3.3):
            fd = -(np.log(ref_discount(t + h)) - np.log(ref_discount(t - h))) / (2 * h)
            self.assertAlmostEqual(float(self.curve.inst_fwd(t)), fd, places=9)

    def test_dtype(self):
        f = self.curve.inst_fwd(1.5)
        self.assertTrue(tf.is_tensor(f), f"expected a tensor, got {type(f).__name__}")
        self.assertEqual(f.dtype, tf.float64)

    def test_gradient_matches_finite_difference(self):
        t, h = 1.5, 1e-6
        with tf.GradientTape() as tape:
            f = self.curve.inst_fwd(t)
        grads = [0.0 if g is None else float(g) for g in tape.gradient(f, self.curve.variables)]
        self.assertTrue(any(g != 0.0 for g in grads))
        for i, var in enumerate(self.curve.variables):
            base = float(var)
            var.assign(base + h)
            up = float(self.curve.inst_fwd(t))
            var.assign(base - h)
            dn = float(self.curve.inst_fwd(t))
            var.assign(base)
            self.assertAlmostEqual(grads[i], (up - dn) / (2 * h), places=6)


class TestHullWhiteUsesInstFwd(unittest.TestCase):
    """Hull-White closed forms consume the curve (inst_fwd for phi): values pinned with the legacy
    HullWhiteProcess (a = 3%, sigma = 1%), now through GaussianShortRateModel (x = r - phi(t)). t = 1 is a pillar:
    alpha(1) and the bond were repinned when inst_fwd became exact (right derivative; the old central difference
    straddled the kink: alpha 0.0238735, bond 0.9572420, A 0.9970701)."""

    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.curve = make_curve()
        self.hw = tq.GaussianShortRateModel.hull_white(self.curve, 0.03, 0.01)

    def test_pinned_values(self):
        zero = tf.zeros([1, 1], tf.float64)
        self.assertAlmostEqual(float(self.curve.inst_fwd(0.0)), 0.02, places=15)                    # r(0)
        self.assertAlmostEqual(float(self.hw.short_rate(1.0, zero)[0]), 0.02304852591595736, places=9)   # alpha(1)
        B = float(self.hw._B(2.0)[0])
        self.assertAlmostEqual(B, 1.941182213858376, places=12)
        x = tf.constant([[0.021]], tf.float64) - (self.hw.short_rate(1.0, zero)[0] - 0.0)
        bond = float(self.hw.zero_bond(1.0, 3.0, x)[0])                    # P(1, 3) at r(1) = 2.1%
        self.assertAlmostEqual(bond, 0.9557102438503723, places=9)
        self.assertAlmostEqual(bond / np.exp(-B * 0.021), 0.995474593063443, places=9)           # A(1, 3)

class TestDividendCurvePV(unittest.TestCase):
    def setUp(self):
        self.curve = make_curve()
        self.divs = DividendCurve(
            reference_date=EVAL_DATE,
            ex_dates=[date(2026, 6, 15), date(2026, 12, 15)],
            amounts=[1.2, 1.3],
            currency=Currency.EUR,
            daycounter_convention=DayCounterConvention.Actual365,
        )

    def test_pv_value_and_dtype(self):
        pv = self.divs.pv_dividends(date(2027, 1, 1), self.curve)
        expected = sum(
            a * ref_discount((d - EVAL_DATE).days / 365.0)
            for d, a in zip(self.divs.ex_dates, self.divs.amounts)
        )
        self.assertAlmostEqual(float(pv), expected, places=13)
        self.assertEqual(pv.dtype, tf.float64)

    def test_pv_only_dividends_before_horizon(self):
        pv = self.divs.pv_dividends(date(2026, 7, 1), self.curve)
        self.assertAlmostEqual(float(pv), 1.2 * ref_discount((date(2026, 6, 15) - EVAL_DATE).days / 365.0), places=13)

    def test_pv_discounts_at_the_payment_dates(self):
        """With payment dates: in the horizon by ex-date, discounted at the payment date (even after the horizon)."""
        pays = [date(2026, 6, 29), date(2026, 12, 29)]
        divs = DividendCurve(reference_date=EVAL_DATE, ex_dates=self.divs.ex_dates, amounts=[1.2, 1.3],
                             currency=Currency.EUR, daycounter_convention=DayCounterConvention.Actual365,
                             payment_dates=pays)
        yf = lambda d: (d - EVAL_DATE).days / 365.0
        self.assertAlmostEqual(float(divs.pv_dividends(date(2027, 1, 1), self.curve)),
                               1.2 * ref_discount(yf(pays[0])) + 1.3 * ref_discount(yf(pays[1])), places=13)
        self.assertAlmostEqual(float(divs.pv_dividends(date(2026, 6, 20), self.curve)),    # ex <= T < pay
                               1.2 * ref_discount(yf(pays[0])), places=13)

    def test_pv_no_dividends(self):
        pv = self.divs.pv_dividends(date(2026, 2, 1), self.curve)
        self.assertEqual(float(pv), 0.0)
        self.assertEqual(pv.dtype, tf.float64)

    def test_amounts_are_a_float64_variable(self):
        (var,) = self.divs.variables
        self.assertIsInstance(var, tf.Variable)
        self.assertEqual(var.dtype, tf.float64)
        self.assertEqual(var.shape, (2,))
        self.assertEqual(self.divs.amounts, [1.2, 1.3])

    def test_pv_gradient_is_discount_factor(self):
        """dPV/dD_i = P(0, ex_i) for ex_i <= horizon, 0 after."""
        (var,) = self.divs.variables
        with tf.GradientTape() as tape:
            pv = self.divs.pv_dividends(date(2026, 7, 1), self.curve)
        grad = tape.gradient(pv, var).numpy()
        self.assertAlmostEqual(grad[0], ref_discount((date(2026, 6, 15) - EVAL_DATE).days / 365.0), places=13)
        self.assertEqual(grad[1], 0.0)

    def test_assign_moves_pv(self):
        (var,) = self.divs.variables
        horizon = date(2027, 1, 1)
        base = float(self.divs.pv_dividends(horizon, self.curve))
        var.assign([2.4, 1.3])
        df1 = ref_discount((date(2026, 6, 15) - EVAL_DATE).days / 365.0)
        self.assertAlmostEqual(float(self.divs.pv_dividends(horizon, self.curve)) - base, 1.2 * df1, places=13)
        self.assertEqual(self.divs.amounts, [2.4, 1.3])


MC_PILLARS = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
MC_RATES = [0.020, 0.024, 0.023, 0.026, 0.029, 0.027]          # humped: discrete forwards up and down


def mc_curve(rates=MC_RATES, pillars=MC_PILLARS) -> RateCurve:
    return RateCurve(EVAL_DATE, list(pillars), list(rates), "MONOTONE_CONVEX", DayCounterConvention.Actual365)


def mc_reference(rates=MC_RATES, pillars=MC_PILLARS):
    """Independent numpy: discrete forwards ``fd_i`` and the node forwards ``f_i`` of Hagan and West (2006)."""
    t = np.concatenate([[0.0], pillars])
    rt = np.concatenate([[0.0], np.array(rates) * np.array(pillars)])
    fd = np.diff(rt) / np.diff(t)
    f = np.empty(len(t))
    f[1:-1] = (np.diff(t)[:-1] * fd[1:] + np.diff(t)[1:] * fd[:-1]) / (t[2:] - t[:-2])
    f[0] = fd[0] - 0.5 * (f[1] - fd[0])
    f[-1] = fd[-1] - 0.5 * (f[-2] - fd[-1])
    return t, fd, f


class TestMonotoneConvex(unittest.TestCase):
    """Monotone convex interpolation (Hagan and West 2006): exact zero rates at the pillars, discrete forwards
    preserved, continuous instantaneous forward equal to the node values, flat forward after the last pillar."""

    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.curve = mc_curve()
        self.t, self.fd, self.f = mc_reference()

    def test_pricer_pins_the_curves(self):
        """``Pricer.price`` pins the market curves (coefficients once under the tape): same price as without pinning,
        rho per pillar = finite differences."""
        from tensorquant.markethandles.marketenvironment import MarketEnvironment
        from tensorquant.markethandles.volatilitysurface import BlackConstantVolatility

        mkt = MarketEnvironment(market={"IR:EUR:ESTR:SPOT": self.curve, "EQ:EUR:X:SPOT": 100.0,
                                        "EQ:EUR:X:VOL": BlackConstantVolatility(EVAL_DATE, 0.2)})
        option = lambda: tq.VanillaOption(Currency.EUR, start_date=EVAL_DATE, end_date=EVAL_DATE + timedelta(days=900),
                                          option_type=tq.OptionType.Put, strike=95.0, underlying="X")
        pricer, opt = tq.BlackScholesPricer(), option()
        pricer.price(opt, mkt, autodiff=True)
        unpinned = float(tq.BlackScholesPricer().calculate_price(option(), mkt))
        self.assertAlmostEqual(float(opt.price), unpinned, places=13)
        grads = pricer.tape.gradient(opt.price, self.curve.variables)
        h = 1e-6
        for var, g in zip(self.curve.variables, grads):
            base = float(var.numpy())
            var.assign(base + h)
            up = float(tq.BlackScholesPricer().calculate_price(option(), mkt))
            var.assign(base - h)
            down = float(tq.BlackScholesPricer().calculate_price(option(), mkt))
            var.assign(base)
            fd = (up - down) / (2 * h)
            self.assertAlmostEqual(0.0 if g is None else float(g), fd, delta=1e-6 * max(1.0, abs(fd)))

    def test_zero_rates_at_the_pillars(self):
        for t, r in zip(MC_PILLARS, MC_RATES):
            self.assertAlmostEqual(float(self.curve.zero_rate(t)), r, places=13)

    def test_discrete_forwards_preserved(self):
        for i in range(1, len(self.t)):
            a, b = self.t[i - 1], self.t[i]
            pa = 1.0 if a == 0.0 else float(self.curve.discount(a))
            fwd = -np.log(float(self.curve.discount(b)) / pa) / (b - a)
            self.assertAlmostEqual(fwd, self.fd[i - 1], places=12)

    def test_instantaneous_forward_continuous_at_the_nodes(self):
        for i, t in enumerate(MC_PILLARS, start=1):
            left, right = float(self.curve.inst_fwd(t - 1e-9)), float(self.curve.inst_fwd(t + 1e-9))
            with self.subTest(pillar=t):
                self.assertAlmostEqual(left, right, places=7)
                self.assertAlmostEqual(float(self.curve.inst_fwd(t)), self.f[i], places=10)
        self.assertAlmostEqual(float(self.curve.inst_fwd(0.0)), self.f[0], places=12)
        self.assertAlmostEqual(float(self.curve.zero_rate(0.0)), self.f[0], places=12)

    def test_inst_fwd_is_minus_dlogP(self):
        for t in (0.1, 0.6, 1.7, 3.3, 7.5, 20.0, 40.0):
            h = 1e-5
            fd = -(np.log(float(self.curve.discount(t + h))) - np.log(float(self.curve.discount(t - h)))) / (2 * h)
            with self.subTest(t=t):
                self.assertAlmostEqual(float(self.curve.inst_fwd(t)), fd, places=7)

    def test_flat_forward_after_the_last_pillar(self):
        for t in (31.0, 45.0):
            self.assertAlmostEqual(float(self.curve.inst_fwd(t)), self.f[-1], places=12)
        expected = MC_RATES[-1] * MC_PILLARS[-1] + self.f[-1] * (45.0 - MC_PILLARS[-1])
        self.assertAlmostEqual(-np.log(float(self.curve.discount(45.0))), expected, places=12)

    def test_monotone_discrete_forwards_give_a_monotone_forward(self):
        """Increasing discrete forwards: the instantaneous forward is non-decreasing (no spurious oscillation)."""
        rates = [0.01, 0.015, 0.02, 0.025, 0.03]
        pillars = [1.0, 2.0, 3.0, 5.0, 10.0]
        curve = mc_curve(rates, pillars)
        grid = np.linspace(0.0, 10.0, 401)
        f = np.array([float(curve.inst_fwd(float(x))) for x in grid])
        self.assertGreaterEqual(np.diff(f).min(), -1e-12)

    def test_flat_input_is_flat(self):
        curve = mc_curve([0.03] * 4, [0.5, 1.0, 5.0, 10.0])
        for t in (0.2, 0.7, 3.0, 8.0, 15.0):
            self.assertAlmostEqual(float(curve.inst_fwd(t)), 0.03, places=13)
            self.assertAlmostEqual(float(curve.discount(t)), np.exp(-0.03 * t), places=13)

    def test_gradients_vs_finite_differences(self):
        for t in (0.6, 3.3, 12.0, 35.0):
            with tf.GradientTape() as tape:
                df = self.curve.discount(t)
            grads = tape.gradient(df, self.curve.variables)
            for j in range(len(MC_RATES)):
                up, dn = list(MC_RATES), list(MC_RATES)
                up[j] += 1e-7                        # small step: near a region boundary the curvature is large
                dn[j] -= 1e-7
                fd = (float(mc_curve(up).discount(t)) - float(mc_curve(dn).discount(t))) / 2e-7
                g = 0.0 if grads[j] is None else float(grads[j])
                with self.subTest(t=t, pillar=j):
                    self.assertAlmostEqual(g, fd, delta=1e-7)

    def test_round_trip_and_validation(self):
        dates = [EVAL_DATE + timedelta(days=int(round(365 * t))) for t in MC_PILLARS]    # snapshots store dates
        curve = RateCurve(EVAL_DATE, dates, MC_RATES, "MONOTONE_CONVEX", DayCounterConvention.Actual365)
        mkt = tq.MarketEnvironment({"IR:EUR:ESTR:SPOT": curve})
        mkt.evaluation_date = EVAL_DATE
        back = tq.MarketEnvironment.from_frames(mkt.to_frames(), EVAL_DATE).get_ir_curve(tq.Currency.EUR)
        self.assertEqual(back.interpolation_type, "MONOTONE_CONVEX")
        for t in (0.6, 3.3, 12.0):
            self.assertAlmostEqual(float(back.discount(t)), float(curve.discount(t)), places=12)
        with self.assertRaises(ValueError):
            RateCurve(EVAL_DATE, MC_PILLARS, MC_RATES, "CUBIC", DayCounterConvention.Actual365)


if __name__ == "__main__":
    unittest.main()
