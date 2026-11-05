"""Rates Monte Carlo: GaussianShortRateModel (G1++/G2++, exact simulation) and SwapExposureGenerator."""

import unittest
from datetime import date, timedelta

import numpy as np
import tensorflow as tf

import tensorquant as tq
from tensorquant.index.curverateindex import IborIndex
from tensorquant.instruments.helpers import SwapGenerator
from tensorquant.models.g2 import G2PlusPlusProcess
from tensorquant.models.hullwhite import HullWhiteProcess
from tensorquant.pricers.fixedflow import FixedCouponDiscounting
from tensorquant.pricers.floatingflow import FloatingCouponDiscounting

EVAL_DATE = date(2026, 5, 15)
A365 = tq.DayCounterConvention.Actual365
PILLARS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
ESTR = [0.020, 0.021, 0.022, 0.024, 0.026, 0.028, 0.030]
EUR6M = [0.023, 0.024, 0.025, 0.027, 0.029, 0.031, 0.033]
HW = dict(a=0.05, sigma=0.01)
G2 = dict(a=0.05, b=0.3, sigma=0.01, eta=0.008, rho=-0.6)


def make_market():
    return tq.MarketEnvironment({
        "IR:EUR:ESTR:SPOT": tq.RateCurve(EVAL_DATE, PILLARS, ESTR, "LINEAR", A365),
        "IR:EUR:6M:SPOT": tq.RateCurve(EVAL_DATE, PILLARS, EUR6M, "LINEAR", A365),
    })


def make_swap(quote=0.028):
    """5Y payer swap, 1Y fixed vs 6M Euribor (SwapGenerator builds payers)."""
    cal = tq.TARGET()
    idx = IborIndex(cal, 6, tq.TimeUnit.Months, tq.Currency.EUR)
    return SwapGenerator(tq.Currency.EUR, 2, "1Y", "6M", tq.BusinessDayConvention.ModifiedFollowing, 1e6,
                         tq.DayCounterConvention.Actual360, tq.DayCounterConvention.Actual360, cal, idx
                         ).build(EVAL_DATE, quote, "5Y")


EXPOSURE_DATES = [EVAL_DATE] + [EVAL_DATE + timedelta(days=91 * k) for k in range(1, 23)]


class RatesTestCase(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = make_market()
        self.curve = self.mkt.get_ir_curve(tq.Currency.EUR)

    def models(self):
        return {"HW": tq.GaussianShortRateModel.hull_white(self.curve, **HW),
                "G2": tq.GaussianShortRateModel.g2pp(self.curve, **G2)}


class TestClosedForms(RatesTestCase):
    def test_bonds_at_zero_reproduce_the_curve(self):
        for name, m in self.models().items():
            for T in (0.5, 3.0, 12.0):
                with self.subTest(model=name, T=T):
                    self.assertAlmostEqual(float(m.zero_bond(0.0, T, tf.zeros([1, m.size()], tf.float64))[0]),
                                           float(self.curve.discount(T)), places=14)

    def test_hull_white_matches_the_legacy_process(self):
        """G1++ in x = r - phi(t) vs HullWhiteProcess in r: same bonds, phi = alpha."""
        m, old = self.models()["HW"], HullWhiteProcess(self.curve, HW["a"], HW["sigma"])
        x = tf.constant([[-0.01], [0.0], [0.015]], tf.float64)
        for t, T in ((1.0, 4.0), (2.0, 7.0)):
            np.testing.assert_allclose(m.zero_bond(t, T, x).numpy(), old.zero_bond(t, T, m.short_rate(t, x)).numpy(),
                                       rtol=1e-9)   # legacy A(S,T) uses inst_fwd (~1e-11)
            self.assertAlmostEqual(float(m.short_rate(t, tf.zeros([1, 1], tf.float64))[0]), float(old.alpha(t)), places=9)
        self.assertAlmostEqual(old.x0, float(self.curve.inst_fwd(0.0)), places=12)   # was AttributeError

    def test_g2pp_matches_the_legacy_process(self):
        m = self.models()["G2"]
        old = G2PlusPlusProcess(self.curve, G2["a"], G2["b"], G2["sigma"], G2["eta"], G2["rho"])
        for x, y in ((0.01, -0.005), (-0.02, 0.003)):
            state = tf.constant([[x, y]], tf.float64)
            self.assertAlmostEqual(float(m.zero_bond(2.0, 6.0, state)[0]), float(old.zero_bond(2.0, 6.0, x, y)), places=9)


class TestSimulation(RatesTestCase):
    N = 50_000

    def test_martingales(self):
        """E[D(0,t)] = P(0,t) and E[D(0,t) P(t,T)] = P(0,T) within 4 standard errors."""
        times = np.arange(1, 21) / 4.0
        for name, m in self.models().items():
            xs, D = m.simulate(times, self.N, seed=11)
            for k, T in ((3, 10.0), (11, 7.0), (19, 10.0)):
                d = D[:, k].numpy()
                v = (D[:, k] * m.zero_bond(times[k], T, xs[:, k])).numpy()
                with self.subTest(model=name, t=times[k]):
                    self.assertLess(abs(d.mean() - float(self.curve.discount(times[k]))), 4 * d.std() / np.sqrt(self.N))
                    self.assertLess(abs(v.mean() - float(self.curve.discount(T))), 4 * v.std() / np.sqrt(self.N))

    def test_factor_moments(self):
        """Var(x_i(t)) = sigma_i^2 (1 - e^{-2 k_i t}) / (2 k_i) and corr(x, y) of G2++."""
        m = self.models()["G2"]
        t = 3.0
        xs, _ = m.simulate([1.0, t], self.N, seed=12)
        x = xs[:, 1].numpy()
        for i, (k, s) in enumerate(((G2["a"], G2["sigma"]), (G2["b"], G2["eta"]))):
            var = s ** 2 * (1 - np.exp(-2 * k * t)) / (2 * k)
            self.assertAlmostEqual(x[:, i].var() / var, 1.0, delta=0.03)
        a, b = G2["a"], G2["b"]
        cov = G2["rho"] * G2["sigma"] * G2["eta"] * (1 - np.exp(-(a + b) * t)) / (a + b)
        corr = cov / np.sqrt(G2["sigma"] ** 2 * (1 - np.exp(-2 * a * t)) / (2 * a) * G2["eta"] ** 2 * (1 - np.exp(-2 * b * t)) / (2 * b))
        self.assertAlmostEqual(np.corrcoef(x[:, 0], x[:, 1])[0, 1], corr, delta=0.02)

    def test_seed_and_invalid_times(self):
        m = self.models()["HW"]
        a, b = m.simulate([0.5, 1.0], 1_000, seed=4), m.simulate([0.5, 1.0], 1_000, seed=4)
        np.testing.assert_array_equal(a[1].numpy(), b[1].numpy())
        with self.assertRaises(ValueError):
            m.simulate([1.0, 0.5], 10)

    def test_rho_by_aad(self):
        """d E[D(0,t) P(t,T)] / d pillar = d P(0,T) / d pillar (the curve enters only through P(0,.))."""
        m = self.models()["HW"]
        with tf.GradientTape() as tape:
            xs, D = m.simulate([2.0], self.N, seed=13)
            value = tf.reduce_mean(D[:, 0] * m.zero_bond(2.0, 7.0, xs[:, 0]))
        with tf.GradientTape() as tape0:
            p0 = self.curve.discount(7.0)
        g, g0 = tape.gradient(value, self.curve.variables), tape0.gradient(p0, self.curve.variables)
        for a, b in zip(g, g0):
            if b is not None:
                self.assertAlmostEqual(float(a) / float(b), 1.0, delta=0.01)


class TestSwapExposure(RatesTestCase):
    def exposure(self, model_name="HW", n_paths=20_000, seed=21, swap=None):
        swap = swap or make_swap()
        gen = tq.SwapExposureGenerator(self.models()[model_name], EXPOSURE_DATES, n_paths=n_paths, seed=seed)
        gen.simulate(swap, self.mkt)
        return gen, swap

    def pv_after(self, swap, d):
        fwd = self.mkt.get_ir_curve(tq.Currency.EUR, "6M")
        f = sum(float(FloatingCouponDiscounting(c).calculate_price(self.curve, fwd))
                for c in swap.floating_leg.leg_flows if c.date > d)
        x = sum(float(FixedCouponDiscounting(c).calculate_price(self.curve)) for c in swap.fixed_leg.leg_flows if c.date > d)
        return f - x

    def test_today_equals_the_swap_pricer(self):
        gen, swap = self.exposure(n_paths=100)
        ref = make_swap()
        tq.SwapPricer().price(ref, self.mkt)
        np.testing.assert_allclose(gen.mtm[:, 0].numpy(), float(ref.price), rtol=1e-12)

    def test_discounted_mtm_is_the_pv_of_the_remaining_flows(self):
        """E[D(0,t) V(t)] = PV(0) of the flows paid after t (martingale), HW and G2++, within 4 SE."""
        for name in ("HW", "G2"):
            gen, swap = self.exposure(name)
            v, D = gen.mtm.numpy(), gen.discount.numpy()
            for k in (4, 10, 16, 20):
                dv = D[:, k] * v[:, k]
                with self.subTest(model=name, date=EXPOSURE_DATES[k]):
                    self.assertLess(abs(dv.mean() - self.pv_after(swap, EXPOSURE_DATES[k])), 4 * dv.std() / np.sqrt(len(dv)))

    def test_profile_identities(self):
        gen, swap = self.exposure(n_paths=5_000)
        ee, ene = gen.expected_exposure.numpy(), gen.expected_negative_exposure.numpy()
        self.assertTrue(np.all(ee >= 0) and np.all(ene <= 0))
        np.testing.assert_allclose(ee + ene, gen.mtm.numpy().mean(axis=0), atol=1e-8)
        self.assertTrue(np.all(gen.pfe(0.975) >= ee - 1e-9))
        after = [k for k, d in enumerate(EXPOSURE_DATES) if d > swap.end_date + timedelta(days=10)]
        self.assertTrue(after)
        self.assertEqual(float(tf.reduce_max(tf.abs(gen.mtm[:, after[0]]))), 0.0)

    def test_discounted_epe_gradient_vs_common_random_numbers_bump(self):
        var = self.curve.variables[4]   # 5Y pillar
        model = self.models()["HW"]
        with tf.GradientTape() as tape:
            gen = tq.SwapExposureGenerator(model, EXPOSURE_DATES, n_paths=5_000, seed=31)
            gen.simulate(make_swap(), self.mkt)
            epe = tf.reduce_sum(gen.discounted_expected_exposure)
        g = float(tape.gradient(epe, var))
        base, out = float(var), []
        for x in (base + 1e-6, base - 1e-6):
            var.assign(x)
            gen = tq.SwapExposureGenerator(model, EXPOSURE_DATES, n_paths=5_000, seed=31)
            gen.simulate(make_swap(), self.mkt)
            out.append(float(tf.reduce_sum(gen.discounted_expected_exposure)))
        var.assign(base)
        self.assertAlmostEqual(g, (out[0] - out[1]) / 2e-6, delta=1e-4 * abs(g))

    def test_exposure_date_before_today_raises(self):
        gen = tq.SwapExposureGenerator(self.models()["HW"], [EVAL_DATE - timedelta(days=1)], n_paths=10)
        with self.assertRaises(ValueError):
            gen.simulate(make_swap(), self.mkt)


if __name__ == "__main__":
    unittest.main()
