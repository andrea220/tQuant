"""Caps / floors / collars on EUR 6M: CapFloorPricer (Bachelier on normal caplet vols), market data, generator."""

import unittest
from datetime import date, timedelta
from math import erf, exp, pi, sqrt

import numpy as np
import tensorflow as tf

import tensorquant as tq
from tensorquant.index.curverateindex import IborIndex
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.synthetic import generate_market_data, sabr_normal_vol
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

EVAL_DATE = date(2026, 5, 15)
A365, A360 = DayCounterConvention.Actual365, DayCounterConvention.Actual360
PILLARS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
STRIKES = [-0.005, 0.0, 0.01, 0.02, 0.03, 0.04, 0.06]
EXPIRIES = [0.5, 1.0, 2.0, 5.0, 10.0]


def make_market(normal_vols=None, vol_type="normal"):
    """ESTR + 6M curves, EUR 6M caplet vols (default: a normal smile around 2.5%)."""
    vols = normal_vols if normal_vols is not None else \
        [[0.0080 + 0.03 * (k - 0.025) ** 2 / 0.0025 * 0.01 + 0.0005 * t for k in STRIKES] for t in EXPIRIES]
    surface = VolatilitySurface(EVAL_DATE, None, DayCounter(A365), STRIKES, EXPIRIES, vols, vol_type=vol_type)
    return MarketEnvironment({
        "IR:EUR:ESTR:SPOT": tq.RateCurve(EVAL_DATE, PILLARS, [0.020, 0.021, 0.022, 0.024, 0.026, 0.028], "LINEAR", A365),
        "IR:EUR:6M:SPOT": tq.RateCurve(EVAL_DATE, PILLARS, [0.023, 0.024, 0.025, 0.027, 0.029, 0.031], "LINEAR", A365),
        "IR:EUR:6M:VOL": surface,
    })


def generator(exclude_first=True):
    cal = tq.TARGET()
    return tq.CapFloorGenerator(tq.Currency.EUR, 2, "6M", tq.BusinessDayConvention.ModifiedFollowing, 1e6, A360, cal,
                                IborIndex(cal, 6, tq.TimeUnit.Months, tq.Currency.EUR), exclude_first)


def bachelier(F, K, s, T, phi):
    d = (F - K) / (s * sqrt(T))
    return phi * (F - K) * 0.5 * (1 + erf(phi * d / sqrt(2))) + s * sqrt(T) * exp(-0.5 * d * d) / sqrt(2 * pi)


class CapTestCase(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = make_market()
        self.disc = self.mkt.get_ir_curve(tq.Currency.EUR)
        self.fwd = self.mkt.get_ir_curve(tq.Currency.EUR, "6M")
        self.surface = self.mkt.get_ir_vol_surface(tq.Currency.EUR, "6M")

    def price(self, product, autodiff=False):
        pricer = tq.PricerAssignment.create(product)
        pricer.price(product, self.mkt, autodiff=autodiff)
        return pricer

    def forward(self, c):
        tau = c.index.daycounter.year_fraction(c.ref_period_start, c.ref_period_end)
        return (float(self.fwd.discount(c.ref_period_start)) / float(self.fwd.discount(c.ref_period_end)) - 1) / tau


class TestPricing(CapTestCase):
    def test_bachelier_closed_form_per_caplet(self):
        for ot, phi in ((tq.OptionType.Call, 1.0), (tq.OptionType.Put, -1.0)):
            for K in (0.015, 0.025, 0.035):
                prod = generator().build(EVAL_DATE, K, "5Y", ot)
                self.price(prod)
                expected = 0.0
                for c in prod.caplets:
                    T = DayCounter(A365).year_fraction(EVAL_DATE, c.fixing_date)
                    s = float(self.surface.volatility(K, T))
                    expected += c.nominal * c.accrual_period * float(self.disc.discount(c.date)) * \
                        bachelier(self.forward(c), K, s, T, phi)
                with self.subTest(type=ot.name, K=K):
                    self.assertAlmostEqual(float(prod.price), expected, delta=1e-8 * abs(expected) + 1e-8)

    def test_cap_floor_parity(self):
        """cap - floor = sum N tau P(pay) (F - K): model-free (a forward-starting swap on the caplet periods)."""
        for K in (0.01, 0.027, 0.05):
            cap, floor = generator().build(EVAL_DATE, K, "7Y"), generator().build(EVAL_DATE, K, "7Y", tq.OptionType.Put)
            self.price(cap)
            self.price(floor)
            swap = sum(c.nominal * c.accrual_period * float(self.disc.discount(c.date)) * (self.forward(c) - K)
                       for c in cap.caplets)
            with self.subTest(K=K):
                self.assertAlmostEqual(float(cap.price) - float(floor.price), swap, delta=1e-7)

    def test_zero_vol_is_intrinsic(self):
        self.mkt = make_market(normal_vols=[[1e-10] * len(STRIKES) for _ in EXPIRIES])
        cap = generator().build(EVAL_DATE, 0.026, "5Y")
        self.price(cap)
        intrinsic = sum(c.nominal * c.accrual_period * float(self.disc.discount(c.date)) * max(self.forward(c) - 0.026, 0)
                        for c in cap.caplets)
        self.assertAlmostEqual(float(cap.price), intrinsic, delta=1e-6)

    def test_lognormal_vols_use_black(self):
        lognormal = [[0.30] * len(STRIKES) for _ in EXPIRIES]
        self.mkt = make_market(normal_vols=lognormal, vol_type="lognormal")
        cap = generator().build(EVAL_DATE, 0.025, "3Y")
        self.price(cap)
        expected = 0.0
        for c in cap.caplets:
            T = DayCounter(A365).year_fraction(EVAL_DATE, c.fixing_date)
            F, s = self.forward(c), 0.30 * sqrt(T)
            d1 = (np.log(F / 0.025) + 0.5 * s * s) / s
            N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
            expected += c.nominal * c.accrual_period * float(self.disc.discount(c.date)) * (F * N(d1) - 0.025 * N(d1 - s))
        self.assertAlmostEqual(float(cap.price), expected, delta=1e-8 * expected)

    def test_collar_is_cap_minus_floor(self):
        col = generator().build_collar(EVAL_DATE, 0.035, 0.02, "5Y")
        self.price(col)
        cap, floor = generator().build(EVAL_DATE, 0.035, "5Y"), generator().build(EVAL_DATE, 0.02, "5Y", tq.OptionType.Put)
        self.price(cap)
        self.price(floor)
        self.assertAlmostEqual(float(col.price), float(cap.price) - float(floor.price), places=8)
        with self.assertRaises(ValueError):
            tq.Collar(floor, cap)

    def test_first_period_excluded_and_fixed_caplet(self):
        full = generator(exclude_first=False).build(EVAL_DATE, 0.02, "2Y")
        spot = generator().build(EVAL_DATE, 0.02, "2Y")
        self.assertEqual(len(full.caplets), len(spot.caplets) + 1)
        # move after the second caplet's fixing: it pays its intrinsic value on the index fixing
        c = spot.caplets[0]
        c.index.add_fixing(c.fixing_date, 0.031)
        tq.Settings.evaluation_date = c.fixing_date + timedelta(days=1)
        self.price(spot)
        df = float(self.disc.discount(c.date))
        self.assertAlmostEqual(float(spot.caplet_values[0]), c.nominal * c.accrual_period * df * (0.031 - 0.02), places=6)

    def test_float64_and_factory(self):
        cap = generator().build(EVAL_DATE, 0.025, "5Y")
        self.assertIsInstance(tq.PricerAssignment.create(cap), tq.CapFloorPricer)
        self.price(cap)
        self.assertEqual(cap.price.dtype, tf.float64)


class TestGreeks(CapTestCase):
    def test_vega_per_node_and_rho_per_pillar_vs_bumps(self):
        make = lambda: generator().build(EVAL_DATE, 0.025, "5Y")
        cap = make()
        pricer = self.price(cap, autodiff=True)
        matrix = self.surface.volatility_matrix
        grads = pricer.tape.gradient(cap.price, [matrix] + self.disc.variables + self.fwd.variables)
        vega, rhos = grads[0].numpy(), grads[1:]

        def reprice():
            p = make()
            self.price(p)
            return float(p.price)

        base, h = matrix.numpy().copy(), 1e-7
        for i, j in ((1, 3), (3, 4), (2, 2)):
            out = []
            for sgn in (1, -1):
                m = base.copy()
                m[i, j] += sgn * h
                matrix.assign(m)
                out.append(reprice())
            matrix.assign(base)
            with self.subTest(node=(i, j)):
                self.assertAlmostEqual(vega[i, j], (out[0] - out[1]) / (2 * h), delta=1e-4 * abs(vega).max())
        for var, g in zip(self.disc.variables + self.fwd.variables, rhos):
            b, out = float(var), []
            for x in (b + 1e-6, b - 1e-6):
                var.assign(x)
                out.append(reprice())
            var.assign(b)
            self.assertAlmostEqual(0.0 if g is None else float(g), (out[0] - out[1]) / 2e-6, delta=1e-2)
        self.assertTrue(any(g is not None for g in rhos[:len(self.disc.variables)]))   # discounting
        self.assertTrue(any(g is not None for g in rhos[len(self.disc.variables):]))   # forwarding


class TestMarketData(unittest.TestCase):
    def test_generated_eur6m_curve_and_caplet_vols(self):
        tables = generate_market_data(EVAL_DATE, 7, (("EUR", 1), ("GBP", 1), ("USD", 1)))
        mkt = MarketEnvironment.from_frames(tables, EVAL_DATE)
        tq.Settings.evaluation_date = EVAL_DATE
        surface = mkt.get_ir_vol_surface(tq.Currency.EUR, "6M")
        self.assertEqual(surface.vol_type, "normal")
        self.assertIn("IR:EUR:6M:VOL", mkt.risk_factors())
        estr, e6m = mkt.get_ir_curve(tq.Currency.EUR), mkt.get_ir_curve(tq.Currency.EUR, "6M")
        spreads = np.array(e6m.rates) - np.array(estr.rates)
        self.assertTrue(np.all((spreads > 0.0015) & (spreads < 0.005)))
        back = MarketEnvironment.from_frames(mkt.to_frames(), EVAL_DATE)
        np.testing.assert_array_equal(back.get_ir_vol_surface(tq.Currency.EUR, "6M").volatility_matrix.numpy(),
                                      surface.volatility_matrix.numpy())
        self.assertEqual(back.get_ir_vol_surface(tq.Currency.EUR, "6M").vol_type, "normal")

    def test_generated_caplet_prices_are_arbitrage_free(self):
        """Per expiry, undiscounted caplet prices on the strike grid: decreasing and convex in K (positive density)."""
        from tensorquant.markethandles.synthetic import add_months
        tables = generate_market_data(EVAL_DATE, 7, (("EUR", 1), ("GBP", 1), ("USD", 1)))
        curve = MarketEnvironment.from_frames(tables, EVAL_DATE).get_ir_curve(tq.Currency.EUR, "6M")
        vols = tables["ir_vol.csv"]
        for expiry, grp in vols.groupby("expiry"):
            grp = grp.sort_values("strike")
            T = (expiry - EVAL_DATE).days / 365
            K, s = grp["strike"].to_numpy(), grp["vol"].to_numpy()
            end = add_months(expiry, 6)
            F = (float(curve.discount(expiry)) / float(curve.discount(end)) - 1) / DayCounter(A360).year_fraction(expiry, end)
            calls = np.array([bachelier(F, k, v, T, 1.0) for k, v in zip(K, s)])
            with self.subTest(expiry=expiry):
                self.assertTrue(np.all(np.diff(calls) < 0))
                self.assertTrue(np.all(np.diff(calls, 2) > 0))

    def test_sabr_normal_atm_and_symmetry(self):
        v = sabr_normal_vol(0.03, np.array([0.03]), 2.0, 0.008, 0.0, 0.3)[0]
        self.assertAlmostEqual(v, 0.008 * (1 + 2 / 24 * 0.09 * 2.0), places=12)
        up, dn = sabr_normal_vol(0.03, np.array([0.04, 0.02]), 2.0, 0.008, 0.0, 0.3)
        self.assertAlmostEqual(up, dn, places=12)   # rho = 0: symmetric smile in F - K

    def test_ir_vol_type_validation(self):
        with self.assertRaises(ValueError):
            VolatilitySurface(EVAL_DATE, None, DayCounter(A365), [0.01, 0.02], [1.0, 2.0], [[0.01] * 2] * 2, vol_type="shifted")
        with self.assertRaises(ValueError):
            make_market().get_ir_vol_surface(tq.Currency.USD, "6M")


if __name__ == "__main__":
    unittest.main()
