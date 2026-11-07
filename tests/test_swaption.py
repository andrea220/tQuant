"""Swaptions: SwaptionVolCube, SwaptionPricer (Bachelier / Black), Gaussian model swaptions, G2++ calibration."""

import unittest
from datetime import date

import numpy as np
import tensorflow as tf
from scipy.optimize import brentq
from scipy.stats import norm

import tensorquant as tq
from tensorquant.calibration.shortrate import _swaption_instruments, calibrate_gaussian_short_rate
from tensorquant.index.curverateindex import IborIndex
from tensorquant.instruments.helpers import SwapGenerator
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.pricers.swaption import swap_annuity_and_forward
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention
from markers import slow

EVAL_DATE = date(2026, 5, 15)
SMALL = (("EUR", 1), ("GBP", 1), ("USD", 1))
A360 = tq.DayCounterConvention.Actual360


def market():
    tq.Settings.evaluation_date = EVAL_DATE
    return tq.generate_market(EVAL_DATE, seed=7, tickers_per_currency=SMALL)


def swaption_generator(notional=1e6):
    cal = tq.TARGET()
    idx = IborIndex(cal, 6, tq.TimeUnit.Months, tq.Currency.EUR)
    return tq.SwaptionGenerator(SwapGenerator(tq.Currency.EUR, 2, "1Y", "6M", tq.BusinessDayConvention.ModifiedFollowing,
                                              notional, A360, A360, cal, idx))


def cap_generator():
    cal = tq.TARGET()
    return tq.CapFloorGenerator(tq.Currency.EUR, 2, "6M", tq.BusinessDayConvention.ModifiedFollowing, 1.0, A360, cal,
                                IborIndex(cal, 6, tq.TimeUnit.Months, tq.Currency.EUR))


def bachelier(F, K, v, T, phi=1.0):
    d = (F - K) / (v * np.sqrt(T))
    return phi * (F - K) * norm.cdf(phi * d) + v * np.sqrt(T) * norm.pdf(d)


class TestVolCube(unittest.TestCase):
    def cube(self):
        vols = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3) * 1e-4 + 0.006
        return tq.SwaptionVolCube(EVAL_DATE, DayCounter(DayCounterConvention.Actual365), [1.0, 5.0], [2.0, 5.0, 10.0],
                                  [-0.01, 0.0, 0.01], vols), vols

    def test_nodes_interpolation_extrapolation(self):
        cube, vols = self.cube()
        self.assertAlmostEqual(float(cube.volatility(5.0, 5.0, 0.0)), vols[1, 1, 1], places=15)
        mid = float(cube.volatility(3.0, 3.5, -0.005))
        self.assertAlmostEqual(mid, vols[:, 0:2, 0:2].mean(), places=15)                  # trilinear midpoint
        self.assertAlmostEqual(float(cube.volatility(0.1, 30.0, 0.05)), vols[0, 2, 2], places=15)   # flat outside

    def test_gradient_and_validation(self):
        cube, _ = self.cube()
        with tf.GradientTape() as tape:
            v = cube.volatility(3.0, 5.0, 0.0)
        g = tape.gradient(v, cube.volatilities).numpy()
        self.assertAlmostEqual(g.sum(), 1.0, places=14)
        self.assertEqual(int(np.count_nonzero(g)), 2)
        with self.assertRaises(ValueError):
            tq.SwaptionVolCube(EVAL_DATE, DayCounter(DayCounterConvention.Actual365), [1.0, 5.0], [2.0, 5.0],
                               [-0.01, 0.0, 0.01], np.zeros((2, 3, 3)))


class TestSwaptionPricer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mkt = market()

    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE

    def build(self, K, payer=True, expiry="2Y", tenor="5Y"):
        return swaption_generator().build(EVAL_DATE, expiry, tenor, K, payer)

    def test_bachelier_closed_form_and_parity(self):
        probe = self.build(0.03)
        A, F = (float(v) for v in swap_annuity_and_forward(probe.swap, self.mkt))
        cube = self.mkt.get_swaption_vol_cube(tq.Currency.EUR, "6M")
        T = DayCounter(DayCounterConvention.Actual365).year_fraction(EVAL_DATE, probe.expiry)
        for off in (-0.01, 0.0, 0.0075):
            K = F + off
            pay, rec = self.build(K, True), self.build(K, False)
            tq.PricerAssignment.create(pay).price(pay, self.mkt)
            tq.PricerAssignment.create(rec).price(rec, self.mkt)
            vol = float(cube.volatility(T, 5.0, off))
            with self.subTest(offset=off):
                self.assertAlmostEqual(float(pay.price), A * bachelier(F, K, vol, T), delta=1e-8 * A)
                self.assertAlmostEqual(float(pay.price) - float(rec.price), A * (F - K), delta=1e-8 * A)

    def test_forward_rate_zeroes_the_swap(self):
        probe = self.build(0.03, expiry="1Y", tenor="10Y")
        _, F = swap_annuity_and_forward(probe.swap, self.mkt)
        at_par = swaption_generator().swap_generator.build(probe.expiry, float(F), "10Y")
        tq.SwapPricer().price(at_par, self.mkt)
        self.assertAlmostEqual(float(at_par.price), 0.0, delta=1e-6)

    def test_lognormal_cube_uses_black(self):
        mkt = market()
        cube = mkt.get_swaption_vol_cube(tq.Currency.EUR, "6M")
        black_cube = tq.SwaptionVolCube(EVAL_DATE, cube.daycounter, cube.expiries, cube.tenors, cube.strike_offsets,
                                        np.full(cube.volatilities.shape, 0.25), vol_type="lognormal")
        mkt.set_swaption_vol_cube(tq.Currency.EUR, "6M", black_cube)
        probe = self.build(0.03)
        A, F = (float(v) for v in swap_annuity_and_forward(probe.swap, mkt))
        T = DayCounter(DayCounterConvention.Actual365).year_fraction(EVAL_DATE, probe.expiry)
        sw = self.build(F * 1.1)
        tq.SwaptionPricer().price(sw, mkt)
        K, s = F * 1.1, 0.25 * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * s * s) / s
        self.assertAlmostEqual(float(sw.price), A * (F * norm.cdf(d1) - K * norm.cdf(d1 - s)), delta=1e-8 * A)

    @slow
    def test_vega_per_node_and_rho_vs_bumps(self):
        make = lambda: self.build(0.031, expiry="3Y", tenor="7Y")
        sw = make()
        pricer = tq.SwaptionPricer()
        pricer.price(sw, self.mkt, autodiff=True)
        cube = self.mkt.get_swaption_vol_cube(tq.Currency.EUR, "6M")
        curves = [self.mkt.get_ir_curve(tq.Currency.EUR), self.mkt.get_ir_curve(tq.Currency.EUR, "6M")]
        grads = pricer.tape.gradient(sw.price, [cube.volatilities] + curves[0].variables + curves[1].variables)

        def reprice():
            s = make()
            tq.SwaptionPricer().price(s, self.mkt)
            return float(s.price)

        vega, base = grads[0].numpy(), cube.volatilities.numpy().copy()
        for idx in zip(*np.nonzero(vega)):
            out = []
            for sgn in (1, -1):
                m = base.copy()
                m[idx] += sgn * 1e-7
                cube.volatilities.assign(m)
                out.append(reprice())
            cube.volatilities.assign(base)
            self.assertAlmostEqual(vega[idx], (out[0] - out[1]) / 2e-7, delta=1e-4 * np.abs(vega).max())
        for var, g in zip(curves[0].variables + curves[1].variables, grads[1:]):
            b, out = float(var), []
            for x in (b + 1e-6, b - 1e-6):
                var.assign(x)
                out.append(reprice())
            var.assign(b)
            self.assertAlmostEqual(0.0 if g is None else float(g), (out[0] - out[1]) / 2e-6, delta=1e-2)

    def test_invalid(self):
        sw = self.build(0.03)
        tq.Settings.evaluation_date = sw.expiry
        with self.assertRaises(ValueError):
            tq.SwaptionPricer().price(sw, self.mkt)
        tq.Settings.evaluation_date = EVAL_DATE
        self.assertIsInstance(tq.PricerAssignment.create(sw), tq.SwaptionPricer)


class TestSwaptionMarketData(unittest.TestCase):
    def test_cube_round_trip_and_accessors(self):
        mkt = market()
        cube = mkt.get_swaption_vol_cube(tq.Currency.EUR, "6M")
        self.assertIn("IR:EUR:6M:SWVOL", mkt.risk_factors())
        back = MarketEnvironment.from_frames(mkt.to_frames(), EVAL_DATE).get_swaption_vol_cube(tq.Currency.EUR, "6M")
        np.testing.assert_array_equal(back.volatilities.numpy(), cube.volatilities.numpy())
        self.assertEqual((back.expiries, back.tenors, back.strike_offsets), (cube.expiries, cube.tenors, cube.strike_offsets))
        with self.assertRaises(ValueError):
            mkt.get_swaption_vol_cube(tq.Currency.USD, "6M")
        with self.assertRaises(TypeError):
            mkt.set_swaption_vol_cube(tq.Currency.EUR, "6M", 0.01)

    def test_synthetic_cube_is_arbitrage_free(self):
        """Per (expiry, tenor): Bachelier payer prices decreasing and convex in the strike."""
        cube = market().get_swaption_vol_cube(tq.Currency.EUR, "6M")
        vols, offs = cube.volatilities.numpy(), np.array(cube.strike_offsets)
        for i, T in enumerate(cube.expiries):
            for j in range(len(cube.tenors)):
                prices = np.array([bachelier(0.0, k, v, T) for k, v in zip(offs, vols[i, j])])
                slopes = np.diff(prices) / np.diff(offs)
                with self.subTest(expiry=T, tenor=cube.tenors[j]):
                    self.assertTrue(np.all(slopes < 0) and np.all(np.diff(slopes) > 0))


def implied_normal(price, F, K, T, A):
    return brentq(lambda v: A * bachelier(F, K, v, T) - price, 1e-7, 0.1, xtol=1e-14)


class TestModelSwaptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mkt = market()
        cls.disc = cls.mkt.get_ir_curve(tq.Currency.EUR)

    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE

    @slow
    def test_gaussian_swap_rate_approximation_vs_monte_carlo(self):
        """ATM normal vols of the approximation within 0.5bp + 2 SE of the exact (MC) model price."""
        models = {"HW": tq.GaussianShortRateModel.hull_white(self.disc, 0.03, 0.009),
                  "G2": tq.GaussianShortRateModel.g2pp(self.disc, 0.05, 0.6, 0.009, 0.008, -0.7)}
        rows = _swaption_instruments(self.mkt, swaption_generator(1.0), [("1Y", "5Y"), ("5Y", "10Y")], (0.0,))
        n = 10_000
        for name, m in models.items():
            for r in rows:
                price, F, A, vol = m.swaption_from_arrays(strike=r["K"], **r["arrays"])
                te = r["arrays"]["te"]
                xs, D = m.simulate([te], n, seed=3)
                x = xs[:, 0]
                ann = sum(w * m.zero_bond(te, t, x) for t, w in zip(r["arrays"]["pay_t"], r["arrays"]["pay_w"]))
                flo = sum(w * m.zero_bond(te, t, x) for t, w in zip(r["arrays"]["st_t"], r["arrays"]["st_w"])) - \
                    sum(w * m.zero_bond(te, t, x) for t, w in zip(r["arrays"]["en_t"], r["arrays"]["en_w"]))
                pay = (D[:, 0] * tf.maximum(flo - r["K"] * ann, 0.0)).numpy()
                mc, se = pay.mean(), pay.std() / np.sqrt(n)
                iv_mc = implied_normal(mc, float(F), r["K"], r["t"], float(A))
                iv_se = abs(implied_normal(mc + 2 * se, float(F), r["K"], r["t"], float(A)) - iv_mc)
                with self.subTest(model=name, swaption=(r["expiry"], r["tenor"])):
                    self.assertLess(abs(float(vol) - iv_mc), 0.5e-4 + iv_se)
                    self.assertAlmostEqual(float(F), r["F"], places=14)

    @slow
    def test_exact_formula_vs_monte_carlo(self):
        """Jamshidian (HW, piecewise HW) / Brigo-Mercurio (G2++) prices within 3 SE of MC at three strikes;
        payer - receiver = A (F - K); AAD gradient to a model parameter = finite difference."""
        models = {"HW": tq.GaussianShortRateModel.hull_white(self.disc, 0.03, 0.009),
                  "HWpw": tq.GaussianShortRateModel.hull_white(self.disc, 0.03, [0.006, 0.009, 0.012], sigma_times=[2.0, 6.0]),
                  "G2": tq.GaussianShortRateModel.g2pp(self.disc, 0.05, 0.6, 0.009, 0.008, -0.7)}
        rows = _swaption_instruments(self.mkt, swaption_generator(1.0), [("1Y", "5Y"), ("5Y", "10Y")], (-0.01, 0.0, 0.01))
        n = 10_000
        for name, m in models.items():
            legs = {}                                                  # MC legs per swaption, shared by the strikes
            for r in rows:
                te = r["arrays"]["te"]
                payer = m.swaption_from_arrays(strike=r["K"], method="exact", **r["arrays"])
                receiver = m.swaption_from_arrays(strike=r["K"], payer=False, method="exact", **r["arrays"])
                if (r["expiry"], r["tenor"]) not in legs:
                    xs, D = m.simulate([te], n, seed=3)
                    x = xs[:, 0]
                    bonds = lambda key: sum(w * m.zero_bond(te, t, x)
                                            for t, w in zip(r["arrays"][key + "_t"], r["arrays"][key + "_w"]))
                    legs[(r["expiry"], r["tenor"])] = (D[:, 0], bonds("st") - bonds("en"), bonds("pay"))
                disc, floating, annuity = legs[(r["expiry"], r["tenor"])]
                pay = (disc * tf.maximum(floating - r["K"] * annuity, 0.0)).numpy()
                with self.subTest(model=name, swaption=(r["expiry"], r["tenor"]), offset=r["K"] - r["F"]):
                    self.assertLess(abs(float(payer[0]) - pay.mean()), 3 * pay.std() / np.sqrt(n))
                    self.assertAlmostEqual(float(payer[0]) - float(receiver[0]), float(payer[2]) * (r["F"] - r["K"]),
                                           places=13)
                    self.assertAlmostEqual(r["annuity"] * bachelier(r["F"], r["K"], float(payer[3]), r["t"]), float(payer[0]),
                                           places=13)                                             # implied normal vol
        r = rows[4]
        for factors in (1, 2):
            def price(s):
                m = (tq.GaussianShortRateModel.hull_white(self.disc, 0.03, s) if factors == 1
                     else tq.GaussianShortRateModel.g2pp(self.disc, 0.05, 0.6, s, 0.008, -0.7))
                return m.swaption_from_arrays(strike=r["K"], method="exact", **r["arrays"])[0]
            s = tf.Variable(np.float64(0.009))
            with tf.GradientTape() as tape:
                v = price(s)
            fd = (float(price(0.009 + 1e-6)) - float(price(0.009 - 1e-6))) / 2e-6
            self.assertAlmostEqual(float(tape.gradient(v, s)) / fd, 1.0, places=7)
        with self.assertRaises(ValueError):
            models["HW"].swaption_from_arrays(strike=r["K"], method="other", **r["arrays"])

    @slow
    def test_g2pp_recovers_its_own_parameters(self):
        """Caplet and swaption vols generated by a G2++: the joint calibration finds the parameters back."""
        truth = dict(a=0.06, b=0.5, sigma=0.009, eta=0.007, rho=-0.6)
        model = tq.GaussianShortRateModel.g2pp(self.disc, **truth)
        pairs = [(e, t) for e in ("1Y", "3Y", "7Y") for t in ("2Y", "5Y", "10Y")]
        rows = _swaption_instruments(self.mkt, swaption_generator(1.0), pairs, (0.0,))
        cube_vols = {(r["expiry"], r["tenor"]): float(model.swaption_from_arrays(strike=r["K"], **r["arrays"])[3])
                     for r in rows}
        a365 = DayCounter(DayCounterConvention.Actual365)
        expiries = sorted({r["expiry"] for r in rows})
        tenors = ["2Y", "5Y", "10Y"]
        vols = [[[cube_vols[(e, t)]] * 3 for t in tenors] for e in expiries]
        mkt = market()
        mkt.set_swaption_vol_cube(tq.Currency.EUR, "6M", tq.SwaptionVolCube(
            EVAL_DATE, a365, [a365.year_fraction(EVAL_DATE, e) for e in expiries], [2.0, 5.0, 10.0],
            [-0.01, 0.0, 0.01], vols))
        _, params, report = calibrate_gaussian_short_rate(mkt, None, 2, term=None,
                                                          swaption_generator=swaption_generator(1.0), swaptions=pairs)
        self.assertLess(report.error_bp.abs().max(), 0.01)
        for k in ("a", "sigma", "eta", "rho"):
            self.assertAlmostEqual(params[k], truth[k], delta=0.02 * abs(truth[k]) + 1e-4, msg=k)

    @slow
    def test_joint_calibration_on_the_synthetic_market(self):
        """The synthetic caplet vols and cube come from one G2++: the joint calibration (exact swaptions) fits
        both within 0.3bp and finds the generating parameters (conventions differ slightly: dates, lags)."""
        from tensorquant.markethandles.synthetic import _rate_model_parameters, _streams
        truth = _rate_model_parameters(_streams(7)["rate_vols"])
        pairs = [(e, t) for e in ("1Y", "2Y", "5Y", "10Y") for t in ("2Y", "5Y", "10Y")]
        _, params, report = calibrate_gaussian_short_rate(self.mkt, cap_generator(), 2, swaption_generator=swaption_generator(1.0),
                                                          swaptions=pairs, swaption_method="exact")
        self.assertEqual(set(report.instrument), {"caplet", "swaption"})
        self.assertLess(report.error_bp.abs().max(), 0.3)
        self.assertEqual(params["at_bound"], [])
        for k in ("a", "b", "sigma", "eta", "rho"):
            self.assertAlmostEqual(params[k], truth[k], delta=0.05 * abs(truth[k]), msg=k)
        with self.assertRaises(ValueError):
            calibrate_gaussian_short_rate(self.mkt, cap_generator(), 2, swaption_generator=swaption_generator(1.0),
                                          swaptions=pairs, swaption_method="other")


if __name__ == "__main__":
    unittest.main()
