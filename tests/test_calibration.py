"""Calibration: caplet vol stripping from cap quotes; Gaussian short rate caplets and calibration to caplet vols."""

import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import brentq
from scipy.stats import norm

import tensorquant as tq
from tensorquant.calibration.capletstripping import _bachelier_call, _caplet_data, cap_flat_vols, strip_caplet_vols
from tensorquant.calibration.shortrate import calibrate_gaussian_short_rate
from tensorquant.index.curverateindex import IborIndex
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

EVAL_DATE = date(2026, 5, 15)
SMALL = (("EUR", 1), ("GBP", 1), ("USD", 1))
TERMS = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
STRIKES = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]


def market():
    tq.Settings.evaluation_date = EVAL_DATE
    return tq.generate_market(EVAL_DATE, seed=7, tickers_per_currency=SMALL)


def generator():
    cal = tq.TARGET()
    return tq.CapFloorGenerator(tq.Currency.EUR, 2, "6M", tq.BusinessDayConvention.ModifiedFollowing, 1.0,
                                tq.DayCounterConvention.Actual360, cal, IborIndex(cal, 6, tq.TimeUnit.Months, tq.Currency.EUR))


class TestCapletStripping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mkt = market()
        cls.original = cls.mkt.get_ir_vol_surface(tq.Currency.EUR, "6M")
        cls.quotes = cap_flat_vols(cls.mkt, generator(), TERMS, STRIKES)
        cls.stripped = strip_caplet_vols(cls.mkt, generator(), cls.quotes)

    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE

    def test_stripped_vols_reprice_every_cap_quote(self):
        """With the stripped caplet vols, CapFloorPricer reprices each cap at its flat-vol price."""
        self.mkt.set_ir_vol_surface(tq.Currency.EUR, "6M", self.stripped)
        try:
            for _, q in self.quotes.iterrows():
                cap = generator().build(EVAL_DATE, q.strike, q.term)
                tq.CapFloorPricer().price(cap, self.mkt)
                flat = sum(w * _bachelier_call(F, q.strike, q.vol, t)
                           for t, F, w, _ in _caplet_data(self.mkt, generator(), q.strike, q.term))
                with self.subTest(term=q.term, K=q.strike):
                    self.assertAlmostEqual(float(cap.price) / flat, 1.0, delta=1e-10)
        finally:
            self.mkt.set_ir_vol_surface(tq.Currency.EUR, "6M", self.original)

    def test_close_to_the_original_caplet_vols(self):
        """Piecewise-constant stripping of a smooth (SABR) caplet surface: within 3bp."""
        diff = [float(self.stripped.volatility(k, t)) - float(self.original.volatility(k, t))
                for t in self.stripped.maturity for k in STRIKES]
        self.assertLess(np.max(np.abs(diff)), 3e-4)
        self.assertEqual(self.stripped.vol_type, "normal")
        self.assertEqual(len(self.stripped.maturity), 19)   # caplets of the 10Y cap (first period excluded)

    def test_flat_quotes_strip_to_flat_caplets(self):
        quotes = pd.DataFrame([{"term": t, "strike": k, "vol": 0.008} for t in ("2Y", "5Y") for k in (0.01, 0.03)])
        surface = strip_caplet_vols(self.mkt, generator(), quotes)
        np.testing.assert_allclose(surface.volatility_matrix.numpy(), 0.008, atol=1e-12)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            strip_caplet_vols(self.mkt, generator(), self.quotes.iloc[:-1])          # incomplete grid
        with self.assertRaises(ValueError):
            strip_caplet_vols(self.mkt, generator(), self.quotes, vol_type="lognormal")
        with self.assertRaises(ValueError):
            self.mkt.set_ir_vol_surface(tq.Currency.USD, "SOFR", self.stripped)        # no VOL entry in the map
        with self.assertRaises(TypeError):
            self.mkt.set_ir_vol_surface(tq.Currency.EUR, "6M", 0.01)


class TestModelCaplets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mkt = market()
        cls.curve = cls.mkt.get_ir_curve(tq.Currency.EUR)
        idx_curve = cls.mkt.get_ir_curve(tq.Currency.EUR, "6M")
        cls.fix = EVAL_DATE + timedelta(days=3 * 365)
        cls.start, cls.end = cls.fix + timedelta(days=2), cls.fix + timedelta(days=184)
        cls.tau = (cls.end - cls.start).days / 360
        cls.basis = (float(idx_curve.discount(cls.start)) / float(idx_curve.discount(cls.end))) / \
            (float(cls.curve.discount(cls.start)) / float(cls.curve.discount(cls.end)))

    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE

    def models(self):
        return {"HW": tq.GaussianShortRateModel.hull_white(self.curve, 0.05, 0.009),
                "G2": tq.GaussianShortRateModel.g2pp(self.curve, 0.05, 0.4, 0.008, 0.006, -0.5)}

    def test_closed_form_vs_monte_carlo(self):
        n = 200_000
        for name, m in self.models().items():
            xs, D = m.simulate([m.time(self.fix)], n, seed=19)
            x = xs[:, 0]
            ps, pe = m.zero_bond(self.fix, self.start, x), m.zero_bond(self.fix, self.end, x)
            for K, is_cap in ((0.02, True), (0.035, True), (0.04, False)):
                phi = 1.0 if is_cap else -1.0
                payoff = (D[:, 0] * tf.maximum(phi * (self.basis * ps - (1 + self.tau * K) * pe), 0.0)).numpy()
                closed = float(m.caplet(self.fix, self.start, self.end, K, self.tau, self.basis, is_cap))
                with self.subTest(model=name, K=K, cap=is_cap):
                    self.assertLess(abs(payoff.mean() - closed), 3.5 * payoff.std() / np.sqrt(n))

    def test_cap_floor_parity_and_gradients(self):
        m = self.models()["G2"]
        for K in (0.01, 0.03):
            cap = float(m.caplet(self.fix, self.start, self.end, K, self.tau, self.basis, True))
            floor = float(m.caplet(self.fix, self.start, self.end, K, self.tau, self.basis, False))
            fwd = self.basis * float(self.curve.discount(self.start)) - (1 + self.tau * K) * float(self.curve.discount(self.end))
            self.assertAlmostEqual(cap - floor, fwd, places=14)
        with tf.GradientTape() as tape:
            v = self.models()["HW"].caplet(self.fix, self.start, self.end, 0.03, self.tau, self.basis)
        grads = tape.gradient(v, self.curve.variables)
        self.assertTrue(any(g is not None and float(g) != 0.0 for g in grads))


def model_implied_surface(mkt, model, strikes):
    """Normal caplet vols implied by *model* on (fixing times of the 10Y cap) x strikes."""
    disc, fwd = mkt.get_ir_curve(tq.Currency.EUR), mkt.get_ir_curve(tq.Currency.EUR, "6M")
    a365 = DayCounter(DayCounterConvention.Actual365)
    times, rows = [], []
    for c in generator().build(EVAL_DATE, 0.0, "10Y").caplets:
        s, e = c.ref_period_start, c.ref_period_end
        tau = c.index.daycounter.year_fraction(s, e)
        F = (float(fwd.discount(s)) / float(fwd.discount(e)) - 1) / tau
        basis = (float(fwd.discount(s)) / float(fwd.discount(e))) / (float(disc.discount(s)) / float(disc.discount(e)))
        t = a365.year_fraction(EVAL_DATE, c.fixing_date)
        df = float(disc.discount(e))
        row = []
        for K in strikes:
            target = float(model.caplet(c.fixing_date, s, e, K, tau, basis)) / (tau * df)
            row.append(brentq(lambda v: _bachelier_call(F, K, v, t) - target, 1e-8, 0.2, xtol=1e-15))
        times.append(t)
        rows.append(row)
    return VolatilitySurface(EVAL_DATE, None, a365, list(strikes), times, rows, vol_type="normal")


class TestShortRateCalibration(unittest.TestCase):
    def setUp(self):
        self.mkt = market()
        self.original = self.mkt.get_ir_vol_surface(tq.Currency.EUR, "6M")

    def tearDown(self):
        self.mkt.set_ir_vol_surface(tq.Currency.EUR, "6M", self.original)

    def test_recovers_hull_white_parameters(self):
        """Caplet vols generated by Hull-White (a = 6%, sigma = 90bp): the calibration finds them back."""
        strikes = [0.01, 0.02, 0.03, 0.04]
        truth = tq.GaussianShortRateModel.hull_white(self.mkt.get_ir_curve(tq.Currency.EUR), 0.06, 0.009)
        self.mkt.set_ir_vol_surface(tq.Currency.EUR, "6M", model_implied_surface(self.mkt, truth, strikes))
        _, params, report = calibrate_gaussian_short_rate(self.mkt, generator(), 1, strikes=strikes)
        self.assertAlmostEqual(params["a"], 0.06, delta=1e-6)
        self.assertAlmostEqual(params["sigma"], 0.009, delta=1e-8)
        self.assertLess(report.error_bp.abs().max(), 1e-4)
        self.assertEqual(params["at_bound"], [])

    def test_g2pp_fits_its_own_caplets(self):
        strikes = [0.01, 0.025, 0.04]
        truth = tq.GaussianShortRateModel.g2pp(self.mkt.get_ir_curve(tq.Currency.EUR), 0.08, 0.6, 0.009, 0.007, -0.4)
        self.mkt.set_ir_vol_surface(tq.Currency.EUR, "6M", model_implied_surface(self.mkt, truth, strikes))
        _, _, report = calibrate_gaussian_short_rate(self.mkt, generator(), 2, strikes=strikes)
        self.assertLess(report.error_bp.abs().max(), 0.05)

    def test_atm_fit_on_the_synthetic_sabr_market(self):
        _, hw, rep_hw = calibrate_gaussian_short_rate(self.mkt, generator(), 1)
        _, g2, rep_g2 = calibrate_gaussian_short_rate(self.mkt, generator(), 2)
        self.assertLess(np.sqrt((rep_hw.error_bp ** 2).mean()), 10.0)   # 2 constant params: term structure only
        self.assertLess(rep_g2.error_bp.abs().max(), 1.0)
        self.assertEqual(len(rep_hw), 19)
        self.assertIn("at_bound", hw)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            calibrate_gaussian_short_rate(self.mkt, generator(), 3)
        lognormal = VolatilitySurface(EVAL_DATE, None, DayCounter(DayCounterConvention.Actual365), [0.01, 0.03],
                                      [1.0, 5.0], [[0.3, 0.3], [0.3, 0.3]], vol_type="lognormal")
        self.mkt.set_ir_vol_surface(tq.Currency.EUR, "6M", lognormal)
        with self.assertRaises(ValueError):
            calibrate_gaussian_short_rate(self.mkt, generator(), 1)


if __name__ == "__main__":
    unittest.main()
