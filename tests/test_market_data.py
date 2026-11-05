"""Market data: FX spots/forwards/vols, CorrelationMatrix, from_frames/to_frames/to_folder, synthetic generator."""

import tempfile
import unittest
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import tensorflow as tf

import tensorquant as tq
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.synthetic import generate_market_data

EVAL_DATE = date(2026, 5, 15)
SMALL = (("EUR", 3), ("GBP", 1), ("USD", 1))
EURUSD, GBPUSD = 1.085, 1.27


def flat(rate):
    return tq.RateCurve(EVAL_DATE, [1.0, 5.0, 10.0], [rate] * 3, "LINEAR", tq.DayCounterConvention.Actual365)


def fx_market():
    return MarketEnvironment({
        "IR:EUR:ESTR:SPOT": flat(0.02), "IR:USD:SOFR:SPOT": flat(0.04), "IR:GBP:SONIA:SPOT": flat(0.045),
        "FX:USD:EURUSD:SPOT": EURUSD, "FX:USD:GBPUSD:SPOT": GBPUSD,
    })


class TestFX(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = fx_market()

    def test_spot_direct_inverse_cross(self):
        for b, q, expected in (("EUR", "USD", EURUSD), ("USD", "EUR", 1 / EURUSD), ("EUR", "GBP", EURUSD / GBPUSD),
                               ("GBP", "EUR", GBPUSD / EURUSD), (tq.Currency.USD, tq.Currency.USD, 1.0)):
            with self.subTest(pair=f"{b}{q}"):
                self.assertAlmostEqual(float(self.mkt.get_fx_spot(b, q)), expected, places=14)
        with self.assertRaises(ValueError):
            self.mkt.get_fx_spot("EUR", "JPY")

    def test_cross_gradient(self):
        rf = self.mkt.risk_factors()
        eu, gu = rf["FX:USD:EURUSD:SPOT"][0], rf["FX:USD:GBPUSD:SPOT"][0]
        with tf.GradientTape() as tape:
            x = self.mkt.get_fx_spot("EUR", "GBP")
        g_eu, g_gu = tape.gradient(x, [eu, gu])
        self.assertAlmostEqual(float(g_eu), 1 / GBPUSD, places=13)
        self.assertAlmostEqual(float(g_gu), -EURUSD / GBPUSD ** 2, places=13)

    def test_forward_covered_interest_parity_and_gradients(self):
        T = EVAL_DATE + timedelta(days=730)
        t = 730 / 365
        fwd = self.mkt.get_fx_forward("EUR", "USD", T)
        self.assertAlmostEqual(float(fwd), EURUSD * np.exp((0.04 - 0.02) * t), places=12)
        self.assertAlmostEqual(float(self.mkt.get_fx_forward("EUR", "GBP", T)),
                               EURUSD / GBPUSD * np.exp((0.045 - 0.02) * t), places=12)
        eur, usd = self.mkt.get_ir_curve(tq.Currency.EUR), self.mkt.get_ir_curve(tq.Currency.USD)
        with tf.GradientTape() as tape:
            fwd = self.mkt.get_fx_forward("EUR", "USD", T)
        g_eur, g_usd = tape.gradient(fwd, [eur.variables, usd.variables])
        self.assertAlmostEqual(sum(float(g) for g in g_eur if g is not None), -t * float(fwd), places=10)
        self.assertAlmostEqual(sum(float(g) for g in g_usd if g is not None), t * float(fwd), places=10)

    def test_fx_vol_surface(self):
        mkt = generate_and_build()
        surface = mkt.get_fx_vol_surface("EUR", "USD")
        self.assertIn("FX:USD:EURUSD:VOL", mkt.risk_factors())
        self.assertGreater(float(surface.volatility(EURUSD, 1.0)), 0.0)
        with self.assertRaises(ValueError):
            mkt.get_fx_vol_surface("USD", "EUR")       # inverse surfaces are not derived
        self.assertAlmostEqual(float(mkt.get_fx_spot("EUR", "USD")), EURUSD, places=14)   # VOL key ignored by spot


class TestCorrelationMatrix(unittest.TestCase):
    NAMES = ["EQ:EUR:A", "EQ:EUR:B", "EQ:USD:C"]
    PAIRS = [("EQ:EUR:A", "EQ:EUR:B", 0.5), ("EQ:EUR:A", "EQ:USD:C", 0.3), ("EQ:USD:C", "EQ:EUR:B", 0.4)]

    def test_from_pairs_submatrix_cholesky(self):
        c = tq.CorrelationMatrix.from_pairs(self.PAIRS)
        self.assertEqual(c.names, self.NAMES)
        sub = c.submatrix(["EQ:USD:C", "EQ:EUR:A"]).numpy()
        np.testing.assert_allclose(sub, [[1.0, 0.3], [0.3, 1.0]])
        L = c.cholesky(self.NAMES).numpy()
        np.testing.assert_allclose(L @ L.T, c.submatrix(self.NAMES).numpy(), atol=1e-14)
        self.assertEqual(float(c.correlation("EQ:EUR:B", "EQ:USD:C")), 0.4)
        with self.assertRaises(KeyError):
            c.submatrix(["EQ:EUR:X"])

    def test_validation(self):
        bad = {"asymmetric": [[1, 0.5], [0.4, 1]], "diagonal": [[1, 0.5], [0.5, 0.9]],
               "range": [[1, 1.2], [1.2, 1]], "not PSD": [[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]]}
        for name, m in bad.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                tq.CorrelationMatrix([f"f{i}" for i in range(len(m))], m)
        with self.assertRaises(ValueError):
            tq.CorrelationMatrix.from_pairs([("a", "b", 0.5), ("b", "a", 0.6)])
        with self.assertRaises(ValueError):
            tq.CorrelationMatrix(["a", "a"], np.eye(2))

    def test_gradient_of_a_pair_hits_both_entries(self):
        c = tq.CorrelationMatrix.from_pairs(self.PAIRS)
        with tf.GradientTape() as tape:
            total = tf.reduce_sum(c.submatrix(["EQ:EUR:A", "EQ:EUR:B"]))
        g = tape.gradient(total, c.matrix).numpy()
        self.assertEqual(g[0, 1], 1.0)
        self.assertEqual(g[1, 0], 1.0)
        self.assertEqual(g[0, 2], 0.0)


def generate_and_build():
    tq.Settings.evaluation_date = EVAL_DATE
    return tq.generate_market(EVAL_DATE, seed=7, tickers_per_currency=SMALL)


class TestFramesAndFolders(unittest.TestCase):
    def setUp(self):
        self.mkt = generate_and_build()

    def assert_same_market(self, a, b):
        ra, rb = a.risk_factors(), b.risk_factors()
        self.assertEqual(sorted(ra), sorted(rb))
        for key in ra:
            for x, y in zip(ra[key], rb[key]):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=key)
        for t in ("TICKER1", "TICKER5"):
            ea, eb = a.get_equity(t), b.get_equity(t)
            self.assertEqual(ea.dividends.ex_dates, eb.dividends.ex_dates)
            self.assertEqual(ea.dividends.payment_dates, eb.dividends.payment_dates)
            self.assertEqual(list(ea.vol_surface.maturity), list(eb.vol_surface.maturity))
            self.assertEqual(float(ea.forward(EVAL_DATE + timedelta(days=500))),
                             float(eb.forward(EVAL_DATE + timedelta(days=500))))

    def test_frames_round_trip(self):
        self.assert_same_market(self.mkt, MarketEnvironment.from_frames(self.mkt.to_frames(), EVAL_DATE))

    def test_folder_round_trip_and_latest_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            folder = self.mkt.to_folder(root, "0900")
            self.assertEqual(folder, Path(root) / EVAL_DATE.strftime("%Y%m%d") / "0900")
            self.assertEqual(sorted(p.name for p in folder.iterdir()), sorted(MarketEnvironment.MARKET_DATA_FILES))
            self.mkt.to_folder(root, "1700")
            back = MarketEnvironment.from_folder(root)                     # latest date, latest snapshot
            self.assertEqual((back.evaluation_date, back.calibration_set), (EVAL_DATE, "1700"))
            self.assert_same_market(self.mkt, back)
            self.assertEqual(MarketEnvironment.from_folder(root, EVAL_DATE, "0900").calibration_set, "0900")
        with tempfile.TemporaryDirectory() as root, self.assertRaises(FileNotFoundError):
            MarketEnvironment.from_folder(root)

    def test_same_price_after_round_trip(self):
        back = MarketEnvironment.from_frames(self.mkt.to_frames(), EVAL_DATE)
        prices = []
        for mkt in (self.mkt, back):
            eq = mkt.get_equity("TICKER2")
            opt = tq.VanillaOption(eq.currency, start_date=EVAL_DATE, end_date=EVAL_DATE + timedelta(days=365),
                                   option_type=tq.OptionType.Call, strike=round(float(eq.spot), 2),
                                   underlying="TICKER2", exercise_type=tq.ExerciseType.European)
            tq.BlackScholesPricer(dividend_model="discrete").price(opt, mkt)
            prices.append(float(opt.price))
        self.assertEqual(prices[0], prices[1])


class TestSyntheticGenerator(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE

    def test_deterministic_and_seeded(self):
        a, b = generate_market_data(EVAL_DATE, 7, SMALL), generate_market_data(EVAL_DATE, 7, SMALL)
        for name in a:
            self.assertTrue(a[name].equals(b[name]), name)
        c = generate_market_data(EVAL_DATE, 8, SMALL)
        self.assertFalse(a["eq_vol.csv"].equals(c["eq_vol.csv"]))

    def test_layout_and_ranges(self):
        t = generate_market_data(EVAL_DATE, 7, SMALL)
        self.assertEqual(sorted(t), sorted(MarketEnvironment.MARKET_DATA_FILES))
        self.assertEqual(t["eq_spot.csv"]["ccy"].tolist(), ["EUR", "EUR", "EUR", "GBP", "USD"])
        corr = t["correlations.csv"]["corr"]
        self.assertTrue(corr.between(0.30, 0.80).all())
        self.assertEqual(len(corr), 5 * 4 // 2)

    def test_surfaces_free_of_static_arbitrage(self):
        mkt = generate_and_build()
        for ticker in ("TICKER1", "TICKER4", "TICKER5"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = tq.LocalVolatilityModel.from_equity(mkt.get_equity(ticker))
            with self.subTest(ticker=ticker):
                self.assertEqual(caught, [])
                self.assertGreater(model.arbitrage["min_g"], 0.0)

    def test_ssvi_atm_on_the_forward(self):
        """At the ATM-forward strike the SSVI total variance is theta = atm(T)^2 T: vols interpolate around it."""
        mkt = generate_and_build()
        eq = mkt.get_equity("TICKER1")
        for months, d in ((12, date(2027, 5, 15)), (60, date(2031, 5, 15))):
            T = (d - EVAL_DATE).days / 365
            F = float(eq.forward(d))
            iv_atm_fwd = float(eq.vol_surface.volatility(F, T))
            self.assertTrue(0.10 < iv_atm_fwd < 0.45)

    def test_curves_are_smooth(self):
        mkt = generate_and_build()
        for ccy in (tq.Currency.EUR, tq.Currency.GBP, tq.Currency.USD):
            curve = mkt.get_ir_curve(ccy)
            f = np.array([float(curve.inst_fwd(float(x))) for x in np.linspace(0.1, 29.9, 300)])
            with self.subTest(ccy=ccy.name):
                self.assertLess(np.abs(np.diff(f)).max(), 0.0025)   # < 25bp between neighbours (linear zero interp)


if __name__ == "__main__":
    unittest.main()
