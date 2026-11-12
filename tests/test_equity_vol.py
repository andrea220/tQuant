"""Equity vols from listed options: implied volatility, implied forward / repo by put-call parity with
de-Americanization (``calibration/equityvol.py``)."""

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import tensorquant as tq
from markers import slow
from tensorquant.calibration.equityvol import (GRID_MONEYNESS, GRID_TENORS, LognormalMixture,
                                               black_implied_vol, build_equity_vol, calibrate_snapshot_vols,
                                               implied_forward,
                                               implied_repo_curve, implied_volatility)
from tensorquant.markethandles.dividendcurve import DividendCurve
from tensorquant.markethandles.ircurve import RateCurve
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.utils import Currency
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

EVAL_DATE = date(2026, 5, 15)
A365 = DayCounterConvention.Actual365
S0 = 100.0
EX_DATES = [date(2026, 9, 15), date(2027, 3, 15), date(2027, 9, 15), date(2028, 3, 15)]
AMOUNTS = [1.20, 1.25, 1.30, 1.35]
EXPIRIES = [EVAL_DATE + timedelta(days=d) for d in (91, 365, 730)]
REPO_TRUE = [0.003, 0.006, 0.010]                 # zero repo rates at the expiries
STRIKES = np.arange(70.0, 131.0, 5.0)


def surface() -> VolatilitySurface:
    """Skewed smile (the quotes use the vol at each strike)."""
    strikes = [60.0, 80.0, 100.0, 120.0, 140.0]
    return VolatilitySurface(reference_date=EVAL_DATE, calendar=None, daycounter=DayCounter(A365), strike=strikes,
                             maturity=[0.25, 1.0, 2.0],
                             volatility_matrix=[[0.36, 0.29, 0.23, 0.20, 0.21], [0.32, 0.27, 0.22, 0.19, 0.19],
                                                [0.30, 0.26, 0.22, 0.20, 0.19]])


def make_market(repo) -> MarketEnvironment:
    return MarketEnvironment(market={
        "IR:EUR:ESTR:SPOT": RateCurve(EVAL_DATE, [0.25, 0.5, 1.0, 2.0, 5.0], [0.026, 0.027, 0.028, 0.029, 0.032],
                                      "LINEAR", A365),
        "EQ:EUR:X:SPOT": S0,
        "EQ:EUR:X:REPO": repo,
        "EQ:EUR:X:DIV": DividendCurve(reference_date=EVAL_DATE, ex_dates=EX_DATES, amounts=AMOUNTS,
                                      currency=Currency.EUR, daycounter_convention=A365),
        "EQ:EUR:X:VOL": surface(),
    })


def true_market() -> MarketEnvironment:
    return make_market(RateCurve(EVAL_DATE, EXPIRIES, REPO_TRUE, "LINEAR", A365))


def quotes(exercise: str = "American", half_spread: float = 0.0) -> pd.DataFrame:
    """Calls and puts at every strike and expiry, priced by ``BlackScholesPricer`` on the true market (vol at the
    strike); bid / ask = price -/+ ``half_spread`` x price (at least 0.005)."""
    tq.Settings.evaluation_date = EVAL_DATE
    mkt, rows = true_market(), []
    for expiry in EXPIRIES:
        for k in STRIKES:
            for cp in ("C", "P"):
                opt = tq.VanillaOption(Currency.EUR, start_date=EVAL_DATE, end_date=expiry,
                                       option_type=tq.OptionType.Call if cp == "C" else tq.OptionType.Put,
                                       strike=float(k), underlying="X", exercise_type=getattr(tq.ExerciseType, exercise))
                tq.BlackScholesPricer().price(opt, mkt, autodiff=False)
                p = float(opt.price)
                h = max(half_spread * p, 0.005) if half_spread else 0.0
                rows.append({"expiry": expiry, "strike": float(k), "option_type": cp, "exercise": exercise,
                             "bid": p - h, "ask": p + h})
    return pd.DataFrame(rows)


class TestImpliedVolatility(unittest.TestCase):
    def test_round_trip_european_and_american(self):
        from tensorquant.pricers.black import bjerksund_stensland_calc, blackscholes_calc
        import tensorflow as tf

        k = np.array([70.0, 90.0, 100.0, 110.0, 140.0])
        sigma = np.array([0.35, 0.25, 0.22, 0.20, 0.30])
        s_net, r, q, t = 97.0, 0.03, 0.01, 1.3
        for american in (False, True):
            calc = bjerksund_stensland_calc if american else blackscholes_calc
            for cp, ot in (("C", tq.OptionType.Call), ("P", tq.OptionType.Put)):
                c = lambda x: tf.constant(x, tf.float64)
                prices = calc(c(s_net), c(k), c(r), c(sigma), c(t), c(q), ot).numpy()
                vols = implied_volatility(prices, k, cp, s_net, r, q, t, american=american)
                np.testing.assert_allclose(vols, sigma, rtol=0, atol=1e-10)

    def test_price_outside_bounds_is_nan(self):
        vols = implied_volatility(np.array([0.0, 1e6]), np.array([100.0, 100.0]), "C", 100.0, 0.02, 0.0, 1.0)
        self.assertTrue(np.isnan(vols).all())


class _ForwardCase(unittest.TestCase):
    """Calibration on a market with zero repo (start guess): the forwards and repos of the true market are found."""

    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.guess = make_market(0.0).get_equity("X")
        self.truth = true_market().get_equity("X")

    def check(self, frame: pd.DataFrame, rtol: float):
        for expiry, z in zip(EXPIRIES, REPO_TRUE):
            res = implied_forward(self.guess, expiry, frame[frame["expiry"] == expiry])
            f_true = float(self.truth.forward(expiry))
            self.assertAlmostEqual(res["forward"] / f_true, 1.0, delta=rtol)
            self.assertAlmostEqual(res["repo"], z, delta=rtol * 10)
            self.assertLessEqual(res["strike_low"], res["forward"])
            self.assertLessEqual(res["forward"], res["strike_high"])
        return res


class TestImpliedForwardEuropean(_ForwardCase):
    @classmethod
    def setUpClass(cls):
        cls.european = quotes("European")

    def test_european_parity(self):
        res = self.check(self.european, 1e-10)
        self.assertLessEqual(res["iterations"], 2)          # no de-Americanization: parity (again if the strikes move)

    def test_repo_curve_reprices_the_forwards(self):
        curve, report = implied_repo_curve(self.guess, self.european)
        self.assertIsInstance(curve, RateCurve)
        self.assertEqual(list(curve.dates), EXPIRIES)
        np.testing.assert_allclose([float(v) for v in curve.variables], REPO_TRUE, rtol=0, atol=1e-9)
        self.assertEqual(list(report["expiry"]), EXPIRIES)
        mkt = make_market(curve)
        for expiry, f in zip(report["expiry"], report["forward"]):
            self.assertAlmostEqual(float(mkt.get_equity("X").forward(expiry)) / f, 1.0, delta=1e-12)

    def test_oscillating_strike_pairs(self):
        """Noisy parity: the pair below the forward implies a forward above it and the pair above one below (the
        iteration jumps between the two pairs). The forward is the mean of the two, flagged as not converged."""
        expiry, df = EXPIRIES[0], float(self.guess.discount_curve.discount(EXPIRIES[0]))
        parity = {95.0: 102.0, 100.0: 100.0, 105.0: 99.0}          # K + (C - P) / DF at each strike
        rows = []
        for k, f in parity.items():
            put = 10.0
            rows += [{"strike": k, "option_type": "P", "bid": put, "ask": put},
                     {"strike": k, "option_type": "C", "bid": put + df * (f - k), "ask": put + df * (f - k)}]
        res = implied_forward(self.guess, expiry, pd.DataFrame(rows), max_iter=20)
        self.assertFalse(res["converged"])
        self.assertAlmostEqual(res["forward"], 0.5 * (101.0 + 99.5), places=6)
        res = implied_forward(self.guess, expiry, self.european[self.european["expiry"] == expiry])
        self.assertTrue(res["converged"])

    def test_missing_quotes_raise(self):
        with self.assertRaises(ValueError):
            implied_forward(self.guess, EXPIRIES[0], self.european.iloc[0:0])
        with self.assertRaises(ValueError):
            implied_forward(self.guess, EXPIRIES[0], self.european, dividend_model="escrowed")


@slow
class TestImpliedForwardAmerican(_ForwardCase):
    """De-Americanization (about 2 s an expiry): quotes from ``BlackScholesPricer`` American (continuous model)."""

    @classmethod
    def setUpClass(cls):
        cls.american = quotes("American")
        wide = cls.american.copy()                          # bid / ask = price -/+ max(1%, 0.005)
        h = np.maximum(0.01 * wide["bid"], 0.005)
        cls.wide = wide.assign(bid=wide["bid"] - h, ask=wide["ask"] + h)

    def test_american_de_americanization(self):
        res = self.check(self.american, 1e-9)
        self.assertGreater(res["iterations"], 1)

    def test_bid_ask_range_contains_true_forward(self):
        for expiry in EXPIRIES:
            res = implied_forward(self.guess, expiry, self.wide[self.wide["expiry"] == expiry])
            f_true = float(self.truth.forward(expiry))
            self.assertLess(res["forward_bid"], f_true)
            self.assertGreater(res["forward_ask"], f_true)
            self.assertAlmostEqual(res["forward"] / f_true, 1.0, delta=1e-6)   # symmetric spreads


class TestLognormalMixture(unittest.TestCase):
    """Smile of one expiry: mixture of lognormals with a default mass (``Q`` at zero)."""

    F, T, DF = 100.0, 1.5, 0.96
    K = np.linspace(40.0, 200.0, 33)

    def test_single_lognormal_is_black(self):
        m = LognormalMixture(weights=[1.0], means=[1.0], stdevs=[0.25 * np.sqrt(self.T)], default=0.0)
        np.testing.assert_allclose(m.implied_vols(self.F, self.K, self.T), 0.25, rtol=0, atol=1e-12)

    def test_forward_parity_and_convexity(self):
        m = LognormalMixture(weights=[0.2, 0.5, 0.29], means=[0.7, 1.0, 1.1], stdevs=[0.6, 0.3, 0.2], default=0.01)
        means = np.array(m.means)
        self.assertAlmostEqual(float(np.dot(m.weights, means)), 1.0, places=14)      # E[S_T] = F
        call, put = m.price(self.F, self.K, self.DF, "C"), m.price(self.F, self.K, self.DF, "P")
        np.testing.assert_allclose(call - put, self.DF * (self.F - self.K), rtol=0, atol=1e-11)
        self.assertTrue((np.diff(call, 2) > 0).all())                                 # density > 0
        self.assertGreater(m.price(self.F, np.array([1e-6]), 1.0, "P")[0] / 1e-6, 0.0099)   # put floor Q K

    @slow
    def test_fit_recovers_a_mixture_smile(self):
        true = LognormalMixture(weights=[0.15, 0.55, 0.299], means=[0.75, 1.0, 1.06], stdevs=[0.55, 0.28, 0.2],
                                default=0.001)
        vols = true.implied_vols(self.F, self.K, self.T)
        fit = LognormalMixture.fit(self.F, self.T, self.K, vols)
        np.testing.assert_allclose(fit.implied_vols(self.F, self.K, self.T), vols, rtol=0, atol=2e-4)
        self.assertAlmostEqual(float(np.dot(fit.weights, fit.means)), 1.0, places=12)

    def test_fit_skewed_smile(self):
        """Equity-like skew (vols of the test surface at 1Y, 60%-140%): within 0.3 vol points."""
        k = np.linspace(60.0, 140.0, 17)
        vols = np.array([float(surface().volatility(strike=float(x), tenor=1.0)) for x in k])
        fit = LognormalMixture.fit(100.0, 1.0, k, vols)
        self.assertLess(np.abs(fit.implied_vols(100.0, k, 1.0) - vols).max(), 3e-3)

    def test_black_implied_vol_round_trip(self):
        m = LognormalMixture(weights=[1.0], means=[1.0], stdevs=[0.3], default=0.0)
        for cp in ("C", "P"):
            undiscounted = m.price(self.F, self.K, 1.0, cp)
            np.testing.assert_allclose(black_implied_vol(undiscounted, self.F, self.K, 1.0, cp), 0.3, atol=1e-10)


class _SurfaceCase(unittest.TestCase):
    """Whole construction on the synthetic quotes: repo curve, smiles, surface on the standard grid up to 6Y."""

    exercise = "European"

    @classmethod
    def setUpClass(cls):
        cls.quotes = quotes(cls.exercise)
        tq.Settings.evaluation_date = EVAL_DATE
        cls.guess = make_market(0.0).get_equity("X")
        cls.truth = true_market().get_equity("X")
        cls.surface, cls.repo, cls.report = build_equity_vol(cls.guess, cls.quotes)

    def test_repo_curve(self):
        np.testing.assert_allclose([float(v) for v in self.repo.variables], REPO_TRUE, rtol=0, atol=1e-8)

    def test_grid(self):
        self.assertEqual(list(self.surface.maturity), list(GRID_TENORS))
        np.testing.assert_allclose(self.surface.strike, S0 * np.asarray(GRID_MONEYNESS), rtol=1e-14)
        matrix = self.surface.volatility_matrix.numpy()
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue(((matrix > 0.05) & (matrix < 1.0)).all())

    def test_quoted_vols_reproduced(self):
        """At the quoted expiries, within the quoted strikes: the smile of the quotes within 0.5 vol points."""
        for expiry in EXPIRIES:
            t = (expiry - EVAL_DATE).days / 365.0
            row = self.report[self.report["expiry"] == expiry].iloc[0]
            self.assertLess(row["fit_rms"], 0.005)
            f = float(self.truth.forward(expiry))
            for k in (80.0, 90.0, 100.0, 110.0, 120.0):
                m = self.report_smile(expiry)
                market = float(surface().volatility(strike=k, tenor=t))
                self.assertAlmostEqual(float(m.implied_vols(f, [k], t)[0]), market, delta=0.005)

    def report_smile(self, expiry):
        return self.report[self.report["expiry"] == expiry].iloc[0]["mixture"]

    def test_total_variance_increases(self):
        """ATM-forward total variance increases with the maturity (no calendar arbitrage at the money)."""
        w = []
        for t in GRID_TENORS:
            f = float(self.truth.forward(t))
            w.append(float(self.surface.volatility(strike=f, tenor=t)) ** 2 * t)
        self.assertTrue((np.diff(w) > 0).all())

    def test_long_end(self):
        """Beyond the last expiry (2Y): finite, smooth, ATM vol between 10% and 40%, skew kept (90% > 110%)."""
        for t in (3.0, 4.0, 5.0, 6.0):
            f = float(self.truth.forward(t))
            atm = float(self.surface.volatility(strike=f, tenor=t))
            self.assertTrue(0.10 < atm < 0.40)
            self.assertGreater(float(self.surface.volatility(strike=0.9 * f, tenor=t)),
                               float(self.surface.volatility(strike=1.1 * f, tenor=t)))


class TestSurfaceEuropean(_SurfaceCase):
    exercise = "European"

    def test_short_expiries_left_out_of_the_repo_curve(self):
        _, repo, report = build_equity_vol(self.guess, self.quotes, min_repo_days=100)    # first expiry: 91 days
        self.assertEqual(list(repo.dates), EXPIRIES[1:])
        self.assertEqual(len(report), len(EXPIRIES))                                         # its smile is kept

    def test_settlement_only_quotes(self):
        """Quotes without bid / ask (settlement ``last``) are used, with a lower weight."""
        frame = self.quotes.assign(last=0.5 * (self.quotes["bid"] + self.quotes["ask"]))
        last = frame["expiry"] == EXPIRIES[-1]
        frame.loc[last, ["bid", "ask"]] = np.nan
        surf, repo, report = build_equity_vol(self.guess, frame)
        row = report[report["expiry"] == EXPIRIES[-1]].iloc[0]
        self.assertEqual(row["source"], "settlement")
        np.testing.assert_allclose(float(repo.variables[-1]), REPO_TRUE[-1], atol=1e-8)


@slow
class TestSurfaceAmerican(_SurfaceCase):
    exercise = "American"


class TestSnapshotVols(unittest.TestCase):
    """``calibrate_snapshot_vols``: ``eq_option_quotes.csv`` of a snapshot -> ``eq_vol.csv``, ``eq_repo.csv``,
    ``eq_vol_calibration.csv``, read back by ``from_folder``."""

    def test_snapshot_round_trip(self):
        tq.Settings.evaluation_date = EVAL_DATE
        frame = quotes("European")
        frame = frame.assign(ticker="X", ccy="EUR", expiry=[e.isoformat() for e in frame["expiry"]],
                             last=np.nan, quote_date=EVAL_DATE.isoformat(), ref_spot=S0, spot_source="snapshot")
        bad = frame.iloc[:6].assign(ticker="Y")                  # an underlying not in the market: reported
        with tempfile.TemporaryDirectory() as tmp:
            folder = make_market(0.0).to_folder(tmp, "1200")
            pd.concat([frame, bad]).to_csv(folder / "eq_option_quotes.csv", index=False)
            written = calibrate_snapshot_vols(folder)
            self.assertEqual(written, ["eq_vol.csv", "eq_repo.csv", "eq_vol_calibration.csv", "eq_vol_quotes.csv"])
            qv = pd.read_csv(folder / "eq_vol_quotes.csv")
            x = qv[(qv["ticker"] == "X") & qv["used"]]
            self.assertTrue((x["iv_bid"] <= x["iv_mid"] + 1e-12).all() and (x["iv_mid"] <= x["iv_ask"] + 1e-12).all())
            self.assertLess((x["model_vol"] - x["iv_mid"]).abs().max(), 0.01)
            cal = pd.read_csv(folder / "eq_vol_calibration.csv")
            row = cal[cal["ticker"] == "X"].iloc[0]
            m = LognormalMixture(json.loads(row["weights"]), json.loads(row["means"]), json.loads(row["stdevs"]),
                                 row["default"])                       # the smile is redrawn from the parameters
            first = x[x["expiry"] == row["expiry"]]
            np.testing.assert_allclose(m.implied_vols(row["forward"], first["strike"], row["t"]), first["model_vol"],
                                       atol=1e-9)
            report = pd.read_csv(folder / "eq_vol_calibration.csv")
            self.assertEqual(set(report["ticker"]), {"X", "Y"})
            self.assertTrue(report.loc[report["ticker"] == "Y", "error"].notna().all())
            self.assertEqual(int((report["ticker"] == "X").sum()), len(EXPIRIES))
            mkt = MarketEnvironment.from_folder(tmp, EVAL_DATE, "1200")
            eq = mkt.get_equity("X")
            self.assertIsInstance(eq.repo, RateCurve)
            np.testing.assert_allclose([float(v) for v in eq.repo.variables], REPO_TRUE, rtol=0, atol=1e-4)
            surf = eq.vol_surface
            self.assertEqual(len(surf.maturity), len(GRID_TENORS))
            np.testing.assert_allclose(surf.strike, S0 * np.asarray(GRID_MONEYNESS), rtol=1e-12)
            self.assertAlmostEqual(float(surf.volatility(strike=100.0, tenor=1.0)),
                                   float(surface().volatility(strike=100.0, tenor=1.0)), delta=0.005)


if __name__ == "__main__":
    unittest.main()
