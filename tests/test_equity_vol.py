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
from scipy.stats import norm
from tensorquant.calibration.equityvol import (GRID_MONEYNESS, GRID_TENORS, LognormalMixture, _carry_smile, _smile_w,
                                               _wing_spec,
                                               listed_tenors,
                                               black_implied_vol, build_equity_vol, calibrate_snapshot_vols, implied_spot,
                                               extrapolate_dividends, implied_forward,
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

    def test_fit_without_point_mass(self):
        """Smooth skewed smile with 0.2 vol pt noise on sparse strikes: without the vol floor the fit puts 15% of the
        probability on a lognormal of vol 0.6% (an almost point mass, a jump of 5% in P(S_T < K) over 1% of the
        forward); with it no lognormal below 1/4 of the at-the-money vol, a smooth density and the fit within 0.05 vol
        points of the unconstrained one."""
        k = np.array([50, 60, 70, 80, 85, 90, 95, 97.5, 100, 102.5, 105, 110, 120, 128, 138, 149.0])
        rng = np.random.default_rng(22)
        x = np.log(k / 100.0)
        vols = 0.26 - 0.25 * rng.uniform(0.5, 1.5) * x + 0.3 * rng.uniform(0.3, 1.5) * x ** 2 \
            + rng.normal(0.0, 0.002, len(k))
        grid, h = np.linspace(40.0, 180.0, 141), 1e-4

        def largest_step(m):                                  # largest increase of P(S_T < K) over 1% of F
            cdf = (m.price(100.0, grid + h, 1.0, "P") - m.price(100.0, grid - h, 1.0, "P")) / (2.0 * h)
            return float(np.diff(cdf).max())

        free = LognormalMixture.fit(100.0, 1.0, k, vols, min_vol_fraction=0.0)
        self.assertGreater(free.weights[free.stdevs < 0.03].sum(), 0.10)
        self.assertGreater(largest_step(free), 0.04)
        fit = LognormalMixture.fit(100.0, 1.0, k, vols)
        atm = float(np.interp(100.0, k, vols))
        self.assertGreaterEqual(fit.stdevs.min(), 0.25 * atm * (1.0 - 1e-9))
        self.assertLess(largest_step(fit), 0.02)
        self.assertLess(fit.rms - free.rms, 5e-4)
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
        listed = [(e - EVAL_DATE).days / 365.0 for e in EXPIRIES]                 # the quoted expiries
        self.assertEqual(list(self.surface.maturity), listed + [3.0, 4.0, 5.0, 6.0])   # then one a year to 6Y
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

    def test_no_butterfly_arbitrage(self):
        """Every grid slice (30%-200% of the spot): undiscounted call prices convex in the strike (puts below the
        forward, the same convexity without the rounding of deep in-the-money calls)."""
        strikes = np.asarray(self.surface.strike, float)
        matrix = self.surface.volatility_matrix.numpy()
        for t, vols in zip(self.surface.maturity, matrix):
            f = float(self.truth.forward(t))
            sd = vols * np.sqrt(t)
            d1 = np.log(f / strikes) / sd + 0.5 * sd
            call = f * norm.cdf(d1) - strikes * norm.cdf(d1 - sd)
            prices = np.where(strikes < f, call - (f - strikes), call)
            conv_c = np.diff(np.diff(call) / np.diff(strikes))
            conv_p = np.diff(np.diff(prices) / np.diff(strikes))
            conv = np.where(strikes[2:] < f, conv_p, conv_c)
            self.assertGreaterEqual(conv.min(), -1e-9, f"T = {t}")

    def test_no_calendar_arbitrage(self):
        """Total variance non-decreasing in the maturity at fixed forward moneyness, over the whole grid (wings beyond
        the quotes included: they are straight in total variance and raised to the earlier expiries)."""
        strikes = np.asarray(self.surface.strike, float)
        matrix = self.surface.volatility_matrix.numpy()
        t = np.asarray(self.surface.maturity, float)
        k_fwd = [np.log(strikes / float(self.truth.forward(tt))) for tt in t]
        for i in range(len(t) - 1):
            k = k_fwd[i + 1]
            inside = (k >= k_fwd[i][0]) & (k <= k_fwd[i][-1])
            w_before = np.interp(k, k_fwd[i], matrix[i] ** 2 * t[i])
            w_after = matrix[i + 1] ** 2 * t[i + 1]
            self.assertGreaterEqual((w_after - w_before)[inside].min(), -1e-4, f"T = {t[i]} -> {t[i + 1]}")

    def test_wings_beyond_the_quotes(self):
        """Beyond the outermost quotes of an expiry: a straight total variance in log-moneyness (the junction found,
        slope within Lee's bounds), never decreasing away from the money."""
        for _, row in self.report[self.report["mixture"].notna()].iterrows():
            spec = _wing_spec(row["mixture"], row["t"], row["k_lo"], row["k_hi"])
            for side, sign in (("lo", -1.0), ("hi", 1.0)):
                self.assertIsNotNone(spec[side], f"{row['expiry']} {side}")
                kj, wj, slope = spec[side]
                self.assertTrue(0.0 <= sign * slope <= 2.0)
                k = kj + sign * np.linspace(0.0, 1.0, 11)
                w, reliable = _smile_w(row["mixture"], row["t"], k, spec)
                np.testing.assert_allclose(w, wj + slope * (k - kj), rtol=0, atol=1e-12)
                self.assertTrue(reliable.all())

    def test_total_variance_increases(self):
        """ATM-forward total variance increases with the maturity (no calendar arbitrage at the money)."""
        w = []
        for t in self.surface.maturity:
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

    def test_listed_tenors(self):
        self.assertEqual(listed_tenors([0.02, 0.5, 2.2], 6.0), [0.02, 0.5, 2.2, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(listed_tenors([0.5, 7.0], 6.0), [0.5, 7.0])

    def test_fixed_grid(self):
        surf, _, _ = build_equity_vol(self.guess, self.quotes, tenors=GRID_TENORS)
        self.assertEqual(list(surf.maturity), list(GRID_TENORS))

    def test_short_expiries_left_out_of_the_repo_curve(self):
        _, repo, report = build_equity_vol(self.guess, self.quotes, min_repo_days=100)    # first expiry: 91 days
        self.assertEqual(list(repo.dates), EXPIRIES[1:])
        self.assertEqual(len(report), len(EXPIRIES))                                         # its smile is kept

    def test_settlement_prices_dropped_with_a_market(self):
        """With a bid / ask market (Bloomberg's rule: 3 good expiries or 10% of them) an expiry with settlement
        prices only (``last``, no bid / ask) is dropped: forwards, repo and smiles from the quoted expiries."""
        frame = self.quotes.assign(last=0.5 * (self.quotes["bid"] + self.quotes["ask"]))
        frame.loc[frame["expiry"] == EXPIRIES[-1], ["bid", "ask"]] = np.nan
        surf, repo, report = build_equity_vol(self.guess, frame)
        self.assertNotIn(EXPIRIES[-1], set(report["expiry"]))
        self.assertEqual(len(repo.variables), len(EXPIRIES) - 1)
        np.testing.assert_allclose([float(v) for v in repo.variables], REPO_TRUE[:-1], rtol=0, atol=1e-8)

    def test_narrow_expiry_borrows_the_smile(self):
        """An expiry quoted only within 90%-110% of the forward: its own quotes fitted as usual, its wings guided by the
        next expiry's smile carried to its maturity and scaled to its at-the-money vol (not a free mixture tail)."""
        f0 = float(self.truth.forward(EXPIRIES[0]))
        first = self.quotes["expiry"] == EXPIRIES[0]
        near = (self.quotes["strike"] > 0.92 * f0) & (self.quotes["strike"] < 1.08 * f0)
        surf, repo, report = build_equity_vol(self.guess, self.quotes[~first | near])
        row = report[report["expiry"] == EXPIRIES[0]].iloc[0]
        lend = report[report["expiry"] == EXPIRIES[1]].iloc[0]
        self.assertEqual(row["smile_from"], EXPIRIES[1])
        self.assertIsNone(lend["smile_from"])
        self.assertLess(row["fit_rms"], 0.002)                         # its own quotes within 0.2 vol pts
        t, m = row["t"], row["mixture"]
        guide = _carry_smile(lend["mixture"], lend["t"], t)
        x = np.array([0.6, 0.7, 1.3])                                   # wings: within 2 vol pts of the guide's shape
        shift = float(m.implied_vols(1.0, [1.0], t)[0] - guide.implied_vols(1.0, [1.0], t)[0])
        np.testing.assert_allclose(m.implied_vols(1.0, x, t), guide.implied_vols(1.0, x, t) + shift, rtol=0, atol=0.02)

    def test_settlement_prices_without_a_market(self):
        """Without any bid / ask the settlement prices are used (the forwards and repo recovered from them)."""
        frame = self.quotes.assign(last=0.5 * (self.quotes["bid"] + self.quotes["ask"]), bid=np.nan, ask=np.nan)
        surf, repo, report = build_equity_vol(self.guess, frame)
        self.assertTrue((report["source"] == "settlement").all())
        np.testing.assert_allclose([float(v) for v in repo.variables], REPO_TRUE, rtol=0, atol=1e-8)


@slow
class TestSurfaceAmerican(_SurfaceCase):
    exercise = "American"


class TestImpliedSpot(unittest.TestCase):
    """Spot implied by the options (index with a delayed price): the forward of the first expiry from 3 days, plus
    the dividends to it, discounted: that expiry has zero repo."""

    def test_spot_from_the_first_month_forward(self):
        tq.Settings.evaluation_date = EVAL_DATE
        frame = quotes("European")
        delayed = make_market(0.0)
        delayed_eq = delayed.get_equity("X")
        delayed_eq.spot.assign(S0 * 1.005)                                         # a stale index price
        spot, anchor = implied_spot(delayed_eq, frame)
        self.assertEqual(anchor, EXPIRIES[0])                                      # 91 days: the first one
        truth = true_market().get_equity("X")
        t = (anchor - EVAL_DATE).days / 365.0
        expected = (S0 - float(truth.pv_dividends(anchor))) * np.exp(-REPO_TRUE[0] * t) \
            + float(truth.pv_dividends(anchor))
        self.assertAlmostEqual(spot, expected, delta=1e-8)


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
            self.assertEqual(len(surf.maturity), len(EXPIRIES) + 4)
            np.testing.assert_allclose(surf.strike, S0 * np.asarray(GRID_MONEYNESS), rtol=1e-12)
            self.assertAlmostEqual(float(surf.volatility(strike=100.0, tenor=1.0)),
                                   float(surface().volatility(strike=100.0, tenor=1.0)), delta=0.005)

    def test_implied_spot_in_the_snapshot(self):
        """``spot_source`` "implied ...": the spot is implied by the options, written to ``eq_spot.csv`` too, and
        the repo of the anchor expiry is zero."""
        tq.Settings.evaluation_date = EVAL_DATE
        frame = quotes("European")
        frame = frame.assign(ticker="X", ccy="EUR", expiry=[e.isoformat() for e in frame["expiry"]],
                             last=np.nan, quote_date=EVAL_DATE.isoformat(), ref_spot=S0 * 1.005,
                             spot_source="implied from the options")
        with tempfile.TemporaryDirectory() as tmp:
            folder = make_market(0.0).to_folder(tmp, "1200")
            frame.to_csv(folder / "eq_option_quotes.csv", index=False)
            calibrate_snapshot_vols(folder)
            spot = float(pd.read_csv(folder / "eq_spot.csv").set_index("ticker").loc["X", "spot"])
            self.assertLess(abs(spot / S0 - 1.0), 0.001)                           # not the stale 100.5
            report = pd.read_csv(folder / "eq_vol_calibration.csv")
            self.assertAlmostEqual(float(report["ref_spot"].iloc[0]), spot, delta=1e-9)
            self.assertTrue(report["spot_source"].iloc[0].startswith("implied"))
            anchor = report[report["expiry"] == EXPIRIES[0].isoformat()].iloc[0]
            self.assertAlmostEqual(float(anchor["repo"]), 0.0, delta=1e-9)


class TestDividendExtrapolation(unittest.TestCase):
    """Index dividends beyond the projections at a constant yield (``extrapolate_dividends``)."""

    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.eq = make_market(0.0).get_equity("X")             # semiannual dividends to 2028-03-15

    def forward(self, t, items, repo):
        """Independent forward just before *t* on all the dividends (ex, pay, amount) before it."""
        disc = self.eq.discount_curve
        pv = sum(a * float(disc.discount(p)) for e, p, a in items if e < t)
        return (S0 - pv) * np.exp(-repo * (t - EVAL_DATE).days / 365.0) / float(disc.discount(t))

    def test_constant_yield_to_the_horizon(self):
        repo = 0.004
        extra = extrapolate_dividends(self.eq, 10.0, repo)
        horizon = date(2036, 5, 15)
        self.assertTrue(len(extra))
        self.assertLessEqual(max(extra["ex_date"]), horizon)
        self.assertEqual(min(extra["ex_date"]), date(2028, 9, 15))            # the last year repeated: Sep, Mar
        self.assertEqual(sorted({(d.month, d.day) for d in extra["ex_date"]}), [(3, 15), (9, 15)])
        self.assertTrue((extra["ex_date"] == extra["pay_date"]).all())
        items = [(e, e, a) for e, a in zip(EX_DATES, AMOUNTS)] + list(
            zip(extra["ex_date"], extra["pay_date"], extra["amount"]))
        base = {(e.month, e.day): a / self.forward(e, items, repo) for e, a in zip(EX_DATES[2:], AMOUNTS[2:])}
        for e, a in zip(extra["ex_date"], extra["amount"]):                 # same yield as the dividend it repeats
            self.assertAlmostEqual(a / self.forward(e, items, repo), base[(e.month, e.day)], delta=1e-12)
        self.assertGreater(extra["amount"].iloc[-1], extra["amount"].iloc[0])   # grows with the forward
        self.assertGreater(extra.attrs["yield"], 0.02)
        self.assertEqual(extra.attrs["last_projected"], EX_DATES[-1])

    def test_nothing_to_extrapolate(self):
        self.assertEqual(len(extrapolate_dividends(self.eq, 1.0)), 0)       # projections beyond the horizon
        tq.Settings.evaluation_date = EVAL_DATE
        no_div = MarketEnvironment(market={"IR:EUR:ESTR:SPOT": self.eq.discount_curve, "EQ:EUR:Y:SPOT": S0,
                                           "EQ:EUR:Y:DIVYIELD": 0.02}).get_equity("Y")
        self.assertEqual(len(extrapolate_dividends(no_div, 10.0)), 0)

    def test_index_snapshot(self):
        """An index of the snapshot (``Index`` ticker in ``snapshot_info.json``): dividends to 10 years written with
        ``source`` = extrapolated, the forward of the listed expiries unchanged (the repo absorbs nothing more), and a
        rerun gives the same file; a single stock is left as it is."""
        frame = quotes("European")
        frame = frame.assign(ticker="X", ccy="EUR", expiry=[e.isoformat() for e in frame["expiry"]],
                             last=np.nan, quote_date=EVAL_DATE.isoformat(), ref_spot=S0, spot_source="snapshot")
        with tempfile.TemporaryDirectory() as tmp:
            folder = make_market(0.0).to_folder(tmp, "1200")
            frame.to_csv(folder / "eq_option_quotes.csv", index=False)
            (folder / "snapshot_info.json").write_text(json.dumps({"universe": ["X Index", "ESTRON Index"]}))
            written = calibrate_snapshot_vols(folder)
            self.assertIn("eq_dividends.csv", written)
            divs = pd.read_csv(folder / "eq_dividends.csv")
            self.assertTrue((folder / "eq_dividends_download.csv").is_file())          # the download kept apart
            self.assertEqual(set(divs["source"]), {"download", "extrapolated"})       # no dividend between cutoff and expiry
            self.assertEqual(int((divs["source"] == "download").sum()), len(EX_DATES))
            self.assertLessEqual(pd.to_datetime(divs["ex_date"]).max().date(), date(2036, 5, 15))
            first = divs.copy()
            report = pd.read_csv(folder / "eq_vol_calibration.csv")
            self.assertTrue(report["div_extrapolated_yield"].notna().all())
            self.assertTrue(report["div_cutoff"].notna().all())
            mkt = MarketEnvironment.from_folder(tmp, EVAL_DATE, "1200")             # read back with the extrapolation
            eq = mkt.get_equity("X")
            self.assertEqual(len(eq.dividends), len(divs))
            for row in report.itertuples():                                      # listed forwards repriced
                self.assertAlmostEqual(float(eq.forward(date.fromisoformat(row.expiry))), row.forward,
                                       delta=1e-6 * S0)
            calibrate_snapshot_vols(folder)                                      # rerun: same dividends
            pd.testing.assert_frame_equal(pd.read_csv(folder / "eq_dividends.csv"), first)
        with tempfile.TemporaryDirectory() as tmp:                               # no snapshot_info: not an index
            folder = make_market(0.0).to_folder(tmp, "1200")
            frame.to_csv(folder / "eq_option_quotes.csv", index=False)
            self.assertNotIn("eq_dividends.csv", calibrate_snapshot_vols(folder))


class TestImpliedIndexDividends(unittest.TestCase):
    """Index dividends implied by the listed forwards with the repo held (``implied_index_dividends`` in
    ``calibrate_snapshot_vols``): a true market (quarterly dividends 0.8, flat repo 0.4%), options priced on it to 5
    years, projections biased +30% beyond 2 years and missing beyond 3: the true dividends and the flat repo are
    recovered."""

    TRUE_REPO = 0.004
    EXPIRIES = [EVAL_DATE + timedelta(days=365 * n) for n in (1, 2, 3, 4, 5)]

    @staticmethod
    def ex_dates(until):
        out, d = [], date(2026, 8, 15)
        while d <= until:
            out.append(d)
            d = date(d.year + (d.month + 3 > 12), (d.month + 2) % 12 + 1, 15)
        return out

    def market(self, ex_dates, amounts, repo):
        curve = tq.RateCurve(EVAL_DATE, [0.25, 0.5, 1.0, 2.0, 5.0, 10.0], [0.026, 0.027, 0.028, 0.029, 0.032, 0.033],
                             "LINEAR", A365)
        return MarketEnvironment(market={
            "IR:EUR:ESTR:SPOT": curve, "EQ:EUR:X:SPOT": S0, "EQ:EUR:X:REPO": repo,
            "EQ:EUR:X:DIV": DividendCurve(reference_date=EVAL_DATE, ex_dates=ex_dates, amounts=amounts,
                                          currency=Currency.EUR, daycounter_convention=A365),
            "EQ:EUR:X:VOL": surface()})

    def test_true_dividends_and_flat_repo(self):
        tq.Settings.evaluation_date = EVAL_DATE
        true_dates = self.ex_dates(date(2032, 5, 15))
        truth = self.market(true_dates, [0.8] * len(true_dates), self.TRUE_REPO)
        rows = []
        for expiry in self.EXPIRIES:
            for k in (85.0, 95.0, 100.0, 105.0, 115.0):
                for cp in ("C", "P"):
                    opt = tq.VanillaOption(Currency.EUR, EVAL_DATE, expiry,
                                           tq.OptionType.Call if cp == "C" else tq.OptionType.Put, k, underlying="X")
                    tq.BlackScholesPricer().price(opt, truth)
                    p = float(opt.price)
                    rows.append({"ticker": "X", "ccy": "EUR", "expiry": expiry.isoformat(), "strike": k,
                                 "option_type": cp, "exercise": "European", "bid": p - 0.01, "ask": p + 0.01,
                                 "last": np.nan, "quote_date": EVAL_DATE.isoformat(), "ref_spot": S0,
                                 "spot_source": "snapshot"})
        cutoff = date(2028, 5, 15)
        bbg_dates = self.ex_dates(date(2029, 5, 15))                                  # projections to 3 years
        bbg = [0.8 if d <= cutoff else 1.04 for d in bbg_dates]                       # +30% beyond the cutoff
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.market(bbg_dates, bbg, 0.0).to_folder(tmp, "1200")
            pd.DataFrame(rows).to_csv(folder / "eq_option_quotes.csv", index=False)
            (folder / "snapshot_info.json").write_text(json.dumps({"universe": ["X Index"]}))
            calibrate_snapshot_vols(folder)
            divs = pd.read_csv(folder / "eq_dividends.csv")
            divs["ex"] = pd.to_datetime(divs["ex_date"]).dt.date
            self.assertEqual(set(divs["source"]), {"download", "implied", "extrapolated"})
            self.assertTrue((divs.loc[divs["source"] == "download", "ex"] <= cutoff).all())
            implied = divs[divs["source"] == "implied"]
            self.assertEqual(list(implied["ex"]), [d for d in true_dates if cutoff < d <= self.EXPIRIES[-1]])
            np.testing.assert_allclose(implied["amount"], 0.8, rtol=1e-6)             # the true dividends
            report = pd.read_csv(folder / "eq_vol_calibration.csv")
            self.assertAlmostEqual(float(report["div_repo_held"].iloc[0]), self.TRUE_REPO, delta=1e-6)
            repo = pd.read_csv(folder / "eq_repo.csv")
            np.testing.assert_allclose(repo["rate"], self.TRUE_REPO, atol=1e-6)      # flat at every expiry
            first = pd.read_csv(folder / "eq_dividends.csv")
            calibrate_snapshot_vols(folder)                                           # rerun: same dividends
            pd.testing.assert_frame_equal(pd.read_csv(folder / "eq_dividends.csv"), first)


if __name__ == "__main__":
    unittest.main()
