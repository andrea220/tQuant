"""Equity: read-only view of an underlying's market data (spot, repo, dividends, vol, discount curve)."""

import unittest
from datetime import date, timedelta

import numpy as np
import tensorflow as tf

import tensorquant as tq
from tensorquant.markethandles.dividendcurve import DividendCurve
from tensorquant.markethandles.equity import Equity
from tensorquant.markethandles.ircurve import RateCurve
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.utils import Currency
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

EVAL_DATE = date(2026, 5, 15)
PILLARS = [0.25, 0.5, 1.0, 2.0, 5.0]
RATES = [0.026, 0.027, 0.028, 0.029, 0.032]
S0, REPO, DIV_YIELD = 100.0, -0.005, 0.015
EX_DATES = [date(2026, 9, 15), date(2027, 3, 15), date(2027, 9, 15)]
AMOUNTS = [1.20, 1.25, 1.30]
EXPIRY = date(2027, 5, 15)  # 365 days: two dividends before expiry


def yf(d: date) -> float:
    return (d - EVAL_DATE).days / 365.0


def ref_df(t: float) -> float:
    return float(np.exp(-np.interp(t, PILLARS, RATES) * t))


def ref_pv(T: date) -> float:
    return sum(a * ref_df(yf(d)) for d, a in zip(EX_DATES, AMOUNTS) if EVAL_DATE < d <= T)


def make_market() -> MarketEnvironment:
    surface = VolatilitySurface(
        reference_date=EVAL_DATE,
        calendar=None,
        daycounter=DayCounter(DayCounterConvention.Actual365),
        strike=[80.0, 100.0, 120.0],
        maturity=[0.5, 1.0, 2.0],
        volatility_matrix=[[0.26, 0.22, 0.21], [0.25, 0.21, 0.20], [0.24, 0.20, 0.19]],
    )
    return MarketEnvironment(market={
        "IR:EUR:ESTR:SPOT": RateCurve(EVAL_DATE, PILLARS, RATES, "LINEAR", DayCounterConvention.Actual365),
        # DISC: discrete dividends (DIVYIELD present but ignored)
        "EQ:EUR:DISC:SPOT": S0,
        "EQ:EUR:DISC:REPO": REPO,
        "EQ:EUR:DISC:DIVYIELD": DIV_YIELD,
        "EQ:EUR:DISC:DIV": DividendCurve(
            reference_date=EVAL_DATE, ex_dates=EX_DATES, amounts=AMOUNTS, currency=Currency.EUR,
            daycounter_convention=DayCounterConvention.Actual365,
        ),
        "EQ:EUR:DISC:VOL": surface,
        # CONT: no discrete dividends -> continuous yield
        "EQ:EUR:CONT:SPOT": S0,
        "EQ:EUR:CONT:REPO": REPO,
        "EQ:EUR:CONT:DIVYIELD": DIV_YIELD,
        # BARE: spot only -> no dividends, no repo, no vol
        "EQ:EUR:BARE:SPOT": S0,
    })


class TestGetEquity(unittest.TestCase):
    def setUp(self):
        self.mkt = make_market()
        self.eq = self.mkt.get_equity("DISC", ccy=Currency.EUR)

    def test_is_view_on_market_objects(self):
        self.assertIsInstance(self.eq, Equity)
        self.assertEqual(self.eq.ticker, "DISC")
        self.assertEqual(self.eq.currency, Currency.EUR)
        self.assertIs(self.eq.spot, self.mkt.get_eq_spot("DISC", ccy=Currency.EUR))
        self.assertIs(self.eq.repo, self.mkt.get_eq_repo("DISC", ccy=Currency.EUR))
        self.assertIs(self.eq.dividends, self.mkt.get_eq_dividends("DISC", ccy=Currency.EUR))
        self.assertIs(self.eq.vol_surface, self.mkt.get_eq_vol_surface("DISC", ccy=Currency.EUR))
        self.assertIs(self.eq.discount_curve, self.mkt.get_ir_curve(Currency.EUR))

    def test_currency_resolved_when_omitted(self):
        self.assertEqual(self.mkt.get_equity("DISC").currency, Currency.EUR)

    def test_bump_in_market_is_seen(self):
        self.mkt.get_eq_spot("DISC", ccy=Currency.EUR).assign(110.0)
        self.assertEqual(float(self.eq.spot), 110.0)

    def test_unknown_ticker(self):
        with self.assertRaises(ValueError):
            self.mkt.get_equity("NOPE")


class TestDividendModel(unittest.TestCase):
    """Option (a): DIVYIELD is used only when there is no discrete dividend schedule."""

    def setUp(self):
        self.mkt = make_market()

    def test_discrete_when_div_present(self):
        eq = self.mkt.get_equity("DISC")
        self.assertEqual(eq.dividend_model, "discrete")
        self.assertIsNone(eq.div_yield)

    def test_continuous_when_only_divyield(self):
        eq = self.mkt.get_equity("CONT")
        self.assertEqual(eq.dividend_model, "continuous")
        self.assertIsNone(eq.dividends)
        self.assertIs(eq.div_yield, self.mkt.get_eq_div_yield("CONT"))

    def test_bare(self):
        eq = self.mkt.get_equity("BARE")
        self.assertEqual(eq.dividend_model, "none")
        self.assertIsNone(eq.vol_surface)
        self.assertEqual(float(eq.repo), 0.0)
        self.assertEqual(float(eq.pv_dividends(EXPIRY)), 0.0)


class TestForward(unittest.TestCase):
    def setUp(self):
        self.mkt = make_market()

    def test_pv_dividends(self):
        eq = self.mkt.get_equity("DISC")
        pv = eq.pv_dividends(EXPIRY)
        self.assertEqual(pv.dtype, tf.float64)
        self.assertAlmostEqual(float(pv), ref_pv(EXPIRY), places=12)
        self.assertEqual(float(self.mkt.get_equity("CONT").pv_dividends(EXPIRY)), 0.0)

    def test_forward_discrete(self):
        t = yf(EXPIRY)
        expected = (S0 - ref_pv(EXPIRY)) * np.exp(-REPO * t) / ref_df(t)
        f = self.mkt.get_equity("DISC").forward(EXPIRY)
        self.assertEqual(f.dtype, tf.float64)
        self.assertAlmostEqual(float(f), expected, places=10)

    def test_forward_continuous(self):
        t = yf(EXPIRY)
        expected = S0 * np.exp(-(DIV_YIELD + REPO) * t) / ref_df(t)
        self.assertAlmostEqual(float(self.mkt.get_equity("CONT").forward(EXPIRY)), expected, places=10)

    def test_forward_bare(self):
        t = yf(EXPIRY)
        self.assertAlmostEqual(float(self.mkt.get_equity("BARE").forward(EXPIRY)), S0 / ref_df(t), places=10)

    def test_forward_at_evaluation_date_is_spot(self):
        self.assertAlmostEqual(float(self.mkt.get_equity("DISC").forward(EVAL_DATE)), S0, places=12)

    def test_date_and_year_fraction_agree(self):
        eq = self.mkt.get_equity("DISC")
        self.assertAlmostEqual(float(eq.forward(EXPIRY)), float(eq.forward(yf(EXPIRY))), places=12)

    def test_matches_black_scholes_pricer_forward(self):
        tq.Settings.evaluation_date = EVAL_DATE
        for ticker in ("DISC",):
            opt = tq.VanillaOption(Currency.EUR, start_date=EVAL_DATE, end_date=EXPIRY,
                                   option_type=tq.OptionType.Call, strike=100.0, underlying=ticker,
                                   exercise_type=tq.ExerciseType.European)
            for model in ("discrete", "continuous"):
                with self.subTest(model=model):
                    tq.BlackScholesPricer(dividend_model=model).price(opt, self.mkt)
                    f = float(self.mkt.get_equity(ticker).forward(EXPIRY))
                    self.assertAlmostEqual(float(opt.forward) / f, 1.0, places=5)  # pricer is float32

    def test_implied_div_yield(self):
        eq = self.mkt.get_equity("DISC")
        t = yf(EXPIRY)
        q = float(eq.implied_div_yield(EXPIRY))
        self.assertAlmostEqual(S0 * np.exp(-q * t), S0 - ref_pv(EXPIRY), places=10)
        self.assertAlmostEqual(float(self.mkt.get_equity("CONT").implied_div_yield(EXPIRY)), DIV_YIELD, places=15)
        with self.assertRaises(ValueError):
            eq.implied_div_yield(EVAL_DATE)


class TestForwardGradients(unittest.TestCase):
    def setUp(self):
        self.mkt = make_market()
        self.t = yf(EXPIRY)

    def test_discrete_spot_and_repo(self):
        eq = self.mkt.get_equity("DISC")
        with tf.GradientTape() as tape:
            f = eq.forward(EXPIRY)
        d_spot, d_repo = tape.gradient(f, [eq.spot, eq.repo])
        # cash dividends do not move with S
        self.assertAlmostEqual(float(d_spot), np.exp(-REPO * self.t) / ref_df(self.t), places=10)
        self.assertAlmostEqual(float(d_repo), -self.t * float(f), places=8)

    def test_continuous_spot(self):
        eq = self.mkt.get_equity("CONT")
        with tf.GradientTape() as tape:
            f = eq.forward(EXPIRY)
        self.assertAlmostEqual(float(tape.gradient(f, eq.spot)), float(f) / S0, places=10)

    def test_curve_pillars_vs_finite_difference(self):
        eq = self.mkt.get_equity("DISC")
        variables = eq.discount_curve.variables
        with tf.GradientTape() as tape:
            f = eq.forward(EXPIRY)
        grads = [0.0 if g is None else float(g) for g in tape.gradient(f, variables)]
        h = 1e-6
        for i, var in enumerate(variables):
            base = float(var)
            var.assign(base + h)
            up = float(eq.forward(EXPIRY))
            var.assign(base - h)
            dn = float(eq.forward(EXPIRY))
            var.assign(base)
            self.assertAlmostEqual(grads[i], (up - dn) / (2 * h), places=5)

    def test_forward_gradient_wrt_dividends(self):
        """dF/dD_i = -P(0, ex_i) e^{-repo T} / P(0, T) for ex_i <= T, 0 after."""
        eq = self.mkt.get_equity("DISC")
        (var,) = eq.dividends.variables
        with tf.GradientTape() as tape:
            F = eq.forward(EXPIRY)
        grad = tape.gradient(F, var).numpy()
        T = yf(EXPIRY)
        for g, d in zip(grad, EX_DATES):
            expected = -ref_df(yf(d)) * np.exp(-REPO * T) / ref_df(T) if d <= EXPIRY else 0.0
            self.assertAlmostEqual(g, expected, places=12)

    def test_risk_factors(self):
        eq = self.mkt.get_equity("DISC")
        rf = eq.risk_factors()
        self.assertEqual(sorted(rf), ["EQ:EUR:DISC:DIV", "EQ:EUR:DISC:REPO", "EQ:EUR:DISC:SPOT", "EQ:EUR:DISC:VOL",
                                      "IR:EUR:ESTR:SPOT"])
        self.assertIs(rf["EQ:EUR:DISC:SPOT"][0], eq.spot)
        self.assertIs(rf["EQ:EUR:DISC:DIV"][0], eq.dividends.variables[0])
        cont = self.mkt.get_equity("CONT").risk_factors()
        self.assertIn("EQ:EUR:CONT:DIVYIELD", cont)
        self.assertNotIn("EQ:EUR:BARE:REPO", self.mkt.get_equity("BARE").risk_factors())


class TestSummary(unittest.TestCase):
    def test_summary(self):
        eq = make_market().get_equity("DISC")
        s = eq.summary(EXPIRY)
        for field in ("ticker", "currency", "spot", "repo", "dividend model", "PV dividends",
                      "forward", "ATM-forward vol", "discount curve"):
            self.assertIn(field, s.index)
        self.assertAlmostEqual(s.loc["forward", "value"], float(eq.forward(EXPIRY)), places=12)
        self.assertEqual(s.loc["ticker", "value"], "DISC")


if __name__ == "__main__":
    unittest.main()
