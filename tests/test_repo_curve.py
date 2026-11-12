"""Repo term structure: ``EQ:<CCY>:<TICKER>:REPO`` as a :class:`RateCurve` of zero repo rates (or a scalar, = flat)."""

import unittest
from datetime import date

import numpy as np
import tensorflow as tf

import tensorquant as tq
from tensorquant.markethandles.dividendcurve import DividendCurve
from tensorquant.markethandles.ircurve import RateCurve
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.utils import Currency
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

EVAL_DATE = date(2026, 5, 15)
A365 = DayCounterConvention.Actual365
PILLARS = [0.25, 0.5, 1.0, 2.0, 5.0]
RATES = [0.026, 0.027, 0.028, 0.029, 0.032]
REPO_DATES = [date(2026, 11, 13), date(2027, 5, 15), date(2028, 5, 14), date(2031, 5, 15)]
REPO_PILLARS = [(d - EVAL_DATE).days / 365.0 for d in REPO_DATES]
REPO_RATES = [0.004, 0.006, 0.009, 0.012]          # zero repo rates, continuously compounded
S0 = 100.0
EX_DATES = [date(2026, 9, 15), date(2027, 3, 15), date(2027, 9, 15)]
AMOUNTS = [1.20, 1.25, 1.30]
EXPIRY = date(2027, 11, 15)                       # 1.5 years: between two repo pillars


def yf(d: date) -> float:
    return (d - EVAL_DATE).days / 365.0


def ref_repo(t: float) -> float:
    """Linear zero repo rate, flat outside the pillars."""
    return float(np.interp(t, REPO_PILLARS, REPO_RATES))


def surface() -> VolatilitySurface:
    return VolatilitySurface(
        reference_date=EVAL_DATE, calendar=None, daycounter=DayCounter(A365),
        strike=[80.0, 100.0, 120.0], maturity=[0.5, 1.0, 2.0],
        volatility_matrix=[[0.26, 0.22, 0.21], [0.25, 0.21, 0.20], [0.24, 0.20, 0.19]])


def make_market(repo) -> MarketEnvironment:
    """Market with one equity ``X`` (discrete dividends) and the given repo (curve or scalar)."""
    return MarketEnvironment(market={
        "IR:EUR:ESTR:SPOT": RateCurve(EVAL_DATE, PILLARS, RATES, "LINEAR", A365),
        "EQ:EUR:X:SPOT": S0,
        "EQ:EUR:X:REPO": repo,
        "EQ:EUR:X:DIV": DividendCurve(reference_date=EVAL_DATE, ex_dates=EX_DATES, amounts=AMOUNTS,
                                      currency=Currency.EUR, daycounter_convention=A365),
        "EQ:EUR:X:VOL": surface(),
    })


def repo_curve(rates=REPO_RATES) -> RateCurve:
    return RateCurve(EVAL_DATE, REPO_DATES, list(rates), "LINEAR", A365)


def option(option_type: str = "Call", strike: float = 100.0, exercise: str = "European") -> tq.VanillaOption:
    return tq.VanillaOption(Currency.EUR, start_date=EVAL_DATE, end_date=EXPIRY,
                            option_type=getattr(tq.OptionType, option_type), strike=strike, underlying="X",
                            exercise_type=getattr(tq.ExerciseType, exercise))


def bs_price(mkt: MarketEnvironment, opt: tq.VanillaOption) -> float:
    pricer = tq.BlackScholesPricer()
    pricer.price(opt, mkt, autodiff=False)
    return float(opt.price)


class TestRepoCurve(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = make_market(repo_curve())
        self.eq = self.mkt.get_equity("X")
        self.t = yf(EXPIRY)

    def test_market_accepts_curve_and_risk_factors(self):
        curve = self.mkt.get_eq_repo("X", ccy=Currency.EUR)
        self.assertIsInstance(curve, RateCurve)
        self.assertIs(self.eq.repo, curve)
        self.assertEqual(self.mkt.risk_factors()["EQ:EUR:X:REPO"], curve.variables)
        self.assertEqual(self.eq.risk_factors()["EQ:EUR:X:REPO"], curve.variables)
        with self.assertRaises(TypeError):
            make_market("not a repo")

    def test_repo_rate_is_zero_rate_to_maturity(self):
        for t in (0.3, 0.75, self.t, 3.0, 7.0):
            self.assertAlmostEqual(float(self.eq.repo_rate(t)), ref_repo(t), places=12)
        self.assertAlmostEqual(float(self.eq.repo_rate(EXPIRY)), ref_repo(self.t), places=12)

    def test_forward_uses_repo_to_maturity(self):
        df = float(self.eq.discount_curve.discount(EXPIRY))
        expected = (S0 - float(self.eq.pv_dividends(EXPIRY))) * np.exp(-ref_repo(self.t) * self.t) / df
        self.assertAlmostEqual(float(self.eq.forward(EXPIRY)), expected, places=10)

    def test_flat_curve_equals_scalar(self):
        flat = make_market(repo_curve([0.007] * 4))
        scalar = make_market(0.007)
        self.assertAlmostEqual(float(flat.get_equity("X").forward(EXPIRY)),
                               float(scalar.get_equity("X").forward(EXPIRY)), places=12)
        for exercise in ("European", "American"):
            self.assertAlmostEqual(bs_price(flat, option("Put", 95.0, exercise)),
                                   bs_price(scalar, option("Put", 95.0, exercise)), places=12)

    def test_black_scholes_uses_repo_to_maturity(self):
        """Pricing with the curve = pricing with the scalar repo ``z(T)`` of the option maturity."""
        scalar = make_market(ref_repo(self.t))
        for opt_type, k in (("Call", 105.0), ("Put", 90.0)):
            self.assertAlmostEqual(bs_price(self.mkt, option(opt_type, k)), bs_price(scalar, option(opt_type, k)),
                                   places=12)

    def test_gbm_mc_uses_repo_to_maturity(self):
        scalar = make_market(ref_repo(self.t))
        prices = []
        for mkt in (self.mkt, scalar):
            opt = option("Call", 100.0)
            tq.VanillaMCPricer(n_paths=2000, n_steps=4, seed=7).price(opt, mkt, autodiff=False)
            prices.append(float(opt.price))
        self.assertAlmostEqual(prices[0], prices[1], places=10)

    def test_forward_gradient_per_pillar_vs_finite_differences(self):
        curve = self.eq.repo
        with tf.GradientTape() as tape:
            f = self.eq.forward(EXPIRY)
        grads = tape.gradient(f, curve.variables)
        h = 1e-6
        for i, var in enumerate(curve.variables):
            base = float(var.numpy())
            var.assign(base + h)
            up = float(self.eq.forward(EXPIRY))
            var.assign(base - h)
            down = float(self.eq.forward(EXPIRY))
            var.assign(base)
            fd = (up - down) / (2 * h)
            g = 0.0 if grads[i] is None else float(grads[i])
            self.assertAlmostEqual(g, fd, delta=1e-6 * max(1.0, abs(fd)))
        # only the two pillars around 1.5y move the forward
        self.assertEqual([g is not None and abs(float(g)) > 0 for g in grads], [False, True, True, False])

    def test_black_scholes_repo_vega_vs_bumps(self):
        curve = self.eq.repo
        opt = option("Put", 95.0)
        pricer = tq.BlackScholesPricer()
        pricer.price(opt, self.mkt, autodiff=True)
        grads = pricer.tape.gradient(opt.price, curve.variables)
        h = 1e-6
        for i, var in enumerate(curve.variables):
            base = float(var.numpy())
            var.assign(base + h)
            up = bs_price(self.mkt, option("Put", 95.0))
            var.assign(base - h)
            down = bs_price(self.mkt, option("Put", 95.0))
            var.assign(base)
            g = 0.0 if grads[i] is None else float(grads[i])
            self.assertAlmostEqual(g, (up - down) / (2 * h), delta=1e-5)

    def test_frames_round_trip(self):
        frames = self.mkt.to_frames()
        self.assertIn("eq_repo.csv", frames)
        repo = frames["eq_repo.csv"]
        self.assertEqual(list(repo.columns), ["ticker", "ccy", "pillar", "rate"])
        self.assertEqual(len(repo), len(REPO_PILLARS))
        self.assertTrue(frames["eq_spot.csv"]["repo"].isna().all())
        back = MarketEnvironment.from_frames(frames, EVAL_DATE)
        eq = back.get_equity("X")
        self.assertIsInstance(eq.repo, RateCurve)
        np.testing.assert_allclose([float(v) for v in eq.repo.variables], REPO_RATES, rtol=0, atol=1e-15)
        for t in (0.3, self.t, 7.0):        # (the forward also moves with the float IR pillars rounded to days)
            self.assertAlmostEqual(float(eq.repo_rate(t)), float(self.eq.repo_rate(t)), places=14)


if __name__ == "__main__":
    unittest.main()
