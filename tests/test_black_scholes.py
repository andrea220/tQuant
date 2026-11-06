"""BlackScholesPricer (European BS + American Bjerksund-Stensland 1993).

PRICES / GREEKS were pinned from the float32 implementation before the refactor to market risk
factors (float64): tolerances cover the float32 round-off of the pinned values.
"""

import unittest
from datetime import date
from math import erf, exp, log, sqrt

import tensorflow as tf

import tensorquant as tq
from tensorquant.markethandles.dividendcurve import DividendCurve
from tensorquant.markethandles.ircurve import RateCurve
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.utils import Currency
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention
from markers import slow

EVAL_DATE = date(2026, 5, 15)
EXPIRIES = {"1Y": date(2027, 5, 15), "6M": date(2026, 11, 15)}
PILLARS = [0.25, 0.5, 1.0, 2.0, 5.0]
RATES = [0.026, 0.027, 0.028, 0.029, 0.032]
S0, REPO, DIV_YIELD = 100.0, -0.005, 0.015
EX_DATES = [date(2026, 9, 15), date(2027, 3, 15), date(2027, 9, 15)]
AMOUNTS = [1.20, 1.25, 1.30]
VOLS = [[0.26, 0.22, 0.21], [0.25, 0.21, 0.20], [0.24, 0.20, 0.19]]


def make_surface() -> VolatilitySurface:
    return VolatilitySurface(
        reference_date=EVAL_DATE, calendar=None, daycounter=DayCounter(DayCounterConvention.Actual365),
        strike=[80.0, 100.0, 120.0], maturity=[0.5, 1.0, 2.0], volatility_matrix=VOLS,
    )


def make_market() -> MarketEnvironment:
    """DISC: discrete dividends (DIVYIELD ignored). CONT: continuous DIVYIELD only. NOVOL: no surface."""
    return MarketEnvironment(market={
        "IR:EUR:ESTR:SPOT": RateCurve(EVAL_DATE, PILLARS, RATES, "LINEAR", DayCounterConvention.Actual365),
        "EQ:EUR:DISC:SPOT": S0,
        "EQ:EUR:DISC:REPO": REPO,
        "EQ:EUR:DISC:DIVYIELD": DIV_YIELD,
        "EQ:EUR:DISC:DIV": DividendCurve(
            reference_date=EVAL_DATE, ex_dates=EX_DATES, amounts=AMOUNTS, currency=Currency.EUR,
            daycounter_convention=DayCounterConvention.Actual365,
        ),
        "EQ:EUR:DISC:VOL": make_surface(),
        "EQ:EUR:CONT:SPOT": S0,
        "EQ:EUR:CONT:REPO": REPO,
        "EQ:EUR:CONT:DIVYIELD": DIV_YIELD,
        "EQ:EUR:CONT:VOL": make_surface(),
        "EQ:EUR:NOVOL:SPOT": S0,
    })


def make_option(expiry: date, option_type: str, strike: float, exercise: str = "European",
                underlying: str = "DISC") -> tq.VanillaOption:
    return tq.VanillaOption(
        Currency.EUR, start_date=EVAL_DATE, end_date=expiry,
        option_type=getattr(tq.OptionType, option_type), strike=strike, underlying=underlying,
        exercise_type=getattr(tq.ExerciseType, exercise),
    )


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black(F: float, K: float, sigma: float, T: float, DF: float, call: bool) -> float:
    d1 = (log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if call:
        return DF * (F * norm_cdf(d1) - K * norm_cdf(d2))
    return DF * (K * norm_cdf(-d2) - F * norm_cdf(-d1))


PRICES = {
    ('1Y', 'discrete', 'European', 'Call', 80.0): 22.36396026611328,
    ('1Y', 'discrete', 'European', 'Call', 95.0): 11.49752426147461,
    ('1Y', 'discrete', 'European', 'Call', 100.0): 8.593135833740234,
    ('1Y', 'discrete', 'European', 'Call', 110.0): 4.662223815917969,
    ('1Y', 'discrete', 'European', 'Call', 125.0): 1.5758552551269531,
    ('1Y', 'discrete', 'European', 'Put', 80.0): 2.0767898559570312,
    ('1Y', 'discrete', 'European', 'Put', 95.0): 5.796176910400391,
    ('1Y', 'discrete', 'European', 'Put', 100.0): 7.753730773925781,
    ('1Y', 'discrete', 'European', 'Put', 110.0): 13.546703338623047,
    ('1Y', 'discrete', 'European', 'Put', 125.0): 25.046157836914062,
    ('1Y', 'discrete', 'American', 'Call', 80.0): 22.36396026611328,
    ('1Y', 'discrete', 'American', 'Call', 95.0): 11.49752426147461,
    ('1Y', 'discrete', 'American', 'Call', 100.0): 8.593135833740234,
    ('1Y', 'discrete', 'American', 'Call', 110.0): 4.662223815917969,
    ('1Y', 'discrete', 'American', 'Call', 125.0): 1.5758552551269531,
    ('1Y', 'discrete', 'American', 'Put', 80.0): 2.0794830322265625,
    ('1Y', 'discrete', 'American', 'Put', 95.0): 5.838886260986328,
    ('1Y', 'discrete', 'American', 'Put', 100.0): 7.852787017822266,
    ('1Y', 'discrete', 'American', 'Put', 110.0): 13.913063049316406,
    ('1Y', 'discrete', 'American', 'Put', 125.0): 26.696258544921875,
    ('1Y', 'continuous', 'European', 'Call', 80.0): 22.363971710205078,
    ('1Y', 'continuous', 'European', 'Call', 95.0): 11.49752426147461,
    ('1Y', 'continuous', 'European', 'Call', 100.0): 8.593135833740234,
    ('1Y', 'continuous', 'European', 'Call', 110.0): 4.662227630615234,
    ('1Y', 'continuous', 'European', 'Call', 125.0): 1.575852394104004,
    ('1Y', 'continuous', 'European', 'Put', 80.0): 2.0767927169799805,
    ('1Y', 'continuous', 'European', 'Put', 95.0): 5.796173095703125,
    ('1Y', 'continuous', 'European', 'Put', 100.0): 7.753730773925781,
    ('1Y', 'continuous', 'European', 'Put', 110.0): 13.546703338623047,
    ('1Y', 'continuous', 'European', 'Put', 125.0): 25.046157836914062,
    ('1Y', 'continuous', 'American', 'Call', 80.0): 22.36404800415039,
    ('1Y', 'continuous', 'American', 'Call', 95.0): 11.497550964355469,
    ('1Y', 'continuous', 'American', 'Call', 100.0): 8.593147277832031,
    ('1Y', 'continuous', 'American', 'Call', 110.0): 4.662208557128906,
    ('1Y', 'continuous', 'American', 'Call', 125.0): 1.5758590698242188,
    ('1Y', 'continuous', 'American', 'Put', 80.0): 2.076995849609375,
    ('1Y', 'continuous', 'American', 'Put', 95.0): 5.799549102783203,
    ('1Y', 'continuous', 'American', 'Put', 100.0): 7.762073516845703,
    ('1Y', 'continuous', 'American', 'Put', 110.0): 13.584186553955078,
    ('1Y', 'continuous', 'American', 'Put', 125.0): 25.269269943237305,
    ('6M', 'discrete', 'European', 'Call', 80.0): 20.99669647216797,
    ('6M', 'discrete', 'European', 'Call', 95.0): 9.30013656616211,
    ('6M', 'discrete', 'European', 'Call', 100.0): 6.359889984130859,
    ('6M', 'discrete', 'European', 'Call', 110.0): 2.6761722564697266,
    ('6M', 'discrete', 'European', 'Call', 125.0): 0.5166506767272949,
    ('6M', 'discrete', 'European', 'Put', 80.0): 0.8548803329467773,
    ('6M', 'discrete', 'European', 'Put', 95.0): 3.955474853515625,
    ('6M', 'discrete', 'European', 'Put', 100.0): 5.947620391845703,
    ('6M', 'discrete', 'European', 'Put', 110.0): 12.128669738769531,
    ('6M', 'discrete', 'European', 'Put', 125.0): 24.766311645507812,
    ('6M', 'discrete', 'American', 'Call', 80.0): 20.99669647216797,
    ('6M', 'discrete', 'American', 'Call', 95.0): 9.30013656616211,
    ('6M', 'discrete', 'American', 'Call', 100.0): 6.359889984130859,
    ('6M', 'discrete', 'American', 'Call', 110.0): 2.6761722564697266,
    ('6M', 'discrete', 'American', 'Call', 125.0): 0.5166506767272949,
    ('6M', 'discrete', 'American', 'Put', 80.0): 0.8549270629882812,
    ('6M', 'discrete', 'American', 'Put', 95.0): 3.959644317626953,
    ('6M', 'discrete', 'American', 'Put', 100.0): 5.9621734619140625,
    ('6M', 'discrete', 'American', 'Put', 110.0): 12.226358413696289,
    ('6M', 'discrete', 'American', 'Put', 125.0): 25.55814552307129,
    ('6M', 'continuous', 'European', 'Call', 80.0): 20.996688842773438,
    ('6M', 'continuous', 'European', 'Call', 95.0): 9.300132751464844,
    ('6M', 'continuous', 'European', 'Call', 100.0): 6.359889984130859,
    ('6M', 'continuous', 'European', 'Call', 110.0): 2.6761722564697266,
    ('6M', 'continuous', 'European', 'Call', 125.0): 0.5166592597961426,
    ('6M', 'continuous', 'European', 'Put', 80.0): 0.8548765182495117,
    ('6M', 'continuous', 'European', 'Put', 95.0): 3.955474853515625,
    ('6M', 'continuous', 'European', 'Put', 100.0): 5.947620391845703,
    ('6M', 'continuous', 'European', 'Put', 110.0): 12.128669738769531,
    ('6M', 'continuous', 'European', 'Put', 125.0): 24.766311645507812,
    ('6M', 'continuous', 'American', 'Call', 80.0): 20.996702194213867,
    ('6M', 'continuous', 'American', 'Call', 95.0): 9.30013656616211,
    ('6M', 'continuous', 'American', 'Call', 100.0): 6.359889984130859,
    ('6M', 'continuous', 'American', 'Call', 110.0): 2.6761703491210938,
    ('6M', 'continuous', 'American', 'Call', 125.0): 0.51666259765625,
    ('6M', 'continuous', 'American', 'Put', 80.0): 0.854888916015625,
    ('6M', 'continuous', 'American', 'Put', 95.0): 3.955677032470703,
    ('6M', 'continuous', 'American', 'Put', 100.0): 5.948421478271484,
    ('6M', 'continuous', 'American', 'Put', 110.0): 12.13587760925293,
    ('6M', 'continuous', 'American', 'Put', 125.0): 24.85438346862793,
}
GREEKS = {  # (delta, vega = sum over surface nodes, dPrice/dT)
    ('1Y', 'discrete', 'European', 'Call'): (0.5608089566230774, 38.7131462097168, 5.6303229331970215),
    ('1Y', 'discrete', 'European', 'Put'): (-0.44420358538627625, 38.7131462097168, 2.417246103286743),
    ('1Y', 'discrete', 'American', 'Call'): (0.5608089566230774, 38.7131462097168, 5.6303229331970215),
    ('1Y', 'discrete', 'American', 'Put'): (-0.4570416212081909, 37.32555389404297, 2.6102218627929688),
    ('1Y', 'continuous', 'European', 'Call'): (0.5472882986068726, 38.7131462097168, 4.294689655303955),
    ('1Y', 'continuous', 'European', 'Put'): (-0.4334942102432251, 38.7131462097168, 3.4751698970794678),
    ('1Y', 'continuous', 'American', 'Call'): (0.5472884774208069, 38.71328353881836, 4.294700622558594),
    ('1Y', 'continuous', 'American', 'Put'): (-0.4347325563430786, 38.62875747680664, 3.499122381210327),
    ('6M', 'discrete', 'European', 'Call'): (0.5430927872657776, 27.905302047729492, 7.632745742797852),
    ('6M', 'discrete', 'European', 'Put'): (-0.459430992603302, 27.905302047729492, 4.473140716552734),
    ('6M', 'discrete', 'American', 'Call'): (0.5430927872657776, 27.905302047729492, 7.632745742797852),
    ('6M', 'discrete', 'American', 'Put'): (-0.4622631072998047, 27.609243392944336, 4.536881446838379),
    ('6M', 'continuous', 'European', 'Call'): (0.5366332530975342, 27.905302047729492, 6.359026908874512),
    ('6M', 'continuous', 'European', 'Put'): (-0.45396655797958374, 27.905302047729492, 5.550647258758545),
    ('6M', 'continuous', 'American', 'Call'): (0.5366333723068237, 27.905319213867188, 6.359025955200195),
    ('6M', 'continuous', 'American', 'Put'): (-0.45414477586746216, 27.891075134277344, 5.555542945861816),
}


class BSTestCase(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = make_market()

    def price(self, opt, model="discrete", autodiff=False):
        pricer = tq.BlackScholesPricer(dividend_model=model)
        pricer.price(opt, self.mkt, autodiff=autodiff)
        return pricer

    def bump_and_reprice(self, opt_factory, var, h, model="discrete"):
        base = float(var)
        out = []
        for x in (base + h, base - h):
            var.assign(x)
            opt = opt_factory()
            self.price(opt, model)
            out.append(float(opt.price))
        var.assign(base)
        return (out[0] - out[1]) / (2 * h)


class TestPinnedValues(BSTestCase):
    @slow
    def test_prices(self):
        for (ename, model, ex, ot, K), expected in PRICES.items():
            with self.subTest(expiry=ename, model=model, exercise=ex, type=ot, K=K):
                opt = make_option(EXPIRIES[ename], ot, K, ex)
                self.price(opt, model)
                self.assertAlmostEqual(float(opt.price), expected, delta=2e-5 + 2e-6 * abs(expected))

    @slow
    def test_greeks(self):
        surface = self.mkt.get_eq_vol_surface("DISC")
        for (ename, model, ex, ot), (delta, vega, theta) in GREEKS.items():
            with self.subTest(expiry=ename, model=model, exercise=ex, type=ot):
                opt = make_option(EXPIRIES[ename], ot, 100.0, ex)
                pricer = self.price(opt, model, autodiff=True)
                g_spot, g_vol, g_t = pricer.tape.gradient(
                    opt.price, [opt.spot, surface.volatility_matrix, opt.time_to_maturity])
                self.assertAlmostEqual(float(g_spot), delta, delta=2e-4)
                self.assertAlmostEqual(float(tf.reduce_sum(g_vol)), vega, delta=2e-3 + 1e-4 * abs(vega))
                self.assertAlmostEqual(float(g_t), theta, delta=2e-3 + 1e-4 * abs(theta))


class TestIndependentChecks(BSTestCase):
    def test_european_vs_closed_form(self):
        for ticker in ("DISC", "CONT"):
            eq = self.mkt.get_equity(ticker)
            for ename, expiry in EXPIRIES.items():
                T = (expiry - EVAL_DATE).days / 365
                F = float(eq.forward(expiry))
                DF = float(eq.discount_curve.discount(expiry))
                for K in (80.0, 100.0, 125.0):
                    sigma = float(eq.vol_surface.volatility(K, T))
                    for ot in ("Call", "Put"):
                        with self.subTest(ticker=ticker, expiry=ename, K=K, type=ot):
                            opt = make_option(expiry, ot, K, underlying=ticker)
                            self.price(opt)
                            self.assertAlmostEqual(float(opt.price), black(F, K, sigma, T, DF, ot == "Call"), places=10)

    def test_put_call_parity(self):
        eq = self.mkt.get_equity("DISC")
        expiry = EXPIRIES["1Y"]
        DF = float(eq.discount_curve.discount(expiry))
        for model in ("discrete", "continuous"):
            call, put = make_option(expiry, "Call", 105.0), make_option(expiry, "Put", 105.0)
            self.price(call, model)
            self.price(put, model)
            with self.subTest(model=model):
                self.assertAlmostEqual(float(call.price) - float(put.price),
                                       DF * (float(eq.forward(expiry)) - 105.0), places=10)

    def test_strike_is_float64(self):
        """103.37 is not representable in float32: parity holds to 1e-10 only with a float64 strike."""
        eq = self.mkt.get_equity("DISC")
        expiry = EXPIRIES["1Y"]
        K = 103.37
        call, put = make_option(expiry, "Call", K), make_option(expiry, "Put", K)
        self.assertEqual(call.strike.dtype, tf.float64)
        self.assertEqual(float(call.strike), K)
        self.price(call)
        self.price(put)
        DF = float(eq.discount_curve.discount(expiry))
        self.assertAlmostEqual(float(call.price) - float(put.price),
                               DF * (float(eq.forward(expiry)) - K), places=10)

    def test_forward_is_the_equity_forward(self):
        for ticker in ("DISC", "CONT"):
            opt = make_option(EXPIRIES["1Y"], "Call", 100.0, underlying=ticker)
            self.price(opt)
            self.assertAlmostEqual(float(opt.forward),
                                   float(self.mkt.get_equity(ticker).forward(EXPIRIES["1Y"])), places=10)

    def test_american_not_below_european(self):
        for model in ("discrete", "continuous"):
            for ot in ("Call", "Put"):
                with self.subTest(model=model, type=ot):
                    eu = make_option(EXPIRIES["1Y"], ot, 100.0)
                    am = make_option(EXPIRIES["1Y"], ot, 100.0, "American")
                    self.price(eu, model)
                    self.price(am, model)
                    self.assertGreaterEqual(float(am.price), float(eu.price) - 1e-12)


class TestRiskFactors(BSTestCase):
    def test_price_is_float64(self):
        for ex in ("European", "American"):
            opt = make_option(EXPIRIES["1Y"], "Put", 100.0, ex)
            self.price(opt)
            self.assertEqual(opt.price.dtype, tf.float64)

    def test_no_local_market_variables(self):
        pricer = self.price(make_option(EXPIRIES["1Y"], "Call", 100.0))
        for attr in ("_s", "_k", "_r", "_q", "_sigma", "_t", "_f"):
            self.assertFalse(hasattr(pricer, attr), attr)

    def test_spot_is_the_market_variable(self):
        opt = make_option(EXPIRIES["1Y"], "Call", 100.0)
        self.price(opt)
        self.assertIs(opt.spot, self.mkt.get_eq_spot("DISC"))

    def test_one_tape_reaches_all_market_factors(self):
        opt = make_option(EXPIRIES["1Y"], "Call", 100.0)
        pricer = self.price(opt, autodiff=True)
        grads = pricer.tape.gradient(opt.price, self.mkt.risk_factors())
        self.assertNotEqual(float(grads["EQ:EUR:DISC:SPOT"][0]), 0.0)
        self.assertNotEqual(float(grads["EQ:EUR:DISC:REPO"][0]), 0.0)
        self.assertNotEqual(float(tf.reduce_sum(grads["EQ:EUR:DISC:VOL"][0])), 0.0)
        self.assertTrue(any(g is not None and float(g) != 0.0 for g in grads["IR:EUR:ESTR:SPOT"]))
        self.assertIsNone(grads["EQ:EUR:CONT:SPOT"][0])

    def test_delta_vs_bump(self):
        for ticker in ("DISC", "CONT"):
            factory = lambda: make_option(EXPIRIES["1Y"], "Call", 100.0, underlying=ticker)
            opt = factory()
            pricer = self.price(opt, autodiff=True)
            spot = self.mkt.get_eq_spot(ticker)
            with self.subTest(ticker=ticker):
                self.assertAlmostEqual(float(pricer.tape.gradient(opt.price, spot)),
                                       self.bump_and_reprice(factory, spot, 1e-4), places=6)

    def test_rho_per_pillar_vs_bump(self):
        factory = lambda: make_option(EXPIRIES["1Y"], "Put", 100.0)
        opt = factory()
        pricer = self.price(opt, autodiff=True)
        curve = self.mkt.get_ir_curve(Currency.EUR)
        grads = pricer.tape.gradient(opt.price, curve.variables)
        for var, g in zip(curve.variables, grads):
            self.assertAlmostEqual(0.0 if g is None else float(g),
                                   self.bump_and_reprice(factory, var, 1e-6), places=4)

    def test_vega_per_node_vs_bump(self):
        factory = lambda: make_option(EXPIRIES["6M"], "Call", 95.0)
        opt = factory()
        pricer = self.price(opt, autodiff=True)
        matrix = self.mkt.get_eq_vol_surface("DISC").volatility_matrix
        grad = pricer.tape.gradient(opt.price, matrix).numpy()
        base = matrix.numpy().copy()
        h = 1e-6
        for i in range(base.shape[0]):
            for j in range(base.shape[1]):
                bumped = []
                for sign in (1, -1):
                    m = base.copy()
                    m[i, j] += sign * h
                    matrix.assign(m)
                    o = factory()
                    self.price(o)
                    bumped.append(float(o.price))
                matrix.assign(base)
                self.assertAlmostEqual(grad[i, j], (bumped[0] - bumped[1]) / (2 * h), places=5)


class TestDividendSensitivities(BSTestCase):
    def bump_dividend(self, factory, i, h, model):
        var = self.mkt.get_eq_dividends("DISC").variables[0]
        base = var.numpy().copy()
        out = []
        for sign in (1, -1):
            bumped = base.copy()
            bumped[i] += sign * h
            var.assign(bumped)
            opt = factory()
            self.price(opt, model)
            out.append(float(opt.price))
        var.assign(base)
        return (out[0] - out[1]) / (2 * h)

    def test_price_gradient_wrt_dividends_vs_bump(self):
        var = self.mkt.get_eq_dividends("DISC").variables[0]
        for model in ("discrete", "continuous"):
            for ot in ("Call", "Put"):
                factory = lambda ot=ot: make_option(EXPIRIES["1Y"], ot, 100.0)
                opt = factory()
                pricer = self.price(opt, model, autodiff=True)
                grad = pricer.tape.gradient(opt.price, var).numpy()
                for i in range(len(grad)):
                    with self.subTest(model=model, type=ot, dividend=i):
                        self.assertAlmostEqual(grad[i], self.bump_dividend(factory, i, 1e-5, model), places=6)

    def test_dividend_after_expiry_has_no_sensitivity(self):
        var = self.mkt.get_eq_dividends("DISC").variables[0]
        ex_dates = self.mkt.get_eq_dividends("DISC").ex_dates
        opt = make_option(EXPIRIES["1Y"], "Call", 100.0)
        pricer = self.price(opt, autodiff=True)
        grad = pricer.tape.gradient(opt.price, var).numpy()
        after = [i for i, d in enumerate(ex_dates) if d > EXPIRIES["1Y"]]
        self.assertTrue(after)
        for i in after:
            self.assertEqual(grad[i], 0.0)

    def test_discrete_dividend_gradient_is_minus_delta_times_df(self):
        """Discrete model: S* = S - PV, so dV/dD_i = -dV/dS * P(0, ex_i) for ex_i <= T."""
        eq = self.mkt.get_equity("DISC")
        opt = make_option(EXPIRIES["1Y"], "Call", 100.0)
        pricer = self.price(opt, autodiff=True)
        d_spot, d_div = pricer.tape.gradient(opt.price, [eq.spot, eq.dividends.variables[0]])
        for g, d in zip(d_div.numpy(), eq.dividends.ex_dates):
            if d <= EXPIRIES["1Y"]:
                self.assertAlmostEqual(g, -float(d_spot) * float(eq.discount_curve.discount(d)), places=12)


class TestEquityDividendModels(BSTestCase):
    def test_continuous_divyield_underlying(self):
        """CONT has no DIV: q = DIVYIELD + repo whatever the pricer's dividend_model."""
        eq = self.mkt.get_equity("CONT")
        expiry = EXPIRIES["1Y"]
        T = (expiry - EVAL_DATE).days / 365
        DF = float(eq.discount_curve.discount(expiry))
        F = S0 * exp(-(DIV_YIELD + REPO) * T) / DF
        sigma = float(eq.vol_surface.volatility(100.0, T))
        for model in ("discrete", "continuous"):
            with self.subTest(model=model):
                opt = make_option(expiry, "Call", 100.0, underlying="CONT")
                self.price(opt, model)
                self.assertAlmostEqual(float(opt.price), black(F, 100.0, sigma, T, DF, True), places=10)

    def test_continuous_model_delta_is_sticky_yield(self):
        """dividend_model='continuous' on discrete dividends: q_eff is held fixed when S moves."""
        eq = self.mkt.get_equity("DISC")
        expiry = EXPIRIES["1Y"]
        opt_d, opt_c = make_option(expiry, "Call", 100.0), make_option(expiry, "Call", 100.0)
        p_d, p_c = self.price(opt_d, "discrete", True), self.price(opt_c, "continuous", True)
        d_disc = float(p_d.tape.gradient(opt_d.price, opt_d.spot))
        d_cont = float(p_c.tape.gradient(opt_c.price, opt_c.spot))
        self.assertAlmostEqual(d_cont / d_disc, 1 - float(eq.pv_dividends(expiry)) / S0, places=10)

    def test_missing_vol_surface(self):
        with self.assertRaises(ValueError):
            self.price(make_option(EXPIRIES["1Y"], "Call", 100.0, underlying="NOVOL"))

    def test_invalid_dividend_model(self):
        with self.assertRaises(ValueError):
            tq.BlackScholesPricer(dividend_model="yield")


if __name__ == "__main__":
    unittest.main()
