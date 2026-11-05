"""VanillaMCPricer: GBM and local vol built from the market (Equity view), float64, AAD greeks on market factors."""

import unittest
from datetime import date

import numpy as np
import tensorflow as tf

import tensorquant as tq
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

from test_black_scholes import EVAL_DATE, EXPIRIES, PILLARS, RATES, S0, make_market, make_option

GBM_PATHS = 200_000


def mc(model=tq.GeometricBrownianMotion, n_paths=GBM_PATHS, n_steps=1, seed=42):
    return tq.VanillaMCPricer(model=model, n_paths=n_paths, n_steps=n_steps, seed=seed)


def flat_vol_market(vol: float = 0.2) -> MarketEnvironment:
    strikes = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 150.0, 170.0, 200.0]
    maturities = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    surface = VolatilitySurface(
        reference_date=EVAL_DATE, calendar=None, daycounter=DayCounter(DayCounterConvention.Actual365),
        strike=strikes, maturity=maturities, volatility_matrix=[[vol] * len(strikes) for _ in maturities],
    )
    return MarketEnvironment(market={
        "IR:EUR:ESTR:SPOT": tq.RateCurve(EVAL_DATE, PILLARS, RATES, "LINEAR", DayCounterConvention.Actual365),
        "EQ:EUR:FLAT:SPOT": S0,
        "EQ:EUR:FLAT:REPO": 0.01,
        "EQ:EUR:FLAT:VOL": surface,
    })


class MCTestCase(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = make_market()

    def bs(self, opt, autodiff=False):
        pricer = tq.BlackScholesPricer(dividend_model="discrete")
        pricer.price(opt, self.mkt, autodiff=autodiff)
        return pricer


class TestGBM(MCTestCase):
    def test_price_vs_black_scholes(self):
        """Escrowed GBM with the surface vol at (K, T): same model as BS discrete, within 3 standard errors."""
        for ticker in ("DISC", "CONT"):
            for ot in ("Call", "Put"):
                for K in (90.0, 100.0, 115.0):
                    opt, ref = make_option(EXPIRIES["1Y"], ot, K, underlying=ticker), \
                        make_option(EXPIRIES["1Y"], ot, K, underlying=ticker)
                    mc().price(opt, self.mkt)
                    self.bs(ref)
                    with self.subTest(ticker=ticker, type=ot, K=K):
                        self.assertLess(abs(float(opt.price) - float(ref.price)), 3 * float(opt.standard_error))

    def test_float64_and_maturity(self):
        opt, ref = make_option(EXPIRIES["6M"], "Call", 100.0), make_option(EXPIRIES["6M"], "Call", 100.0)
        mc().price(opt, self.mkt)
        self.bs(ref)
        self.assertEqual(opt.price.dtype, tf.float64)
        self.assertEqual(float(opt.time_to_maturity), float(ref.time_to_maturity))

    def test_seed_reproducible(self):
        prices = []
        for seed in (7, 7, 8):
            opt = make_option(EXPIRIES["1Y"], "Call", 100.0)
            mc(n_paths=10_000, seed=seed).price(opt, self.mkt)
            prices.append(float(opt.price))
        self.assertEqual(prices[0], prices[1])
        self.assertNotEqual(prices[0], prices[2])

    def test_steps_do_not_change_gbm(self):
        """GBM is simulated exactly: the terminal law does not depend on n_steps (up to MC noise)."""
        one, many = make_option(EXPIRIES["1Y"], "Call", 100.0), make_option(EXPIRIES["1Y"], "Call", 100.0)
        mc(n_steps=1).price(one, self.mkt)
        mc(n_steps=12).price(many, self.mkt)
        self.assertLess(abs(float(one.price) - float(many.price)), 4 * float(one.standard_error))

    def test_aad_greeks_reach_market_factors(self):
        opt = make_option(EXPIRIES["1Y"], "Call", 100.0)
        pricer = mc()
        pricer.price(opt, self.mkt, autodiff=True)
        grads = pricer.tape.gradient(opt.price, self.mkt.get_equity("DISC").risk_factors())
        for key in ("EQ:EUR:DISC:SPOT", "EQ:EUR:DISC:REPO", "EQ:EUR:DISC:VOL", "EQ:EUR:DISC:DIV", "IR:EUR:ESTR:SPOT"):
            with self.subTest(factor=key):
                self.assertTrue(any(g is not None and float(tf.reduce_sum(tf.abs(g))) > 0 for g in grads[key]))

    def test_delta_vega_vs_black_scholes(self):
        opt, ref = make_option(EXPIRIES["1Y"], "Call", 100.0), make_option(EXPIRIES["1Y"], "Call", 100.0)
        eq = self.mkt.get_equity("DISC")
        sources = [eq.spot, eq.vol_surface.volatility_matrix]
        pricer = mc()
        pricer.price(opt, self.mkt, autodiff=True)
        d_mc, v_mc = pricer.tape.gradient(opt.price, sources)
        bs = self.bs(ref, autodiff=True)
        d_bs, v_bs = bs.tape.gradient(ref.price, sources)
        self.assertAlmostEqual(float(d_mc) / float(d_bs), 1.0, delta=0.01)
        np.testing.assert_allclose(v_mc.numpy(), v_bs.numpy(), rtol=0.03, atol=1e-8)

    def test_delta_vs_common_random_numbers_bump(self):
        """Pathwise AAD delta = derivative of the same MC estimator (bump with the same seed)."""
        spot = self.mkt.get_eq_spot("DISC")
        opt = make_option(EXPIRIES["1Y"], "Call", 100.0)
        pricer = mc()
        pricer.price(opt, self.mkt, autodiff=True)
        delta = float(pricer.tape.gradient(opt.price, spot))
        h, out = 1e-4, []
        for x in (S0 + h, S0 - h):
            spot.assign(x)
            o = make_option(EXPIRIES["1Y"], "Call", 100.0)
            mc().price(o, self.mkt)
            out.append(float(o.price))
        spot.assign(S0)
        self.assertAlmostEqual(delta, (out[0] - out[1]) / (2 * h), delta=1e-4 * abs(delta))

    def test_dividend_gradient_is_minus_delta_times_df(self):
        """Escrowed model: x0 = S - sum D_i P(0, ex_i), so dV/dD_i = -dV/dS P(0, ex_i) for ex_i <= T."""
        eq = self.mkt.get_equity("DISC")
        opt = make_option(EXPIRIES["1Y"], "Put", 100.0)
        pricer = mc()
        pricer.price(opt, self.mkt, autodiff=True)
        d_spot, d_div = pricer.tape.gradient(opt.price, [eq.spot, eq.dividends.variables[0]])
        for g, d in zip(d_div.numpy(), eq.dividends.ex_dates):
            expected = -float(d_spot) * float(eq.discount_curve.discount(d)) if d <= EXPIRIES["1Y"] else 0.0
            self.assertAlmostEqual(g, expected, places=10)

    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            tq.VanillaMCPricer(model=object())
        with self.assertRaises(ValueError):
            mc().price(make_option(EXPIRIES["1Y"], "Call", 100.0, "American"), self.mkt)
        with self.assertRaises(ValueError):
            mc().price(make_option(EXPIRIES["1Y"], "Call", 100.0, underlying="NOVOL"), self.mkt)


class TestLocalVol(MCTestCase):
    LV = dict(model=tq.LocalVolatilityModel, n_paths=20_000, n_steps=50)

    def test_flat_surface_reproduces_black_scholes(self):
        """Flat implied vol: Dupire local vol = implied vol, MC within 3 standard errors of BS."""
        self.mkt = flat_vol_market(0.2)
        for ot, K in (("Call", 100.0), ("Put", 90.0), ("Call", 120.0)):
            opt = make_option(EXPIRIES["1Y"], ot, K, underlying="FLAT")
            ref = make_option(EXPIRIES["1Y"], ot, K, underlying="FLAT")
            mc(**self.LV).price(opt, self.mkt)
            self.bs(ref)
            with self.subTest(type=ot, K=K):
                self.assertLess(abs(float(opt.price) - float(ref.price)), 3 * float(opt.standard_error))

    def test_float64_and_vega_reaches_nodes(self):
        opt = make_option(EXPIRIES["1Y"], "Call", 100.0)
        pricer = mc(**self.LV)
        pricer.price(opt, self.mkt, autodiff=True)
        self.assertEqual(opt.price.dtype, tf.float64)
        vega = pricer.tape.gradient(opt.price, self.mkt.get_eq_vol_surface("DISC").volatility_matrix)
        self.assertIsNotNone(vega)
        self.assertGreater(float(tf.reduce_sum(vega)), 0.0)

    def test_node_vega_vs_common_random_numbers_bump(self):
        matrix = self.mkt.get_eq_vol_surface("DISC").volatility_matrix
        opt = make_option(EXPIRIES["1Y"], "Call", 100.0)
        pricer = mc(**self.LV)
        pricer.price(opt, self.mkt, autodiff=True)
        grad = pricer.tape.gradient(opt.price, matrix).numpy()
        base = matrix.numpy().copy()
        i, j = 1, 1                       # node T = 1Y, K = 100
        h, out = 1e-5, []
        for sign in (1, -1):
            m = base.copy()
            m[i, j] += sign * h
            matrix.assign(m)
            o = make_option(EXPIRIES["1Y"], "Call", 100.0)
            mc(**self.LV).price(o, self.mkt)
            out.append(float(o.price))
        matrix.assign(base)
        self.assertAlmostEqual(grad[i, j], (out[0] - out[1]) / (2 * h), delta=1e-3 * abs(grad[i, j]))



def skewed_market(regrid_ticker: str = "SKEW") -> MarketEnvironment:
    """Narrow (70%-130%), coarse grid with a strong skew: paths leave the grid, as in the marketdata/ snapshot."""
    strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    maturities = [0.25, 0.5, 1.0, 2.0, 3.0]
    iv = [[0.30 - 0.25 * np.log(k / S0) / np.sqrt(max(t, 0.25)) + 0.2 * np.log(k / S0) ** 2 for k in strikes]
          for t in maturities]
    surface = VolatilitySurface(
        reference_date=EVAL_DATE, calendar=None, daycounter=DayCounter(DayCounterConvention.Actual365),
        strike=strikes, maturity=maturities, volatility_matrix=iv,
    )
    return MarketEnvironment(market={
        "IR:EUR:ESTR:SPOT": tq.RateCurve(EVAL_DATE, PILLARS, RATES, "LINEAR", DayCounterConvention.Actual365),
        f"EQ:EUR:{regrid_ticker}:SPOT": S0,
        f"EQ:EUR:{regrid_ticker}:REPO": 0.0,
        f"EQ:EUR:{regrid_ticker}:VOL": surface,
    })


class TestRegrid(unittest.TestCase):
    def test_weights_reproduce_the_nodes(self):
        from tensorquant.models.localvolatility import _strike_weights, _time_weights
        k, t = np.array([70.0, 85.0, 100.0, 120.0, 150.0]), np.array([0.25, 0.5, 1.0, 3.0])
        np.testing.assert_allclose(_strike_weights(k, k), np.eye(len(k)), atol=1e-12)
        np.testing.assert_allclose(_time_weights(t, t), np.eye(len(t)), atol=1e-12)

    def test_flat_surface_stays_flat_and_gradient_reaches_nodes(self):
        from tensorquant.models.localvolatility import regrid_implied_vol
        iv = tf.Variable(np.full((3, 4), 0.25), dtype=tf.float64)
        with tf.GradientTape() as tape:
            new, t_new, k_new = regrid_implied_vol(iv, [0.5, 1.0, 2.0], [80.0, 95.0, 110.0, 130.0])
            total = tf.reduce_sum(new)
        np.testing.assert_allclose(new.numpy(), 0.25, atol=1e-12)
        self.assertGreater(len(k_new), 4)
        self.assertEqual(t_new[-1], 2.0)
        self.assertTrue(np.all(np.abs(tape.gradient(total, iv).numpy()) > 0))  # spline weights may be < 0

    def test_skewed_smile_repriced_at_the_nodes(self):
        """MC implied vols at the 2Y nodes within 1.5 vol points with re-gridding; the market grid is far worse."""
        from scipy.optimize import brentq
        from scipy.stats import norm
        tq.Settings.evaluation_date = EVAL_DATE
        mkt = skewed_market()
        eq = mkt.get_equity("SKEW")
        expiry = date(2028, 5, 14)  # 2Y
        T = eq.vol_surface.daycounter.year_fraction(EVAL_DATE, expiry)
        DF, F = float(eq.discount_curve.discount(expiry)), float(eq.forward(expiry))
        t_grid = tf.linspace(tf.constant(0.0, tf.float64), tf.constant(T, tf.float64), 101)[1:]
        dw = tf.random.stateless_normal([40_000, 100], seed=[3, 0], dtype=tf.float64)

        def implied(price, K):
            f = lambda s: DF * (F * norm.cdf((np.log(F / K) + 0.5 * s * s * T) / (s * np.sqrt(T)))
                                - K * norm.cdf((np.log(F / K) - 0.5 * s * s * T) / (s * np.sqrt(T)))) - price
            return brentq(f, 1e-4, 3.0)

        worst = {}
        for regrid in (True, False):
            s_t = tq.LocalVolatilityModel.from_equity(eq, expiry, regrid=regrid).evolve(t_grid, dw).numpy()[:, -1]
            worst[regrid] = max(abs(implied(np.mean(DF * np.maximum(s_t - K, 0.0)), K) - float(eq.vol_surface.volatility(K, T)))
                                for K in eq.vol_surface.strike)
        self.assertLess(worst[True], 0.015)
        self.assertGreater(worst[False], 2 * worst[True])


if __name__ == "__main__":
    unittest.main()
