"""VanillaMCPricer: GBM and local vol built from the market (Equity view), float64, AAD greeks on market factors."""

import unittest
from datetime import date, timedelta

import numpy as np
import tensorflow as tf

import tensorquant as tq
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

from markers import slow
from test_black_scholes import EVAL_DATE, EXPIRIES, PILLARS, RATES, S0, make_market, make_option

GBM_PATHS = 20_000


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
        self.assertAlmostEqual(float(d_mc) / float(d_bs), 1.0, delta=0.02)
        np.testing.assert_allclose(v_mc.numpy(), v_bs.numpy(), rtol=0.05, atol=1e-8)

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
    LV = dict(model=tq.LocalVolatilityModel, n_paths=5_000, n_steps=50)

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

    @slow
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



def ssvi_market(eta: float = 1.0, rho: float = -0.6, atm: float = 0.25, ticker: str = "SSVI") -> MarketEnvironment:
    """Narrow (70%-130%), coarse grid sampled from an arbitrage-free SSVI surface (power-law phi,
    eta (1 + |rho|) <= 2), with discrete dividends and repo: paths leave the grid, forward has jumps."""
    strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    maturities = [0.25, 0.5, 1.0, 2.0, 3.0]
    base = MarketEnvironment(market={
        "IR:EUR:ESTR:SPOT": tq.RateCurve(EVAL_DATE, PILLARS, RATES, "LINEAR", DayCounterConvention.Actual365),
        f"EQ:EUR:{ticker}:SPOT": S0,
        f"EQ:EUR:{ticker}:REPO": 0.005,
        f"EQ:EUR:{ticker}:DIV": make_market().get_eq_dividends("DISC"),
    })
    eq = base.get_equity(ticker)
    iv = []
    for t in maturities:
        F = float(eq.forward(EVAL_DATE + timedelta(days=int(round(t * 365)))))
        theta = atm ** 2 * t
        phi = eta / np.sqrt(theta * (1.0 + theta))
        row = []
        for k in strikes:
            y = np.log(k / F)
            w = 0.5 * theta * (1 + rho * phi * y + np.sqrt((phi * y + rho) ** 2 + 1 - rho ** 2))
            row.append(np.sqrt(w / t))
        iv.append(row)
    surface = VolatilitySurface(
        reference_date=EVAL_DATE, calendar=None, daycounter=DayCounter(DayCounterConvention.Actual365),
        strike=strikes, maturity=maturities, volatility_matrix=iv,
    )
    market = {k: v for k, v in {
        "IR:EUR:ESTR:SPOT": base.get_ir_curve(tq.Currency.EUR),
        f"EQ:EUR:{ticker}:SPOT": S0,
        f"EQ:EUR:{ticker}:REPO": 0.005,
        f"EQ:EUR:{ticker}:DIV": make_market().get_eq_dividends("DISC"),
        f"EQ:EUR:{ticker}:VOL": surface,
    }.items()}
    return MarketEnvironment(market=market)


def otm_implied_vol(price: float, F: float, K: float, T: float, DF: float) -> float:
    from scipy.optimize import brentq
    from scipy.stats import norm

    def black_otm(s):
        d1 = (np.log(F / K) + 0.5 * s * s * T) / (s * np.sqrt(T))
        call = DF * (F * norm.cdf(d1) - K * norm.cdf(d1 - s * np.sqrt(T)))
        return call - DF * (F - K) if K < F else call

    return brentq(lambda s: black_otm(s) - price, 1e-4, 3.0)


class TestNaturalCubicSpline(unittest.TestCase):
    def test_vs_scipy_and_linear_wings(self):
        from scipy.interpolate import CubicSpline
        from tensorquant.numericalhandles.interpolation import natural_cubic_spline
        x = np.array([-1.0, -0.4, 0.0, 0.3, 1.1])
        f = np.array([0.5, 0.2, 0.1, 0.15, 0.4])
        q = np.linspace(-0.9, 1.0, 40)
        v, d1, d2 = natural_cubic_spline(x, f, q)
        cs = CubicSpline(x, f, bc_type="natural")
        np.testing.assert_allclose(v.numpy(), cs(q), atol=1e-14)
        np.testing.assert_allclose(d1.numpy(), cs(q, 1), atol=1e-13)
        np.testing.assert_allclose(d2.numpy(), cs(q, 2), atol=1e-12)
        out = np.array([-2.0, 2.0])
        v, d1, d2 = natural_cubic_spline(x, f, out)
        np.testing.assert_allclose(v.numpy(), [f[0] + cs(x[0], 1) * (-1.0), f[-1] + cs(x[-1], 1) * 0.9], atol=1e-13)
        np.testing.assert_allclose(d2.numpy(), 0.0)

    def test_gradient_wrt_knots_and_values(self):
        from tensorquant.numericalhandles.interpolation import natural_cubic_spline
        x = tf.Variable([-1.0, -0.4, 0.0, 0.3, 1.1], dtype=tf.float64)
        f = tf.Variable([0.5, 0.2, 0.1, 0.15, 0.4], dtype=tf.float64)
        q = tf.constant([-1.5, -0.2, 0.5, 1.5], tf.float64)
        with tf.GradientTape() as tape:
            total = tf.reduce_sum(natural_cubic_spline(x, f, q)[0])
        gx, gf = tape.gradient(total, [x, f])
        h = 1e-6
        for var, g in ((x, gx), (f, gf)):
            base = var.numpy().copy()
            for i in range(len(base)):
                out = []
                for sgn in (1, -1):
                    b = base.copy()
                    b[i] += sgn * h
                    var.assign(b)
                    out.append(float(tf.reduce_sum(natural_cubic_spline(x, f, q)[0])))
                var.assign(base)
                self.assertAlmostEqual(float(g[i]), (out[0] - out[1]) / (2 * h), places=6)


class TestLocalVolForward(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = ssvi_market()
        self.eq = self.mkt.get_equity("SSVI")

    def simulate(self, T, n_paths=10_000, seed=3, **kw):
        model = tq.LocalVolatilityModel.from_equity(self.eq, **kw)
        n = max(50, int(50 * T))
        t_grid = tf.linspace(tf.constant(0.0, tf.float64), tf.constant(T, tf.float64), n + 1)[1:]
        dw = tf.random.stateless_normal([n_paths, n], seed=[seed, 0], dtype=tf.float64)
        return model, t_grid, model.evolve(t_grid, dw)

    @slow
    def test_grid_up_to_the_maturity(self):
        """The local vol built up to a maturity (vol nodes to the first one from it) = the whole surface on the same
        draws: same paths, same vega on the nodes used, zero vega after them."""
        vols = self.eq.vol_surface.volatility_matrix
        t_grid = tf.linspace(tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64), 51)[1:]
        dw = tf.random.stateless_normal([2_000, 50], seed=[7, 0], dtype=tf.float64)
        out = []
        for maturity in (None, 1.0):
            with tf.GradientTape() as tape:
                model = tq.LocalVolatilityModel.from_equity(self.eq, maturity)
                paths = model.evolve(t_grid, dw)
                value = tf.reduce_mean(tf.nn.relu(paths[:, -1] - 100.0))
            out.append((paths.numpy(), tape.gradient(value, vols).numpy(), model.local_vol_grid[0].shape[0]))
        (p_all, g_all, rows_all), (p_cut, g_cut, rows_cut) = out
        self.assertLess(rows_cut, rows_all)                                       # a shorter grid
        np.testing.assert_allclose(p_cut, p_all, rtol=0, atol=1e-10)
        np.testing.assert_allclose(g_cut, g_all, rtol=0, atol=1e-10)
        k = int(np.searchsorted(np.asarray(self.eq.vol_surface.maturity), 1.0 - 1e-12)) + 1
        self.assertTrue((g_cut[k:] == 0).all())

    def test_no_arbitrage_reported_on_ssvi(self):
        model = tq.LocalVolatilityModel.from_equity(self.eq)
        self.assertEqual(model.arbitrage["arbitrage_fraction"], 0.0)
        self.assertGreater(model.arbitrage["min_g"], 0.0)

    def test_forward_is_the_market_forward(self):
        """S_t = F(t) X_t with the martingale correction: MC mean = Equity.forward at every step (dividend jumps)."""
        _, t_grid, paths = self.simulate(2.0, n_paths=5_000)
        for k in range(0, len(t_grid), 25):
            self.assertAlmostEqual(float(tf.reduce_mean(paths[:, k])), float(self.eq.forward(float(t_grid[k]))), places=8)

    def test_ssvi_smile_repriced_at_the_nodes(self):
        """Arbitrage-free surface on a narrow grid: MC implied vols (OTM options) at the 1Y and 3Y nodes, a gross
        check (10k paths: MC noise ~0.5 vol points)."""
        vs = self.eq.vol_surface
        for T, tol in ((1.0, 1.5), (3.0, 2.0)):
            expiry = EVAL_DATE + timedelta(days=int(round(T * 365)))
            _, _, paths = self.simulate(T)
            s_t = paths[:, -1].numpy()
            F, DF = float(self.eq.forward(expiry)), float(self.eq.discount_curve.discount(expiry))
            for K in vs.strike:
                payoff = DF * (np.maximum(K - s_t, 0.0) if K < F else np.maximum(s_t - K, 0.0))
                with self.subTest(T=T, K=K):
                    self.assertLess(abs(otm_implied_vol(payoff.mean(), F, K, T, DF) - float(vs.volatility(K, T))), tol / 100)

    def test_arbitrage_warning(self):
        """Butterfly arbitrage (smile too convex) is detected and reported."""
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        iv = [[0.20 + 3.0 * np.log(k / S0) ** 2 for k in strikes] for _ in range(3)]
        surface = VolatilitySurface(
            reference_date=EVAL_DATE, calendar=None, daycounter=DayCounter(DayCounterConvention.Actual365),
            strike=strikes, maturity=[0.5, 1.0, 2.0], volatility_matrix=iv,
        )
        mkt = MarketEnvironment(market={
            "IR:EUR:ESTR:SPOT": tq.RateCurve(EVAL_DATE, PILLARS, RATES, "LINEAR", DayCounterConvention.Actual365),
            "EQ:EUR:ARB:SPOT": S0,
            "EQ:EUR:ARB:VOL": surface,
        })
        with self.assertWarns(UserWarning):
            model = tq.LocalVolatilityModel.from_equity(mkt.get_equity("ARB"))
        self.assertGreater(model.arbitrage["arbitrage_fraction"], 0.0)

    @slow
    def test_greeks_vs_common_random_numbers_bumps(self):
        """Pathwise AAD delta and node vega of the pricer = derivatives of the same MC estimator."""
        pricer_args = dict(model=tq.LocalVolatilityModel, n_paths=3_000, n_steps=50, seed=5)
        make = lambda: make_option(EXPIRIES["1Y"], "Call", 100.0, underlying="SSVI")
        opt = make()
        pricer = tq.VanillaMCPricer(**pricer_args)
        pricer.price(opt, self.mkt, autodiff=True)
        spot, matrix = self.mkt.get_eq_spot("SSVI"), self.mkt.get_eq_vol_surface("SSVI").volatility_matrix
        d_spot, d_vol = pricer.tape.gradient(opt.price, [spot, matrix])

        def reprice():
            o = make()
            tq.VanillaMCPricer(**pricer_args).price(o, self.mkt)
            return float(o.price)

        h, out = 1e-3, []
        for x in (S0 + h, S0 - h):
            spot.assign(x)
            out.append(reprice())
        spot.assign(S0)
        self.assertAlmostEqual(float(d_spot), (out[0] - out[1]) / (2 * h), delta=1e-3 * abs(float(d_spot)))

        base, (i, j) = matrix.numpy().copy(), (2, 3)   # node T = 1Y, K = 100
        out = []
        for sgn in (1, -1):
            m = base.copy()
            m[i, j] += sgn * 1e-5
            matrix.assign(m)
            out.append(reprice())
        matrix.assign(base)
        self.assertAlmostEqual(float(d_vol[i, j]), (out[0] - out[1]) / 2e-5, delta=2e-3 * abs(float(d_vol[i, j])))


if __name__ == "__main__":
    unittest.main()
