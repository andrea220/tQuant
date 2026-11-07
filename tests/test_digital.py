"""DigitalOption: Black-Scholes closed form, parities with vanillas, call-spread limit, Monte Carlo (hard / smoothed)."""

import unittest
from datetime import date

import numpy as np
import tensorflow as tf
from scipy.stats import norm

import tensorquant as tq

from test_black_scholes import EVAL_DATE, EXPIRIES, make_market
from test_vanilla_mc import flat_vol_market

CASH, ASSET = tq.PayoffType.CashOrNothing, tq.PayoffType.AssetOrNothing
EXPIRY = EXPIRIES["1Y"]


def digital(option_type, strike, payoff_type=CASH, cash=1.0, underlying="CONT", expiry=EXPIRY):
    return tq.DigitalOption(tq.Currency.EUR, EVAL_DATE, expiry, getattr(tq.OptionType, option_type), strike,
                            payoff_type=payoff_type, cash_amount=cash, underlying=underlying)


def vanilla(option_type, strike, underlying="CONT", expiry=EXPIRY):
    return tq.VanillaOption(tq.Currency.EUR, start_date=EVAL_DATE, end_date=expiry,
                            option_type=getattr(tq.OptionType, option_type), strike=strike, underlying=underlying)


def bs_price(product, mkt, model="continuous", autodiff=False):
    pricer = tq.BlackScholesPricer(dividend_model=model)
    pricer.price(product, mkt, autodiff=autodiff)
    return pricer


class TestBlackScholes(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = make_market()

    def test_closed_form(self):
        """Cash-or-nothing A P(0,T) N(phi d2), asset-or-nothing P(0,T) F N(phi d1), on the forward and the vol at K."""
        for ticker in ("CONT", "DISC"):
            eq = self.mkt.get_equity(ticker)
            F, DF = float(eq.forward(EXPIRY)), float(eq.discount_curve.discount(EXPIRY))
            T = (EXPIRY - EVAL_DATE).days / 365
            for K in (85.0, 100.0, 115.0):
                sigma = float(eq.vol_surface.volatility(K, T))
                d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                for ot, phi in (("Call", 1.0), ("Put", -1.0)):
                    cash, asset = digital(ot, K, CASH, 2.5, ticker), digital(ot, K, ASSET, underlying=ticker)
                    bs_price(cash, self.mkt, "discrete")
                    bs_price(asset, self.mkt, "discrete")
                    with self.subTest(ticker=ticker, K=K, type=ot):
                        self.assertAlmostEqual(float(cash.price), 2.5 * DF * norm.cdf(phi * d2), places=12)
                        self.assertAlmostEqual(float(asset.price), DF * F * norm.cdf(phi * d1), places=10)

    def test_parities(self):
        """Cash call + put = A P(0,T); asset call + put = P(0,T) F; asset - K cash = vanilla (same vol at K)."""
        eq = self.mkt.get_equity("CONT")
        F, DF = float(eq.forward(EXPIRY)), float(eq.discount_curve.discount(EXPIRY))
        for K in (90.0, 110.0):
            p = {}
            for ot in ("Call", "Put"):
                for pt in (CASH, ASSET):
                    d = digital(ot, K, pt)
                    bs_price(d, self.mkt)
                    p[(ot, pt)] = float(d.price)
            self.assertAlmostEqual(p[("Call", CASH)] + p[("Put", CASH)], DF, places=13)
            self.assertAlmostEqual(p[("Call", ASSET)] + p[("Put", ASSET)], DF * F, places=11)
            for ot, phi in (("Call", 1.0), ("Put", -1.0)):
                v = vanilla(ot, K)
                bs_price(v, self.mkt)
                self.assertAlmostEqual(phi * (p[(ot, ASSET)] - K * p[(ot, CASH)]), float(v.price), places=11)

    def test_call_spread_limit_flat_and_skew(self):
        """(C(K-h) - C(K+h)) / 2h -> the digital on a flat surface; on a skewed one it converges to
        digital - vega * dsigma/dK (the slope of the smile, which the BS digital at the vol of K misses)."""
        flat = flat_vol_market(0.2)
        K, h = 100.0, 1e-3
        spread = []
        for k in (K - h, K + h):
            v = vanilla("Call", k, "FLAT")
            bs_price(v, flat)
            spread.append(float(v.price))
        d = digital("Call", K, underlying="FLAT")
        bs_price(d, flat)
        self.assertAlmostEqual((spread[0] - spread[1]) / (2 * h), float(d.price), places=7)

        eq = self.mkt.get_equity("CONT")
        T = (EXPIRY - EVAL_DATE).days / 365
        K = 90.0                                                   # inside a smile segment: dsigma/dK = -0.002
        spread = []
        for k in (K - h, K + h):
            v = vanilla("Call", k)
            bs_price(v, self.mkt)
            spread.append(float(v.price))
        d = digital("Call", K)
        bs_price(d, self.mkt)
        v = vanilla("Call", K)
        pricer = bs_price(v, self.mkt, autodiff=True)
        vega = float(tf.reduce_sum(pricer.tape.gradient(v.price, eq.vol_surface.volatility_matrix)))
        slope = (float(eq.vol_surface.volatility(K + 1.0, T)) - float(eq.vol_surface.volatility(K - 1.0, T))) / 2.0
        self.assertAlmostEqual(slope, -0.002, places=12)
        self.assertAlmostEqual((spread[0] - spread[1]) / (2 * h), float(d.price) - vega * slope, places=6)
        self.assertGreater((spread[0] - spread[1]) / (2 * h) - float(d.price), 0.02)   # negative skew: digital call up

    def test_greeks_and_factory(self):
        """Cash digital delta = A P(0,T) n(d2) / (S sigma sqrt T) (continuous model) by AAD; factory -> BS."""
        eq = self.mkt.get_equity("CONT")
        d = digital("Call", 100.0)
        self.assertIsInstance(tq.PricerAssignment.create(d), tq.BlackScholesPricer)
        pricer = bs_price(d, self.mkt, autodiff=True)
        delta = float(pricer.tape.gradient(d.price, eq.spot))
        F, DF = float(eq.forward(EXPIRY)), float(eq.discount_curve.discount(EXPIRY))
        T = (EXPIRY - EVAL_DATE).days / 365
        sigma = float(eq.vol_surface.volatility(100.0, T))
        d2 = (np.log(F / 100.0) - 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        self.assertAlmostEqual(delta, DF * norm.pdf(d2) / (float(eq.spot) * sigma * np.sqrt(T)), places=12)
        with self.assertRaises(ValueError):
            tq.DigitalOption(tq.Currency.EUR, EVAL_DATE, EXPIRY, tq.OptionType.Call, 100.0,
                             payoff_type=tq.PayoffType.PlainVanilla)


class TestMonteCarlo(unittest.TestCase):
    N = 20_000

    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = make_market()

    def mc(self, product, smoothing, autodiff=False, seed=5):
        pricer = tq.VanillaMCPricer(n_paths=self.N, n_steps=1, seed=seed, digital_smoothing=smoothing)
        pricer.price(product, self.mkt, autodiff=autodiff)
        return pricer

    def test_gbm_hard_equals_closed_form(self):
        """GBM with the vol at K, hard indicator: = Black-Scholes within 3 SE (cash call, cash put, asset call)."""
        for ot, pt in (("Call", CASH), ("Put", CASH), ("Call", ASSET)):
            mc, ref = digital(ot, 105.0, pt), digital(ot, 105.0, pt)
            self.mc(mc, 0.0)
            bs_price(ref, self.mkt)
            with self.subTest(type=ot, payoff=pt.value):
                self.assertLess(abs(float(mc.price) - float(ref.price)), 3 * float(mc.standard_error))

    def test_smoothing_bias_and_greeks(self):
        """Hard indicator: zero pathwise delta. Sigmoid of width 1% of K: on the same paths the price moves by less
        than 1% (smoothing bias), and the AAD delta is the Black-Scholes digital delta within 6%."""
        eq = self.mkt.get_equity("CONT")
        hard = digital("Call", 100.0)
        pricer = self.mc(hard, 0.0, autodiff=True)
        self.assertIsNone(pricer.tape.gradient(hard.price, eq.spot))      # the indicator cuts the graph: no delta
        smooth, ref = digital("Call", 100.0), digital("Call", 100.0)
        pricer = self.mc(smooth, 0.01, autodiff=True)
        ref_pricer = bs_price(ref, self.mkt, autodiff=True)
        self.assertAlmostEqual(float(smooth.price) / float(hard.price), 1.0, delta=0.01)
        mc_delta = float(pricer.tape.gradient(smooth.price, eq.spot))
        bs_delta = float(ref_pricer.tape.gradient(ref.price, eq.spot))
        self.assertAlmostEqual(mc_delta / bs_delta, 1.0, delta=0.06)


if __name__ == "__main__":
    unittest.main()
