"""BarrierOption: Black-Scholes closed form (Reiner-Rubinstein, barrier at expiry, Broadie-Glasserman-Kou) and Monte
Carlo (Brownian bridge for continuous monitoring, discrete dates)."""

import unittest
from datetime import date, timedelta

import numpy as np
import tensorflow as tf

import tensorquant as tq

from markers import slow
from test_black_scholes import EVAL_DATE, EXPIRIES
from test_vanilla_mc import flat_vol_market

EXPIRY = EXPIRIES["1Y"]
BT, OT = tq.BarrierType, tq.OptionType
# (K, H) below and above the spot (100), strike on both sides of the barrier
DOWN_CASES = ((100.0, 90.0), (85.0, 90.0))
UP_CASES = ((100.0, 115.0), (120.0, 115.0))


def barrier(ot, K, H, bt, monitoring=None, underlying="FLAT"):
    return tq.BarrierOption(tq.Currency.EUR, EVAL_DATE, EXPIRY, ot, K, H, bt, monitoring_dates=monitoring,
                            underlying=underlying)


def vanilla(ot, K, underlying="FLAT"):
    return tq.VanillaOption(tq.Currency.EUR, start_date=EVAL_DATE, end_date=EXPIRY, option_type=ot, strike=K,
                            underlying=underlying)


def all_cases():
    for bt in BT:
        for ot in (OT.Call, OT.Put):
            for K, H in (DOWN_CASES if bt.is_down else UP_CASES):
                yield bt, ot, K, H


def bs(product, mkt, autodiff=False):
    pricer = tq.BlackScholesPricer()
    pricer.price(product, mkt, autodiff=autodiff)
    return float(product.price), pricer


def daily_dates():
    n = (EXPIRY - EVAL_DATE).days
    return [EVAL_DATE + timedelta(days=i) for i in range(1, n + 1)]


class TestBlackScholes(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = flat_vol_market(0.2)

    def test_in_plus_out_is_vanilla(self):
        """Knock-in + knock-out = vanilla: continuous, at expiry, daily (BGK)."""
        for monitoring in (None, [EXPIRY], daily_dates()):
            for ot in (OT.Call, OT.Put):
                for K, H in DOWN_CASES + UP_CASES:
                    ins, outs = (BT.DownIn, BT.DownOut) if H < 100 else (BT.UpIn, BT.UpOut)
                    v, _ = bs(vanilla(ot, K), self.mkt)
                    i, _ = bs(barrier(ot, K, H, ins, monitoring), self.mkt)
                    o, _ = bs(barrier(ot, K, H, outs, monitoring), self.mkt)
                    with self.subTest(monitoring=None if monitoring is None else len(monitoring), ot=ot, K=K, H=H):
                        self.assertAlmostEqual(i + o, v, places=10)
                        self.assertGreaterEqual(min(i, o), -1e-12)

    def test_limits(self):
        """Far barrier: knock-in -> 0, knock-out -> vanilla; barrier already crossed: knock-in = vanilla."""
        v, _ = bs(vanilla(OT.Put, 100.0), self.mkt)
        self.assertLess(bs(barrier(OT.Put, 100.0, 1.0, BT.DownIn), self.mkt)[0], 1e-12)
        self.assertAlmostEqual(bs(barrier(OT.Put, 100.0, 1.0, BT.DownOut), self.mkt)[0], v, places=12)
        self.assertAlmostEqual(bs(barrier(OT.Put, 100.0, 120.0, BT.DownIn), self.mkt)[0], v, places=12)
        self.assertAlmostEqual(bs(barrier(OT.Put, 100.0, 120.0, BT.DownOut), self.mkt)[0], 0.0, places=12)

    def test_monitoring_ordering(self):
        """Knock-in value: at expiry only < daily < continuous (more chances to hit); daily close to continuous."""
        at_t, _ = bs(barrier(OT.Put, 100.0, 80.0, BT.DownIn, [EXPIRY]), self.mkt)
        daily, _ = bs(barrier(OT.Put, 100.0, 80.0, BT.DownIn, daily_dates()), self.mkt)
        cont, _ = bs(barrier(OT.Put, 100.0, 80.0, BT.DownIn), self.mkt)
        self.assertLess(at_t, daily)
        self.assertLess(daily, cont)
        self.assertLess(cont - daily, 0.05 * cont)

    def test_at_expiry_decomposition(self):
        """Down-and-in put observed at expiry = put(H) + (K - H) cash digital put(H)."""
        K, H = 100.0, 80.0
        p, _ = bs(barrier(OT.Put, K, H, BT.DownIn, [EXPIRY]), self.mkt)
        put_h, _ = bs(vanilla(OT.Put, H), self.mkt)
        dig, _ = bs(tq.DigitalOption(tq.Currency.EUR, EVAL_DATE, EXPIRY, OT.Put, H, underlying="FLAT"), self.mkt)
        self.assertAlmostEqual(p, put_h + (K - H) * dig, places=12)

    @slow
    def test_delta_vega_vs_bumps(self):
        """AAD delta and vega (sum over the nodes) of the continuous closed form vs central bumps."""
        eq = self.mkt.get_equity("FLAT")
        surface = eq.vol_surface.volatility_matrix
        for bt, ot, K, H in all_cases():
            product = barrier(ot, K, H, bt)
            _, pricer = bs(product, self.mkt, autodiff=True)
            delta, vega = pricer.tape.gradient(product.price, [eq.spot, surface])
            s0, m0 = float(eq.spot), surface.numpy()
            h = 1e-3
            eq.spot.assign(s0 + h); up, _ = bs(barrier(ot, K, H, bt), self.mkt)
            eq.spot.assign(s0 - h); dn, _ = bs(barrier(ot, K, H, bt), self.mkt)
            eq.spot.assign(s0)
            surface.assign(m0 + 1e-5); vu, _ = bs(barrier(ot, K, H, bt), self.mkt)
            surface.assign(m0 - 1e-5); vd, _ = bs(barrier(ot, K, H, bt), self.mkt)
            surface.assign(m0)
            with self.subTest(bt=bt, ot=ot, K=K, H=H):
                self.assertAlmostEqual(float(delta), (up - dn) / (2 * h), places=5)
                self.assertAlmostEqual(float(tf.reduce_sum(vega)), (vu - vd) / 2e-5, places=3)

    def test_factory_and_validation(self):
        self.assertIsInstance(tq.PricerAssignment.create(barrier(OT.Put, 100.0, 80.0, BT.DownIn)),
                              tq.BlackScholesPricer)
        with self.assertRaises(ValueError):
            barrier(OT.Put, 100.0, -1.0, BT.DownIn)
        with self.assertRaises(ValueError):
            tq.BarrierOption(tq.Currency.EUR, EVAL_DATE, EXPIRY, OT.Put, 100.0, 80.0, BT.DownIn,
                             monitoring_dates=[date(2028, 5, 15)], underlying="FLAT")
        with self.assertRaises(ValueError):    # a single monitoring date before the expiry
            bs(barrier(OT.Put, 100.0, 80.0, BT.DownIn, [EXPIRIES["6M"]]), self.mkt)


class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = flat_vol_market(0.2)

    def mc(self, product, n_paths=4_000, n_steps=50, seed=1, smoothing=0.0, autodiff=False):
        pricer = tq.VanillaMCPricer(n_paths=n_paths, n_steps=n_steps, seed=seed, digital_smoothing=smoothing)
        pricer.price(product, self.mkt, autodiff=autodiff)
        return float(product.price), float(product.standard_error), pricer

    def test_continuous_vs_closed_form(self):
        """GBM with the Brownian-bridge survival = Reiner-Rubinstein within 3 SE, all 8 types."""
        for bt, ot, K, H in all_cases():
            price, se, _ = self.mc(barrier(ot, K, H, bt))
            ref, _ = bs(barrier(ot, K, H, bt), self.mkt)
            with self.subTest(bt=bt, ot=ot, K=K, H=H):
                self.assertLess(abs(price - ref), 3 * se + 1e-12)

    def test_at_expiry_vs_closed_form(self):
        """Barrier observed at expiry only: MC (hard indicator) = exact decomposition within 3 SE."""
        for bt in (BT.DownIn, BT.DownOut):
            product = barrier(OT.Put, 100.0, 80.0, bt, [EXPIRY])
            price, se, _ = self.mc(product, n_steps=1)
            ref, _ = bs(barrier(OT.Put, 100.0, 80.0, bt, [EXPIRY]), self.mkt)
            self.assertLess(abs(price - ref), 3 * se)

    def test_discrete_monthly_vs_bgk(self):
        """Monthly monitoring: MC on the dates = BGK-shifted closed form within 3 SE + 1% (BGK is an approximation)."""
        months = [d for d in (EVAL_DATE + timedelta(days=30 * i) for i in range(1, 13))
                  if d < EXPIRY] + [EXPIRY]
        product = barrier(OT.Put, 100.0, 85.0, BT.DownIn, months)
        price, se, _ = self.mc(product, n_steps=12)
        ref, _ = bs(barrier(OT.Put, 100.0, 85.0, BT.DownIn, months), self.mkt)
        self.assertLess(abs(price - ref), 3 * se + 0.01 * ref)

    def test_in_plus_out_is_vanilla(self):
        """Same paths: knock-in + knock-out = vanilla path by path (continuous and discrete)."""
        for monitoring in (None, [EXPIRY]):
            i, _, _ = self.mc(barrier(OT.Put, 100.0, 85.0, BT.DownIn, monitoring), n_paths=2_000, n_steps=20)
            o, _, _ = self.mc(barrier(OT.Put, 100.0, 85.0, BT.DownOut, monitoring), n_paths=2_000, n_steps=20)
            v, _, _ = self.mc(vanilla(OT.Put, 100.0), n_paths=2_000, n_steps=20)
            self.assertAlmostEqual(i + o, v, places=10)

    def test_delta_vs_bump(self):
        """Pathwise AAD delta of the bridge estimator (continuous in the path) vs a common-random-numbers bump."""
        eq = self.mkt.get_equity("FLAT")
        product = barrier(OT.Put, 100.0, 85.0, BT.DownIn)
        _, _, pricer = self.mc(product, n_paths=2_000, n_steps=20, autodiff=True)
        delta = float(pricer.tape.gradient(product.price, eq.spot))
        s0, h = float(eq.spot), 0.5
        eq.spot.assign(s0 + h); up, _, _ = self.mc(barrier(OT.Put, 100.0, 85.0, BT.DownIn), n_paths=2_000, n_steps=20)
        eq.spot.assign(s0 - h); dn, _, _ = self.mc(barrier(OT.Put, 100.0, 85.0, BT.DownIn), n_paths=2_000, n_steps=20)
        eq.spot.assign(s0)
        self.assertAlmostEqual(delta, (up - dn) / (2 * h), delta=0.01 * abs(delta) + 1e-3)


if __name__ == "__main__":
    unittest.main()
