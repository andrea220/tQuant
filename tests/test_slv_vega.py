"""Heston SLV vega per node: the leverage is calibrated separately (conditional expectation ``E[v | X]`` stored on a
grid) and the market greeks map the model greeks through it by the adjoint of that calibration (``par_greeks``).

The reference is the two-stage model differentiated by bumping: bump a vol node, recalibrate the leverage on the same
particles and nodes (both tables), reprice on the same paths (central differences). The adjoint must give the same
derivative, for a single node, its neighbour and a parallel shift, at 1Y and at 5Y (stability over a long
calibration).

Smoothing for risk: the price comes from the Richardson-corrected table, the greeks from the smooth 41-node one, so
the adjoint is the derivative of a slightly different (smoother) model than the bumped one: 3% on the node, 5% on
the parallel shift (1Y: 1.03). Off the node the gap can reach 0.05 vega per vol point (1Y, strike below: 0.02 vs
0.07), not tested.
"""

import unittest
from datetime import date, timedelta

import numpy as np
import tensorflow as tf

import tensorquant as tq

from markers import slow
from test_heston import TRUTH

EVAL_DATE = date(2026, 5, 15)
SMALL = (("EUR", 1), ("GBP", 1), ("USD", 1))
SETTINGS = dict(n_particles=20_000, steps_per_year=40)
PATHS = dict(n_paths=20_000, seed=3)
H = 1e-3


def ssvi_equity():
    tq.Settings.evaluation_date = EVAL_DATE
    mkt = tq.generate_market(EVAL_DATE, seed=7, tickers_per_currency=SMALL)
    return mkt, mkt.get_equity("TICKER1")


def put(eq, days):
    """Put struck at the 80% node of the synthetic surface (196.39)."""
    return tq.VanillaOption(eq.currency, start_date=EVAL_DATE, end_date=EVAL_DATE + timedelta(days=days),
                            option_type=tq.OptionType.Put, strike=float(eq.vol_surface.strike[2]),
                            underlying="TICKER1")


class TestSLVVega(unittest.TestCase):
    def two_stage_price(self, mkt, eq, model, days, n_steps):
        """Recalibrate the leverage of *model* on the current market (same particles, same nodes) and reprice."""
        bumped = tq.HestonSLVModel(model.heston, **model.settings)
        bumped.calibrate_leverage(eq, nodes=model.leverage_table["nodes"])
        product = put(eq, days)
        tq.VanillaMCPricer(model=bumped, n_steps=n_steps, **PATHS).price(product, mkt)
        return float(product.price)

    def check_vega(self, days, row, n_steps):
        mkt, eq = ssvi_equity()
        model = tq.HestonSLVModel(tq.HestonModel(**TRUTH), horizon=days / 365, **SETTINGS)
        model.calibrate_leverage(eq)
        product = put(eq, days)
        pricer = tq.VanillaMCPricer(model=model, n_steps=n_steps, **PATHS)
        pricer.price(product, mkt, autodiff=True)
        grads = pricer.tape.gradient(product.price, {"model": model.parameters, "market": eq.risk_factors()})
        direct = {k: [None if g is None else g.numpy() for g in v] for k, v in grads["market"].items()}
        greeks = tq.par_greeks(model, {k: v.numpy() for k, v in grads["model"].items()}, direct, equity=eq)
        vega = greeks["EQ:EUR:TICKER1:VOL"][0] / 100                                  # per vol point
        self.assertTrue(np.all(np.isfinite(vega)))

        vol = eq.vol_surface.volatility_matrix
        base = vol.numpy().copy()

        def bump(shift):
            prices = []
            for sign in (1.0, -1.0):
                vol.assign(base + sign * shift)
                prices.append(self.two_stage_price(mkt, eq, model, days, n_steps))
            vol.assign(base)
            return (prices[0] - prices[1]) / (2 * H) / 100

        for col in (2, 4):
            shift = np.zeros_like(base)
            shift[row, col] = H
            with self.subTest(node=(row, col)):
                self.assertAlmostEqual(vega[row, col], bump(shift), delta=0.03 * abs(vega[row, 2]) + 0.005)
        self.assertAlmostEqual(vega.sum() / bump(np.full_like(base, H)), 1.0, delta=0.05)
        return vega

    @slow
    def test_one_year_put(self):
        """1Y 80% put: adjoint par vega = bumps of the two-stage model (node, neighbour, parallel)."""
        self.check_vega(365, row=3, n_steps=40)

    @slow
    def test_five_year_put_is_stable(self):
        """5Y 80% put (200 calibration steps): the adjoint stays finite and equals the bumps."""
        self.check_vega(1826, row=7, n_steps=100)

    @slow
    def test_class_path_equals_instance_par_greeks(self):
        """The class path (leverage calibrated inside the pricing tape, one backward pass) = the instance's direct
        greeks + the adjoint through the stored calibration, on the same particles."""
        mkt, eq = ssvi_equity()
        model = tq.HestonSLVModel(tq.HestonModel(**TRUTH), horizon=1.0, **SETTINGS)
        model.calibrate_leverage(eq)
        product = put(eq, 365)
        pricer = tq.VanillaMCPricer(model=model, n_steps=40, **PATHS)
        pricer.price(product, mkt, autodiff=True)
        grads = pricer.tape.gradient(product.price, {"model": model.parameters, "market": eq.risk_factors()})
        direct = {k: [None if g is None else g.numpy() for g in v] for k, v in grads["market"].items()}
        par = tq.par_greeks(model, {k: v.numpy() for k, v in grads["model"].items()}, direct, equity=eq)

        with tf.GradientTape() as tape:
            process = tq.HestonSLVProcess.from_equity(eq, maturity=1.0, heston=model.heston.process(eq),
                                                      **model.settings)
            t_grid = np.arange(1, 41) / 40.0
            dw = tf.random.stateless_normal([PATHS["n_paths"], 40, 2], seed=[PATHS["seed"], 0], dtype=tf.float64)
            s_t = process.evolve(t_grid, dw, observation_times=[1.0])[:, 0]
            price = tf.cast(eq.discount_curve.discount(1.0), tf.float64) * tf.reduce_mean(
                tf.maximum(float(eq.vol_surface.strike[2]) - s_t, 0.0))
        one_pass = tape.gradient(price, eq.risk_factors())
        self.assertAlmostEqual(float(price), float(product.price), places=10)
        for key, blocks in one_pass.items():
            for a, b in zip(blocks, par[key]):
                if a is not None:
                    np.testing.assert_allclose(a.numpy(), b, rtol=1e-6, atol=1e-9, err_msg=key)


if __name__ == "__main__":
    unittest.main()
