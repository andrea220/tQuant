"""Equity models as instances (``EquityModel``): ``Model.calibrate`` / ``model.process(equity)`` vs the class path.

- the pricers give the same price with a calibrated instance as with the model class (same paths);
- local vol: the same greeks (the vol nodes are read at pricing in both paths);
- Heston, displaced diffusion: model greeks mapped through the stored calibration Jacobian (``par_greeks``) = the AAD
  greeks through the calibration of the class path; SLV: the same through the adjoint of the leverage calibration;
- models built by hand (no Jacobian), serialization round trip, ticker and argument checks.
"""

import unittest
from datetime import date

import numpy as np
import tensorflow as tf

import tensorquant as tq

from markers import slow
from test_model_pins import autocallable, call, market

PATHS = dict(n_paths=4_000, n_steps=24, seed=3)


def price_and_greeks(model, product, mkt, eq):
    """Price, model greeks ``{name: dV/dp}`` and market greeks ``{key: [dV/dvariable]}`` in one backward pass."""
    pricer = tq.VanillaMCPricer(model=model, **PATHS)
    pricer.price(product, mkt, autodiff=True)
    params = model.parameters if isinstance(model, tq.EquityModel) else {}
    grads = pricer.tape.gradient(product.price, {"model": params, "market": eq.risk_factors()})
    market = {k: [np.zeros(v.shape) if g is None else g.numpy() for g, v in zip(gs, eq.risk_factors()[k])]
              for k, gs in grads["market"].items()}
    return float(product.price), {k: v.numpy() for k, v in grads["model"].items()}, market


def assert_greeks_close(test, a: dict, b: dict, rtol, atol):
    test.assertEqual(set(a), set(b))
    for key in a:
        for x, y in zip(a[key], b[key]):
            np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=key)


class TestLocalVolInstance(unittest.TestCase):
    def setUp(self):
        self.mkt, self.eq = market()

    def test_instance_equals_class(self):
        """No parameters: same price and the same greeks (vega per node straight from the vol nodes)."""
        model = tq.LocalVolatilityModel.calibrate(self.eq)
        self.assertEqual((model.ticker, model.parameters, model.jacobian), ("TICKER1", {}, None))
        p_cls, _, g_cls = price_and_greeks(tq.LocalVolatilityModel, call(self.eq), self.mkt, self.eq)
        p_ins, _, g_ins = price_and_greeks(model, call(self.eq), self.mkt, self.eq)
        self.assertEqual(p_cls, p_ins)
        assert_greeks_close(self, g_cls, g_ins, rtol=1e-12, atol=1e-12)
        assert_greeks_close(self, g_ins, tq.par_greeks(model, {}, g_ins), rtol=0, atol=0)

    def test_settings_and_autocallable(self):
        """Settings reach the process; the autocallable pricer takes an instance too (same price as the class)."""
        coarse = tq.LocalVolatilityModel(n_y=120)
        self.assertEqual(coarse.process(self.eq).local_vol_grid[2].shape[0], 120)
        prices = []
        for model in (tq.LocalVolatilityModel, tq.LocalVolatilityModel()):
            product = autocallable(self.eq)
            tq.AutocallableMCPricer(model=model, n_paths=2_000, seed=5, max_time_step=1 / 12).price(product, self.mkt)
            prices.append(float(product.price))
        self.assertEqual(prices[0], prices[1])
        with self.assertRaises(TypeError):
            tq.LocalVolatilityModel(bad_setting=1)


class TestHestonInstance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mkt, cls.eq = market()
        cls.model = tq.HestonModel.calibrate(cls.eq)

    def setUp(self):
        tq.Settings.evaluation_date = date(2026, 5, 15)

    @slow
    def test_par_greeks_equal_greeks_through_the_calibration(self):
        """Same parameters and price as the class path; dV/dp . dp/dm + direct = AAD through the calibration."""
        process = tq.HestonModel.from_equity(self.eq)
        for name, value in self.model.parameters.items():
            self.assertAlmostEqual(float(value), float(process.parameters[name]), places=14)
        self.assertEqual(self.model.ticker, "TICKER1")
        self.assertLess(self.model.report.rms_error_volpts.max(), 20.0)
        p_cls, _, g_cls = price_and_greeks(tq.HestonModel, call(self.eq), self.mkt, self.eq)
        p_ins, model_greeks, direct = price_and_greeks(self.model, call(self.eq), self.mkt, self.eq)
        self.assertAlmostEqual(p_ins / p_cls, 1.0, places=12)
        assert_greeks_close(self, g_cls, tq.par_greeks(self.model, model_greeks, direct), rtol=1e-6, atol=1e-8)
        vol_key = "EQ:EUR:TICKER1:VOL"
        self.assertGreater(np.abs(g_cls[vol_key][0]).sum(), 10.0)              # vega spread over the surface

    @slow
    def test_serialization_round_trip(self):
        frames = self.model.to_frames()
        back = tq.HestonModel.from_frames(frames)
        self.assertEqual((back.ticker, back.calibration_date), (self.model.ticker, self.model.calibration_date))
        for name, value in self.model.parameters.items():
            self.assertEqual(float(back.parameters[name]), float(value))
        for name, per_key in self.model.jacobian.items():
            for key, blocks in per_key.items():
                for a, b in zip(blocks, back.jacobian[name][key]):
                    np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            tq.LocalVolatilityModel.from_frames(frames)

    def test_by_hand_and_checks(self):
        """A model built by hand prices, has model greeks, no Jacobian; ticker mismatch and bad models raise."""
        manual = tq.HestonModel(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.6)
        self.assertIsNone(manual.jacobian)
        price, model_greeks, _ = price_and_greeks(manual, call(self.eq), self.mkt, self.eq)
        self.assertGreater(price, 0.0)
        self.assertGreater(model_greeks["v0"], 0.0)                              # more variance, dearer call
        with self.assertRaises(ValueError):
            tq.par_greeks(manual, model_greeks)
        with self.assertRaises(ValueError):
            tq.HestonModel(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-1.0)
        other = tq.HestonModel(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.6)
        other.ticker = "TICKER2"
        with self.assertRaises(ValueError):
            tq.VanillaMCPricer(model=other, **PATHS).price(call(self.eq), self.mkt)
        for bad in ("heston", manual.process(self.eq)):
            with self.assertRaises(TypeError):
                tq.VanillaMCPricer(model=bad)


class TestDisplacedDiffusionInstance(unittest.TestCase):
    @slow
    def test_par_greeks_equal_greeks_through_the_calibration(self):
        mkt, eq = market()
        model = tq.DisplacedDiffusionModel.calibrate(eq)
        p_cls, _, g_cls = price_and_greeks(tq.DisplacedDiffusionModel, call(eq), mkt, eq)
        p_ins, model_greeks, direct = price_and_greeks(model, call(eq), mkt, eq)
        self.assertAlmostEqual(p_ins / p_cls, 1.0, places=12)
        assert_greeks_close(self, g_cls, tq.par_greeks(model, model_greeks, direct), rtol=1e-6, atol=1e-8)
        back = tq.DisplacedDiffusionModel.from_frames(model.to_frames())
        np.testing.assert_array_equal(back.parameters["beta_fwd"].numpy(), model.parameters["beta_fwd"].numpy())
        self.assertEqual(back.settings, model.settings)


class TestHestonSLVInstance(unittest.TestCase):
    @slow
    def test_instance_equals_class(self):
        """Heston held in both paths, the same E[v | X] rows (calibrated up to 5Y for the instance, up to the expiry
        inside the tape for the class): same price; the class greeks (one pass through the leverage calibration) =
        the instance's direct greeks + dV/dE mapped by the adjoint (par_greeks with the equity); the instance
        round-trips through its tables."""
        mkt, eq = market()
        model = tq.HestonSLVModel.calibrate(eq)
        self.assertEqual(list(model.parameters), ["expectation"])
        p_cls, _, g_cls = price_and_greeks(tq.HestonSLVModel, call(eq), mkt, eq)
        p_ins, model_greeks, direct = price_and_greeks(model, call(eq), mkt, eq)
        self.assertAlmostEqual(p_ins / p_cls, 1.0, places=12)
        assert_greeks_close(self, g_cls, tq.par_greeks(model, model_greeks, direct, equity=eq), rtol=1e-6, atol=1e-8)
        with self.assertRaises(ValueError):
            tq.par_greeks(model, model_greeks, direct)                        # the adjoint needs the market
        back = tq.HestonSLVModel.from_frames(model.to_frames())
        self.assertEqual(back.settings, model.settings)
        self.assertEqual(float(back.heston.parameters["xi"]), float(model.heston.parameters["xi"]))
        self.assertEqual(back.local_vol.settings, model.local_vol.settings)
        np.testing.assert_array_equal(back.parameters["expectation"].numpy(), model.parameters["expectation"].numpy())
        np.testing.assert_array_equal(back.leverage_table["nodes"], model.leverage_table["nodes"])


if __name__ == "__main__":
    unittest.main()
