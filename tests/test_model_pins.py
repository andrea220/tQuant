"""Pinned prices, greeks and calibrated parameters of the equity Monte Carlo models (regression safety).

Numbers computed with the code before the model / process split (models as instances, ``model.process(equity)``):
the Monte Carlo pricers called with a model **class** must keep giving them. Few paths and steps: these are
regression pins, not accuracy checks.
"""

import unittest
from datetime import date

import numpy as np
import tensorflow as tf

import tensorquant as tq

from markers import slow

EVAL_DATE = date(2026, 5, 15)
SMALL = (("EUR", 1), ("GBP", 1), ("USD", 1))
RTOL = 1e-8


def market():
    tq.Settings.evaluation_date = EVAL_DATE
    mkt = tq.generate_market(EVAL_DATE, seed=7, tickers_per_currency=SMALL)
    return mkt, mkt.get_equity("TICKER1")


def call(eq):
    """1Y call struck at the 1Y forward."""
    expiry = date(2027, 5, 14)
    return tq.VanillaOption(eq.currency, start_date=EVAL_DATE, end_date=expiry, option_type=tq.OptionType.Call,
                            strike=round(float(eq.forward(expiry)), 2), underlying="TICKER1")


def autocallable(eq):
    """2Y phoenix, semi-annual coupons and autocalls from 1Y, put barrier 80%."""
    cal = tq.TARGET()
    mf = tq.BusinessDayConvention.ModifiedFollowing
    sg = tq.ScheduleGenerator(cal, mf)
    end = cal.advance(EVAL_DATE, 2, tq.TimeUnit.Years, mf)
    cfix = sg.generate(EVAL_DATE, end, 6, tq.TimeUnit.Months)[1:]
    afix = sg.generate(cal.advance(EVAL_DATE, 1, tq.TimeUnit.Years, mf), end, 6, tq.TimeUnit.Months)
    return tq.AutocallableOption(
        ccy=eq.currency, notional=100.0, start_date=EVAL_DATE, end_date=end, strike=float(eq.spot),
        coupon_fixing_dates=cfix, coupon_payment_dates=cfix, coupon_rates=[4.0], coupon_barriers=[80.0], memory=False,
        autocall_fixing_dates=afix, autocall_payment_dates=afix, autocall_barrier=[100.0], payoff_barrier=80.0,
        underlying="TICKER1")


def greeks(pricer, product, mkt, eq):
    """``[price, delta, vega sum, vega |sum|, vega at (1Y, ATM), vega at (6M, 80%)]`` (vega per unit vol)."""
    pricer.price(product, mkt, autodiff=True)
    delta, vega = pricer.tape.gradient(product.price, [eq.spot, eq.vol_surface.volatility_matrix])
    v = vega.numpy()
    return [float(product.price), float(delta), float(v.sum()), float(np.abs(v).sum()), float(v[3, 4]), float(v[2, 2])]


PINS = {
    'vanilla_lv': [20.082009562527, 0.515926372727, 97.166309277369, 128.417816625189, 78.475094840817, -1.465142786815],
    'vanilla_heston': [19.098402190726, 0.558591457995, 91.248605600356, 183.967403910103, 13.502593464935, -2.386649205459],
    'vanilla_dd': [19.89005568691, 0.475998424297, 94.637810614771, 96.662994343986, 26.590914145174, -0.061308643417],
    'vanilla_slv': [19.463165927289, 0.537332104045, 93.640036276162, 143.978074799629, 54.926747379648, -6.975946325967],
    'autocall_lv': [0.626508884383, 0.185697085104, -60.657136253783, 233.925991066973, -0.879028842357, 0.286760035786],
    'autocall_heston': [1.361782349158, 0.141159431188, -54.125855548605, 146.705039792001, -9.183400682061, 1.238948382337],
    'heston_params': [0.059312323705, 0.722280684703, 0.039834245025, 0.451180868984, -0.742564800306],
    'dd_params': [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.228204339385, 0.222918221022, 0.208786206801,
                  0.194924044653, 0.180890185724, 0.170470483952, 0.157306403068, 0.136040150227],
}


class TestPins(unittest.TestCase):
    def setUp(self):
        self.mkt, self.eq = market()

    def check(self, key, values):
        np.testing.assert_allclose(values, PINS[key], rtol=RTOL, atol=1e-12, err_msg=key)

    def vanilla(self, model):
        return greeks(tq.VanillaMCPricer(model=model, n_paths=4_000, n_steps=24, seed=3), call(self.eq), self.mkt, self.eq)

    def autocall(self, model):
        return greeks(tq.AutocallableMCPricer(model=model, n_paths=2_000, seed=5, max_time_step=1 / 12),
                      autocallable(self.eq), self.mkt, self.eq)

    def test_local_vol(self):
        self.check("vanilla_lv", self.vanilla(tq.LocalVolatilityModel))
        self.check("autocall_lv", self.autocall(tq.LocalVolatilityModel))

    @slow
    def test_heston(self):
        self.check("heston_params", [float(v) for v in tq.HestonModel.from_equity(self.eq).parameters.values()])
        self.check("vanilla_heston", self.vanilla(tq.HestonModel))
        self.check("autocall_heston", self.autocall(tq.HestonModel))

    @slow
    def test_displaced_diffusion(self):
        dd = tq.DisplacedDiffusionModel.from_equity(self.eq)
        self.check("dd_params", np.concatenate([p.numpy() for p in dd.forward_parameters]))
        self.check("vanilla_dd", self.vanilla(tq.DisplacedDiffusionModel))

    @slow
    def test_heston_slv(self):
        self.check("vanilla_slv", self.vanilla(tq.HestonSLVModel))


if __name__ == "__main__":
    unittest.main()
