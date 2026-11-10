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
    # repinned when the leverage moved from the pricing paths to a calibrated E[v | X] table, price on the
    # Richardson-corrected table, greeks on the 41-node one (was 19.463165927289, 0.537332104045, 93.640036276162,
    # 143.978074799629, 54.926747379648, -6.975946325967)
    'vanilla_slv': [19.601883807291, 0.539073753733, 93.490532242752, 147.74109398196, 60.420870026788, -7.682180122143],
    'autocall_lv': [0.626508884383, 0.185697085104, -60.657136253783, 233.925991066973, -0.879028842357, 0.286760035786],
    'autocall_heston': [1.361782349158, 0.141159431188, -54.125855548605, 146.705039792001, -9.183400682061, 1.238948382337],
    'heston_params': [0.059312323705, 0.722280684703, 0.039834245025, 0.451180868984, -0.742564800306],
    'dd_params': [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.228204339385, 0.222918221022, 0.208786206801,
                  0.194924044653, 0.180890185724, 0.170470483952, 0.157306403068, 0.136040150227],
}


# Calibration Jacobians {parameter|market key|variable index: [sum, abs sum]}, pinned before the implicit-function
# derivatives were precomputed (custom gradient): Heston on market(), displaced diffusion on dd_noisy().
HESTON_JACOBIAN = {'kappa|EQ:EUR:TICKER1:DIV|0': [0.0518485695373, 0.0767859529302],
 'kappa|EQ:EUR:TICKER1:REPO|0': [1.84993776961, 1.84993776961],
 'kappa|EQ:EUR:TICKER1:SPOT|0': [-0.0313719837331, 0.0313719837331],
 'kappa|EQ:EUR:TICKER1:VOL|0': [-0.791175794292, 120.90342532],
 'kappa|IR:EUR:ESTR:SPOT|0': [-0.814879980787, 0.814879980787],
 'kappa|IR:EUR:ESTR:SPOT|1': [-1.0820613651, 1.0820613651],
 'kappa|IR:EUR:ESTR:SPOT|10': [0.0, 0.0],
 'kappa|IR:EUR:ESTR:SPOT|11': [0.0, 0.0],
 'kappa|IR:EUR:ESTR:SPOT|2': [-0.0352431973939, 0.0352431973939],
 'kappa|IR:EUR:ESTR:SPOT|3': [5.62486907437, 5.62486907437],
 'kappa|IR:EUR:ESTR:SPOT|4': [7.20728230725, 7.20728230725],
 'kappa|IR:EUR:ESTR:SPOT|5': [0.409152098437, 0.409152098437],
 'kappa|IR:EUR:ESTR:SPOT|6': [-13.6978616758, 13.6978616758],
 'kappa|IR:EUR:ESTR:SPOT|7': [0.0, 0.0],
 'kappa|IR:EUR:ESTR:SPOT|8': [0.0, 0.0],
 'kappa|IR:EUR:ESTR:SPOT|9': [0.0, 0.0],
 'rho|EQ:EUR:TICKER1:DIV|0': [-0.00480796321857, 0.00480796321857],
 'rho|EQ:EUR:TICKER1:REPO|0': [-1.22514708667, 1.22514708667],
 'rho|EQ:EUR:TICKER1:SPOT|0': [0.00127280383844, 0.00127280383844],
 'rho|EQ:EUR:TICKER1:VOL|0': [0.454685560662, 20.3408339679],
 'rho|IR:EUR:ESTR:SPOT|0': [-0.0164081250959, 0.0164081250959],
 'rho|IR:EUR:ESTR:SPOT|1': [-0.00316022231423, 0.00316022231423],
 'rho|IR:EUR:ESTR:SPOT|10': [0.0, 0.0],
 'rho|IR:EUR:ESTR:SPOT|11': [0.0, 0.0],
 'rho|IR:EUR:ESTR:SPOT|2': [0.0707059950506, 0.0707059950506],
 'rho|IR:EUR:ESTR:SPOT|3': [0.141683236647, 0.141683236647],
 'rho|IR:EUR:ESTR:SPOT|4': [0.0566119480366, 0.0566119480366],
 'rho|IR:EUR:ESTR:SPOT|5': [0.0757720204692, 0.0757720204692],
 'rho|IR:EUR:ESTR:SPOT|6': [0.950292712533, 0.950292712533],
 'rho|IR:EUR:ESTR:SPOT|7': [0.0, 0.0],
 'rho|IR:EUR:ESTR:SPOT|8': [0.0, 0.0],
 'rho|IR:EUR:ESTR:SPOT|9': [0.0, 0.0],
 'theta|EQ:EUR:TICKER1:DIV|0': [0.00117732880114, 0.00120656480183],
 'theta|EQ:EUR:TICKER1:REPO|0': [0.345544972678, 0.345544972678],
 'theta|EQ:EUR:TICKER1:SPOT|0': [1.46553840481e-05, 1.46553840481e-05],
 'theta|EQ:EUR:TICKER1:VOL|0': [0.574235454717, 1.58080888852],
 'theta|IR:EUR:ESTR:SPOT|0': [0.00241288511695, 0.00241288511695],
 'theta|IR:EUR:ESTR:SPOT|1': [0.0112874452804, 0.0112874452804],
 'theta|IR:EUR:ESTR:SPOT|10': [0.0, 0.0],
 'theta|IR:EUR:ESTR:SPOT|11': [0.0, 0.0],
 'theta|IR:EUR:ESTR:SPOT|2': [0.0180541367799, 0.0180541367799],
 'theta|IR:EUR:ESTR:SPOT|3': [0.00239828259552, 0.00239828259552],
 'theta|IR:EUR:ESTR:SPOT|4': [-0.0521042769651, 0.0521042769651],
 'theta|IR:EUR:ESTR:SPOT|5': [-0.109908007925, 0.109908007925],
 'theta|IR:EUR:ESTR:SPOT|6': [-0.232476771039, 0.232476771039],
 'theta|IR:EUR:ESTR:SPOT|7': [0.0, 0.0],
 'theta|IR:EUR:ESTR:SPOT|8': [0.0, 0.0],
 'theta|IR:EUR:ESTR:SPOT|9': [0.0, 0.0],
 'v0|EQ:EUR:TICKER1:DIV|0': [0.0014042935329, 0.00163273114708],
 'v0|EQ:EUR:TICKER1:REPO|0': [0.0540800985791, 0.0540800985791],
 'v0|EQ:EUR:TICKER1:SPOT|0': [-0.00141059472008, 0.00141059472008],
 'v0|EQ:EUR:TICKER1:VOL|0': [0.529362575302, 1.19558297119],
 'v0|IR:EUR:ESTR:SPOT|0': [-0.0165749683094, 0.0165749683094],
 'v0|IR:EUR:ESTR:SPOT|1': [-0.0308434888502, 0.0308434888502],
 'v0|IR:EUR:ESTR:SPOT|10': [0.0, 0.0],
 'v0|IR:EUR:ESTR:SPOT|11': [0.0, 0.0],
 'v0|IR:EUR:ESTR:SPOT|2': [-0.0293254584267, 0.0293254584267],
 'v0|IR:EUR:ESTR:SPOT|3': [0.0118468375325, 0.0118468375325],
 'v0|IR:EUR:ESTR:SPOT|4': [0.0455302257328, 0.0455302257328],
 'v0|IR:EUR:ESTR:SPOT|5': [0.0190009920942, 0.0190009920942],
 'v0|IR:EUR:ESTR:SPOT|6': [-0.0560824717307, 0.0560824717307],
 'v0|IR:EUR:ESTR:SPOT|7': [0.0, 0.0],
 'v0|IR:EUR:ESTR:SPOT|8': [0.0, 0.0],
 'v0|IR:EUR:ESTR:SPOT|9': [0.0, 0.0],
 'xi|EQ:EUR:TICKER1:DIV|0': [0.0159296459124, 0.0222732110813],
 'xi|EQ:EUR:TICKER1:REPO|0': [0.433866669168, 0.433866669168],
 'xi|EQ:EUR:TICKER1:SPOT|0': [-0.0105325092038, 0.0105325092038],
 'xi|EQ:EUR:TICKER1:VOL|0': [2.63995000325, 37.7088132536],
 'xi|IR:EUR:ESTR:SPOT|0': [-0.270498388818, 0.270498388818],
 'xi|IR:EUR:ESTR:SPOT|1': [-0.31507641026, 0.31507641026],
 'xi|IR:EUR:ESTR:SPOT|10': [0.0, 0.0],
 'xi|IR:EUR:ESTR:SPOT|11': [0.0, 0.0],
 'xi|IR:EUR:ESTR:SPOT|2': [0.0524656030225, 0.0524656030225],
 'xi|IR:EUR:ESTR:SPOT|3': [1.62492954039, 1.62492954039],
 'xi|IR:EUR:ESTR:SPOT|4': [1.8605515509, 1.8605515509],
 'xi|IR:EUR:ESTR:SPOT|5': [-0.0481483061791, 0.0481483061791],
 'xi|IR:EUR:ESTR:SPOT|6': [-3.47899422419, 3.47899422419],
 'xi|IR:EUR:ESTR:SPOT|7': [0.0, 0.0],
 'xi|IR:EUR:ESTR:SPOT|8': [0.0, 0.0],
 'xi|IR:EUR:ESTR:SPOT|9': [0.0, 0.0]}
DD_JACOBIAN = {'beta_fwd|EQ:EUR:DD:DIVYIELD|0': [0.220125193507, 2.15187764738],
 'beta_fwd|EQ:EUR:DD:REPO|0': [0.220125193507, 2.15187764738],
 'beta_fwd|EQ:EUR:DD:SPOT|0': [0.0024415657295, 0.0341732771151],
 'beta_fwd|EQ:EUR:DD:VOL|0': [8.41363593393, 247.85573925],
 'beta_fwd|IR:EUR:ESTR:SPOT|0': [0.330417559635, 1.13158585731],
 'beta_fwd|IR:EUR:ESTR:SPOT|1': [-0.282814339498, 1.28802020372],
 'beta_fwd|IR:EUR:ESTR:SPOT|2': [-0.267728413645, 0.267728413645],
 'beta_fwd|IR:EUR:ESTR:SPOT|3': [0.0, 0.0],
 'sigma_fwd|EQ:EUR:DD:DIVYIELD|0': [0.332978393147, 0.332978393147],
 'sigma_fwd|EQ:EUR:DD:REPO|0': [0.332978393147, 0.332978393147],
 'sigma_fwd|EQ:EUR:DD:SPOT|0': [-0.0017728596712, 0.0017728596712],
 'sigma_fwd|EQ:EUR:DD:VOL|0': [2.99971340786, 7.97888097373],
 'sigma_fwd|IR:EUR:ESTR:SPOT|0': [0.00386453327234, 0.0382748939448],
 'sigma_fwd|IR:EUR:ESTR:SPOT|1': [-0.0331871409101, 0.119906483024],
 'sigma_fwd|IR:EUR:ESTR:SPOT|2': [-0.303655785509, 0.303655785509],
 'sigma_fwd|IR:EUR:ESTR:SPOT|3': [0.0, 0.0]}


class TestPins(unittest.TestCase):
    def setUp(self):
        self.mkt, self.eq = market()

    def check(self, key, values, rtol=RTOL):
        np.testing.assert_allclose(values, PINS[key], rtol=rtol, atol=1e-12, err_msg=key)

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
        # the particle calibration amplifies rounding: a 1e-15 relative change of the Heston parameters (eager vs
        # compiled implicit step) moved these numbers by ~4e-7, so 1e-6 here
        self.check("vanilla_slv", self.vanilla(tq.HestonSLVModel), rtol=1e-6)


def dd_noisy():
    """Displaced diffusion market (test_displaced_diffusion) with a deterministic +-0.4 vol pt pattern: interior
    betas and non-zero residuals (the residual terms of the implicit derivative count)."""
    from test_displaced_diffusion import dd_market
    mkt = dd_market()
    vol = mkt.get_equity("DD").vol_surface.volatility_matrix
    n, m = vol.shape
    vol.assign(vol.numpy() + 0.004 * np.sin(np.arange(n * m).reshape(n, m)))
    return mkt, mkt.get_equity("DD")


def jacobian_summary(model) -> dict:
    out = {}
    for name, per_key in model.jacobian.items():
        for key, blocks in per_key.items():
            for j, b in enumerate(blocks):
                out[f"{name}|{key}|{j}"] = [float(np.sum(b)), float(np.abs(b).sum())]
    return out


class TestCalibrationJacobianPins(unittest.TestCase):
    """The stored calibration Jacobians dp/d(market) (sum and abs sum per parameter, key and variable)."""

    def check(self, model, pins):
        got = jacobian_summary(model)
        self.assertEqual(set(got), set(pins))
        for key, values in pins.items():
            np.testing.assert_allclose(got[key], values, rtol=RTOL, atol=1e-10, err_msg=key)

    @slow
    def test_heston(self):
        self.check(tq.HestonModel.calibrate(market()[1]), HESTON_JACOBIAN)

    def test_displaced_diffusion(self):
        self.check(tq.DisplacedDiffusionModel.calibrate(dd_noisy()[1]), DD_JACOBIAN)


if __name__ == "__main__":
    unittest.main()
