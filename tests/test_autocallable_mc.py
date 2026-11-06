"""AutocallableMCPricer: investor's side, model built from the market, TF payoff with smoothed digitals (AAD greeks)."""

import unittest
from datetime import date

import numpy as np
import tensorflow as tf

import tensorquant as tq
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

EVAL_DATE = date(2026, 5, 15)
RATE, VOL, S0 = 0.021, 0.25, 100.0
# Old pricer (GBM(mu=r, sigma=25%, x0=100) instance, ACT/365 grid), 4 seeds x 200k paths: coupons and put
OLD = {False: (0.101431, 0.089427), True: (0.127471, 0.089352)}


def flat_market(vol: float = VOL) -> MarketEnvironment:
    strikes = [40.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0]
    maturities = [0.5, 1.0, 2.0, 3.0, 5.0, 6.0]
    surface = VolatilitySurface(
        reference_date=EVAL_DATE, calendar=None, daycounter=DayCounter(DayCounterConvention.Actual365),
        strike=strikes, maturity=maturities, volatility_matrix=[[vol] * len(strikes) for _ in maturities],
    )
    return MarketEnvironment(market={
        "IR:EUR:ESTR:SPOT": tq.RateCurve(EVAL_DATE, [1.0, 5.0, 10.0], [RATE] * 3, "LINEAR", DayCounterConvention.Actual365),
        "EQ:EUR:UND:SPOT": S0,
        "EQ:EUR:UND:REPO": 0.0,
        "EQ:EUR:UND:VOL": surface,
    })


def make_autocallable(memory=False, coupon_barrier=80.0, autocall_barrier=100.0, payoff_barrier=80.0):
    cal = tq.TARGET()
    mf = tq.BusinessDayConvention.ModifiedFollowing
    sg = tq.ScheduleGenerator(cal, mf)
    end = cal.advance(EVAL_DATE, 5, tq.TimeUnit.Years, mf)
    cfix = sg.generate(EVAL_DATE, end, 6, tq.TimeUnit.Months)[1:]
    afix = sg.generate(cal.advance(EVAL_DATE, 1, tq.TimeUnit.Years, mf), end, 6, tq.TimeUnit.Months)
    return tq.AutocallableOption(
        ccy=tq.Currency.EUR, notional=1.0, start_date=EVAL_DATE, end_date=end, strike=S0,
        coupon_fixing_dates=cfix, coupon_payment_dates=cfix, coupon_rates=[4.0], coupon_barriers=[coupon_barrier],
        memory=memory, autocall_fixing_dates=afix, autocall_payment_dates=afix, autocall_barrier=[autocall_barrier],
        payoff_barrier=payoff_barrier, underlying="UND",
    )


def numpy_payoff(product, paths, column, fixings, curve):
    """Independent hard-barrier replica (numpy), investor's side: (coupons, put) per path."""
    s_all = paths.numpy()
    n = s_all.shape[0]
    K = product.strike
    alive, unpaid = np.ones(n, bool), np.zeros(n)
    coupons, put = np.zeros(n), np.zeros(n)
    for d in fixings:
        s = s_all[:, column[d]]
        if d in product.coupon_fixing_dates:
            i = product.coupon_fixing_dates.index(d)
            above = s >= product.coupon_barriers[i] / 100 * K
            paid = alive & above
            amount = product.coupon_rates[i] / 100 * ((1 + unpaid) if product.memory else 1.0)
            df = float(curve.discount(product.coupon_payment_dates[i]))
            coupons += np.where(paid, amount, 0.0) * df
            if d == product.coupon_fixing_dates[-1]:
                ki = alive & (s < product.payoff_barrier / 100 * K)
                put += np.where(ki, np.maximum(K - s, 0) / K, 0.0) * df
            unpaid = np.where(paid, 0.0, np.where(alive & ~above, unpaid + 1, unpaid))
        if d in product.autocall_fixing_dates:
            j = product.autocall_fixing_dates.index(d)
            alive &= ~(s >= product.autocall_barrier[j] / 100 * K)
    return coupons, put


class ACTestCase(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = flat_market()

    def price(self, product, model=tq.GeometricBrownianMotion, eps=0.0, n_paths=50_000, seed=7, autodiff=False):
        pricer = tq.AutocallableMCPricer(model=model, n_paths=n_paths, seed=seed, barrier_smoothing=eps)
        pricer.price(product, self.mkt, autodiff=autodiff)
        return pricer


class TestHardPayoff(ACTestCase):
    def test_numpy_replica_on_the_same_paths(self):
        for memory in (False, True):
            prod = make_autocallable(memory=memory)
            pricer = self.price(prod)
            column = {d: int(np.searchsorted(pricer.time_grid, t)) for d, t in zip(
                pricer.date_grid, [DayCounter(DayCounterConvention.Actual365).year_fraction(EVAL_DATE, d)
                                   for d in pricer.date_grid])}
            c, p = numpy_payoff(prod, pricer.paths, column, pricer.date_grid, self.mkt.get_ir_curve(tq.Currency.EUR))
            with self.subTest(memory=memory):
                self.assertAlmostEqual(float(prod.coupon_pv), c.mean(), places=12)
                self.assertAlmostEqual(float(prod.put_pv), p.mean(), places=12)
                self.assertAlmostEqual(float(prod.price), c.mean() - p.mean(), places=12)

    def test_consistent_with_the_old_pricer(self):
        """Same GBM as the old pricer: coupons and put within 4 SE; the put now enters with a minus (investor)."""
        for memory, (old_c, old_p) in OLD.items():
            prod = make_autocallable(memory=memory)
            pricer = self.price(prod, n_paths=200_000)
            se = float(prod.standard_error)
            with self.subTest(memory=memory):
                self.assertAlmostEqual(float(prod.coupon_pv), old_c, delta=4 * se)
                self.assertAlmostEqual(float(prod.put_pv), old_p, delta=4 * se)
                self.assertAlmostEqual(float(prod.price), float(prod.coupon_pv) - float(prod.put_pv), places=12)

    def test_never_called_all_coupons_paid(self):
        """Coupon barrier 0, autocall unreachable: coupons = sum c P(0, pay_i) exactly."""
        prod = make_autocallable(coupon_barrier=0.0, autocall_barrier=1e6, payoff_barrier=0.0)
        self.price(prod, n_paths=1_000)
        curve = self.mkt.get_ir_curve(tq.Currency.EUR)
        expected = sum(0.04 * float(curve.discount(d)) for d in prod.coupon_payment_dates)
        self.assertAlmostEqual(float(prod.coupon_pv), expected, places=12)
        self.assertEqual(float(prod.put_pv), 0.0)

    def test_called_at_the_first_date(self):
        """Autocall barrier 0: called at the first autocall date, coupons up to it (barrier 0), no put."""
        prod = make_autocallable(coupon_barrier=0.0, autocall_barrier=0.0)
        self.price(prod, n_paths=1_000)
        curve = self.mkt.get_ir_curve(tq.Currency.EUR)
        first = prod.autocall_fixing_dates[0]
        expected = sum(0.04 * float(curve.discount(p))
                       for f, p in zip(prod.coupon_fixing_dates, prod.coupon_payment_dates) if f <= first)
        self.assertAlmostEqual(float(prod.coupon_pv), expected, places=12)
        self.assertEqual(float(prod.put_pv), 0.0)

    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            tq.AutocallableMCPricer(model=object())
        with self.assertRaises(ValueError):
            tq.AutocallableMCPricer(barrier_smoothing=-0.01)
        prod = make_autocallable()
        prod.underlying = None
        with self.assertRaises(ValueError):
            self.price(prod)


class TestSmoothing(ACTestCase):
    def test_smoothing_converges_to_the_hard_payoff(self):
        prod = make_autocallable()
        self.price(prod, eps=0.0)
        hard = float(prod.price)
        diffs = []
        for eps in (0.01, 0.001):
            p = make_autocallable()
            self.price(p, eps=eps)
            diffs.append(abs(float(p.price) - hard))
        self.assertLess(diffs[1], diffs[0])
        self.assertLess(diffs[1], 1e-3)
        self.assertLess(diffs[0], 5e-3)   # eps = 1%: bias well below 0.5% of notional

    def test_hard_payoff_has_no_delta_from_the_digitals(self):
        """eps = 0: AAD sees only the put (continuous part); eps = 1% gives the full delta."""
        spot = self.mkt.get_eq_spot("UND")
        deltas = {}
        for eps in (0.0, 0.01):
            prod = make_autocallable()
            pricer = self.price(prod, eps=eps, autodiff=True)
            deltas[eps] = float(pricer.tape.gradient(prod.price, spot))
        self.assertGreater(abs(deltas[0.01] - deltas[0.0]), 0.1 * abs(deltas[0.01]))


class TestGreeks(ACTestCase):
    def check_vs_bump(self, model, n_paths):
        prod = make_autocallable()
        pricer = self.price(prod, model=model, eps=0.01, n_paths=n_paths, autodiff=True)
        spot = self.mkt.get_eq_spot("UND")
        matrix = self.mkt.get_eq_vol_surface("UND").volatility_matrix
        d_spot, d_vol = pricer.tape.gradient(prod.price, [spot, matrix])

        def reprice():
            p = make_autocallable()
            self.price(p, model=model, eps=0.01, n_paths=n_paths)
            return float(p.price)

        h, out = 1e-3, []
        for x in (S0 + h, S0 - h):
            spot.assign(x)
            out.append(reprice())
        spot.assign(S0)
        self.assertAlmostEqual(float(d_spot), (out[0] - out[1]) / (2 * h), delta=2e-3 * abs(float(d_spot)) + 1e-6)
        vega = float(tf.reduce_sum(d_vol))
        base, out = matrix.numpy().copy(), []
        for sgn in (1, -1):
            matrix.assign(base + sgn * 1e-4)
            out.append(reprice())
        matrix.assign(base)
        self.assertAlmostEqual(vega, (out[0] - out[1]) / 2e-4, delta=5e-3 * abs(vega) + 1e-6)
        return float(d_spot), vega

    def test_gbm_greeks_vs_common_random_numbers_bumps(self):
        delta, vega = self.check_vs_bump(tq.GeometricBrownianMotion, 20_000)
        self.assertGreater(delta, 0.0)   # investor: short the put, long the coupons
        self.assertLess(vega, 0.0)

    def test_local_vol_default_greeks_vs_bumps(self):
        self.assertIs(tq.AutocallableMCPricer()._model, tq.LocalVolatilityModel)
        prod = make_autocallable()
        pricer = self.price(prod, model=tq.LocalVolatilityModel, eps=0.01, n_paths=5_000)
        self.assertEqual(prod.price.dtype, tf.float64)
        self.check_vs_bump(tq.LocalVolatilityModel, 5_000)

    def test_displaced_diffusion_flat_surface(self):
        """Flat smile: the displaced diffusion calibrates to beta = 1 (lognormal), the price is the GBM one within
        4 SE; delta and vega through the calibration vs common-random-numbers bumps."""
        model = tq.DisplacedDiffusionModel.from_equity(self.mkt.get_equity("UND"))
        np.testing.assert_allclose(model.forward_parameters[0].numpy(), 1.0, atol=1e-6)
        gbm, dd = make_autocallable(), make_autocallable()
        self.price(gbm, model=tq.GeometricBrownianMotion, n_paths=50_000)
        self.price(dd, model=tq.DisplacedDiffusionModel, n_paths=50_000)
        se = float(np.hypot(float(gbm.standard_error), float(dd.standard_error)))
        self.assertLess(abs(float(gbm.price) - float(dd.price)), 4 * se)
        self.check_vs_bump(tq.DisplacedDiffusionModel, 5_000)

    def test_heston_flat_surface(self):
        """Flat smile: Heston fits it with v0 = theta = vol^2 (vol of vol and mean reversion nearly degenerate),
        the price is the GBM one within 4 SE; the AAD greeks reach spot and vol nodes (finite)."""
        model = tq.HestonModel.from_equity(self.mkt.get_equity("UND"))
        self.assertAlmostEqual(float(model.parameters["v0"]) ** 0.5, VOL, delta=1e-3)
        self.assertAlmostEqual(float(model.parameters["theta"]) ** 0.5, VOL, delta=1e-3)
        self.assertLess(model.calibration.max_error_volpts.max(), 0.01)
        gbm, hes = make_autocallable(), make_autocallable()
        self.price(gbm, model=tq.GeometricBrownianMotion, n_paths=20_000)
        pricer = self.price(hes, model=tq.HestonModel, eps=0.0, n_paths=20_000, autodiff=True)
        se = float(np.hypot(float(gbm.standard_error), float(hes.standard_error)))
        self.assertLess(abs(float(gbm.price) - float(hes.price)), 4 * se)
        d_spot, d_vol = pricer.tape.gradient(hes.price, [self.mkt.get_eq_spot("UND"),
                                                         self.mkt.get_eq_vol_surface("UND").volatility_matrix])
        self.assertTrue(np.isfinite(float(d_spot)) and np.all(np.isfinite(d_vol.numpy())))

    def test_local_vol_flat_surface_close_to_gbm(self):
        """Flat surface: local vol = GBM up to MC noise and the proportional vs escrowed dividends (none here)."""
        gbm, lv = make_autocallable(), make_autocallable()
        self.price(gbm, model=tq.GeometricBrownianMotion, n_paths=50_000)
        self.price(lv, model=tq.LocalVolatilityModel, n_paths=50_000)
        se = float(np.hypot(float(gbm.standard_error), float(lv.standard_error)))
        self.assertLess(abs(float(gbm.price) - float(lv.price)), 4 * se)


if __name__ == "__main__":
    unittest.main()
