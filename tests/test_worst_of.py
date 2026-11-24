"""Worst-of autocallable: several underlyings simulated jointly (MultiAssetProcess), levels on min_i S_i / K_i."""

import unittest
from datetime import date

import numpy as np
import tensorflow as tf

import tensorquant as tq
from markers import slow
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

EVAL_DATE = date(2026, 5, 15)
S_A, S_B, VOL_A, VOL_B, RHO = 100.0, 50.0, 0.25, 0.30, 0.6
R_EUR, REPO, FX_VOL, EURUSD, RHO_X = 0.02, 0.01, 0.10, 1.10, 0.3


def flat_curve(rate):
    return tq.RateCurve(EVAL_DATE, [1.0, 5.0, 10.0], [rate] * 3, "LINEAR", DayCounterConvention.Actual365)


def flat_surface(vol, spot):
    strikes = [spot * m for m in (0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0)]
    maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    return VolatilitySurface(reference_date=EVAL_DATE, calendar=None,
                             daycounter=DayCounter(DayCounterConvention.Actual365), strike=strikes,
                             maturity=maturities, volatility_matrix=[[vol] * len(strikes) for _ in maturities])


def market(rho=RHO, b_ccy="EUR", b_repo=REPO) -> MarketEnvironment:
    """Two stocks A (EUR) and B (EUR or USD), flat vols, equal EUR and USD rates, EURUSD with a flat vol."""
    a, b = "EQ:EUR:A", f"EQ:{b_ccy}:B"
    corr = tq.CorrelationMatrix.from_pairs([(a, b, rho), (a, "FX:USD:EURUSD", 0.0), (b, "FX:USD:EURUSD", RHO_X)])
    return MarketEnvironment(market={
        "IR:EUR:ESTR:SPOT": flat_curve(R_EUR), "IR:USD:SOFR:SPOT": flat_curve(R_EUR),
        "EQ:EUR:A:SPOT": S_A, "EQ:EUR:A:REPO": REPO, "EQ:EUR:A:VOL": flat_surface(VOL_A, S_A),
        f"{b}:SPOT": S_B, f"{b}:REPO": b_repo, f"{b}:VOL": flat_surface(VOL_B, S_B),
        "FX:USD:EURUSD:SPOT": EURUSD, "FX:USD:EURUSD:VOL": flat_surface(FX_VOL, EURUSD),
        "CORR:ALL:GLOBAL:SPOT": corr,
    })


def worst_of(underlying=("A", "B"), strike=(S_A, S_B), strike_date=None, start=EVAL_DATE, memory=False):
    cal = tq.TARGET()
    mf = tq.BusinessDayConvention.ModifiedFollowing
    end = cal.advance(start, 3, tq.TimeUnit.Years, mf)
    fix = tq.ScheduleGenerator(cal, mf).generate(start, end, 6, tq.TimeUnit.Months)[1:]
    return tq.AutocallableOption(
        ccy=tq.Currency.EUR, notional=1.0, start_date=EVAL_DATE, end_date=end,
        strike=None if strike is None else list(strike), strike_date=strike_date,
        coupon_fixing_dates=fix, coupon_payment_dates=fix, coupon_rates=[3.0], coupon_barriers=[80.0], memory=memory,
        autocall_fixing_dates=fix[1:], autocall_payment_dates=fix[1:], autocall_barrier=[100.0], payoff_barrier=70.0,
        underlying=list(underlying))


def numpy_payoff(product, performance, column, curve):
    """Independent hard-barrier replica (numpy) on the worst performance ``[n_paths, n_fixings]``."""
    n = performance.shape[0]
    alive, unpaid = np.ones(n, bool), np.zeros(n)
    coupons, put = np.zeros(n), np.zeros(n)
    for d in sorted(column):
        p = performance[:, column[d]]
        if d in product.coupon_fixing_dates:
            i = product.coupon_fixing_dates.index(d)
            above = p >= product.coupon_barriers[i] / 100
            paid = alive & above
            amount = product.coupon_rates[i] / 100 * ((1 + unpaid) if product.memory else 1.0)
            df = float(curve.discount(product.coupon_payment_dates[i]))
            coupons += np.where(paid, amount, 0.0) * df
            if d == product.coupon_fixing_dates[-1]:
                ki = alive & (p < product.payoff_barrier / 100)
                put += np.where(ki, product.payoff_participation * np.maximum(product.put_strike / 100 - p, 0), 0) * df
            unpaid = np.where(paid, 0.0, np.where(alive & ~above, unpaid + 1, unpaid))
        if d in product.autocall_fixing_dates:
            alive &= ~(p >= product.autocall_barrier[product.autocall_fixing_dates.index(d)] / 100)
    return coupons, put


class WorstOfCase(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = market()

    def pricer(self, model=tq.GeometricBrownianMotion, eps=0.0, n_paths=2_000, seed=7):
        return tq.AutocallableMCPricer(model=model, n_paths=n_paths, seed=seed, barrier_smoothing=eps,
                                       max_time_step=1 / 12)


class TestProduct(WorstOfCase):
    def test_flags_and_validation(self):
        prod = worst_of()
        self.assertTrue(prod.is_worst_of)
        self.assertEqual(prod.underlyings, ["A", "B"])
        self.assertEqual(prod.strikes, [S_A, S_B])
        self.assertFalse(worst_of(underlying=["A"], strike=[S_A]).is_worst_of)
        with self.assertRaises(ValueError):
            worst_of(underlying=["A", "A"])
        with self.assertRaises(ValueError):
            worst_of(strike=[S_A])


class TestPaths(WorstOfCase):
    def test_hard_payoff_equals_numpy_replica(self):
        """Spot start and forward start (each strike fixed on its own path at the strike date)."""
        start = date(2026, 11, 16)
        for prod in (worst_of(), worst_of(strike=None, strike_date=start, start=start), worst_of(memory=True)):
            pricer = self.pricer()
            pricer.price(prod, self.mkt)
            paths = pricer.paths.numpy()
            column = {d: i for i, d in enumerate(pricer.date_grid)}
            strikes = (paths[:, column[prod.strike_date], :] if prod.strike is None
                       else np.array(prod.strikes)[None, :])
            performance = (paths / strikes[:, None, :]).min(axis=2)
            np.testing.assert_allclose(pricer.performance.numpy(), performance, rtol=1e-12)
            coupons, put = numpy_payoff(prod, performance, column, self.mkt.get_ir_curve(tq.Currency.EUR))
            with self.subTest(forward_start=prod.strike is None, memory=prod.memory):
                self.assertAlmostEqual(float(prod.price), float(np.mean(coupons - put)), places=12)

    def test_correlation_and_marginals(self):
        """GBM and local vol (flat smile): each underlying's mean = its forward within 3 SE at the last fixing, the
        log returns correlated as in the market."""
        for model in (tq.GeometricBrownianMotion, tq.LocalVolatilityModel):
            pricer = self.pricer(model=model, n_paths=4_000)
            pricer.price(worst_of(), self.mkt)
            paths, t = pricer.paths.numpy(), pricer.time_grid
            for i, (ticker, vol) in enumerate((("A", VOL_A), ("B", VOL_B))):
                s_t = paths[:, -1, i]
                forward = float(self.mkt.get_equity(ticker).forward(pricer.date_grid[-1]))
                with self.subTest(model=model.__name__, ticker=ticker):
                    self.assertLess(abs(s_t.mean() - forward), 3 * s_t.std() / np.sqrt(len(s_t)))
            rho = np.corrcoef(np.log(paths[:, 0, 0]), np.log(paths[:, 0, 1]))[0, 1]
            with self.subTest(model=model.__name__, what="correlation"):
                self.assertAlmostEqual(rho, RHO, delta=0.04)

    def test_heston_rejected(self):
        with self.assertRaises(ValueError):
            self.pricer(model=tq.HestonModel).price(worst_of(), self.mkt)


class TestPrice(WorstOfCase):
    def test_correlation_ordering_and_single_limit(self):
        """Investor's side: lower correlation, lower worst performance, lower price (same draws); the worst-of is
        below the single-underlying products on A and B."""
        prices = []
        for rho in (0.2, 0.9):
            prod = worst_of()
            self.pricer(eps=0.01).price(prod, market(rho=rho))
            prices.append(float(prod.price))
        self.assertLess(prices[0], prices[1])
        for ticker, spot in (("A", S_A), ("B", S_B)):
            single = worst_of()
            single = tq.AutocallableOption(
                ccy=tq.Currency.EUR, notional=1.0, start_date=EVAL_DATE, end_date=single.end_date, strike=spot,
                coupon_fixing_dates=single.coupon_fixing_dates, coupon_payment_dates=single.coupon_payment_dates,
                coupon_rates=[3.0], coupon_barriers=[80.0], memory=False,
                autocall_fixing_dates=single.autocall_fixing_dates,
                autocall_payment_dates=single.autocall_payment_dates, autocall_barrier=[100.0], payoff_barrier=70.0,
                underlying=ticker)
            self.pricer(eps=0.01).price(single, market(rho=0.9))
            with self.subTest(ticker=ticker):
                self.assertLess(prices[1], float(single.price))

    def test_quanto_underlying_is_a_repo_shift(self):
        """GBM, equal rates: B in USD paid in EUR (quanto) = B in EUR with the repo shifted by rho_X sigma_B sigma_X,
        path by path (same draws)."""
        mkt = market(b_ccy="USD")
        rho_q, sigma_x = mkt.get_quanto("B", tq.Currency.USD, tq.Currency.EUR, 1.0)
        shifted = market(b_ccy="EUR", b_repo=REPO + float(rho_q) * VOL_B * float(sigma_x))
        quanto, plain = worst_of(), worst_of()
        self.pricer(eps=0.01).price(quanto, mkt)
        self.pricer(eps=0.01).price(plain, shifted)
        self.assertAlmostEqual(float(quanto.price), float(plain.price), places=10)
        self.assertEqual(set(quanto.quanto), {"B"})


class TestGreeks(WorstOfCase):
    def test_delta_and_correlation_vs_bumps(self):
        """GBM, smoothed barriers: AAD delta per underlying and the correlation sensitivity (both entries of the
        pair) vs central common-random-numbers bumps."""
        prod = worst_of()
        pricer = self.pricer(eps=0.02, n_paths=4_000)
        pricer.price(prod, self.mkt, autodiff=True)
        factors = self.mkt.risk_factors()
        keys = ["EQ:EUR:A:SPOT", "EQ:EUR:B:SPOT", "CORR:ALL:GLOBAL:SPOT"]
        grads = pricer.tape.gradient(prod.price, {k: factors[k] for k in keys})
        corr = self.mkt.get_correlation_matrix()
        i, j = corr.names.index("EQ:EUR:A"), corr.names.index("EQ:EUR:B")
        g_corr = grads["CORR:ALL:GLOBAL:SPOT"][0].numpy()
        aad = {"A": float(grads["EQ:EUR:A:SPOT"][0]), "B": float(grads["EQ:EUR:B:SPOT"][0]),
               "rho": g_corr[i, j] + g_corr[j, i]}

        def reprice(mkt):
            p = worst_of()
            self.pricer(eps=0.02, n_paths=4_000).price(p, mkt)
            return float(p.price)

        bumps = {}
        for ticker, spot in (("A", S_A), ("B", S_B)):
            h = 0.01 * spot
            up, dn = market(), market()
            var = lambda m: m.risk_factors()[f"EQ:EUR:{ticker}:SPOT"][0]
            var(up).assign(spot + h)
            var(dn).assign(spot - h)
            bumps[ticker] = (reprice(up) - reprice(dn)) / (2 * h)
        bumps["rho"] = (reprice(market(rho=RHO + 0.01)) - reprice(market(rho=RHO - 0.01))) / 0.02
        for k in bumps:
            with self.subTest(greek=k):
                self.assertAlmostEqual(aad[k], bumps[k], delta=0.03 * abs(bumps[k]) + 1e-6)


@slow
class TestLocalVol(WorstOfCase):
    def test_local_vol_flat_smile_close_to_gbm(self):
        """Flat surfaces: local vol = GBM within 3 SE."""
        prices = {}
        for model in (tq.GeometricBrownianMotion, tq.LocalVolatilityModel):
            prod = worst_of()
            self.pricer(model=model, eps=0.01, n_paths=4_000, seed=11).price(prod, self.mkt)
            prices[model] = (float(prod.price), float(prod.standard_error))
        (gbm, se_g), (lv, se_l) = prices.values()
        self.assertLess(abs(gbm - lv), 3 * np.hypot(se_g, se_l))


if __name__ == "__main__":
    unittest.main()
