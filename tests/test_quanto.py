"""Quanto products: paid in a currency other than the underlying's at a fixed rate (payment_ccy)."""

import unittest
from datetime import date

import numpy as np
import tensorflow as tf
from scipy.stats import norm

import tensorquant as tq
from markers import slow
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

EVAL_DATE = date(2026, 5, 15)
EXPIRY = date(2027, 5, 14)
S0, VOL, FX_VOL, EURUSD, RHO = 100.0, 0.25, 0.10, 1.10, 0.4      # RHO: corr(underlying, EURUSD)
R_EUR, R_USD, REPO = 0.02, 0.04, 0.005


def flat_curve(rate):
    return tq.RateCurve(EVAL_DATE, [1.0, 5.0, 10.0], [rate] * 3, "LINEAR", DayCounterConvention.Actual365)


def flat_surface(vol, strikes, maturities=(0.25, 0.5, 1.0, 2.0, 3.0, 5.0)):
    return VolatilitySurface(reference_date=EVAL_DATE, calendar=None,
                             daycounter=DayCounter(DayCounterConvention.Actual365), strike=list(strikes),
                             maturity=list(maturities), volatility_matrix=[[vol] * len(strikes) for _ in maturities])


def market(rho=RHO, r_usd=R_USD, repo=REPO, fx_vol=FX_VOL) -> MarketEnvironment:
    """A USD underlying (UND) and a EUR one (EUND), flat vols, EURUSD with a flat vol, their correlations."""
    strikes = [40.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0]
    corr = tq.CorrelationMatrix.from_pairs([("EQ:USD:UND", "FX:USD:EURUSD", rho), ("EQ:EUR:EUND", "FX:USD:EURUSD", rho),
                                            ("EQ:USD:UND", "EQ:EUR:EUND", 0.0)])
    return MarketEnvironment(market={
        "IR:EUR:ESTR:SPOT": flat_curve(R_EUR), "IR:USD:SOFR:SPOT": flat_curve(r_usd),
        "EQ:USD:UND:SPOT": S0, "EQ:USD:UND:REPO": repo, "EQ:USD:UND:VOL": flat_surface(VOL, strikes),
        "EQ:EUR:EUND:SPOT": S0, "EQ:EUR:EUND:REPO": repo, "EQ:EUR:EUND:VOL": flat_surface(VOL, strikes),
        "FX:USD:EURUSD:SPOT": EURUSD, "FX:USD:EURUSD:VOL": flat_surface(fx_vol, [0.5, 1.0, 1.1, 1.5, 2.0]),
        "CORR:ALL:GLOBAL:SPOT": corr,
    })


def tenor(d=EXPIRY):
    return DayCounter(DayCounterConvention.Actual365).year_fraction(EVAL_DATE, d)


def black_put(forward, strike, vol, t, df):
    sd = vol * np.sqrt(t)
    d1 = np.log(forward / strike) / sd + 0.5 * sd
    return df * (strike * norm.cdf(-(d1 - sd)) - forward * norm.cdf(-d1))


def put(payment=tq.Currency.EUR, strike=95.0, ticker="UND", ccy=tq.Currency.USD, exercise=tq.ExerciseType.European):
    return tq.VanillaOption(ccy, EVAL_DATE, EXPIRY, tq.OptionType.Put, strike, underlying=ticker,
                            exercise_type=exercise, payment_ccy=payment)


class QuantoCase(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = market()


class TestQuantoInputs(QuantoCase):
    def test_direct_and_inverse_pair(self):
        """X = value of 1 unit of the underlying's currency in the payment one: EUR -> USD is EURUSD (rho as stored),
        USD -> EUR its inverse (opposite correlation, same vol)."""
        rho, sx = self.mkt.get_quanto("EUND", tq.Currency.EUR, tq.Currency.USD, 1.0)
        self.assertAlmostEqual(float(rho), RHO, places=12)
        self.assertAlmostEqual(float(sx), FX_VOL, places=12)
        rho, sx = self.mkt.get_quanto("UND", tq.Currency.USD, tq.Currency.EUR, 1.0)
        self.assertAlmostEqual(float(rho), -RHO, places=12)
        self.assertAlmostEqual(float(sx), FX_VOL, places=12)
        with self.assertRaises(ValueError):
            self.mkt.get_quanto("UND", tq.Currency.USD, tq.Currency.GBP, 1.0)

    def test_product_flags(self):
        self.assertFalse(put(payment=None).is_quanto)
        self.assertEqual(put(payment=None).payment_ccy, tq.Currency.USD)
        self.assertTrue(put().is_quanto)


class TestBlackScholesQuanto(QuantoCase):
    def price(self, product, mkt=None):
        pricer = tq.BlackScholesPricer(dividend_model="continuous")
        pricer.price(product, mkt or self.mkt, autodiff=True)
        return pricer

    def test_closed_form(self):
        """USD underlying paid in EUR: DF_EUR x Black(F e^{-rho_X sigma sigma_X T}), rho_X = -corr(S, EURUSD)."""
        t = tenor()
        prod = put()
        self.price(prod)
        forward = S0 * np.exp((R_USD - REPO) * t)
        quanto_forward = forward * np.exp(-(-RHO) * VOL * FX_VOL * t)
        expected = black_put(quanto_forward, 95.0, VOL, t, np.exp(-R_EUR * t))
        self.assertAlmostEqual(float(prod.price), expected, places=10)
        self.assertAlmostEqual(float(prod.forward), quanto_forward, places=10)

    def test_same_currency_and_zero_correlation(self):
        """Paid in its own currency: the plain price; zero correlation: the plain forward, discounted in EUR."""
        plain, own = put(payment=None), put(payment=tq.Currency.USD)
        self.price(plain)
        self.price(own)
        self.assertEqual(float(plain.price), float(own.price))
        t = tenor()
        prod = put()
        self.price(prod, market(rho=0.0))
        expected = black_put(S0 * np.exp((R_USD - REPO) * t), 95.0, VOL, t, np.exp(-R_EUR * t))
        self.assertAlmostEqual(float(prod.price), expected, places=10)

    def test_greeks_to_correlation_and_fx_vol(self):
        """AAD sensitivities to the correlation (both entries of the pair) and to the FX vol nodes = bumps."""
        prod = put()
        pricer = self.price(prod)
        corr = self.mkt.get_correlation_matrix()
        fx_surface = self.mkt.get_fx_vol_surface("EUR", "USD")
        g_corr, g_fx = pricer.tape.gradient(prod.price, [corr.variables[0], fx_surface.volatility_matrix])
        i, j = corr.names.index("EQ:USD:UND"), corr.names.index("FX:USD:EURUSD")
        aad_rho = float(g_corr.numpy()[i, j] + g_corr.numpy()[j, i])
        aad_fx = float(tf.reduce_sum(g_fx))
        h = 1e-5
        def bump(rho):
            bumped = put()
            self.price(bumped, market(rho=rho))
            return float(bumped.price)

        fd_rho = (bump(RHO + h) - bump(RHO - h)) / (2 * h)
        self.assertAlmostEqual(aad_rho, fd_rho, delta=1e-6 * max(1.0, abs(fd_rho)))
        base = float(prod.price)
        fx_surface.volatility_matrix.assign_add(tf.fill(fx_surface.volatility_matrix.shape, tf.constant(h, tf.float64)))
        up = put()
        self.price(up)
        fx_surface.volatility_matrix.assign_add(tf.fill(fx_surface.volatility_matrix.shape, tf.constant(-h, tf.float64)))
        self.assertAlmostEqual(aad_fx, (float(up.price) - base) / h, delta=1e-4 * max(1.0, abs(aad_fx)))
        self.assertNotEqual(aad_rho, 0.0)

    def test_american_quanto_rejected(self):
        with self.assertRaises(ValueError):
            self.price(put(exercise=tq.ExerciseType.American))


class TestMonteCarloQuanto(QuantoCase):
    def mc(self, model, n_paths=6_000):
        prod = put()
        tq.VanillaMCPricer(model=model, n_paths=n_paths, n_steps=24, seed=3).price(prod, self.mkt, autodiff=False)
        return prod

    def test_gbm_and_local_vol_vs_closed_form(self):
        """On a flat smile the GBM and the local vol quanto drift give the Black-Scholes quanto price within 3 SE."""
        bs = put()
        tq.BlackScholesPricer(dividend_model="continuous").price(bs, self.mkt, autodiff=False)
        for model in (tq.GeometricBrownianMotion, tq.LocalVolatilityModel):
            prod = self.mc(model)
            with self.subTest(model=model.__name__):
                self.assertLess(abs(float(prod.price) - float(bs.price)), 3.0 * float(prod.standard_error))
                self.assertIsNotNone(prod.quanto_correlation)

    def test_heston_rejected(self):
        with self.assertRaises(ValueError):
            self.mc(tq.HestonModel(v0=0.06, kappa=1.0, theta=0.06, xi=0.3, rho=-0.5), n_paths=100)


class TestAutocallableQuanto(QuantoCase):
    def autocallable(self, payment):
        cal = tq.TARGET()
        mf = tq.BusinessDayConvention.ModifiedFollowing
        sg = tq.ScheduleGenerator(cal, mf)
        end = cal.advance(EVAL_DATE, 3, tq.TimeUnit.Years, mf)
        fix = sg.generate(EVAL_DATE, end, 6, tq.TimeUnit.Months)[1:]
        return tq.AutocallableOption(
            ccy=tq.Currency.USD, notional=1.0, start_date=EVAL_DATE, end_date=end, strike=S0,
            coupon_fixing_dates=fix, coupon_payment_dates=fix, coupon_rates=[3.0], coupon_barriers=[80.0],
            memory=False, autocall_fixing_dates=fix[1:], autocall_payment_dates=fix[1:], autocall_barrier=[100.0],
            payoff_barrier=70.0, underlying="UND", payment_ccy=payment)

    def test_gbm_quanto_is_a_repo_shift(self):
        """GBM: the quanto drift is a carry shift rho_X sigma sigma_X; with equal rates in both currencies the quanto
        autocallable equals the plain one on a market with that repo, path by path (same draws)."""
        mkt = market(r_usd=R_EUR)
        rho, sx = mkt.get_quanto("UND", tq.Currency.USD, tq.Currency.EUR, 1.0)
        shifted = market(r_usd=R_EUR, repo=REPO + float(rho) * VOL * float(sx))
        quanto, plain = self.autocallable(tq.Currency.EUR), self.autocallable(None)
        pricer = lambda: tq.AutocallableMCPricer(model=tq.GeometricBrownianMotion, n_paths=2_000, seed=5)
        pricer().price(quanto, mkt, autodiff=False)
        pricer().price(plain, shifted, autodiff=False)
        self.assertAlmostEqual(float(quanto.price), float(plain.price), places=10)

    @slow
    def test_local_vol_quanto_moves_with_the_correlation(self):
        """Local vol: a positive correlation with X lowers the drift (fewer autocalls and coupons, more put): a lower
        price for the investor."""
        prices = []
        for rho in (-0.5, 0.5):
            prod = self.autocallable(tq.Currency.EUR)
            tq.AutocallableMCPricer(n_paths=4_000, seed=5).price(prod, market(rho=rho), autodiff=False)
            prices.append(float(prod.price))
        self.assertLess(prices[0], prices[1])             # rho(S, EURUSD) = -0.5 -> rho_X = +0.5: lower drift


class TestComposite(QuantoCase):
    """Composite: the payoff on the underlying in the payment currency, Y = S X (here a USD underlying in EUR)."""

    def composite_put(self, strike):
        return tq.VanillaOption(tq.Currency.USD, EVAL_DATE, EXPIRY, tq.OptionType.Put, strike, underlying="UND",
                                payment_ccy=tq.Currency.EUR, fx_mode="composite")

    def closed_form(self, strike):
        t = tenor()
        x0 = 1.0 / EURUSD                                           # 1 USD in EUR
        fx_forward = x0 * np.exp((R_EUR - R_USD) * t)
        forward = S0 * np.exp((R_USD - REPO) * t) * fx_forward
        rho = -RHO                                                  # corr(S, X) with X = 1 / EURUSD
        sigma = np.sqrt(VOL ** 2 + FX_VOL ** 2 + 2 * rho * VOL * FX_VOL)
        return black_put(forward, strike, sigma, t, np.exp(-R_EUR * t))

    def test_flags(self):
        prod = self.composite_put(90.0)
        self.assertTrue(prod.is_composite)
        self.assertFalse(prod.is_quanto)
        with self.assertRaises(ValueError):
            tq.VanillaOption(tq.Currency.USD, EVAL_DATE, EXPIRY, tq.OptionType.Put, 90.0, underlying="UND",
                             payment_ccy=tq.Currency.EUR, fx_mode="flexo")

    def test_black_scholes_closed_form(self):
        """DF_EUR Black(F_S F_X, K, sigma_Y), sigma_Y^2 = sigma_S^2 + sigma_X^2 + 2 rho sigma_S sigma_X."""
        strike = 0.95 * S0 / EURUSD
        prod = self.composite_put(strike)
        tq.BlackScholesPricer(dividend_model="continuous").price(prod, self.mkt)
        self.assertAlmostEqual(float(prod.price), self.closed_form(strike), places=10)

    def test_monte_carlo_vs_closed_form(self):
        """GBM and local vol (flat smile): S with the quanto drift, X lognormal correlated on the same draws."""
        strike = 0.95 * S0 / EURUSD
        expected = self.closed_form(strike)
        for model in (tq.GeometricBrownianMotion, tq.LocalVolatilityModel):
            prod = self.composite_put(strike)
            tq.VanillaMCPricer(model=model, n_paths=6_000, n_steps=24, seed=3).price(prod, self.mkt)
            with self.subTest(model=model.__name__):
                self.assertLess(abs(float(prod.price) - expected), 3.0 * float(prod.standard_error))

    def test_autocallable_without_fx_risk(self):
        """FX vol near zero and equal rates: X stays at its spot, Y / Y0 = S / S0, so the composite autocallable
        (strike S0 X0, paid in EUR) equals the plain one, same draws."""
        mkt = market(r_usd=R_EUR, fx_vol=1e-12)
        x0 = 1.0 / EURUSD
        cases = TestAutocallableQuanto("test_gbm_quanto_is_a_repo_shift")
        plain = cases.autocallable(None)
        composite = tq.AutocallableOption(
            ccy=tq.Currency.USD, notional=1.0, start_date=EVAL_DATE, end_date=plain.end_date, strike=S0 * x0,
            coupon_fixing_dates=plain.coupon_fixing_dates, coupon_payment_dates=plain.coupon_payment_dates,
            coupon_rates=plain.coupon_rates, coupon_barriers=plain.coupon_barriers, memory=False,
            autocall_fixing_dates=plain.autocall_fixing_dates, autocall_payment_dates=plain.autocall_payment_dates,
            autocall_barrier=plain.autocall_barrier, payoff_barrier=70.0, underlying="UND",
            payment_ccy=tq.Currency.EUR, fx_mode="composite")
        pricer = lambda: tq.AutocallableMCPricer(model=tq.GeometricBrownianMotion, n_paths=2_000, seed=5)
        pricer().price(plain, mkt)
        pricer().price(composite, mkt)
        self.assertAlmostEqual(float(composite.price), float(plain.price), places=8)


class TestPriceCurrency(QuantoCase):
    def test_converted_at_the_fx_spot_with_its_delta(self):
        """A USD put priced in EUR: the USD price / EURUSD; its AAD derivative to the EURUSD spot is -price / EURUSD^2;
        the Monte Carlo standard error converted too."""
        usd = put(payment=None)
        tq.BlackScholesPricer().price(usd, self.mkt)
        eur = put(payment=None)
        pricer = tq.BlackScholesPricer()
        pricer.price(eur, self.mkt, autodiff=True, price_currency=tq.Currency.EUR)
        self.assertAlmostEqual(float(eur.price), float(usd.price) / EURUSD, places=12)
        self.assertEqual(eur.price_currency, tq.Currency.EUR)
        fx = self.mkt.risk_factors()["FX:USD:EURUSD:SPOT"][0]
        self.assertAlmostEqual(float(pricer.tape.gradient(eur.price, fx)), -float(usd.price) / EURUSD ** 2, places=10)
        mc_usd, mc_eur = put(payment=None), put(payment=None)
        tq.VanillaMCPricer(n_paths=1_000, n_steps=4, seed=2).price(mc_usd, self.mkt)
        tq.VanillaMCPricer(n_paths=1_000, n_steps=4, seed=2).price(mc_eur, self.mkt, price_currency=tq.Currency.EUR)
        self.assertAlmostEqual(float(mc_eur.standard_error), float(mc_usd.standard_error) / EURUSD, places=12)
        same = put(payment=None)
        tq.BlackScholesPricer().price(same, self.mkt, price_currency=tq.Currency.USD)
        self.assertEqual(float(same.price), float(usd.price))


if __name__ == "__main__":
    unittest.main()
