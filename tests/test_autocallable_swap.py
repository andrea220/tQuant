"""Autocallable swap (``AutocallableSwap`` / ``AutocallableSwapPricer``) and the side of the autocallables: funding
leg = floating leg of a swap without an autocall, cut at the autocall, identities with the autocallable on the same
paths, payer = -receiver, projections, rho and delta against bumps."""

import unittest
from datetime import date

import numpy as np

import tensorquant as tq

EVAL_DATE = date(2026, 5, 15)
N_PATHS = 4_000


def market():
    tq.Settings.evaluation_date = EVAL_DATE
    return tq.generate_market(EVAL_DATE, seed=7, tickers_per_currency=(("EUR", 1), ("GBP", 1), ("USD", 1)))


class TestAutocallableSwap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mkt = market()
        cls.ticker = next(k for k in cls.mkt.risk_factors() if k.startswith("EQ:EUR:") and k.endswith(":VOL")).split(":")[2]
        cal, mf = tq.TARGET(), tq.BusinessDayConvention.ModifiedFollowing
        sg = tq.ScheduleGenerator(cal, mf)
        cls.strike_date = cal.advance(EVAL_DATE, 10, tq.TimeUnit.Days, mf)          # every fixing in the future
        cls.end = cal.advance(cls.strike_date, 3, tq.TimeUnit.Years, mf)
        cls.coupons = sg.generate(cls.strike_date, cls.end, 6, tq.TimeUnit.Months)[1:]
        cls.autocalls = sg.generate(cal.advance(cls.strike_date, 1, tq.TimeUnit.Years, mf), cls.end, 6,
                                    tq.TimeUnit.Months)

    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE

    def product(self, cls=tq.AutocallableSwap, barrier=100.0, side=tq.SwapType.Receiver, **kwargs):
        if cls is tq.AutocallableSwap:
            kwargs = {"funding_curve": "6M", "funding_margin": 30.0, **kwargs}
        return cls(ccy=tq.Currency.EUR, notional=100.0, start_date=EVAL_DATE, end_date=self.end, strike=None,
                   strike_date=self.strike_date, coupon_fixing_dates=self.coupons, coupon_payment_dates=self.coupons,
                   coupon_rates=[4.0], coupon_barriers=[70.0], memory=True, autocall_fixing_dates=self.autocalls,
                   autocall_payment_dates=self.autocalls, autocall_barrier=[barrier], payoff_barrier=60.0,
                   underlying=self.ticker, side=side, **kwargs)

    def price(self, product, smoothing=0.01, autodiff=False):
        pricer = tq.AutocallableSwapPricer(n_paths=N_PATHS, seed=3, barrier_smoothing=smoothing)
        pricer.price(product, self.mkt, autodiff=autodiff)
        return pricer

    def test_periods(self):
        """Funding periods from the strike date to each coupon payment, ACT/360 accruals."""
        p = self.product()
        self.assertEqual(p.funding_start_dates, [self.strike_date] + self.coupons[:-1])
        self.assertEqual(p.funding_payment_dates, self.coupons)
        a360 = tq.DayCounter(tq.DayCounterConvention.Actual360)
        np.testing.assert_allclose(p.funding_accruals, [a360.year_fraction(s, e) for s, e in
                                                        zip(p.funding_start_dates, p.funding_end_dates)])

    def test_no_autocall_is_a_floating_leg(self):
        """Autocall out of reach: the funding leg = the floating leg of a swap on the same periods (EUR 6M + 30 bp,
        discounted on ESTR), to machine precision."""
        p = self.product(barrier=1e6)
        self.price(p)
        index = tq.IborIndex(tq.TARGET(), 6, tq.TimeUnit.Months, tq.Currency.EUR)
        n = len(p.funding_payment_dates)
        leg = tq.FloatingRateLeg(p.funding_payment_dates, p.funding_start_dates, p.funding_end_dates, [1.0] * n,
                                 [1] * n, [0.003] * n, index, tq.DayCounter(tq.DayCounterConvention.Actual360))
        swap_leg = tq.FloatingLegDiscounting(leg).calculate_price(self.mkt.get_ir_curve(tq.Currency.EUR),
                                                                  self.mkt.get_ir_curve(tq.Currency.EUR, "6M"))
        self.assertAlmostEqual(float(p.funding_pv), float(swap_leg), delta=1e-13)

    def test_called_at_the_first_date(self):
        """Autocall certain at the first observation (hard barriers): only the periods up to it are paid."""
        p = self.product(barrier=0.0)
        self.price(p, smoothing=0.0)
        ref = self.product(barrier=1e6)
        self.price(ref, smoothing=0.0)
        disc, proj = self.mkt.get_ir_curve(tq.Currency.EUR), self.mkt.get_ir_curve(tq.Currency.EUR, "6M")
        expected = sum((float(proj.discount(s)) / float(proj.discount(e)) - 1.0 + 0.003 * tau) * float(disc.discount(e))
                       for s, e, tau in zip(p.funding_start_dates, p.funding_end_dates, p.funding_accruals)
                       if e <= self.autocalls[0])
        self.assertAlmostEqual(float(p.funding_pv), expected, delta=1e-12)
        self.assertLess(float(p.funding_pv), float(ref.funding_pv))

    def test_identities_on_the_same_paths(self):
        """Swap = autocallable - funding (same paths); payer = - receiver; the autocallable's side too."""
        swap = self.product()
        self.price(swap)
        option = self.product(cls=tq.AutocallableOption)
        tq.AutocallableMCPricer(n_paths=N_PATHS, seed=3).price(option, self.mkt)
        self.assertAlmostEqual(float(swap.price), float(option.price) - 100.0 * float(swap.funding_pv), delta=1e-10)
        payer = self.product(side=tq.SwapType.Payer)
        self.price(payer)
        self.assertAlmostEqual(float(payer.price), -float(swap.price), delta=1e-10)
        option_payer = self.product(cls=tq.AutocallableOption, side=tq.SwapType.Payer)
        tq.AutocallableMCPricer(n_paths=N_PATHS, seed=3).price(option_payer, self.mkt)
        self.assertAlmostEqual(float(option_payer.price), -float(option.price), delta=1e-10)

    def test_validation(self):
        with self.assertRaises(ValueError):
            tq.AutocallableMCPricer(n_paths=100).price(self.product(), self.mkt)
        with self.assertRaises(ValueError):
            self.product(side="Payer")

    def test_projections(self):
        """Hard barriers: the funding flows discounted = minus the funding leg; the issuer's flows are the client's
        with the other sign."""
        p = self.product()
        pricer = self.price(p, smoothing=0.0)
        flows = pricer.projections()["cash_flows"]
        disc = self.mkt.get_ir_curve(tq.Currency.EUR)
        pv = sum(r["funding"] * float(disc.discount(r["date"])) for r in flows)
        self.assertAlmostEqual(pv, -float(p.funding_pv), delta=1e-12)
        self.assertTrue(all(r["funding"] <= 0 for r in flows))
        payer = self.product(side=tq.SwapType.Payer)
        mirrored = self.price(payer, smoothing=0.0).projections()["cash_flows"]
        for a, b in zip(flows, mirrored):
            for kind in ("coupon", "redemption", "put", "funding", "total"):
                self.assertAlmostEqual(a[kind], -b[kind], delta=1e-14)

    def test_rho_and_delta_vs_bumps(self):
        """AAD rho on a 6M and an ESTR pillar and the delta (funding cut by the autocall included) = central bumps on
        the same paths."""
        p = self.product()
        pricer = self.price(p, autodiff=True)
        factors = self.mkt.risk_factors()
        keys = ["IR:EUR:6M:SPOT", "IR:EUR:ESTR:SPOT", f"EQ:EUR:{self.ticker}:SPOT"]
        grads = pricer.tape.gradient(p.price, {k: factors[k] for k in keys})

        def bumped(var, h):
            base = var.numpy().copy()
            var.assign(base + h)
            try:
                q = self.product()
                self.price(q)
                return float(q.price)
            finally:
                var.assign(base)

        for key in keys[:2]:                            # one variable per pillar: the largest rho
            g = [0.0 if x is None else float(x) for x in grads[key]]
            i = int(np.argmax(np.abs(g)))
            h = 1e-5
            fd = (bumped(factors[key][i], h) - bumped(factors[key][i], -h)) / (2 * h)
            self.assertAlmostEqual(g[i] / fd, 1.0, delta=1e-4, msg=key)
        spot = factors[keys[2]][0]
        h = 0.01 * float(spot.numpy())
        fd = (bumped(spot, h) - bumped(spot, -h)) / (2 * h)
        self.assertAlmostEqual(float(np.sum(grads[keys[2]][0])) / fd, 1.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
