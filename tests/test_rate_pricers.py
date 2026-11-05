"""Rate pricers: Deposit, FRA, Swap, OIS on a synthetic two-curve market (ESTR discounting, EUR 6M forwarding)."""

import unittest
from datetime import date

import tensorflow as tf

import tensorquant as tq
from tensorquant.index.curverateindex import IborIndex, OvernightIndex
from tensorquant.instruments.helpers import DepositGenerator, FraGenerator, OisGenerator, SwapGenerator
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.timehandles.daycounter import DayCounter

EVAL_DATE = date(2026, 5, 15)
A360, A365 = tq.DayCounterConvention.Actual360, tq.DayCounterConvention.Actual365
PILLARS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
ESTR_RATES = [0.020, 0.021, 0.022, 0.024, 0.026, 0.028]
EUR6M_RATES = [0.023, 0.024, 0.025, 0.027, 0.029, 0.031]
NOTIONAL = 1e6
CAL = tq.TARGET()
MF = tq.BusinessDayConvention.ModifiedFollowing

# Pinned prices. Before the step-3 fixes: deposit -71.3696210316848 (unchanged), fra 442.55638853151595
# (P(0,T2) instead of P(0,T1), index dates), swap 4436.106339597871 and ois 2542.5690338979766 (coupons
# discounted at the coupon day-count year fraction on an ACT/365 curve; OIS rate on the curve day count).
PRICES = {
    "deposit": -71.3696210316848,
    "fra": 423.1593272750896,
    "swap": 4434.221205069334,
    "ois": 1892.1651524309927,
}


def make_market(eur6m_rates=EUR6M_RATES) -> MarketEnvironment:
    return MarketEnvironment({
        "IR:EUR:ESTR:SPOT": tq.RateCurve(EVAL_DATE, PILLARS, ESTR_RATES, "LINEAR", A365),
        "IR:EUR:6M:SPOT": tq.RateCurve(EVAL_DATE, PILLARS, list(eur6m_rates), "LINEAR", A365),
    })


def ibor6m() -> IborIndex:
    return IborIndex(CAL, 6, tq.TimeUnit.Months, tq.Currency.EUR)


def generators() -> dict:
    return {
        "deposit": DepositGenerator(tq.Currency.EUR, 2, MF, A360, NOTIONAL, CAL),
        "fra": FraGenerator(tq.Currency.EUR, 2, 2, "6M", MF, NOTIONAL, A360, CAL, ibor6m()),
        "swap": SwapGenerator(tq.Currency.EUR, 2, "1Y", "6M", MF, NOTIONAL, A360, A360, CAL, ibor6m()),
        "ois": OisGenerator(tq.Currency.EUR, 2, "1Y", "1Y", MF, NOTIONAL, A360, A360, CAL,
                            OvernightIndex(CAL, tq.Currency.EUR)),
    }


TERMS = {"deposit": (0.021, "6M"), "fra": (0.025, "6M-12M"), "swap": (0.028, "5Y"), "ois": (0.023, "2Y")}


def make_products(quotes: dict | None = None) -> dict:
    quotes = quotes or {}
    gens = generators()
    return {name: gens[name].build(EVAL_DATE, quotes.get(name, q), term) for name, (q, term) in TERMS.items()}


def make_product(name: str, quote: float | None = None):
    q, term = TERMS[name]
    return generators()[name].build(EVAL_DATE, q if quote is None else quote, term)


class RateTestCase(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE
        self.mkt = make_market()

    def price(self, product, autodiff=False):
        pricer = tq.PricerAssignment.create(product)
        pricer.price(product, self.mkt, autodiff=autodiff)
        return pricer


class TestPinnedValues(RateTestCase):
    def test_prices(self):
        for name, prod in make_products().items():
            with self.subTest(product=name):
                self.price(prod)
                self.assertAlmostEqual(float(prod.price), PRICES[name], delta=1e-6)

    def test_price_is_float64_tensor(self):
        for name, prod in make_products().items():
            with self.subTest(product=name):
                self.price(prod)
                self.assertIsInstance(prod.price, tf.Tensor)
                self.assertEqual(prod.price.dtype, tf.float64)


class TestGradients(RateTestCase):
    def test_pillar_gradients_vs_bump(self):
        """dNPV/dr_i by AAD on every pillar of both curves, against central bumps of the same variable."""
        h = 1e-6
        for name in TERMS:
            prod = make_product(name)
            pricer = self.price(prod, autodiff=True)
            factors = self.mkt.risk_factors()
            grads = pricer.tape.gradient(prod.price, factors)
            for key, variables in factors.items():
                for i, (var, g) in enumerate(zip(variables, grads[key])):
                    base = float(var)
                    out = []
                    for x in (base + h, base - h):
                        var.assign(x)
                        p = make_product(name)
                        self.price(p)
                        out.append(float(p.price))
                    var.assign(base)
                    with self.subTest(product=name, curve=key, pillar=i):
                        g = 0.0 if g is None else float(g)
                        self.assertAlmostEqual(g, (out[0] - out[1]) / (2 * h), delta=1e-2 + 1e-7 * abs(g))


class TestIndependentChecks(RateTestCase):
    def test_deposit_closed_form(self):
        """PV = N [P(0, s) - (1 + r tau) P(0, e)], tau ACT/360 on the deposit dates."""
        depo = make_product("deposit")
        self.price(depo)
        curve = self.mkt.get_ir_curve(tq.Currency.EUR)
        tau = DayCounter(A360).year_fraction(depo.start_date, depo.end_date)
        expected = NOTIONAL * (float(curve.discount(depo.start_date))
                               - (1 + TERMS["deposit"][0] * tau) * float(curve.discount(depo.end_date)))
        self.assertAlmostEqual(float(depo.price), expected, delta=1e-8)

    def test_deposit_on_an_act360_curve(self):
        """Discount factors read at the dates: the closed form holds for any curve day count."""
        self.mkt = MarketEnvironment({
            "IR:EUR:ESTR:SPOT": tq.RateCurve(EVAL_DATE, PILLARS, ESTR_RATES, "LINEAR", A360),
            "IR:EUR:6M:SPOT": tq.RateCurve(EVAL_DATE, PILLARS, EUR6M_RATES, "LINEAR", A360),
        })
        self.test_deposit_closed_form()

    def test_fra_standard_formula(self):
        """NPV = N tau (F - K) P_disc(0, T1) / (1 + F tau), F from the 6M curve on the FRA dates."""
        fra = make_product("fra")
        self.price(fra)
        disc, fwd = self.mkt.get_ir_curve(tq.Currency.EUR), self.mkt.get_ir_curve(tq.Currency.EUR, "6M")
        t1, t2 = fra.start_date, fra.end_date
        tau = DayCounter(A360).year_fraction(t1, t2)
        F = (float(fwd.discount(t1)) / float(fwd.discount(t2)) - 1) / tau
        expected = NOTIONAL * tau * (F - TERMS["fra"][0]) * float(disc.discount(t1)) / (1 + F * tau)
        self.assertAlmostEqual(float(fra.price), expected, delta=1e-8)

    def test_fixed_leg_is_sum_of_discounted_amounts(self):
        """Fixed leg = sum_i amount_i P(0, pay_i), P at the payment date (curve day count)."""
        swap = make_product("swap")
        self.price(swap)
        curve = self.mkt.get_ir_curve(tq.Currency.EUR)
        expected = sum(float(c.amount) * float(curve.discount(c.date)) for c in swap.fixed_leg.leg_flows)
        self.assertAlmostEqual(abs(float(swap.fixed_leg.price)), expected, delta=1e-7)

    def test_single_curve_floating_legs_telescope(self):
        """Same curve for forwarding and discounting: floating leg = N (P(0, start) - P(0, end))."""
        self.mkt = make_market(eur6m_rates=ESTR_RATES)
        curve = self.mkt.get_ir_curve(tq.Currency.EUR)
        for name in ("swap", "ois"):
            prod = make_product(name)
            self.price(prod)
            flows = prod.floating_leg.leg_flows
            expected = NOTIONAL * (float(curve.discount(flows[0].accrual_start_date))
                                   - float(curve.discount(flows[-1].date)))
            with self.subTest(product=name):
                self.assertAlmostEqual(abs(float(prod.floating_leg.price)), expected, delta=1e-6)

    def test_npv_is_zero_at_the_fair_rate(self):
        """NPV is linear in the fixed rate: K* = K0 - NPV(K0) / annuity gives NPV(K*) = 0."""
        for name in ("swap", "ois"):
            k0 = 0.03
            p0, p1 = make_product(name, k0), make_product(name, k0 + 1e-4)
            self.price(p0)
            self.price(p1)
            annuity = (float(p1.price) - float(p0.price)) / 1e-4
            fair = make_product(name, k0 - float(p0.price) / annuity)
            self.price(fair)
            with self.subTest(product=name):
                self.assertAlmostEqual(float(fair.price), 0.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
