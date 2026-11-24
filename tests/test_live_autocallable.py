"""Live (seasoned) autocallables: the state of the trade from the fixings already observed (market or trade)."""

import unittest
from datetime import date, timedelta

import numpy as np
import tensorflow as tf

import tensorquant as tq
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

TRADE = date(2025, 6, 16)                     # trade and strike date, one year before the evaluation
EVAL = date(2026, 6, 22)
RATE, VOL, S0, K = 0.021, 0.25, 100.0, 95.0  # K: the initial fixing (struck one year ago)
LAG = timedelta(days=14)                      # payment two weeks after each fixing


def flat_surface(spot):
    strikes = [spot * m for m in (0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0)]
    return VolatilitySurface(reference_date=EVAL, calendar=None, daycounter=DayCounter(DayCounterConvention.Actual365),
                             strike=strikes, maturity=[0.5, 1.0, 2.0, 3.0, 5.0],
                             volatility_matrix=[[VOL] * len(strikes) for _ in range(5)])


def schedule():
    cal = tq.TARGET()
    mf = tq.BusinessDayConvention.ModifiedFollowing
    end = cal.advance(TRADE, 4, tq.TimeUnit.Years, mf)
    fix = tq.ScheduleGenerator(cal, mf).generate(TRADE, end, 6, tq.TimeUnit.Months)[1:]
    return fix, end


FIX, END = schedule()
PAST = [d for d in FIX if d < EVAL]           # two observations already fixed (Dec 2025, Jun 2026)


def market(fixings=None, rho=0.5) -> MarketEnvironment:
    """A (EUR) and B (EUR), flat 25% vols; fixings ``{ticker: {date: value}}``."""
    data = {
        "IR:EUR:ESTR:SPOT": tq.RateCurve(EVAL, [1.0, 5.0, 10.0], [RATE] * 3, "LINEAR", DayCounterConvention.Actual365),
        "CORR:ALL:GLOBAL:SPOT": tq.CorrelationMatrix.from_pairs([("EQ:EUR:A", "EQ:EUR:B", rho)]),
    }
    for ticker in ("A", "B"):
        data.update({f"EQ:EUR:{ticker}:SPOT": S0, f"EQ:EUR:{ticker}:REPO": 0.0,
                     f"EQ:EUR:{ticker}:VOL": flat_surface(S0)})
    for ticker, values in (fixings or {}).items():
        data[f"EQ:EUR:{ticker}:FIXINGS"] = tq.Fixings(list(values), list(values.values()))
    return MarketEnvironment(market=data)


def autocallable(underlying="A", memory=True, fixings=None, fix=None, start=TRADE, strike=None, strike_date=TRADE):
    fix = FIX if fix is None else fix
    pay = [d + LAG for d in fix]
    return tq.AutocallableOption(
        ccy=tq.Currency.EUR, notional=1.0, start_date=start, end_date=END + LAG, strike=strike,
        strike_date=strike_date, coupon_fixing_dates=fix, coupon_payment_dates=pay, coupon_rates=[3.0],
        coupon_barriers=[80.0], memory=memory, autocall_fixing_dates=fix[1:], autocall_payment_dates=pay[1:],
        autocall_barrier=[100.0], payoff_barrier=70.0, underlying=underlying, fixings=fixings)


def pricer(eps=0.0, n_paths=2_000, model=tq.GeometricBrownianMotion):
    return tq.AutocallableMCPricer(model=model, n_paths=n_paths, seed=11, barrier_smoothing=eps, max_time_step=1 / 12)


def replica(product, performance: dict, curve):
    """Hard payoff on the performance ``{date: [n]}`` (past fixings as numbers), payments before EVAL left out."""
    n = max(np.size(v) for v in performance.values())
    alive, unpaid, coupons, put = np.ones(n, bool), np.zeros(n), np.zeros(n), np.zeros(n)
    for d in sorted(performance):
        p = np.broadcast_to(performance[d], (n,))
        if d in product.coupon_fixing_dates:
            i = product.coupon_fixing_dates.index(d)
            pay = product.coupon_payment_dates[i]
            df = float(curve.discount(pay)) if pay >= EVAL else 0.0
            above = p >= product.coupon_barriers[i] / 100
            amount = product.coupon_rates[i] / 100 * ((1 + unpaid) if product.memory else 1.0)
            coupons += np.where(alive & above, amount, 0.0) * df
            if d == product.coupon_fixing_dates[-1]:
                ki = alive & (p < product.payoff_barrier / 100)
                put += np.where(ki, product.payoff_participation * np.maximum(product.put_strike / 100 - p, 0), 0) * df
            unpaid = np.where(alive & above, 0.0, np.where(alive & ~above, unpaid + 1, unpaid))
        if d in product.autocall_fixing_dates:
            alive &= ~(p >= product.autocall_barrier[product.autocall_fixing_dates.index(d)] / 100)
    return float(np.mean(coupons - put))


class LiveCase(unittest.TestCase):
    def setUp(self):
        tq.Settings.evaluation_date = EVAL


class TestState(LiveCase):
    def test_numpy_replica_with_past_fixings(self):
        """Struck at 95 one year ago; Dec 2025 at 76 (below the 80% coupon barrier: missed, in memory), Jun 2026 at
        90.25 (95%: coupon paid with the memory, settled after today; no autocall). Hard payoff = numpy replica."""
        values = {TRADE: K, PAST[0]: 0.80 * K - 0.01, PAST[1]: 0.95 * K}
        prod = autocallable()
        p = pricer()
        p.price(prod, market({"A": values}))
        curve = market().get_ir_curve(tq.Currency.EUR)
        performance = {d: values[d] / K for d in PAST}
        paths = p.paths.numpy()
        performance.update({d: paths[:, i] / K for i, d in enumerate(p.date_grid)})
        self.assertEqual(p.date_grid, [d for d in FIX if d > EVAL])
        self.assertAlmostEqual(float(prod.price), replica(prod, performance, curve), places=12)
        # the memory coupon of Jun 2026 (two coupons, paid after today) is in the price: certain, discounted
        df = float(curve.discount(PAST[1] + LAG))
        no_memory = autocallable(memory=False)
        pricer().price(no_memory, market({"A": values}))
        self.assertGreater(float(prod.price) - float(no_memory.price), 0.03 * df - 1e-12)

    def test_trade_state(self):
        """The state from the same fixings: strike, coupon missed then paid with the memory, called, nothing after."""
        values = {TRADE: K, PAST[0]: 0.75 * K, PAST[1]: 1.05 * K}
        state = pricer().trade_state(autocallable(), market({"A": values}))
        self.assertEqual(state["strikes"], [K])
        self.assertEqual([r["coupon"] for r in state["rows"]], ["missed", "paid"])
        self.assertEqual(state["rows"][1]["coupon_paid"], 6.0)                  # 3% plus the one in memory
        self.assertEqual([r["autocall"] for r in state["rows"]], [False, True])
        self.assertEqual(state["called_on"], PAST[1])
        self.assertAlmostEqual(state["rows"][0]["performance"], 0.75, places=14)

    def test_called_in_the_past(self):
        """Autocalled in Jun 2026 (105%): only the last coupon, fixed and paid after today, is left: its discounted
        value exactly; once paid, nothing."""
        values = {TRADE: K, PAST[0]: 0.9 * K, PAST[1]: 1.05 * K}
        prod = autocallable(memory=False)
        pricer().price(prod, market({"A": values}))
        df = float(market().get_ir_curve(tq.Currency.EUR).discount(PAST[1] + LAG))
        self.assertAlmostEqual(float(prod.price), 0.03 * df, places=14)
        self.assertAlmostEqual(float(prod.standard_error), 0.0, places=15)
        tq.Settings.evaluation_date = PAST[1] + LAG + timedelta(days=1)
        settled = autocallable(memory=False)
        pricer().price(settled, market({"A": values}))
        self.assertEqual(float(settled.price), 0.0)

    def test_seasoned_trade_equals_a_new_one(self):
        """Past coupons paid, no autocall, no memory: the value is that of a new trade on the remaining dates with the
        same strike (same grid, same draws), plus the Jun 2026 coupon, fixed but paid after today."""
        values = {TRADE: K, PAST[0]: 0.9 * K, PAST[1]: 0.85 * K}
        live = autocallable(memory=False)
        pricer(eps=0.01).price(live, market({"A": values}))
        rest = [d for d in FIX if d > EVAL]
        new = tq.AutocallableOption(
            ccy=tq.Currency.EUR, notional=1.0, start_date=EVAL, end_date=END + LAG, strike=K,
            coupon_fixing_dates=rest, coupon_payment_dates=[d + LAG for d in rest], coupon_rates=[3.0],
            coupon_barriers=[80.0], memory=False, autocall_fixing_dates=rest, autocall_payment_dates=[d + LAG for d in rest],
            autocall_barrier=[100.0], payoff_barrier=70.0, underlying="A")
        pricer(eps=0.01).price(new, market())
        df = float(market().get_ir_curve(tq.Currency.EUR).discount(PAST[1] + LAG))
        self.assertAlmostEqual(float(live.price), float(new.price) + 0.03 * df, places=12)

    def test_fixings_from_the_trade_and_missing_ones(self):
        """A past date without a fixing raises; the trade's fixings fill it (and win over the market's)."""
        values = {TRADE: K, PAST[0]: 0.9 * K, PAST[1]: 0.85 * K}
        with self.assertRaisesRegex(ValueError, "no fixing of A"):
            pricer().price(autocallable(), market({"A": {TRADE: K, PAST[0]: 0.9 * K}}))
        from_trade = autocallable(fixings={PAST[1]: 0.85 * K})
        pricer().price(from_trade, market({"A": {TRADE: K, PAST[0]: 0.9 * K, PAST[1]: 0.5 * K}}))
        reference = autocallable()
        pricer().price(reference, market({"A": values}))
        self.assertEqual(float(from_trade.price), float(reference.price))

    def test_fixing_today_is_the_spot(self):
        """An observation on the evaluation date not in the data is the spot (today's close not known yet)."""
        tq.Settings.evaluation_date = PAST[1]
        values = {TRADE: K, PAST[0]: 0.9 * K}
        spot = autocallable()
        pricer().price(spot, market({"A": values}))
        given = autocallable(fixings={PAST[1]: S0})
        pricer().price(given, market({"A": values}))
        self.assertAlmostEqual(float(spot.price), float(given.price), places=12)

    def test_no_delta_from_the_past(self):
        """Every observation fixed, the last coupon not yet paid: a certain amount, no simulation, no delta."""
        last = FIX[-1]
        tq.Settings.evaluation_date = last + timedelta(days=1)
        values = {TRADE: K, **{d: 0.9 * K for d in FIX}}
        prod = autocallable(memory=False)
        p = pricer(eps=0.01)
        mkt = market({"A": values})
        p.price(prod, mkt, autodiff=True)
        self.assertIsNone(p.paths)
        df = float(mkt.get_ir_curve(tq.Currency.EUR).discount(last + LAG))
        self.assertAlmostEqual(float(prod.price), 0.03 * df, places=14)
        self.assertIsNone(p.tape.gradient(prod.price, mkt.risk_factors()["EQ:EUR:A:SPOT"][0]))


class TestWorstOf(LiveCase):
    def test_numpy_replica_per_underlying(self):
        """Worst-of A / B struck one year ago at 95 / 110: the worst performance of the past fixings on each
        underlying, then the paths."""
        fa = {TRADE: K, PAST[0]: 0.9 * K, PAST[1]: 1.02 * K}
        fb = {TRADE: 110.0, PAST[0]: 0.78 * 110.0, PAST[1]: 0.99 * 110.0}
        prod = autocallable(underlying=["A", "B"])
        p = pricer()
        p.price(prod, market({"A": fa, "B": fb}))
        performance = {d: min(fa[d] / K, fb[d] / 110.0) for d in PAST}
        perf = p.performance.numpy()
        performance.update({d: perf[:, i] for i, d in enumerate(p.date_grid)})
        np.testing.assert_allclose(perf, (p.paths.numpy() / np.array([K, 110.0])).min(axis=2), rtol=1e-12)
        self.assertAlmostEqual(float(prod.price), replica(prod, performance, market().get_ir_curve(tq.Currency.EUR)),
                               places=12)

    def test_fixings_given_on_the_trade(self):
        """Worst-of fixings on the trade: a list in the order of the underlyings."""
        fa = {TRADE: K, PAST[0]: 0.9 * K, PAST[1]: 1.02 * K}
        fb = {TRADE: 110.0, PAST[0]: 0.78 * 110.0, PAST[1]: 0.99 * 110.0}
        on_trade = autocallable(underlying=["A", "B"], fixings={d: [fa[d], fb[d]] for d in fa})
        pricer().price(on_trade, market())
        from_market = autocallable(underlying=["A", "B"])
        pricer().price(from_market, market({"A": fa, "B": fb}))
        self.assertEqual(float(on_trade.price), float(from_market.price))
        with self.assertRaises(ValueError):
            pricer().price(autocallable(underlying=["A", "B"], fixings={TRADE: [K]}), market({"A": fa, "B": fb}))


class TestMarketFixings(LiveCase):
    def test_round_trip_and_not_a_risk_factor(self):
        values = {TRADE: K, PAST[0]: 0.9 * K}
        mkt = market({"A": values})
        self.assertEqual(mkt.get_eq_fixing("A", PAST[0]), 0.9 * K)
        self.assertNotIn("EQ:EUR:A:FIXINGS", mkt.risk_factors())
        frames = mkt.to_frames()
        self.assertEqual(len(frames["eq_fixings.csv"]), 2)
        back = MarketEnvironment.from_frames(frames, EVAL)
        self.assertEqual(back.get_eq_fixings("A").dates, [TRADE, PAST[0]])
        self.assertEqual(back.get_eq_fixing("A", TRADE), K)
        self.assertIsNone(back.get_eq_fixings("B"))
        with self.assertRaises(ValueError):
            back.get_eq_fixing("A", EVAL)


if __name__ == "__main__":
    unittest.main()
