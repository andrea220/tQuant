"""tqstudio Tradepad swap: generators from the curve conventions, legs and cash flows (``swap_legs``) vs the pricer."""

import os
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorquant as tq  # noqa: E402
from tqstudio.views.tradepad import ModelCache, _fields, catalog, price_trade, swap_generators, swap_legs  # noqa: E402

TODAY = date(2026, 5, 15)


def request(**fields):
    base = {"generator": "EUR 6M Euribor IRS", "side": "Pay fixed", "notional": 1e7, "forward": "Spot",
            "tenor": "5Y", "rate": 2.5, "spread_bp": 0.0}
    return {"asset_class": "Rates", "product": "Swap", "fields": {**base, **fields}}


class TestTradepadSwap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tq.Settings.evaluation_date = TODAY
        cls.mkt = tq.generate_market(TODAY, seed=3, tickers_per_currency=(("EUR", 2),))

    def setUp(self):
        tq.Settings.evaluation_date = TODAY

    def headline(self, req):
        return {h[0]: h[1] for h in price_trade(self.mkt, ModelCache(), req)["headline"]}

    def test_generators_of_the_market_curves(self):
        gens = swap_generators(self.mkt)
        self.assertEqual(sorted(gens), ["EUR 6M Euribor IRS", "EUR ESTR OIS", "GBP SONIA OIS", "USD SOFR OIS"])
        g = gens["EUR 6M Euribor IRS"]
        self.assertEqual((g["discount"], g["projection"]), ("IR:EUR:ESTR:SPOT", "IR:EUR:6M:SPOT"))
        self.assertEqual(g["conventions"]["Fixed leg"], "1Y, Thirty360")
        self.assertEqual(gens["GBP SONIA OIS"]["conventions"]["Spot lag"], "T+0")
        products = catalog(self.mkt)["asset_classes"]["Rates"]
        self.assertIn("Swap", products)
        self.assertNotIn("OIS", products)                         # one swap screen, the generator picks the legs

    def test_flows_equal_the_pricer(self):
        """The PV of the two legs' flows (signed by the side) = the pricer's PV, for spot / forward start, pay /
        receive, with a spread, Ibor and OIS."""
        cases = [{}, {"side": "Receive fixed"}, {"forward": "1Y", "spread_bp": 15.0},
                 {"generator": "EUR ESTR OIS", "tenor": "7Y"}, {"generator": "USD SOFR OIS", "forward": "2Y",
                                                                "side": "Receive fixed", "spread_bp": -5.0}]
        for case in cases:
            req = request(**case)
            _, f = _fields(self.mkt, req)
            legs = swap_legs(self.mkt, f)
            with self.subTest(**case):
                self.assertAlmostEqual(legs["npv"], self.headline(req)["PV"], delta=1e-6)
                fixed, floating = legs["legs"]
                self.assertEqual(fixed["direction"], "Receive" if case.get("side") == "Receive fixed" else "Pay")
                self.assertEqual(floating["direction"], "Pay" if case.get("side") == "Receive fixed" else "Receive")
                self.assertTrue(all((r["amount"] < 0) == (fixed["direction"] == "Pay") for r in fixed["rows"]))
                self.assertAlmostEqual(floating["rows"][0]["spread"], case.get("spread_bp", 0.0) / 1e4, places=12)

    def test_par_rate_gives_zero(self):
        for gen in ("EUR 6M Euribor IRS", "EUR ESTR OIS"):
            par = self.headline(request(generator=gen, spread_bp=10.0))["Fair rate (%)"]
            with self.subTest(generator=gen):
                self.assertAlmostEqual(self.headline(request(generator=gen, spread_bp=10.0, rate=par))["PV"], 0.0,
                                       delta=1e-4)

    def test_schedule_and_fixings(self):
        _, f = _fields(self.mkt, request(forward="1Y"))
        fixed, floating = swap_legs(self.mkt, f)["legs"]
        self.assertEqual(len(fixed["rows"]), 5)                    # annual fixed, semiannual 6M floating
        self.assertEqual(len(floating["rows"]), 10)
        cal = tq.TARGET()
        for r in floating["rows"]:                                  # fixing: two TARGET business days before the start
            start, fixing = date.fromisoformat(r["start"]), date.fromisoformat(r["fixing"])
            days = [fixing + timedelta(days=k) for k in range(1, (start - fixing).days + 1)]
            self.assertEqual(sum(cal.is_business_day(d) for d in days), 2)
            self.assertTrue(cal.is_business_day(fixing))
        _, f = _fields(self.mkt, request(generator="EUR ESTR OIS"))
        self.assertTrue(all(r["fixing"] is None for r in swap_legs(self.mkt, f)["legs"][1]["rows"]))   # compounded

    def test_sensitivity_table(self):
        """Rows by maturity, total first: the Zero totals per curve = the rho totals; curves without a stored
        bootstrap Jacobian (the synthetic ones) have no Par column values."""
        out = price_trade(self.mkt, ModelCache(), request(forward="1Y", spread_bp=10.0))
        t = out["sens_table"]
        self.assertEqual(sorted(t["curves"]), ["EUR 6M", "EUR ESTR"])
        self.assertEqual(t["par_available"], [False, False])
        self.assertTrue(all(x is None for x in t["total"]["par"]))
        rho = {r["curve"]: r["total"] for r in out["rho"]}
        for j, curve in enumerate(t["curves"]):
            self.assertAlmostEqual(t["total"]["zero"][j], rho[curve], delta=1e-9)
            self.assertAlmostEqual(sum(r["zero"][j] or 0.0 for r in t["rows"]), rho[curve], delta=1e-9)
        times = [r["T"] for r in t["rows"]]
        self.assertEqual(times, sorted(times))

    def test_unknown_generator(self):
        with self.assertRaises(ValueError):
            _fields(self.mkt, request(generator="JPY TONA OIS"))


if __name__ == "__main__":
    unittest.main()
