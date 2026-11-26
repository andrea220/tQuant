"""tqstudio Tradepad correlations: the matrix of a deal in the Market Data tab (asset / asset, asset / currency),
overridden for one pricing only (the market matrix restored, also after an error)."""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import tensorquant as tq  # noqa: E402
from tqstudio.views.tradepad import GBM_DEBUG, ModelCache, catalog, describe_trade, price_trade  # noqa: E402
from test_quanto import EVAL_DATE as TODAY, market  # noqa: E402  (USD UND, EUR EUND, EURUSD, their correlations)


class TestDealCorrelations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tq.Settings.evaluation_date = TODAY
        cls.mkt = market()
        cls.mkt.evaluation_date = TODAY                 # as a loaded snapshot (the Tradepad reads it)

    def setUp(self):
        tq.Settings.evaluation_date = TODAY

    def request(self, **fields):
        spec = catalog(self.mkt)["asset_classes"]["Exotics"]["Autocallable"]
        base = {f["name"]: f.get("default") for f in spec["fields"] if "name" in f}
        base.update({"payment_ccy": "EUR", "strike_date": TODAY.isoformat()}, **fields)
        return {"asset_class": "Exotics", "product": "Autocallable", "fields": base, "model": GBM_DEBUG}

    def price(self, req):
        out = price_trade(self.mkt, ModelCache(), req, n_paths=2_000)
        return {h[0]: h[1] for h in out["headline"]}["Price"]

    def test_matrix_of_the_deal(self):
        """Worst-of UND (USD) and EUND (EUR) paid in EUR: the two underlyings and EURUSD, as stored; one factor:
        nothing to show."""
        md = describe_trade(self.mkt, self.request(ticker="UND", ticker_2="EUND"))["market_data"]
        c = md["correlations"]
        self.assertEqual(c["names"], ["EQ:USD:UND", "EQ:EUR:EUND", "FX:USD:EURUSD"])
        self.assertEqual(c["labels"], ["UND", "EUND", "EURUSD"])
        np.testing.assert_allclose(c["matrix"], self.mkt.get_correlation(c["names"]).numpy(), atol=1e-14)
        self.assertIsNone(describe_trade(self.mkt, self.request(ticker="EUND"))["market_data"]["correlations"])

    def test_override_for_one_pricing(self):
        """A quanto autocallable on UND paid in EUR: the UND / EURUSD correlation moves the price; the market matrix
        is unchanged after it, after an out-of-range value and after edits that are not positive semi-definite."""
        req = self.request(ticker="UND")
        var = self.mkt.get_correlation_matrix().variables[0]
        before = var.numpy().copy()
        key = "EQ:USD:UND|FX:USD:EURUSD"
        prices = [self.price({**req, "overrides": {"corr": {key: x}}}) for x in (-0.5, 0.5)]
        self.assertNotAlmostEqual(prices[0], prices[1], places=6)
        np.testing.assert_array_equal(var.numpy(), before)
        with self.assertRaises(ValueError):
            self.price({**req, "overrides": {"corr": {key: 1.2}}})
        np.testing.assert_array_equal(var.numpy(), before)
        bad = {"EQ:USD:UND|EQ:EUR:EUND": 0.95, "EQ:USD:UND|FX:USD:EURUSD": 0.95, "EQ:EUR:EUND|FX:USD:EURUSD": -0.95}
        with self.assertRaises(ValueError):
            self.price({**req, "overrides": {"corr": bad}})
        np.testing.assert_array_equal(var.numpy(), before)


if __name__ == "__main__":
    unittest.main()
