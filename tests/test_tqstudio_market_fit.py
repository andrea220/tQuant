"""tqstudio market data: fit of the listed options (Black-Scholes repricing of the quotes behind a vol surface)."""

import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import tensorquant as tq  # noqa: E402
from tqstudio.views.market import fit_view  # noqa: E402

TODAY = date(2026, 11, 13)
TICKER = "TICKER1"


class TestFitView(unittest.TestCase):
    """Quotes priced by BlackScholesPricer at the surface vol: the fit view must reprice them exactly."""

    @classmethod
    def setUpClass(cls):
        tq.Settings.evaluation_date = TODAY
        cls.mkt = tq.generate_market(TODAY, seed=7, tickers_per_currency=(("EUR", 2),))
        eq = cls.mkt.get_equity(TICKER)
        cls.spot = float(eq.spot)
        surface = eq.vol_surface
        rows_vq, rows_oq = [], []
        specs = [(90, 0.85, "P", "European"), (90, 1.10, "C", "European"), (400, 0.70, "P", "American"),
                 (400, 1.25, "C", "American"), (730, 0.95, "P", "American")]
        for i, (days, m, cp, ex) in enumerate(specs):
            expiry = TODAY + timedelta(days=days)
            strike = round(m * cls.spot, 2)
            option = tq.VanillaOption(tq.Currency.EUR, TODAY, expiry,
                                      tq.OptionType.Call if cp == "C" else tq.OptionType.Put, strike,
                                      underlying=TICKER, exercise_type=getattr(tq.ExerciseType, ex))
            pricer = tq.BlackScholesPricer()
            pricer.price(option, cls.mkt)
            price = float(option.price)
            vol = float(option.volatility)
            settlement = i == 4                                      # one settlement-only quote
            rows_vq.append({"ticker": TICKER, "ccy": "EUR", "expiry": expiry.isoformat(), "strike": strike,
                            "option_type": cp, "iv_bid": vol - 0.002, "iv_mid": vol, "iv_ask": vol + 0.002,
                            "settlement": settlement, "used": True, "model_vol": vol})
            earlier = i == 1                                         # bid of the previous session
            rows_oq.append({"ticker": TICKER, "ccy": "EUR", "expiry": expiry.isoformat(), "strike": strike,
                            "option_type": cp, "exercise": ex,
                            "bid": None if settlement else 0.98 * price, "ask": None if settlement else 1.02 * price,
                            "last": price,
                            "last_update": (TODAY - timedelta(days=1 if earlier else 0)).isoformat(),
                            "last_update_time": "" if earlier else f"16:{10 + i:02d}:05",
                            "quote_date": TODAY.isoformat()})
        rows_vq.append({**rows_vq[0], "strike": 1.0, "used": False})  # not used: left out
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        pd.DataFrame(rows_vq).to_csv(folder / "eq_vol_quotes.csv", index=False)
        pd.DataFrame(rows_oq).to_csv(folder / "eq_option_quotes.csv", index=False)
        cls.view = fit_view(cls.mkt, TICKER, folder)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_repriced_exactly(self):
        v = self.view
        self.assertEqual(v["count"], 5)
        for q in v["quotes"]:
            self.assertAlmostEqual(q["surface_price"], q["mid"], delta=1e-10 * self.spot)
            self.assertAlmostEqual(q["smile_price"], q["mid"], delta=1e-10 * self.spot)
            self.assertAlmostEqual(q["surface_vol"], q["vol_mid"], places=12)
        self.assertEqual([q["exercise"] for q in v["quotes"]][:3], ["European", "European", "American"])

    def test_summary(self):
        for model in ("surface", "smile"):
            s = self.view["summary"][model]
            self.assertEqual(s["two_sided"], 4)                    # the settlement quote is not two-sided
            self.assertEqual(s["within_bid_ask"], 1.0)
            self.assertLess(s["rms_pct_spot"], 1e-8)
            self.assertLess(s["max_volpts"], 1e-8)

    def test_settlement_mid_is_the_last_price(self):
        q = next(q for q in self.view["quotes"] if q["settlement"])
        self.assertEqual(q["bid"], q["ask"])
        self.assertEqual(q["mid"], q["bid"])

    def test_smile_vol_differs_from_the_surface(self):
        """Pricing with another vol (the smile's) moves the price: the two views are independent."""
        bumped = Path(self.tmp.name) / "bumped"
        bumped.mkdir(exist_ok=True)
        vq = pd.read_csv(Path(self.tmp.name) / "eq_vol_quotes.csv")
        vq["model_vol"] = vq["model_vol"] + 0.01
        vq.to_csv(bumped / "eq_vol_quotes.csv", index=False)
        pd.read_csv(Path(self.tmp.name) / "eq_option_quotes.csv").to_csv(bumped / "eq_option_quotes.csv", index=False)
        v = fit_view(self.mkt, TICKER, bumped)
        self.assertAlmostEqual(v["summary"]["smile"]["max_volpts"], 1.0, places=9)
        self.assertLess(v["summary"]["surface"]["max_volpts"], 1e-8)
        self.assertTrue(all(q["smile_price"] > q["surface_price"] for q in v["quotes"]))

    def test_quote_times(self):
        v = self.view
        self.assertTrue(v["has_times"])
        self.assertEqual(v["quote_date"], TODAY.isoformat())
        self.assertEqual(v["previous_session"], 1)
        earlier = [q for q in v["quotes"] if q["previous_session"]]
        self.assertEqual(len(earlier), 1)
        self.assertIsNone(earlier[0]["last_update_time"])           # an earlier session: its date only
        self.assertEqual(earlier[0]["last_update"], (TODAY - timedelta(days=1)).isoformat())
        self.assertEqual(v["quotes"][0]["last_update_time"], "16:10:05")

    def test_snapshot_without_times(self):
        """Snapshots downloaded before the time was kept: dates only, no error."""
        old = Path(self.tmp.name) / "old"
        old.mkdir(exist_ok=True)
        oq = pd.read_csv(Path(self.tmp.name) / "eq_option_quotes.csv").drop(columns=["last_update_time"])
        oq.to_csv(old / "eq_option_quotes.csv", index=False)
        pd.read_csv(Path(self.tmp.name) / "eq_vol_quotes.csv").to_csv(old / "eq_vol_quotes.csv", index=False)
        v = fit_view(self.mkt, TICKER, old)
        self.assertFalse(v["has_times"])
        self.assertEqual(v["previous_session"], 1)
        self.assertTrue(all(q["last_update_time"] is None for q in v["quotes"]))

    def test_missing_files(self):
        with tempfile.TemporaryDirectory() as empty, self.assertRaises(FileNotFoundError):
            fit_view(self.mkt, TICKER, Path(empty))


if __name__ == "__main__":
    unittest.main()
