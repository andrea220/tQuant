"""mdbuild with an in-memory source (no Bloomberg): universe parsing, incremental history, correlations, snapshot
tables in the ``MarketEnvironment.from_folder`` layout. Run from ``marketdata``: ``python -m unittest discover -s tests``."""

import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdbuild.history import correlations, read_history, update_history   # noqa: E402
from mdbuild.snapshot import build_tables, curve_quotes, run             # noqa: E402
from mdbuild.universe import Curve, load_curves, load_universe, parse    # noqa: E402

TODAY = date(2026, 11, 12)
FIELDS = {"price": "PX_LAST", "currency": "CRNCY", "name": "NAME", "last_update": "LAST_UPDATE_DT",
          "dividends": "BDVD_ALL_PROJECTIONS", "curve_rates": "CURVE_TENOR_RATES"}
TICKERS = ["AAA IM Equity", "BBB US Equity", "EURUSD Curncy", "ESTRON Index"]
TARGET = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, -0.2], [0.3, -0.2, 1.0]])


class FakeSource:
    """Business-day prices from correlated log returns (TARGET), fixed reference data and dividends; records the
    history requests."""

    def __init__(self):
        dates = pd.bdate_range("2014-01-01", TODAY.isoformat())
        rng = np.random.default_rng(0)
        returns = rng.standard_normal((len(dates), 3)) @ np.linalg.cholesky(TARGET).T * 0.01
        prices = np.exp(np.cumsum(returns, axis=0)) * np.array([20.0, 300.0, 1.1])
        self.prices = {t: pd.Series(prices[:, k], index=dates) for k, t in enumerate(TICKERS[:3])}
        self.calls = []

    def reference(self, tickers, fields):
        data = {"AAA IM Equity": [25.4, "EUR", "AAA SPA", "2026-11-12"],
                "BBB US Equity": [355.6, "USD", "BBB INC", "2026-11-12"],
                "EURUSD Curncy": [1.1092, "USD", "EUR-USD X-RATE", "2026-11-12"],
                "ESTRON Index": [2.439, "EUR", "ESTR", "2026-11-11"]}
        return pd.DataFrame({t: data[t] for t in tickers}, index=fields).T

    def bulk(self, ticker, field):
        if field == "CURVE_TENOR_RATES":
            members = {"YCSW0514 Index": [("1W", "EESWE1Z BGN Curncy", 2.445), ("1Y", "EESWE1 BGN Curncy", 2.912),
                                          ("10Y", "EESWE10 BGN Curncy", 3.13)],
                       "YCSW0045 Index": [("6M", "EUR006M Index", 3.023), ("7M", "EUFR0AG BGN Curncy", 3.13),
                                          ("18M", "EUFR011F BGN Curncy", 3.398), ("2Y", "EUSA2 BGN Curncy", 3.357)]}
            return pd.DataFrame([{"Tenor": t, "Tenor Ticker": k, "Ask Yield": q + 0.002, "Mid Yield": q,
                                  "Bid Yield": q - 0.002, "Last Update": "2026-11-12"} for t, k, q in members[ticker]])
        if ticker != "AAA IM Equity":
            return pd.DataFrame()
        return pd.DataFrame({"Ex-Date": ["2026-10-01", "2026-12-07", "2027-03-22", ""],
                             "Amount Per Share": [0.5, 0.46, 0.28, 0.3]})

    def history(self, tickers, field, start, end):
        self.calls.append((tuple(tickers), start, end))
        rows = []
        for t in tickers:
            s = self.prices[t]
            s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
            rows += [{"date": d, "ticker": t, "value": v} for d, v in s.items()]
        return pd.DataFrame(rows, columns=["date", "ticker", "value"])


class TestUniverse(unittest.TestCase):
    def test_parse(self):
        self.assertEqual((parse("ENI IM Equity").kind, parse("ENI IM Equity").code), ("equity", "ENI_IM"))
        fx = parse("EURUSD Curncy")
        self.assertEqual((fx.kind, fx.code, fx.fx_base, fx.fx_quote), ("fx", "EURUSD", "EUR", "USD"))
        self.assertEqual(parse("SONIO/N Index").code, "SONION")
        for bad in ("ENI IM", "EURUS Curncy", "X Govt"):
            with self.assertRaises(ValueError):
                parse(bad)

    def test_universe_file(self):
        universe = load_universe(Path(__file__).resolve().parents[1] / "config" / "universe.yaml")
        kinds = [i.kind for i in universe]
        self.assertEqual((kinds.count("equity"), kinds.count("fx"), kinds.count("fixing")), (10, 5, 4))


class TestCurves(unittest.TestCase):
    def test_curves_in_the_universe(self):
        curves = load_curves(Path(__file__).resolve().parents[1] / "config" / "universe.yaml")
        self.assertEqual([(c.ccy, c.name, c.kind) for c in curves],
                         [("EUR", "ESTR", "ois"), ("EUR", "6M", "ibor"), ("USD", "SOFR", "ois"), ("GBP", "SONIA", "ois")])

    def test_quotes_of_the_members(self):
        quotes = curve_quotes(FakeSource(), [Curve("EUR", "ESTR", "YCSW0514 Index"), Curve("EUR", "6M", "YCSW0045 Index")],
                              "CURVE_TENOR_RATES")
        self.assertEqual(list(quotes.columns), ["ccy", "curve", "tenor", "instrument", "quote", "bloomberg", "last_update"])
        self.assertEqual(quotes["instrument"].tolist(), ["ois", "ois", "ois", "fixing", "fra", "fra", "swap"])
        self.assertAlmostEqual(quotes.loc[quotes.bloomberg == "EUSA2 BGN Curncy", "quote"].item(), 0.03357)   # mid, decimal


class TestHistory(unittest.TestCase):
    def test_incremental_update(self):
        src = FakeSource()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.parquet"
            first = update_history(path, src, TICKERS[:2], "PX_LAST", 10, date(2026, 6, 30))
            self.assertEqual(len(src.calls), 1)
            self.assertEqual(src.calls[0][1], date(2016, 6, 30))          # ten years back
            full = update_history(path, src, TICKERS[:3], "PX_LAST", 10, TODAY)
            # the new ticker over the whole window, the stored ones only after their last date
            starts = {c[0]: c[1] for c in src.calls[1:]}
            self.assertEqual(starts[("EURUSD Curncy",)], date(2016, 11, 12))
            self.assertEqual(starts[("AAA IM Equity", "BBB US Equity")], date(2026, 7, 1))
            self.assertFalse(full.duplicated(["date", "ticker"]).any())
            pd.testing.assert_frame_equal(read_history(path).reset_index(drop=True), full[["date", "ticker", "value"]]
                                          .reset_index(drop=True))
            # the window moves with today: the oldest days of the first download are trimmed
            self.assertEqual(len(full[full.ticker == "AAA IM Equity"]), len(pd.bdate_range("2016-11-12", TODAY)))
            self.assertLess(full["date"].min(), first["date"].max())

    def test_correlations_of_log_returns(self):
        src = FakeSource()
        hist = src.history(TICKERS[:3], "PX_LAST", date(2016, 1, 1), TODAY)
        names = {"AAA IM Equity": "EQ:EUR:AAA_IM", "BBB US Equity": "EQ:USD:BBB_US", "EURUSD Curncy": "FX:USD:EURUSD"}
        out = correlations(hist, names, 2500)
        self.assertEqual(list(out.columns), ["factor_1", "factor_2", "corr"])
        self.assertEqual(len(out), 3)
        got = {(r.factor_1, r.factor_2): r.corr for r in out.itertuples()}
        self.assertAlmostEqual(got[("EQ:EUR:AAA_IM", "EQ:USD:BBB_US")], 0.6, delta=0.05)
        self.assertAlmostEqual(got[("EQ:USD:BBB_US", "FX:USD:EURUSD")], -0.2, delta=0.05)
        matrix = np.eye(3)
        idx = {n: k for k, n in enumerate(names.values())}
        for (a, b), c in got.items():
            matrix[idx[a], idx[b]] = matrix[idx[b], idx[a]] = c
        self.assertGreater(np.linalg.eigvalsh(matrix).min(), 0.0)


class TestSnapshot(unittest.TestCase):
    def test_tables_in_the_market_layout(self):
        src = FakeSource()
        universe = [parse(t) for t in TICKERS]
        hist = src.history(TICKERS[:3], "PX_LAST", date(2016, 1, 1), TODAY)
        tables = build_tables(src, universe, FIELDS, TODAY, hist, 252)
        self.assertEqual(list(tables["eq_spot.csv"].columns), ["ticker", "ccy", "spot", "repo", "div_yield"])
        self.assertEqual(tables["eq_spot.csv"]["ticker"].tolist(), ["AAA_IM", "BBB_US"])
        div = tables["eq_dividends.csv"]
        self.assertEqual(div["ex_date"].tolist(), ["2026-12-07", "2027-03-22"])        # past and blank dropped
        self.assertTrue((div["pay_date"] == div["ex_date"]).all())
        fx = tables["fx_spot.csv"].iloc[0]
        self.assertEqual((fx.base, fx.quote, fx.spot), ("EUR", "USD", 1.1092))
        fix = tables["ir_fixings.csv"].iloc[0]
        self.assertEqual((fix["index"], fix["fixing_date"]), ("ESTRON", "2026-11-11"))
        self.assertAlmostEqual(fix["value"], 0.02439)
        self.assertEqual(len(tables["correlations.csv"]), 3)

    def test_run_writes_a_snapshot(self):
        settings = {"fields": FIELDS, "history": {"years": 10}, "correlation": {"window_days": 252},
                    "output": {"snapshots": "live", "history": "history"}}
        with tempfile.TemporaryDirectory() as tmp:
            folder = run(FakeSource(), [parse(t) for t in TICKERS], settings, Path(tmp), datetime(2026, 11, 12, 17, 30),
                         [Curve("EUR", "ESTR", "YCSW0514 Index")])
            self.assertEqual(folder, Path(tmp) / "live" / "20261112" / "1730")
            self.assertEqual(sorted(p.name for p in folder.iterdir()),
                             ["correlations.csv", "eq_dividends.csv", "eq_spot.csv", "fx_spot.csv", "ir_fixings.csv",
                              "ir_quotes.csv", "snapshot_info.json"])
            self.assertTrue((Path(tmp) / "history" / "prices.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
