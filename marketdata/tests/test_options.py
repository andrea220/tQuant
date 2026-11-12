"""Listed options for the vol surfaces with an in-memory source: chain parsing and cache, main root, expiry and
strike selection around the forward, four fields, session check. Run from ``marketdata``:
``python -m unittest discover -s tests``."""

import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdbuild.options import (QUOTE_COLUMNS, equity_option_quotes, forward_estimate, load_chain,   # noqa: E402
                             main_root, option_quotes, parse_chain, select_expiries, select_options)
from mdbuild.snapshot import build_tables                                                          # noqa: E402
from mdbuild.universe import parse                                                                 # noqa: E402

TODAY = date(2026, 11, 12)
EXPIRIES = ["11/20/26", "12/18/26", "01/15/27", "02/19/27", "03/19/27", "04/16/27", "06/18/27", "09/17/27",
            "12/17/27", "06/16/28", "12/15/28", "12/20/30", "06/18/32", "06/17/33"]
MONEYNESS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]
OPTIONS = {"chain_max_age_days": 7, "front_expiries": 3, "max_years": 6.2, "moneyness": MONEYNESS,
           "parity_strikes": 2, "stale_days": 3, "rates": {"EUR": "ESTRON Index"}}


def chain_tickers():
    out = [f"AAA IM {e} {cp}{k:g} Equity" for e in EXPIRIES for cp in "CP" for k in range(10, 51)]
    out += [f"AAA5 IM 03/19/27 C{k} Equity" for k in range(20, 31)]          # another root: ignored
    out += ["not an option"]
    return out


class OptionSource:
    """Chain of ``AAA IM Equity`` and quotes of its options (update date ``quote_day``); records the requests."""

    def __init__(self, quote_day=TODAY):
        self.quote_day = quote_day
        self.bulk_calls, self.reference_calls = [], []

    def bulk(self, ticker, field):
        self.bulk_calls.append((ticker, field))
        if field == "OPT_CHAIN" and ticker == "AAA IM Equity":
            return pd.DataFrame({"Security Description": chain_tickers()})
        if field == "BDVD_ALL_PROJECTIONS" and ticker == "AAA IM Equity":
            return pd.DataFrame({"Ex-Date": ["2027-05-18", "2028-05-16"], "Amount Per Share": [1.0, 1.1]})
        return pd.DataFrame()

    def reference(self, tickers, fields):
        self.reference_calls.append((tuple(tickers), tuple(fields)))
        static = {"AAA IM Equity": [25.4, "EUR", "AAA SPA", "2026-11-12"],
                  "ESTRON Index": [2.0, "EUR", "ESTR", "2026-11-11"]}
        rows = {}
        for t in tickers:
            if t in static:
                rows[t] = static[t]
            elif fields == ["OPT_EXER_TYP"]:
                rows[t] = ["American"]
            else:                                         # PX_BID, PX_ASK, PX_LAST, LAST_UPDATE_BID_RT
                k = float(parse_chain([t])["strike"].iloc[0])
                stale = k == 13.0                           # one stale quote
                day = self.quote_day - timedelta(days=10) if stale else self.quote_day
                stamp = "16:29:32" if day == TODAY else day.isoformat()     # a time today, a date before
                rows[t] = [0.5 + k / 100, 0.55 + k / 100, 0.52 + k / 100, stamp]
        return pd.DataFrame(rows, index=fields).T


class TestChain(unittest.TestCase):
    def test_parse_and_root(self):
        chain = parse_chain(chain_tickers())
        self.assertEqual(len(chain), len(EXPIRIES) * 2 * 41 + 11)
        row = chain.iloc[0]
        self.assertEqual((row.root, row.exch, row.expiry, row.option_type, row.strike),
                         ("AAA", "IM", date(2026, 11, 20), "C", 10.0))
        self.assertEqual(main_root("ENI IM Equity"), ("ENI", "IM"))
        self.assertEqual(main_root("ENI IM Equity", {"ENI IM Equity": "ENI3 IM"}), ("ENI3", "IM"))

    def test_cache(self):
        src = OptionSource()
        with tempfile.TemporaryDirectory() as tmp:
            first = load_chain(src, "AAA IM Equity", "AAA_IM", Path(tmp), TODAY, 7)
            self.assertEqual(set(first["root"]), {"AAA"})                       # main root only
            self.assertEqual(set(first["exercise"]), {"American"})
            self.assertEqual(len(src.bulk_calls), 1)
            self.assertEqual(len(src.reference_calls), 1)                       # exercise style: one option
            again = load_chain(src, "AAA IM Equity", "AAA_IM", Path(tmp), TODAY + timedelta(days=6), 7)
            self.assertEqual(len(src.bulk_calls), 1)                            # from the cache
            pd.testing.assert_frame_equal(again[["expiry", "option_type", "strike"]],
                                          first[["expiry", "option_type", "strike"]])
            load_chain(src, "AAA IM Equity", "AAA_IM", Path(tmp), TODAY + timedelta(days=7), 7)
            self.assertEqual(len(src.bulk_calls), 2)                            # refreshed after a week


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.chain = parse_chain(chain_tickers())
        self.chain = self.chain[self.chain["root"] == "AAA"]

    def test_expiries(self):
        out = select_expiries(self.chain["expiry"], TODAY, 3, 6.2)
        self.assertEqual(out[:3], [date(2026, 11, 20), date(2026, 12, 18), date(2027, 1, 15)])
        self.assertTrue(all(e.month in (3, 6, 9, 12) for e in out[3:]))
        self.assertNotIn(date(2027, 4, 16), out)
        self.assertEqual(out[-1], date(2032, 6, 18))                              # 2033 beyond 6.2 years

    def test_forward_estimate(self):
        divs = pd.DataFrame({"ex_date": ["2027-05-18", "2028-05-16"], "amount": [1.0, 1.1]})
        f = forward_estimate(25.4, divs, 0.02, TODAY, date(2027, 12, 17))
        self.assertAlmostEqual(f, (25.4 - 1.0) * 2.718281828459045 ** (0.02 * 400 / 365), places=10)

    def test_strikes_around_the_forward(self):
        f = 21.3
        sel = select_options(self.chain, {date(2027, 12, 17): f}, MONEYNESS, 2)
        self.assertTrue(((sel["option_type"] == "P") | (sel["strike"] >= 19)).all())
        otm = sel[~sel["strike"].isin([20.0, 21.0, 22.0, 23.0])]
        self.assertTrue(((otm["option_type"] == "P") == (otm["strike"] < f)).all())   # out of the money
        for k in (20.0, 21.0, 22.0, 23.0):                                      # parity: both types
            self.assertEqual(set(sel.loc[sel["strike"] == k, "option_type"]), {"C", "P"})
        self.assertFalse(sel["bloomberg"].duplicated().any())
        self.assertEqual(sel["strike"].min(), 11.0)                              # 50% of F
        self.assertEqual(sel["strike"].max(), 32.0)                              # 150% of F
        self.assertLessEqual(len(sel), len(MONEYNESS) + 4)


class TestQuoteDay(unittest.TestCase):
    def test_time_is_today_date_is_that_day(self):
        from mdbuild.options import quote_day
        self.assertEqual(quote_day("16:29:32", TODAY), TODAY)
        self.assertEqual(quote_day("2026-11-11", TODAY), date(2026, 11, 11))
        self.assertEqual(quote_day("11/11/2026", TODAY), date(2026, 11, 11))
        self.assertIsNone(quote_day(None, TODAY))
        self.assertIsNone(quote_day(float("nan"), TODAY))


class TestQuotes(unittest.TestCase):
    def selection(self):
        chain = parse_chain(chain_tickers())
        chain = chain[chain["root"] == "AAA"].assign(exercise="American")
        return select_options(chain, {date(2027, 12, 17): 24.0}, MONEYNESS, 2)

    def test_four_fields_and_stale_quotes(self):
        src = OptionSource()
        out = option_quotes(src, self.selection(), "AAA_IM", "EUR", "AAA IM Equity", 25.4, TODAY, None, 3)
        self.assertEqual(list(out.columns), QUOTE_COLUMNS)
        self.assertEqual(src.reference_calls[-1][1], ("PX_BID", "PX_ASK", "PX_LAST", "LAST_UPDATE_BID_RT"))
        self.assertNotIn(13.0, set(out["strike"]))                              # 10 days old
        self.assertEqual(set(out["quote_date"]), {TODAY.isoformat()})
        self.assertEqual(set(out["ref_spot"]), {25.4})

    def test_session_check_previous_close(self):
        """Quotes of yesterday's session: the reference spot is yesterday's close, not today's spot."""
        yesterday = TODAY - timedelta(days=1)
        history = pd.DataFrame({"date": pd.to_datetime([yesterday, TODAY]), "ticker": "AAA IM Equity",
                                "value": [24.9, 25.4]})
        out = option_quotes(OptionSource(yesterday), self.selection(), "AAA_IM", "EUR", "AAA IM Equity", 25.4,
                            TODAY, history, 3)
        self.assertEqual(set(out["quote_date"]), {yesterday.isoformat()})
        self.assertEqual(set(out["ref_spot"]), {24.9})
        self.assertTrue(out["spot_source"].str.startswith("close").all())

    def test_snapshot_table(self):
        src = OptionSource()
        universe = [parse("AAA IM Equity"), parse("ESTRON Index")]
        days = pd.bdate_range(end=TODAY.isoformat(), periods=300)
        hist = pd.DataFrame({"date": days, "ticker": "AAA IM Equity",
                             "value": 25.4 * (1 + 0.01 * pd.Series(range(300)).mod(7).to_numpy())})
        fields = {"price": "PX_LAST", "currency": "CRNCY", "name": "NAME", "last_update": "LAST_UPDATE_DT",
                  "dividends": "BDVD_ALL_PROJECTIONS", "curve_rates": "CURVE_TENOR_RATES"}
        with tempfile.TemporaryDirectory() as tmp:
            tables = build_tables(src, universe, fields, TODAY, hist, 252, (), OPTIONS, Path(tmp))
        quotes = tables["eq_option_quotes.csv"]
        self.assertEqual(list(quotes.columns), QUOTE_COLUMNS)
        expiries = sorted(set(quotes["expiry"]))
        self.assertEqual(len(expiries), len(select_expiries(parse_chain(chain_tickers())["expiry"], TODAY, 3, 6.2)))
        per_expiry = quotes.groupby("expiry").size()
        self.assertLessEqual(per_expiry.max(), len(MONEYNESS) + 4)              # about 20 options an expiry
        self.assertTrue(set(quotes["ticker"]) == {"AAA_IM"})


if __name__ == "__main__":
    unittest.main()
