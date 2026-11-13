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
# main root: monthlies (third Fridays), one weekly (12/04), quarterlies and long dated; weeklies also on the weekly
# roots 2AAA (11/13) and 4AAA (11/27), as on IDEM
EXPIRIES = ["11/20/26", "12/04/26", "12/18/26", "01/15/27", "02/19/27", "03/19/27", "04/16/27", "06/18/27",
            "09/17/27", "12/17/27", "03/17/28", "06/16/28", "12/15/28", "12/20/30", "06/18/32", "06/17/33"]
WEEKLY_ROOTS = {"2AAA": "11/13/26", "4AAA": "11/27/26"}
MONEYNESS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]
OPTIONS = {"chain_max_age_days": 7, "weekly_expiries": 2, "monthly_months": 3, "quarterly_months": 18,
           "max_years": 6.2, "strike_range": "40%-200%", "moneyness": MONEYNESS,
           "parity_strikes": 2, "stale_days": 3, "rates": {"EUR": "ESTRON Index"}}


def chain_tickers():
    out = [f"AAA IM {e} {cp}{k:g} Equity" for e in EXPIRIES for cp in "CP" for k in range(10, 51)]
    out += [f"{root} IM {e} {cp}{k} Equity" for root, e in WEEKLY_ROOTS.items() for cp in "CP" for k in range(15, 36)]
    out += [f"AAA5 IM 03/19/27 C{k} Equity" for k in range(20, 31)]          # another root: ignored
    out += [f"AAA2 GR 11/13/26 C{k} Equity" for k in range(20, 31)]          # another exchange: ignored
    out += ["not an option"]
    return out


class OptionSource:
    """Chain of ``AAA IM Equity`` (``CHAIN_EXPIRATION_DATES``, ``CHAIN_TICKERS`` with their filters, as Bloomberg:
    American options only, tickers without the yellow key) and quotes of its options (update date ``quote_day``);
    records the requests."""

    SPOT = 25.4

    def __init__(self, quote_day=TODAY):
        self.quote_day = quote_day
        self.bulk_calls, self.reference_calls = [], []

    def _chain(self, exch):
        chain = parse_chain(chain_tickers())
        return chain[(chain["exch"] == exch) & (chain["root"] != "AAA5")]          # AAA5: a European root

    def bulk(self, ticker, field, overrides=None):
        self.bulk_calls.append((ticker, field, dict(overrides or {})))
        o = overrides or {}
        if field == "CHAIN_EXPIRATION_DATES" and ticker == "AAA IM Equity":
            return pd.DataFrame({"Expiration": sorted(set(self._chain(o.get("CHAIN_EXCH_OVRD", "IM"))["expiry"]))})
        if field == "CHAIN_TICKERS" and ticker == "AAA IM Equity":
            chain = self._chain(o.get("CHAIN_EXCH_OVRD", "IM"))
            expiry = pd.Timestamp(o["CHAIN_EXP_DT_OVRD"]).date()
            lo, hi = (float(x.rstrip("%")) / 100 * self.SPOT for x in o["CHAIN_STRIKE_PX_OVRD"].split("-"))
            sel = chain[(chain["expiry"] == expiry) & (chain["option_type"] == o["CHAIN_PUT_CALL_TYPE_OVRD"])
                        & chain["strike"].between(lo, hi)]
            return pd.DataFrame({"Ticker": [t.rsplit(" ", 1)[0] for t in sel["bloomberg"]]})
        if field == "BDVD_ALL_PROJECTIONS" and ticker == "AAA IM Equity":
            return pd.DataFrame({"Ex-Date": ["2027-05-18", "2028-05-16"], "Amount Per Share": [1.0, 1.1]})
        return pd.DataFrame()

    def reference(self, tickers, fields):
        self.reference_calls.append((tuple(tickers), tuple(fields)))
        static = {"AAA IM Equity": [25.4, "EUR", "AAA SPA", "2026-11-12"],
                  "ESTRON Index": [2.0, "EUR", "ESTR", "2026-11-11"]}
        rows = {}
        for t in tickers:
            if t in static and fields[0] == "PX_BID":        # the underlying with the quotes: live spot
                rows[t] = [None, None, 25.6, "16:29:40"]
            elif t in static:
                rows[t] = static[t]
            elif fields == ["OPT_EXER_TYP"]:
                rows[t] = ["American"]
            else:                                         # PX_BID, PX_ASK, PX_LAST, LAST_UPDATE_BID_RT
                k = float(parse_chain([t])["strike"].iloc[0])
                stale = k == 12.0                           # one stale quote (selected: 50% of F = 24)
                day = self.quote_day - timedelta(days=10) if stale else self.quote_day
                stamp = "16:29:32" if day == TODAY else day.isoformat()     # a time today, a date before
                rows[t] = [0.5 + k / 100, 0.55 + k / 100, 0.52 + k / 100, stamp]
        return pd.DataFrame(rows, index=fields).T


class TestChain(unittest.TestCase):
    def test_parse_and_root(self):
        chain = parse_chain(chain_tickers())
        self.assertEqual(len(chain), len(EXPIRIES) * 2 * 41 + 2 * 2 * 21 + 11 + 11)
        row = chain.iloc[0]
        self.assertEqual((row.root, row.exch, row.expiry, row.option_type, row.strike),
                         ("AAA", "IM", date(2026, 11, 20), "C", 10.0))
        self.assertEqual(main_root("ENI IM Equity"), ("ENI", "IM"))
        self.assertEqual(main_root("ENI IM Equity", {"ENI IM Equity": "ENI3 IM"}), ("ENI3", "IM"))
        self.assertEqual(main_root("SX5E Index"), ("SX5E", None))

    def test_index_options(self):
        from mdbuild.options import is_weekly_root
        chain = parse_chain(["SX5E 12/18/26 C5000 Index", "SPX US 12/18/26 P7870 Index", "SPXW US 11/20/26 C3200 Index"])
        self.assertEqual(list(chain["root"]), ["SX5E", "SPX", "SPXW"])
        self.assertEqual(list(chain["exch"]), ["", "US", "US"])
        self.assertEqual(list(chain["strike"]), [5000.0, 7870.0, 3200.0])
        self.assertTrue(is_weekly_root("SPXW", "SPX") and is_weekly_root("2ENI", "ENI"))
        self.assertFalse(is_weekly_root("ENI5", "ENI") or is_weekly_root("SPX", "SPX"))

    def test_chain_by_expiry(self):
        """Expiries (one request), then calls and puts of each selected expiry within the strike range (two requests
        an expiry): the selected expiries only, main and weekly roots, main exchange, strikes 40%-200% of the spot."""
        src = OptionSource()
        with tempfile.TemporaryDirectory() as tmp:
            chain = load_chain(src, "AAA IM Equity", "AAA_IM", Path(tmp), TODAY, OPTIONS)
        selected = select_expiries(parse_chain(chain_tickers())["expiry"], TODAY, 2, 3, 18, 6.2)
        self.assertEqual(sorted(set(chain["expiry"])), selected)
        self.assertEqual(len(src.bulk_calls), 1 + 2 * len(selected))
        self.assertEqual(src.bulk_calls[0][2]["CHAIN_EXP_DT_OVRD"], "ALL")
        ticker_calls = [c[2] for c in src.bulk_calls[1:]]
        self.assertTrue(all(c["CHAIN_EXP_MATCH_OVRD"] == "Exact" and c["CHAIN_EXCH_OVRD"] == "IM"
                            and c["CHAIN_ADJ_CONTRACT_OVRD"] == "N" for c in ticker_calls))
        self.assertEqual(set(chain["root"]), {"AAA", "2AAA"})                    # 4AAA's expiry is not selected
        self.assertEqual(set(chain["exch"]), {"IM"})
        self.assertTrue(chain["strike"].between(0.4 * 25.4, 2.0 * 25.4).all())
        self.assertEqual(set(chain["exercise"]), {"American"})
        self.assertFalse(src.reference_calls)                                   # no field requests

    def test_index_without_exchange(self):
        """An index (no exchange in its tickers): no exchange filter, European exercise."""
        from mdbuild.options import chain_overrides
        self.assertNotIn("CHAIN_EXCH_OVRD", chain_overrides("SX5E Index", "20271217", "C", "40%-200%"))
        self.assertEqual(chain_overrides("ENI IM Equity", "20271217", "C", "40%-200%")["CHAIN_EXCH_OVRD"], "IM")

    def test_cache(self):
        src = OptionSource()
        with tempfile.TemporaryDirectory() as tmp:
            first = load_chain(src, "AAA IM Equity", "AAA_IM", Path(tmp), TODAY, OPTIONS)
            n = len(src.bulk_calls)
            again = load_chain(src, "AAA IM Equity", "AAA_IM", Path(tmp), TODAY, OPTIONS)
            self.assertEqual(len(src.bulk_calls), n)                            # from the cache
            pd.testing.assert_frame_equal(again[["expiry", "option_type", "strike"]],
                                          first[["expiry", "option_type", "strike"]])
            load_chain(src, "AAA IM Equity", "AAA_IM", Path(tmp), TODAY + timedelta(days=1), OPTIONS)
            self.assertGreater(len(src.bulk_calls), n)                          # first expiry (11/13) passed
            m = len(src.bulk_calls)
            load_chain(src, "AAA IM Equity", "AAA_IM", Path(tmp), TODAY + timedelta(days=1), OPTIONS)
            self.assertEqual(len(src.bulk_calls), m)


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.chain = parse_chain(chain_tickers())
        self.chain = self.chain[self.chain["root"] == "AAA"]

    def test_expiries(self):
        """The first 2 expiries, monthlies to 3 months, quarterlies to 18 months, then one a year (the nearest to
        December) to 6.2 years."""
        expiries = parse_chain(chain_tickers())["expiry"]
        out = select_expiries(expiries, TODAY, 2, 3, 18, 6.2)
        self.assertEqual(out, [date(2026, 11, 13), date(2026, 11, 20),                       # weeklies
                               date(2026, 12, 18), date(2027, 1, 15),                         # monthlies
                               date(2027, 3, 19), date(2027, 6, 18), date(2027, 9, 17),       # quarterlies
                               date(2027, 12, 17), date(2028, 3, 17),
                               date(2028, 12, 15), date(2030, 12, 20), date(2032, 6, 18)])    # one a year

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
        self.assertIn("AAA IM Equity", src.reference_calls[-1][0])               # spot in the same request
        self.assertNotIn("AAA IM Equity", set(out["bloomberg"]))
        stale = out[out["strike"] == 12.0]                                      # bid / ask 10 days old: last only
        self.assertTrue(len(stale) and stale["bid"].isna().all() and stale["ask"].isna().all())
        self.assertAlmostEqual(float(stale["last"].iloc[0]), 0.52 + 0.12)
        fresh = out[out["strike"] != 12.0]
        self.assertTrue(fresh["bid"].notna().all() and fresh["ask"].notna().all())
        self.assertEqual(set(out["quote_date"]), {TODAY.isoformat()})
        self.assertEqual(set(out["ref_spot"]), {25.6})                         # live spot, not the snapshot's
        self.assertEqual(set(out["spot_source"]), {"with the quotes"})

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
        self.assertEqual(len(expiries), len(select_expiries(parse_chain(chain_tickers())["expiry"], TODAY,
                                                            2, 3, 18, 6.2)))
        self.assertIn("2026-11-13", expiries)                                    # weekly root
        per_expiry = quotes.groupby("expiry").size()
        self.assertLessEqual(per_expiry.max(), len(MONEYNESS) + 4)              # about 20 options an expiry
        self.assertTrue(set(quotes["ticker"]) == {"AAA_IM"})


if __name__ == "__main__":
    unittest.main()
