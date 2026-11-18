"""tqstudio market data universe: universe file, status in a snapshot, download runner (no Bloomberg)."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import socket  # noqa: E402
from tqstudio.views.universe import (DownloadRuns, add_ticker, allowed_client, app_settings,  # noqa: E402
                                     bloomberg_endpoint, bloomberg_reachable, read_universe, remove_ticker,
                                     resolve_python, server_authorized, universe_view)

UNIVERSE = """\
tickers:
  # Italy
  - ENI IM Equity
  # - UCG IM Equity
  - EURUSD Curncy
  # - EURCHF Curncy
  - ESTRON Index
equity_indices:
  - SX5E Index
  # - SPX Index
implied_spot:
  - SX5E Index
option_roots:
  ENI IM Equity: ENX IM
curves:
  EUR ESTR: YCSW0514 Index
  EUR 6M: YCSW0045 Index
"""

TABLES = {
    "eq_spot.csv": "ticker,ccy,spot,repo,div_yield\nENI_IM,EUR,25.0,,\nSX5E,EUR,5000.0,,\n",
    "eq_dividends.csv": "ticker,ccy,ex_date,pay_date,amount\nENI_IM,EUR,2027-01-20,2027-01-22,0.25\n"
                        "ENI_IM,EUR,2027-05-20,2027-05-22,0.25\n",
    "eq_option_quotes.csv": "ticker,ccy,expiry,strike,option_type,exercise,bid,ask,last,last_update,quote_date,"
                            "ref_spot,spot_source,bloomberg\n"
                            "ENI_IM,EUR,2026-12-18,24,P,American,0.5,0.6,0.55,2026-11-13,2026-11-13,25,with the quotes,x\n"
                            "ENI_IM,EUR,2026-12-18,26,C,American,0.4,0.5,0.45,2026-11-13,2026-11-13,25,with the quotes,y\n",
    "eq_vol_calibration.csv": "ticker,spot_source,expiry,error\nENI_IM,with the quotes,2026-12-18,\n"
                              "ENI_IM,with the quotes,2027-03-19,\nENI_IM,with the quotes,2027-06-18,too few quotes\n",
    "eq_vol_quotes.csv": "ticker,ccy,expiry,strike,option_type,iv_bid,iv_mid,iv_ask,settlement,used,model_vol\n"
                         "ENI_IM,EUR,2026-12-18,24,P,0.20,0.21,0.22,False,True,0.215\n"
                         "ENI_IM,EUR,2026-12-18,26,C,0.18,0.19,0.20,False,True,0.25\n"
                         "ENI_IM,EUR,2027-03-19,26,C,0.18,0.19,0.20,True,True,0.30\n",
    "fx_spot.csv": "base,quote,spot\nEUR,USD,1.10\n",
    "ir_fixings.csv": "index,ccy,fixing_date,value,bloomberg\nESTRON,EUR,2026-11-12,0.0244,ESTRON Index\n",
    "ir_quotes.csv": "ccy,curve,tenor,instrument,quote,bloomberg,last_update\n"
                     "EUR,ESTR,1W,ois,0.0244,a,2026-11-12\nEUR,ESTR,1Y,ois,0.0230,b,2026-11-13\n",
    "ir_curves.csv": "ccy,curve,pillar,value,quote,daycount,interp\nEUR,ESTR,2026-11-24,0.0245,zero,A365,MC\n"
                     "EUR,ESTR,2027-11-24,0.0231,zero,A365,MC\n",
}


class TestMarketUniverse(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.universe = base / "universe.yaml"
        self.universe.write_text(UNIVERSE, encoding="utf-8")
        self.snapshot = base / "live" / "20261113" / "1619"
        self.snapshot.mkdir(parents=True)
        for name, text in TABLES.items():
            (self.snapshot / name).write_text(text, encoding="utf-8")
        info = {"created": "2026-11-13T16:19:25", "history": {"rows": 100, "first": "2016-11-14"},
                "universe": ["ENI IM Equity", "EURUSD Curncy", "ESTRON Index", "SX5E Index"]}
        (self.snapshot / "snapshot_info.json").write_text(json.dumps(info), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_universe(self):
        u = read_universe(self.universe)
        eq = {r["bbg"]: r for r in u["equity"]}
        self.assertEqual(set(eq), {"ENI IM Equity", "UCG IM Equity", "SX5E Index", "SPX Index"})
        self.assertTrue(eq["ENI IM Equity"]["included"])
        self.assertFalse(eq["UCG IM Equity"]["included"])          # commented line: excluded
        self.assertEqual((eq["SX5E Index"]["type"], eq["SPX Index"]["type"]), ("index", "index"))
        self.assertFalse(eq["SPX Index"]["included"])
        self.assertEqual(eq["ENI IM Equity"]["code"], "ENI_IM")
        self.assertEqual([(r["bbg"], r["included"]) for r in u["fx"]], [("EURUSD Curncy", True), ("EURCHF Curncy", False)])
        self.assertEqual([r["code"] for r in u["fixing"]], ["ESTRON"])
        self.assertEqual([(c["name"], c["kind"]) for c in u["curves"]], [("EUR ESTR", "ois"), ("EUR 6M", "ibor")])
        self.assertEqual(u["option_roots"], {"ENI IM Equity": "ENX IM"})
        self.assertEqual(u["implied_spot"], ["SX5E Index"])

    def test_status_in_snapshot(self):
        v = universe_view(self.snapshot, self.universe)
        self.assertEqual(v["snapshot"]["date"], "2026-11-13")
        self.assertEqual(v["snapshot"]["set"], "1619")
        eni = next(r for r in v["equity"] if r["code"] == "ENI_IM")
        self.assertTrue(eni["in_snapshot"])
        self.assertEqual((eni["spot"], eni["ccy"], eni["dividends"], eni["options"]), (25.0, "EUR", 2, 2))
        self.assertEqual(eni["expiries"], 2)                          # the expiry with an error has no smile
        self.assertAlmostEqual(eni["within_bid_ask"], 0.5)            # two-sided quotes only (the settlement is out)
        self.assertEqual(eni["option_root"], "ENX IM")
        sx5e = next(r for r in v["equity"] if r["code"] == "SX5E")
        self.assertTrue(sx5e["implied_spot"])
        self.assertIsNone(sx5e["within_bid_ask"])
        self.assertFalse(next(r for r in v["equity"] if r["code"] == "UCG_IM")["in_snapshot"])
        self.assertEqual([(r["code"], r["in_snapshot"], r["spot"]) for r in v["fx"]],
                         [("EURUSD", True, 1.10), ("EURCHF", False, None)])
        self.assertEqual(v["fixing"][0]["value"], 0.0244)
        estr, six = v["curves"]
        self.assertEqual((estr["quotes"], estr["last_update"], estr["pillars"], estr["bootstrapped"]),
                         (2, "2026-11-13", 2, True))
        self.assertEqual((six["in_snapshot"], six["bootstrapped"]), (False, False))
        json.dumps(v)                                                 # JSON-ready (no numpy types, no NaN)

    def test_added_after_the_snapshot(self):
        """An instrument added to the universe after the download is not in the snapshot."""
        self.universe.write_text(UNIVERSE.replace("  - ENI IM Equity\n", "  - ENI IM Equity\n  - ISP IM Equity\n"),
                                 encoding="utf-8")
        v = universe_view(self.snapshot, self.universe)
        isp = next(r for r in v["equity"] if r["code"] == "ISP_IM")
        self.assertTrue(isp["included"])
        self.assertFalse(isp["in_snapshot"])

    def test_no_snapshot(self):
        v = universe_view(None, self.universe)
        self.assertIsNone(v["snapshot"])
        self.assertFalse(any(r["in_snapshot"] for r in v["equity"]))


class TestAddTicker(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.universe = Path(self.tmp.name) / "universe.yaml"
        self.universe.write_bytes(UNIVERSE.encode("utf-8"))

    def tearDown(self):
        self.tmp.cleanup()

    def active(self, kind):
        return [r["bbg"] for r in read_universe(self.universe)[kind] if r["included"]]

    def test_add_stock_after_the_equities(self):
        out = add_ticker("equity", "isp  im equity", self.universe)
        self.assertEqual(out, {"ticker": "ISP IM Equity", "section": "tickers", "action": "added"})
        lines = self.universe.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[lines.index("  - ISP IM Equity") - 1], "  # - UCG IM Equity")   # with the equities
        self.assertIn("  # Italy", lines)                                                       # comments kept
        self.assertEqual(self.active("equity"), ["ENI IM Equity", "ISP IM Equity", "SX5E Index"])

    def test_uncomment_instead_of_duplicate(self):
        out = add_ticker("equity", "UCG IM Equity", self.universe)
        self.assertEqual(out["action"], "uncommented")
        text = self.universe.read_text(encoding="utf-8")
        self.assertEqual(text.count("UCG IM Equity"), 1)
        self.assertIn("UCG IM Equity", self.active("equity"))
        self.assertEqual(add_ticker("equity", "SPX Index", self.universe)["section"], "equity_indices")
        self.assertIn("SPX Index", self.active("equity"))

    def test_index_fx_and_fixing(self):
        self.assertEqual(add_ticker("equity", "FTSEMIB Index", self.universe)["section"], "equity_indices")
        self.assertIn("FTSEMIB Index", self.active("equity"))
        add_ticker("fx", "EURJPY Curncy", self.universe)
        self.assertEqual(self.active("fx"), ["EURUSD Curncy", "EURJPY Curncy"])
        add_ticker("rates", "EUR003M Index", self.universe)
        self.assertEqual(self.active("fixing"), ["ESTRON Index", "EUR003M Index"])
        self.assertNotIn("EUR003M Index", self.active("equity"))

    def test_rejected(self):
        before = self.universe.read_bytes()
        for cls, ticker in [("equity", "ENI IM Equity"),          # already in
                            ("equity", "EURUSD Curncy"),           # not an equity yellow key
                            ("fx", "EURUSDX Curncy"),              # not a 6-letter pair
                            ("rates", "ENI IM Equity"),
                            ("equity", "ENI"),                     # no yellow key
                            ("credit", "X Corp")]:
            with self.assertRaises(ValueError, msg=f"{cls} {ticker}"):
                add_ticker(cls, ticker, self.universe)
        self.assertEqual(self.universe.read_bytes(), before)       # file untouched

    def test_remove_and_add_back(self):
        out = remove_ticker("equity", "ENI IM Equity", self.universe)
        self.assertEqual(out, {"ticker": "ENI IM Equity", "section": "tickers", "action": "removed"})
        self.assertNotIn("ENI IM Equity", self.active("equity"))
        self.assertIn("  # - ENI IM Equity", self.universe.read_text(encoding="utf-8").splitlines())
        self.assertEqual(add_ticker("equity", "ENI IM Equity", self.universe)["action"], "uncommented")
        self.assertEqual(self.universe.read_text(encoding="utf-8"), UNIVERSE)       # back to the same file

    def test_remove_index_fx_fixing(self):
        """An index is commented in equity_indices only (its implied_spot entry stays: harmless)."""
        self.assertEqual(remove_ticker("equity", "SX5E Index", self.universe)["section"], "equity_indices")
        self.assertNotIn("SX5E Index", self.active("equity"))
        self.assertEqual(read_universe(self.universe)["implied_spot"], ["SX5E Index"])
        remove_ticker("fx", "EURUSD Curncy", self.universe)
        self.assertEqual(self.active("fx"), [])
        remove_ticker("rates", "ESTRON Index", self.universe)
        self.assertEqual(self.active("fixing"), [])

    def test_remove_rejected(self):
        before = self.universe.read_bytes()
        for cls, ticker in [("equity", "UCG IM Equity"),            # already excluded
                            ("equity", "EURUSD Curncy"),            # not an equity
                            ("rates", "YCSW0514 Index"),            # a curve, not a fixing
                            ("credit", "ENI IM Equity")]:
            with self.assertRaises(ValueError, msg=f"{cls} {ticker}"):
                remove_ticker(cls, ticker, self.universe)
        self.assertEqual(self.universe.read_bytes(), before)

    def test_line_endings_kept(self):
        self.universe.write_bytes(UNIVERSE.replace("\n", "\r\n").encode("utf-8"))
        add_ticker("fx", "EURJPY Curncy", self.universe)
        raw = self.universe.read_bytes()
        self.assertEqual(raw.count(b"\n"), raw.count(b"\r\n"))


class TestDownloadRuns(unittest.TestCase):

    def wait(self, runs):
        for _ in range(200):
            status = runs.status()
            if not status["running"]:
                return status
            time.sleep(0.05)
        self.fail("the process did not end")

    def test_run_and_result(self):
        code = "print('history updated'); print('snapshot -> live/20261116/1000')"
        runs = DownloadRuns(cwd=Path(tempfile.gettempdir()), command=[sys.executable, "-c", code])
        self.assertEqual(runs.status(), {"running": False})
        runs.start()
        status = self.wait(runs)
        self.assertEqual(status["exit_code"], 0)
        self.assertEqual(status["result"], "live/20261116/1000")
        self.assertIn("history updated", status["log"])

    def test_one_at_a_time_and_failure(self):
        runs = DownloadRuns(cwd=Path(tempfile.gettempdir()),
                            command=[sys.executable, "-c", "import time; time.sleep(1); raise ValueError('no terminal')"])
        runs.start()
        with self.assertRaises(ValueError):
            runs.start()                                              # a download is already running
        status = self.wait(runs)
        self.assertNotEqual(status["exit_code"], 0)
        self.assertIsNone(status["result"])
        self.assertTrue(any("ValueError" in e for e in status["errors"]))

    def test_python_from_the_settings(self):
        """Without a command: the first Python of the settings that exists and imports blpapi; none: a clear error
        before starting anything."""
        import tqstudio.views.universe as module
        reachable, module.bloomberg_reachable = module.bloomberg_reachable, lambda host, port: True   # a terminal
        self.addCleanup(setattr, module, "bloomberg_reachable", reachable)
        with tempfile.TemporaryDirectory() as tmp:
            settings = Path(tmp) / "settings.yaml"
            settings.write_text("python:\n  - " + str(Path(tmp) / "missing.exe") + "\n  - current\n",
                                encoding="utf-8")
            runs = DownloadRuns(cwd=Path(tmp), settings=settings)
            try:
                import blpapi  # noqa: F401
                has_blpapi = True
            except ImportError:
                has_blpapi = False
            if has_blpapi:                                  # the test Python: `python -m mdbuild` fails here (no module)
                status = self.wait(runs) if runs.start() else None
                self.assertEqual(status["python"], sys.executable)
            settings.write_text("python:\n  - " + str(Path(tmp) / "missing.exe") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                DownloadRuns(cwd=Path(tmp), settings=settings).start()


class TestTerminalCheck(unittest.TestCase):
    """The download checks the Bloomberg port before starting: no terminal, an error at once."""

    def test_closed_and_open_port(self):
        import socket as sk
        server = sk.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            self.assertTrue(bloomberg_reachable("localhost", port))
        finally:
            server.close()
        t0 = time.perf_counter()
        self.assertFalse(bloomberg_reachable("localhost", port))             # nothing listens any more
        self.assertLess(time.perf_counter() - t0, 3.0)

    def test_no_terminal_no_process(self):
        """A closed port: start raises at once (the page shows the popup), nothing is started."""
        import socket as sk
        probe = sk.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()                                                       # a free port: nothing listens
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            (root / "config" / "settings.yaml").write_text(f"bloomberg:\n  host: localhost\n  port: {port}\n",
                                                          encoding="utf-8")
            self.assertEqual(bloomberg_endpoint(root), ("localhost", port))
            runs = DownloadRuns(cwd=root, root=root)
            t0 = time.perf_counter()
            with self.assertRaises(ValueError) as err:
                runs.start()
            self.assertLess(time.perf_counter() - t0, 3.0)
            self.assertIn("Bloomberg terminal", str(err.exception))
            self.assertIsNone(runs.run)                                     # no process


class TestAppSettings(unittest.TestCase):

    def test_resolve_python(self):
        """Candidates in order: a missing path skipped, an interpreter without the module skipped, ``current`` =
        the running Python."""
        self.assertEqual(resolve_python(["C:/no/such/python.exe", "current"], module="json"), sys.executable)
        self.assertEqual(resolve_python([sys.executable], module="json"), sys.executable)
        with self.assertRaises(ValueError) as err:
            resolve_python(["C:/no/such/python.exe", "current"], module="no_such_module_xyz")
        self.assertIn("not found", str(err.exception))
        self.assertIn("no no_such_module_xyz", str(err.exception))

    def test_authorized_pcs_only(self):
        """Only the list counts: no pc by default, not even this one; a local request is this machine by its name."""
        me = socket.gethostname()
        self.assertFalse(allowed_client("127.0.0.1", []))                   # this pc, not in an empty list
        self.assertFalse(allowed_client("127.0.0.1", ["OTHER-PC-XYZ"]))
        self.assertTrue(allowed_client("127.0.0.1", [me]))                  # this pc, by its name
        self.assertTrue(allowed_client("::1", [me.lower()]))                # names case-insensitive
        self.assertFalse(allowed_client(None, [me]))
        self.assertTrue(allowed_client("10.1.2.3", ["10.1.2.3"]))           # another pc by IP
        self.assertFalse(allowed_client("10.1.2.4", ["10.1.2.3", "no-such-host-xyz.invalid"]))

    def test_server_authorized(self):
        """A download needs this machine in the list (the shared app, on a pc without a terminal, never downloads)."""
        me = socket.gethostname()
        self.assertTrue(server_authorized([me]))
        self.assertFalse(server_authorized(["OTHER-PC-XYZ", "10.1.2.3"]))
        self.assertFalse(server_authorized([]))

    def test_settings_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "settings.yaml"
            empty.write_text("marketdata_roots:\n  - C:/x\n", encoding="utf-8")
            self.assertEqual(app_settings(empty), {"authorized_pcs": [], "python": ["current"]})
            self.assertEqual(app_settings(Path(tmp) / "missing.yaml"), {"authorized_pcs": [], "python": ["current"]})
        real = app_settings()                                                 # tqstudio/config.yaml
        self.assertIn(socket.gethostname().upper(), [p.upper() for p in real["authorized_pcs"]])
        self.assertEqual(real["python"][-1], "current")


if __name__ == "__main__":
    unittest.main()
