"""Market data universe (``marketdata/config/universe.yaml``): what is downloaded, its status in a snapshot, and the
download (``python -m mdbuild``) started from the app in a separate process."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
from ..paths import CODE_MARKETDATA, MARKETDATA  # noqa: E402  (shared data; mdbuild code in the repository)

UNIVERSE = MARKETDATA / "config" / "universe.yaml"           # universe lives with the data (per data folder)
UNIVERSE_TEMPLATE = CODE_MARKETDATA / "config" / "universe.yaml"   # the repository's copy, used as read fallback / bootstrap
SETTINGS = CODE_MARKETDATA / "config" / "settings.yaml"      # Bloomberg host/port, fields, history: same on every machine


def _ensure_universe(path: Path = UNIVERSE) -> Path:
    """Universe file in the data folder; copied from :data:`UNIVERSE_TEMPLATE` on first access (parent created)."""
    path = Path(path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(UNIVERSE_TEMPLATE).read_bytes())
    return path


def _mdbuild_universe():
    """``marketdata/mdbuild/universe.py`` loaded by path, once (marketdata is self-contained, not an installed
    package); registered in ``sys.modules`` before running, as its dataclasses need."""
    name = "_tqstudio_mdbuild_universe"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, CODE_MARKETDATA / "mdbuild" / "universe.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def read_universe(path: Path = UNIVERSE) -> dict:
    """The universe file: every instrument (active or not), curves, option roots, implied-spot underlyings. The data
    folder's copy when present, else the repository's :data:`UNIVERSE_TEMPLATE` (read-only fallback).

    Returns:
        dict: ``equity`` / ``fx`` / ``fixing`` lists of ``{bbg, code, type, included, active, models}`` in the
        order of the file (equity ``type``: ``stock`` or ``index``; ``included`` = ``active``; ``models``: the
        instrument's own calibration flags), ``curves`` (``{name, ccy, curve, bbg, kind, active}``), ``option_roots``,
        ``implied_spot``.
    """
    u = _mdbuild_universe()
    path = Path(path)
    if not path.is_file():
        path = Path(UNIVERSE_TEMPLATE)
    entries, curve_entries = u.read_entries(path)
    out = {"equity": [], "fx": [], "fixing": []}
    for e in entries:
        inst = u.instrument(e)
        out[inst.kind].append({"bbg": inst.bbg, "code": inst.code, "included": e["active"], "active": e["active"],
                               "type": ("index" if e.get("equity_index") else "stock") if inst.kind == "equity"
                               else None, "models": dict(e.get("models") or {})})
    out["curves"] = []
    for c in curve_entries:
        curve = u._curve(c["name"], c["ticker"])
        out["curves"].append({"name": curve.label, "ccy": curve.ccy, "curve": curve.name, "bbg": curve.bbg,
                              "kind": curve.kind, "active": c["active"]})
    out["option_roots"] = u.load_option_roots(path)
    out["implied_spot"] = u.load_implied_spot(path)
    return out


def _write(path: Path, entries: list, curves: list) -> None:
    _mdbuild_universe().write_entries(path, entries, curves)


def add_ticker(asset_class: str, ticker: str, path: Path = UNIVERSE) -> dict:
    """Add a Bloomberg ticker to the universe file, active (downloaded at the next full download); an inactive one
    already there is activated instead.

    Equities (``Equity`` yellow key), equity indices (``Index`` under *asset_class* ``"equity"``), FX pairs
    (``Curncy``) and rate fixings (``Index`` under ``"rates"``); a new entry goes after the last one of its kind. Rate
    curves need quote conventions in tQuant and are not added here.

    Args:
        asset_class (str): ``"equity"``, ``"fx"`` or ``"rates"``.
        ticker (str): Bloomberg ticker with its yellow key (``"UCG IM Equity"``, ``"EURJPY Curncy"``).
        path (Path): The universe file.

    Returns:
        dict: ``{"ticker", "action"}`` (``"added"`` or ``"activated"``).

    Raises:
        ValueError: Unknown asset class, a yellow key that does not belong to it, an invalid ticker, or a ticker
            already active in the universe.
    """
    u = _mdbuild_universe()
    if asset_class not in ASSET_CLASSES:
        raise ValueError(f"unknown asset class {asset_class}")
    ticker = " ".join(str(ticker).split())
    parts = ticker.split()
    if len(parts) < 2:
        raise ValueError(f"'{ticker}': give the Bloomberg ticker with its yellow key (e.g. UCG IM Equity)")
    parts = [p.upper() for p in parts[:-1]] + [parts[-1].capitalize()]      # Bloomberg form: ISP IM Equity
    ticker = " ".join(parts)
    if parts[-1] not in ASSET_CLASSES[asset_class]:
        raise ValueError(f"'{ticker}': {asset_class} tickers end with {' or '.join(ASSET_CLASSES[asset_class])}")
    u.parse(ticker)                                         # raises on an invalid FX pair, etc.
    index = asset_class == "equity" and parts[-1] == "Index"
    path = _ensure_universe(path)                           # write side: bootstrap the data copy from the template
    entries, curves = u.read_entries(path)
    existing = next((e for e in entries if e["ticker"].upper() == ticker.upper()), None)
    if existing is not None:
        if existing["active"]:
            raise ValueError(f"'{ticker}' is already in the universe")
        existing["active"] = True
        _write(path, entries, curves)
        return {"ticker": existing["ticker"], "action": "activated"}
    entry = {"ticker": ticker, "active": True, **({"equity_index": True} if index else {})}
    kind = u.instrument(entry).kind
    same = [i for i, e in enumerate(entries) if u.instrument(e).kind == kind
            and bool(e.get("equity_index")) == index]
    entries.insert((same[-1] + 1) if same else len(entries), entry)
    _write(path, entries, curves)
    return {"ticker": ticker, "action": "added"}


def set_active(name: str, active: bool, path: Path = UNIVERSE) -> dict:
    """Switch the ``active`` flag of an instrument (Bloomberg ticker) or a curve (``"EUR ESTR"``) of the universe.

    Returns:
        dict: ``{"name", "active"}``.

    Raises:
        ValueError: If *name* is not in the universe.
    """
    u = _mdbuild_universe()
    path = _ensure_universe(path)
    entries, curves = u.read_entries(path)
    name = " ".join(str(name).split())
    target = next((e for e in entries if e["ticker"].upper() == name.upper()), None) or \
        next((c for c in curves if c["name"].upper() == name.upper()), None)
    if target is None:
        raise ValueError(f"'{name}' is not in the universe")
    target["active"] = bool(active)
    _write(path, entries, curves)
    return {"name": target["name"] if "name" in target else target["ticker"], "active": bool(active)}


def remove_ticker(asset_class: str, ticker: str, path: Path = UNIVERSE) -> dict:
    """Take a ticker out of the full download: its entry becomes inactive (kept in the universe, it can still be
    downloaded alone or activated again). Past snapshots keep their data.

    Args:
        asset_class (str): ``"equity"``, ``"fx"`` or ``"rates"`` (rate fixings; curves: :func:`set_active`).
        ticker (str): The Bloomberg ticker as listed.
        path (Path): The universe file.

    Returns:
        dict: ``{"ticker", "action": "deactivated"}``.

    Raises:
        ValueError: Unknown asset class, or the ticker is not an active one of that asset class.
    """
    if asset_class not in ASSET_CLASSES:
        raise ValueError(f"unknown asset class {asset_class}")
    ticker = " ".join(str(ticker).split())
    kind = {"equity": "equity", "fx": "fx", "rates": "fixing"}[asset_class]
    match = next((r for r in read_universe(path)[kind] if r["included"] and r["bbg"].upper() == ticker.upper()), None)
    if match is None:
        raise ValueError(f"'{ticker}' is not an active {asset_class} ticker of the universe")
    set_active(match["bbg"], False, path)
    return {"ticker": match["bbg"], "action": "deactivated"}


def _read(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _clean(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def universe_view(snapshot: Path | None, path: Path = UNIVERSE) -> dict:
    """The universe and, per instrument, its status in the snapshot folder *snapshot* (``live/<yyyymmdd>/<hhmm>``).

    Equities: spot and its source, dividends, option quotes (count, quote date), expiries with a smile, share of the
    quotes whose model vol is within bid / ask; FX: spot; fixings: value and date; curves: quotes, last update,
    bootstrapped. ``in_snapshot`` is False for an instrument added after the snapshot (or excluded).
    """
    from .calibration import MODELS, _flags, read_config
    uni = read_universe(path)
    config = read_config(universe=path if Path(path).is_file() else None)
    info, folder = {}, Path(snapshot) if snapshot is not None else None
    if folder is not None and (folder / "snapshot_info.json").is_file():
        info = json.loads((folder / "snapshot_info.json").read_text(encoding="utf-8"))
    tables = {n: (_read(folder, n) if folder is not None else pd.DataFrame()) for n in (
        "eq_spot.csv", "eq_dividends.csv", "eq_option_quotes.csv", "eq_vol_calibration.csv", "eq_vol_quotes.csv",
        "fx_spot.csv", "ir_fixings.csv", "ir_quotes.csv", "ir_curves.csv")}
    snap_universe = set(info.get("universe", []))

    def rows_of(name, column, value):
        t = tables[name]
        return t[t[column] == value] if not t.empty and column in t else t.iloc[0:0]

    equity = []
    for r in uni["equity"]:
        code = r["code"]
        spot, divs = rows_of("eq_spot.csv", "ticker", code), rows_of("eq_dividends.csv", "ticker", code)
        opts, cal = rows_of("eq_option_quotes.csv", "ticker", code), rows_of("eq_vol_calibration.csv", "ticker", code)
        vq = rows_of("eq_vol_quotes.csv", "ticker", code)
        fitted = cal[cal["error"].isna()] if "error" in cal else cal
        within = None
        if not vq.empty:
            two = vq[vq["used"].astype(bool) & ~vq["settlement"].astype(bool)].dropna(subset=["iv_bid", "iv_ask", "model_vol"])
            if len(two):
                within = float(((two["model_vol"] >= two["iv_bid"] - 1e-12) & (two["model_vol"] <= two["iv_ask"] + 1e-12)).mean())
        equity.append({**r, "in_snapshot": r["bbg"] in snap_universe and not spot.empty,
                       "flags": _flags(config, "equity", code),         # the models calibrated (own or default)
                       "ccy": spot["ccy"].iloc[0] if not spot.empty else None,
                       "spot": float(spot["spot"].iloc[0]) if not spot.empty else None,
                       "spot_source": cal["spot_source"].iloc[0] if not cal.empty else None,
                       "dividends": int(len(divs)), "options": int(len(opts)),
                       "quote_date": opts["quote_date"].iloc[0] if not opts.empty else None,
                       "expiries": int(len(fitted)), "within_bid_ask": within,
                       "option_root": uni["option_roots"].get(r["bbg"]),
                       "implied_spot": r["bbg"] in uni["implied_spot"]})
    fx = []
    for r in uni["fx"]:
        t = tables["fx_spot.csv"]
        row = t[(t["base"] == r["code"][:3]) & (t["quote"] == r["code"][3:])] if not t.empty else t
        fx.append({**r, "in_snapshot": not row.empty, "spot": float(row["spot"].iloc[0]) if not row.empty else None})
    fixings = []
    for r in uni["fixing"]:
        row = rows_of("ir_fixings.csv", "bloomberg", r["bbg"])
        fixings.append({**r, "in_snapshot": not row.empty,
                        "value": float(row["value"].iloc[0]) if not row.empty else None,
                        "fixing_date": row["fixing_date"].iloc[0] if not row.empty else None})
    curves = []
    q, c = tables["ir_quotes.csv"], tables["ir_curves.csv"]
    for r in uni["curves"]:
        quotes = q[(q["ccy"] == r["ccy"]) & (q["curve"].astype(str) == r["curve"])] if not q.empty else q
        built = c[(c["ccy"] == r["ccy"]) & (c["curve"].astype(str) == r["curve"])] if not c.empty else c
        curves.append({**r, "in_snapshot": not quotes.empty, "quotes": int(len(quotes)),
                       "last_update": quotes["last_update"].max() if not quotes.empty else None,
                       "pillars": int(len(built)), "bootstrapped": not built.empty})
    snapshot_info = None
    if folder is not None:
        snapshot_info = {"date": datetime.strptime(folder.parent.name, "%Y%m%d").date().isoformat(), "set": folder.name,
                         "created": info.get("created"), "history_rows": (info.get("history") or {}).get("rows"),
                         "history_first": (info.get("history") or {}).get("first")}
    if snapshot_info is not None:
        snapshot_info.update(selected=info.get("selected"))
    out = {"snapshot": snapshot_info, "equity": equity, "fx": fx, "fixing": fixings, "curves": curves,
           "models": MODELS["equity"],
           "file": str(Path(path).relative_to(ROOT.parent)) if Path(path).is_relative_to(ROOT.parent) else str(path)}
    return {k: ([{kk: _clean(vv) for kk, vv in row.items()} for row in v]
                if isinstance(v, list) and v and isinstance(v[0], dict) else v)
            for k, v in out.items()}


ASSET_CLASSES = {"equity": ("Equity", "Index"), "fx": ("Curncy",), "rates": ("Index",)}


from ..paths import CONFIG as APP_CONFIG  # noqa: E402  (tqstudio/config.yaml: authorized pcs, Python)
LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")


def app_settings(path: Path = APP_CONFIG) -> dict:
    """The rules of the app in ``tqstudio/config.yaml``: ``authorized_pcs`` (the pcs, by name or IP, allowed to change
    the universe and to start downloads: the ones with a Bloomberg terminal; no pc is allowed by default) and ``python``
    (interpreters for the download, tried in order; ``current`` = the one running tqstudio). Defaults when missing."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {}
    app = data or {}
    return {"authorized_pcs": [str(c).strip() for c in app.get("authorized_pcs") or []],
            "python": [str(c).strip() for c in app.get("python") or ["current"]]}


def _identities(host: str) -> set[str]:
    """Names and addresses of the machine of *host* (this machine, by its name and addresses, for a local request)."""
    if host in LOCAL_HOSTS:
        name = socket.gethostname()
        try:
            addresses = set(socket.gethostbyname_ex(name)[2])
        except OSError:
            addresses = set()
        return {name.lower(), *addresses, *LOCAL_HOSTS}
    return {host.lower()}


def allowed_client(host: str | None, pcs: list[str]) -> bool:
    """Whether a request from *host* comes from one of the authorized *pcs* (names or IPs; names resolved at each
    call, so an IP that changes keeps its name; a local request is this machine, by its name). Only the list counts:
    no pc is allowed by default."""
    if host is None or not pcs:
        return False
    pcs = [p for p in pcs if not (p.startswith("<") and p.endswith(">"))]      # placeholders in config.yaml: skipped
    ids = _identities(host)
    if any(pc.lower() in ids for pc in pcs):                                    # cheap pass: match by name / IP string
        return True
    for pc in pcs:                                                              # slow pass: resolve the remaining names
        try:
            if ids & {info[4][0] for info in socket.getaddrinfo(pc, None)}:
                return True
        except OSError:
            continue
    return False


def server_authorized(pcs: list[str]) -> bool:
    """Whether this machine (the one running tqstudio, where a download runs on its Bloomberg terminal) is one of the
    authorized *pcs*: the shared app, on a machine without a terminal, never starts a download."""
    return allowed_client("127.0.0.1", pcs)


def resolve_python(candidates: list[str], module: str = "blpapi") -> str:
    """The first interpreter of *candidates* that exists and imports *module* (``current`` = ``sys.executable``).

    Raises:
        ValueError: None of them does (the message lists what was tried and why).
    """
    tried = []
    for candidate in candidates:
        exe = sys.executable if candidate == "current" else candidate
        if not Path(exe).is_file():
            tried.append(f"{candidate}: not found")
            continue
        check = subprocess.run([exe, "-c", f"import {module}"], capture_output=True, text=True, timeout=60)
        if check.returncode == 0:
            return exe
        tried.append(f"{candidate}: no {module}")
    raise ValueError("no Python for the download: " + "; ".join(tried))


def bloomberg_endpoint(root: Path | None = None) -> tuple[str, int]:
    """Host and port of the Bloomberg terminal of the downloads (``bloomberg`` in :data:`SETTINGS`, in the repository)."""
    del root                                                # kept for backwards compatibility; settings live in the repo
    data = yaml.safe_load(Path(SETTINGS).read_text(encoding="utf-8")) or {}
    bbg = data.get("bloomberg") or {}
    return str(bbg.get("host", "localhost")), int(bbg.get("port", 8194))


def bloomberg_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    """Whether something listens on the Bloomberg port (the terminal's API service): a TCP connection, in about
    *timeout* seconds at most (``localhost`` as 127.0.0.1: no wait on the IPv6 address first)."""
    address = "127.0.0.1" if host in ("localhost", "::1") else host
    try:
        socket.create_connection((address, int(port)), timeout=timeout).close()
        return True
    except OSError:
        return False


class DownloadRuns:
    """``python -m mdbuild --root <market data folder>`` (the code of the repository, the data of the shared folder)
    started from the app in a separate process, one at a time; its output read back. The Bloomberg terminal must be
    running on this machine. The interpreter is the first of the ``python`` candidates of the settings with ``blpapi``
    (:func:`resolve_python`), unless *command* is given."""

    def __init__(self, cwd: Path = CODE_MARKETDATA, command=None, settings: Path = APP_CONFIG, root: Path = MARKETDATA):
        self.cwd = Path(cwd)
        self.root = Path(root)
        self.command = command
        self.settings = Path(settings)
        self.run = None

    def start(self, select=None, then=None) -> dict:
        """Start a download: every active instrument, or *select* (equities, Bloomberg tickers) with the active rates,
        FX and curves. *then*: called with the new snapshot folder once the download has succeeded (e.g. start the
        calibration of its models); its result or error is reported in the status (``after``)."""
        if self.run is not None and self.run["process"].poll() is None:
            raise ValueError("a download is already running")
        if self.command is None:                       # no process without a terminal: the error at once
            host, port = bloomberg_endpoint()
            if not bloomberg_reachable(host, port):
                raise ValueError(f"ConnectionError: cannot reach the Bloomberg terminal on {host}:{port} "
                                 "(is the terminal running?)")
        universe = str(_ensure_universe(self.root / "config" / "universe.yaml"))     # bootstrap on first download
        command = self.command or [resolve_python(app_settings(self.settings)["python"]), "-u", "-m", "mdbuild",
                                   "--root", str(self.root), "--universe", universe, "--settings", str(SETTINGS)]
        if select:
            command = [*command, "--select", ",".join(select)]
        log = []
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TF_ENABLE_ONEDNN_OPTS": "0"}
        process = subprocess.Popen(command, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)

        def read():
            with process.stdout:
                for line in process.stdout:
                    log.append(line.rstrip())
            if then is not None and process.wait() == 0:      # the next step on the new snapshot
                folder = next((l.split("-> ")[-1].strip() for l in log if l.startswith("snapshot ->")), None)
                try:
                    self.run["after"] = then(Path(folder)) if folder else "no snapshot written"
                except Exception as exc:                       # e.g. a calibration already running
                    self.run["after"] = f"{type(exc).__name__}: {exc}"

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        self.run = {"process": process, "log": log, "reader": reader, "python": command[0], "select": select,
                    "then": then is not None, "after": None,
                    "started": datetime.now().isoformat(timespec="seconds"), "t0": time.perf_counter(), "t1": None}
        return self.status()

    def status(self) -> dict:
        if self.run is None:
            return {"running": False}
        log, code = self.run["log"], self.run["process"].poll()
        if code is not None and self.run["t1"] is None:
            self.run["reader"].join(timeout=2)
            self.run["t1"] = time.perf_counter()
        end = self.run["t1"] or time.perf_counter()
        result = next((l.split("-> ")[-1].strip() for l in log if l.startswith("snapshot ->")), None)
        return {"running": code is None, "exit_code": code, "started": self.run["started"], "python": self.run["python"],
                "select": self.run.get("select"), "calibrate": self.run.get("then"),
                "after": self.run.get("after"),
                "seconds": round(end - self.run["t0"], 1), "result": result, "log": log[-12:],
                "errors": [l for l in log if "Error" in l or "Traceback" in l][-5:]}
