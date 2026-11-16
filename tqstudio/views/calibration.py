"""Calibration tab: calibration sets (list, one set, one underlying with its models, fit and test) and the runs of
``python -m tqstudio.calibrate`` started from the app (separate processes)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import tensorquant as tq
from tensorquant.calibration.models import (STALE_VOLPTS, check_calibration, list_calibration_sets,
                                            load_calibration_set, load_model, model_test)

ROOT = Path(__file__).resolve().parents[1]


def _nan(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{k: _nan(v) for k, v in r.items()} for r in frame.replace({np.nan: None}).to_dict("records")]


def set_folder(root: Path, day: str, calibration_set: str) -> Path:
    folder = Path(root) / date.fromisoformat(day).strftime("%Y%m%d") / calibration_set
    if not (folder / "calibration_info.json").is_file():
        raise FileNotFoundError(f"no calibration set {day} {calibration_set}")
    return folder


def sets_view(root: Path) -> dict:
    """The calibration sets, latest first."""
    return {"sets": [{k: v for k, v in s.items() if k != "path"} for s in list_calibration_sets(root)]}


def set_view(folder: Path) -> dict:
    """One set: info and the grid underlyings x models (status, quality, errors, seconds)."""
    loaded = load_calibration_set(folder)
    summary = loaded["summary"]
    return {"info": loaded["info"], "models": sorted(summary["model"].unique().tolist()),
            "tickers": sorted(summary["ticker"].unique().tolist()), "rows": _records(summary)}


def _labels(maturities, day: date) -> list[str]:
    return [(day + timedelta(days=int(round(365 * t)))).isoformat() for t in maturities]


def underlying_view(folder: Path, ticker: str, source, latest=None) -> dict:
    """One underlying of a set: per model the parameters, the calibration report, the fit on the surface nodes
    (errors map, smiles with the quotes of the source snapshot, ATM-forward term structure) and the drift of the
    stored parameters on the *latest* market (when it is another snapshot).

    Args:
        folder (Path): The calibration set.
        ticker (str): The underlying.
        source (MarketEnvironment): The snapshot the set was calibrated on (evaluation date set by the caller).
        latest (MarketEnvironment, optional): The latest snapshot, for the drift.
    """
    summary = load_calibration_set(folder)["summary"]
    rows = summary[summary["ticker"] == ticker]
    if rows.empty:
        raise KeyError(ticker)
    eq = source.get_equity(ticker)
    surface = eq.vol_surface
    maturities = [float(t) for t in surface.maturity]
    strikes = [float(k) for k in surface.strike]
    s0 = float(eq.spot)
    day = source.evaluation_date
    market = surface.volatility_matrix.numpy()
    out = {"ticker": ticker, "spot": s0, "expiries": _labels(maturities, day), "T": maturities,
           "strikes": strikes, "moneyness": [k / s0 for k in strikes], "market": market.tolist(),
           "atm_market": [float(surface.volatility(strike=float(eq.forward(t)), tenor=t)) for t in maturities],
           "models": []}
    for r in _records(rows):
        entry = {**r, "parameters": json.loads(r["parameters"]) if r.get("parameters") else None}
        if r["status"] == "ok":
            fit = pd.read_csv(folder / ticker / f"{r['model']}_fit.csv")
            model_vols = fit.pivot(index="t", columns="strike", values="model_vol").sort_index().sort_index(axis=1)
            entry["vols"] = model_vols.to_numpy().tolist()
            entry["errors"] = (100.0 * (model_vols.to_numpy() - market)).tolist()
            model = load_model(folder, ticker, r["model"])
            entry["parameter_view"] = parameter_view(model)
            entry.pop("parameters", None)                 # raw values (SLV: the whole table): the view above instead
            if r["model"] != "HestonSLV":
                process = model.process(eq)
                entry["atm"] = [float(process.implied_vol(float(eq.forward(t)), t)) for t in maturities]
            report = folder / ticker / f"{r['model']}_report.csv"
            entry["report"] = _records(pd.read_csv(report)) if report.is_file() else None
            if latest is not None and r["model"] != "HestonSLV" and ticker in [k.split(":")[2] for k in latest.risk_factors()
                                                                             if k.endswith(":VOL")]:
                previous = tq.Settings.evaluation_date            # the drift on the latest market, at its date
                tq.Settings.evaluation_date = latest.evaluation_date
                try:
                    entry["drift"] = {**check_calibration(model, latest.get_equity(ticker)),
                                      "snapshot": f"{latest.evaluation_date.isoformat()} {latest.calibration_set}"}
                finally:
                    tq.Settings.evaluation_date = previous
        out["models"].append(entry)
    out["quotes"] = _quotes(folder, ticker, out["expiries"])
    out["stale_volpts"] = STALE_VOLPTS
    return out


HESTON_LABELS = {"v0": ("v0", "initial variance"), "kappa": ("kappa", "mean reversion speed"),
                 "theta": ("theta", "long-run variance"), "xi": ("xi", "vol of vol"),
                 "rho": ("rho", "spot-variance correlation")}


def _scalar_rows(params: dict, prefix: str = "") -> list[dict]:
    rows = []
    for key, (name, note) in HESTON_LABELS.items():
        if key in params:
            value = float(np.ravel(params[key].numpy() if hasattr(params[key], "numpy") else params[key])[0])
            extra = f" (vol {100 * np.sqrt(max(value, 0.0)):.1f}%)" if key in ("v0", "theta") else ""
            rows.append({"name": prefix + name, "value": value, "note": note + extra})
    return rows


def parameter_view(model) -> dict:
    """The parameters of a stored model, readable: ``scalars`` (``name, value, note``) and an optional ``table``
    (``title, columns, rows``): Heston's five parameters and the Feller condition, the displaced diffusion pieces,
    Heston SLV's Heston part, mixing and a summary of the ``E[v | X]`` table (not its thousands of values)."""
    name = type(model).__name__
    if name == "HestonModel":
        p = {k: float(v.numpy()) for k, v in model.parameters.items()}
        rows = _scalar_rows(p)
        feller = 2.0 * p["kappa"] * p["theta"] - p["xi"] ** 2
        rows.append({"name": "Feller 2 kappa theta - xi^2", "value": feller,
                     "note": "variance stays positive" if feller >= 0 else "violated: the variance can reach 0"})
        return {"scalars": rows, "table": None}
    if name == "DisplacedDiffusionModel":
        ends = [float(t) for t in model.settings["t_nodes"]]
        beta, sigma = (np.ravel(v.numpy()) for v in (model.parameters["beta_fwd"], model.parameters["sigma_fwd"]))
        starts = [0.0] + ends[:-1]
        return {"scalars": [], "table": {
            "title": "Forward pieces (beta: 1 lognormal, 0 normal; sigma of the shifted forward)",
            "columns": ["From (years)", "To (years)", "beta", "sigma"],
            "rows": [[a, b, float(x), float(y)] for a, b, x, y in zip(starts, ends, beta, sigma)]}}
    if name == "HestonSLVModel":
        rows = _scalar_rows({k: float(v.numpy()) for k, v in model.heston.parameters.items()}, "Heston ")
        s = model.settings
        rows.append({"name": "mixing", "value": float(s["mixing"]), "note": "vol of vol scale (1: full Heston xi)"})
        e = model.parameters.get("expectation")
        if e is not None and model.leverage_table is not None:
            values = e.numpy()
            times = np.asarray(model.leverage_table["times"], dtype=float)
            rows.append({"name": "E[v | X] table", "value": None,
                         "note": f"{values.shape[0]} times x {values.shape[1]} nodes, 0 to {times[-1]:.1f} years "
                                 f"({int(s['steps_per_year'])} a year), values {values.min():.4f} to {values.max():.4f}, "
                                 f"{int(s['n_particles']):,} particles"})
        return {"scalars": rows, "table": None}
    rows = []
    for key, v in model.parameters.items():                 # any other model: scalars, vectors summarised
        a = np.ravel(v.numpy())
        rows.append({"name": key, "value": float(a[0]) if a.size == 1 else None,
                     "note": "" if a.size == 1 else f"{a.size} values, {a.min():.4g} to {a.max():.4g}"})
    return {"scalars": rows, "table": None}


def _quotes(folder: Path, ticker: str, expiries: list[str]) -> dict:
    """Bid / mid / ask vols of the option quotes of the source snapshot, per listed expiry (``{}`` without them)."""
    info = load_calibration_set(folder)["info"]
    snap = ROOT.parent / "marketdata" / "live" / date.fromisoformat(info["snapshot"]["date"]).strftime("%Y%m%d") \
        / info["snapshot"]["set"] / "eq_vol_quotes.csv"
    if not snap.is_file():
        return {}
    q = pd.read_csv(snap)
    q = q[(q["ticker"] == ticker) & q["used"].astype(bool)]
    return {e: _records(g[["strike", "option_type", "iv_bid", "iv_mid", "iv_ask", "settlement"]])
            for e, g in q.groupby("expiry") if e in expiries}


def test_view(folder: Path, ticker: str, source, rerun: bool = False, n_paths: int = 20_000) -> dict:
    """The model test of an underlying (cached in the set: ``<ticker>/model_test.csv``)."""
    path = folder / ticker / "model_test.csv"
    seconds = {}
    if rerun or not path.is_file():
        summary = load_calibration_set(folder)["summary"]
        names = summary[(summary["ticker"] == ticker) & (summary["status"] == "ok")]["model"].tolist()
        models = {n: load_model(folder, ticker, n) for n in names}
        frame = model_test(models, source.get_equity(ticker), source, n_paths=n_paths)
        seconds = frame.attrs.get("seconds", {})
        path.parent.mkdir(exist_ok=True)
        frame.to_csv(path, index=False)
    frame = pd.read_csv(path)
    engines = [c for c in frame.columns if not c.endswith(("_se", "_vol")) and c not in
               ("product", "option_type", "t", "expiry", "strike_pct", "barrier_pct", "Black")]
    return {"engines": engines, "rows": _records(frame), "seconds": seconds, "cached": not seconds}


class CalibrationRuns:
    """``python -m tqstudio.calibrate`` started from the app in a separate process; its progress lines read back."""

    def __init__(self):
        self.run = None

    def start(self, options: dict) -> dict:
        if self.run is not None and self.run["process"].poll() is None:
            raise ValueError("a calibration is already running")
        args = [sys.executable, "-u", "-m", "tqstudio.calibrate"]
        for key in ("date", "set", "models", "tickers", "slv"):
            if options.get(key):
                args += [f"--{key}", str(options[key])]
        log = []
        process = subprocess.Popen(args, cwd=ROOT.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)

        def read():
            for line in process.stdout:
                log.append(line.rstrip())

        threading.Thread(target=read, daemon=True).start()
        self.run = {"process": process, "log": log, "started": datetime.now().isoformat(timespec="seconds"),
                    "t0": time.perf_counter(), "options": options}
        return self.status()

    def status(self) -> dict:
        if self.run is None:
            return {"running": False}
        log, code = self.run["log"], self.run["process"].poll()
        progress = [l for l in log if l.startswith("progress ")]
        done = progress[-1].split()[1] if progress else None
        return {"running": code is None, "exit_code": code, "started": self.run["started"],
                "seconds": round(time.perf_counter() - self.run["t0"], 1), "progress": done,
                "current": " ".join(progress[-1].split()[2:]) if progress else None,
                "result": next((l.split("-> ")[-1] for l in log if l.startswith("calibration set ->")), None),
                "errors": [l for l in log if "Error" in l or "Traceback" in l][-5:]}


# ----------------------------------------------------------------------------------------------------------------------
# universe: the models to calibrate per underlying (marketdata/config/calibration.yaml)
# ----------------------------------------------------------------------------------------------------------------------

CONFIG = ROOT.parent / "marketdata" / "config" / "calibration.yaml"
MODELS = {"equity": ["Heston", "DisplacedDiffusion", "HestonSLV"], "rates": ["HullWhite", "G2pp"], "fx": []}
SUPPORTED = {"Heston", "DisplacedDiffusion", "HestonSLV"}     # models of the calibration sets (rates: not yet)


def read_config(path: Path = CONFIG) -> dict:
    """The flags file, completed with an empty section and the defaults of every asset class."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {}
    data = data or {}
    data.setdefault("defaults", {})
    for cls in MODELS:
        data[cls] = data.get(cls) or {}
        data["defaults"][cls] = data["defaults"].get(cls) or {m: False for m in MODELS[cls]}
    return data


def _flags(config: dict, cls: str, name: str) -> dict:
    stored = (config.get(cls) or {}).get(name) or {}
    defaults = config["defaults"].get(cls) or {}
    return {m: bool(stored.get(m, defaults.get(m, False))) for m in MODELS[cls]}


def underlyings(mkt) -> dict:
    """The underlyings of a market by asset class: equities (with a vol surface), rate curves, FX pairs."""
    keys = list(mkt.risk_factors())
    equity = sorted({k.split(":")[2] for k in keys if k.startswith("EQ:") and k.endswith(":VOL")})
    rates = sorted({k.split(":")[1] + ":" + k.split(":")[2] for k in keys if k.startswith("IR:") and k.endswith(":SPOT")})
    fx = sorted({k.split(":")[2] for k in keys if k.startswith("FX:") and k.endswith(":SPOT")})
    return {"equity": equity, "rates": rates, "fx": fx}


def plan(config: dict, mkt) -> dict:
    """``{ticker: [models]}`` of the equities of *mkt* flagged in *config* (the calibration sets' plan)."""
    return {t: [m for m, on in _flags(config, "equity", t).items() if on] for t in underlyings(mkt)["equity"]}


def universe_view(mkt, root: Path, path: Path = CONFIG) -> dict:
    """Per asset class the underlyings of the latest snapshot, their flags and the last calibration of each model."""
    config = read_config(path)
    last = {}
    for s in reversed(list_calibration_sets(root)):           # oldest first: the latest wins
        summary = load_calibration_set(s["path"])["summary"]
        for r in _records(summary):
            last[(r["ticker"], r["model"])] = {"set": s["date"] + " " + s["set"], "status": r["status"],
                                               "quality": r.get("quality"), "rms_volpts": r.get("rms_volpts")}
    out = {"models": MODELS, "supported": sorted(SUPPORTED), "classes": {}}
    for cls, names in underlyings(mkt).items():
        out["classes"][cls] = [{"name": n, "flags": _flags(config, cls, n),
                                "last": {m: last.get((n, m)) for m in MODELS[cls]}} for n in names]
    return out


def save_flags(asset_class: str, name: str, flags: dict, path: Path = CONFIG) -> dict:
    """Store the flags of one underlying (the comment header of the file is kept)."""
    if asset_class not in MODELS:
        raise ValueError(f"unknown asset class {asset_class}")
    config = read_config(path)
    config[asset_class][name] = {m: bool(flags.get(m, False)) for m in MODELS[asset_class]}
    text = Path(path).read_text(encoding="utf-8") if Path(path).is_file() else ""
    header = "".join(line for line in text.splitlines(keepends=True) if line.startswith("#"))
    Path(path).write_text(header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config[asset_class][name]


# ----------------------------------------------------------------------------------------------------------------------
# Heston SLV: when to recalibrate
# ----------------------------------------------------------------------------------------------------------------------

SLV_RULES = {"age_days": 7, "spot_move_pct": 5.0, "surface_change_volpts": 1.0, "fit_volpts": 1.0}
_TENORS, _MONEYNESS = (0.25, 0.5, 1.0, 2.0, 3.0), (0.8, 0.9, 1.0, 1.1, 1.2)


def _surface_grid(eq) -> np.ndarray:
    """Vols at fixed tenors and moneyness of the spot (sticky moneyness): ``[tenors, moneyness]``."""
    s0, vs = float(eq.spot), eq.vol_surface
    return np.array([[float(vs.volatility(strike=m * s0, tenor=t)) for m in _MONEYNESS] for t in _TENORS])


def slv_monitor(root: Path, latest, snapshot) -> dict:
    """Per underlying, its latest Heston SLV calibration against the latest market: age, spot move, surface change
    (rms at fixed tenors and moneyness), ATM 1Y vol then and now, drift of the Heston part (closed form) and the
    verdict with its reasons (the Monte Carlo fit of the SLV on the latest market is :func:`slv_check`).

    Args:
        root (Path): Calibration sets.
        latest (MarketEnvironment): The latest snapshot (evaluation date set by the caller).
        snapshot (callable): ``snapshot(date, set)`` -> the market of a snapshot (sets the evaluation date).
    """
    found = {}
    for s in list_calibration_sets(root):                     # latest first: the first SLV of each underlying
        summary = load_calibration_set(s["path"])["summary"]
        for r in _records(summary[(summary["model"] == "HestonSLV") & (summary["status"] == "ok")]):
            found.setdefault(r["ticker"], (s, r))
    today = latest.evaluation_date
    latest_tickers = set(underlyings(latest)["equity"])
    rows = []
    for ticker, (s, r) in sorted(found.items()):
        row = {"ticker": ticker, "set": f"{s['date']} {s['set']}", "date": s["date"], "path": s["path"],
               "snapshot": f"{s['snapshot']['date']} {s['snapshot']['set']}",
               "age_days": (today - date.fromisoformat(s["snapshot"]["date"])).days,
               "fit_at_calibration": r.get("rms_volpts")}
        if ticker not in latest_tickers:
            rows.append({**row, "verdict": "n/a", "reasons": ["not in the latest snapshot"]})
            continue
        then = snapshot(s["snapshot"]["date"], s["snapshot"]["set"])
        eq_then = then.get_equity(ticker)
        grid_then, spot_then = _surface_grid(eq_then), float(eq_then.spot)
        model = load_model(s["path"], ticker, "HestonSLV")
        tq.Settings.evaluation_date = latest.evaluation_date
        eq_now = latest.get_equity(ticker)
        grid_now, spot_now = _surface_grid(eq_now), float(eq_now.spot)
        heston = check_calibration(model.heston, eq_now)
        row.update(spot_then=spot_then, spot_now=spot_now, spot_move_pct=100.0 * (spot_now / spot_then - 1.0),
                   surface_change_volpts=float(100.0 * np.sqrt(np.mean((grid_now - grid_then) ** 2))),
                   atm1y_then=float(grid_then[2, 2]), atm1y_now=float(grid_now[2, 2]),
                   heston_drift_volpts=heston["rms_volpts"])
        reasons = []
        if row["age_days"] > SLV_RULES["age_days"]:
            reasons.append(f"{row['age_days']} days old")
        if abs(row["spot_move_pct"]) > SLV_RULES["spot_move_pct"]:
            reasons.append(f"spot {row['spot_move_pct']:+.1f}%")
        if row["surface_change_volpts"] > SLV_RULES["surface_change_volpts"]:
            reasons.append(f"surface moved {row['surface_change_volpts']:.1f} vol pts")
        rows.append({**row, "verdict": "Recalibrate" if reasons else "OK", "reasons": reasons})
    tq.Settings.evaluation_date = latest.evaluation_date
    return {"rules": SLV_RULES, "latest": f"{today.isoformat()} {latest.calibration_set}",
            "rows": [{k: v for k, v in r.items() if k != "path"} for r in rows]}


def slv_check(root: Path, latest, ticker: str) -> dict:
    """Monte Carlo fit of the latest Heston SLV of *ticker* on the latest market (rms over the meaningful nodes)."""
    for s in list_calibration_sets(root):
        summary = load_calibration_set(s["path"])["summary"]
        if len(summary[(summary["ticker"] == ticker) & (summary["model"] == "HestonSLV") & (summary["status"] == "ok")]):
            out = check_calibration(load_model(s["path"], ticker, "HestonSLV"), latest.get_equity(ticker))
            return {"ticker": ticker, "fit_volpts": out["rms_volpts"], "max_volpts": out["max_volpts"],
                    "recalibrate": bool(out["rms_volpts"] > SLV_RULES["fit_volpts"])}
    raise KeyError(f"no Heston SLV calibration of {ticker}")

