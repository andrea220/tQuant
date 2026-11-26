"""Calibration sets: the equity models calibrated on a market snapshot, stored apart from it.

A calibration set is a folder ``<root>/<yyyymmdd>/<hhmm>/`` with ``calibration_info.json`` (source snapshot, models,
settings, times), ``summary.csv`` (one row per underlying and model: status ok / error, fit quality good / fair / poor, fit errors,
seconds) and per underlying
``<ticker>/<model>_<table>.csv`` (the tables of :meth:`EquityModel.to_frames`) and ``<ticker>/<model>_fit.csv`` (model
and market vols on the surface nodes). Models are calibrated less often than the market data are downloaded (Heston
and displaced diffusion daily, Heston SLV on request): a pricing loads the latest set and holds the parameters on the
current market (:func:`check_calibration` measures how far they have drifted).

- :func:`calibrate_snapshot_models`: calibrate the equities of a snapshot, write the set.
- :func:`list_calibration_sets`, :func:`load_calibration_set`, :func:`load_model`: read them back.
- :func:`model_fit`: model vs market vols on the nodes of the surface (closed forms for Heston and displaced
  diffusion, Monte Carlo for Heston SLV).
- :func:`check_calibration`: rms error of stored parameters on a (newer) market.
- :func:`model_test`: a battery of European products (vanillas, down-and-in puts observed at expiry, cash digitals)
  priced with each model on shared Monte Carlo paths, against Black (vol of the surface at the strike; digitals by a
  call spread on the smile) and the local vol model.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm

from ..markethandles.equity import Equity
from ..markethandles.utils import Currency, OptionType
from ..models.displaceddiffusion import DisplacedDiffusionModel
from ..models.brownian import simulation_grid
from ..models.heston import HestonModel
from ..models.hestonslv import HestonSLVModel
from ..models.localvolatility import LocalVolatilityModel
from ..numericalhandles.dtypes import DTYPE
from .equityvol import black_implied_vol

__all__ = ["MODEL_CLASSES", "calibrate_snapshot_models", "list_calibration_sets", "load_calibration_set",
           "load_model", "model_fit", "check_calibration", "model_test", "STALE_VOLPTS"]

MODEL_CLASSES = {"Heston": HestonModel, "DisplacedDiffusion": DisplacedDiffusionModel, "HestonSLV": HestonSLVModel,
                 "LocalVol": LocalVolatilityModel}      # local vol: a check of the surface only (pricing rebuilds it)
MC_MODELS = (HestonSLVModel, LocalVolatilityModel)      # no closed form: fitted by Monte Carlo
STALE_VOLPTS = 1.0                   # rms model - market error (vol points) above which a calibration is to redo
QUALITY = ((1.0, "good"), (2.5, "fair"))   # rms of the fit (vol points): good, fair, else poor


# ----------------------------------------------------------------------------------------------------------------------
# fit and drift
# ----------------------------------------------------------------------------------------------------------------------

MIN_PATHS_BEYOND = 30      # a node is measured by the Monte Carlo fit when at least so many paths end beyond it


def _mc_vanilla_vols(model, equity: Equity, maturities, strikes, n_paths: int = 20_000, seed: int = 7,
                     market: np.ndarray = None) -> np.ndarray:
    """Black vols of Monte Carlo out-of-the-money vanillas (shared paths), ``[n_maturities, n_strikes]``.

    With the *market* vols, a node where fewer than :data:`MIN_PATHS_BEYOND` paths are expected beyond the strike
    (the market probability of ending there times *n_paths*) is NaN: its price is a handful of paths or none, its
    implied vol no measure of the model (the 200% calls of SX5E within a year come out at a zero vol)."""
    t_obs = np.asarray(maturities, dtype=float)
    terminal = _simulate(model, equity, t_obs, n_paths, seed)
    out = np.full((len(t_obs), len(strikes)), np.nan)
    for i, t in enumerate(t_obs):
        f = float(equity.forward(float(t)))
        s = terminal[:, i]
        for j, k in enumerate(strikes):
            cp = "P" if k < f else "C"
            if market is not None:
                sd = float(market[i, j]) * np.sqrt(t)
                d2 = (np.log(f / k) - 0.5 * sd ** 2) / sd
                if n_paths * norm.cdf(-d2 if cp == "P" else d2) < MIN_PATHS_BEYOND:
                    continue
            payoff = np.maximum(k - s, 0.0) if cp == "P" else np.maximum(s - k, 0.0)
            out[i, j] = black_implied_vol([payoff.mean()], f, [k], float(t), cp)[0]
    return out


def model_fit(model, equity: Equity, mc_paths: int = 20_000) -> pd.DataFrame:
    """Model and market vols on the nodes of the vol surface of *equity*.

    Args:
        model: A calibrated :class:`HestonModel`, :class:`DisplacedDiffusionModel` (closed forms),
            :class:`HestonSLVModel` or :class:`LocalVolatilityModel` (Monte Carlo).
        equity (Equity): The underlying (its surface is the market).
        mc_paths (int): Paths of the Monte Carlo fit (Heston SLV, local vol).

    Returns:
        pd.DataFrame: ``t``, ``strike``, ``moneyness``, ``market_vol``, ``model_vol``, ``error_volpts`` (model -
        market, vol points).
    """
    surface = equity.vol_surface
    maturities, strikes = [float(t) for t in surface.maturity], [float(k) for k in surface.strike]
    market = surface.volatility_matrix.numpy()
    if isinstance(model, MC_MODELS):
        vols = _mc_vanilla_vols(model, equity, maturities, strikes, mc_paths, market=market)
    else:
        process = model.process(equity)
        vols = np.array([[process.implied_vol(k, t) for k in strikes] for t in maturities])
    s0 = float(equity.spot)
    rows = [{"t": t, "strike": k, "moneyness": k / s0, "market_vol": float(market[i, j]), "model_vol": float(vols[i, j]),
             "error_volpts": 100.0 * float(vols[i, j] - market[i, j])}
            for i, t in enumerate(maturities) for j, k in enumerate(strikes)]
    return pd.DataFrame(rows)


def _fit_errors(fit: pd.DataFrame, min_t: float = 1.0 / 12.0, n_std: float = 2.0) -> dict:
    """rms and max error (vol points) where the market is meaningful for a diffusive model: maturities from *min_t*
    and strikes within *n_std* standard deviations, ``|ln(K / S)| <= n_std sigma_ATM sqrt(T)`` (the wings beyond
    are extrapolated surface, the first days are dominated by jumps)."""
    rows = []
    for t, g in fit.groupby("t"):
        if t < min_t:
            continue
        g = g.sort_values("moneyness")
        atm = float(np.interp(1.0, g["moneyness"], g["market_vol"]))
        rows.append(g[np.abs(np.log(g["moneyness"])) <= n_std * atm * np.sqrt(t)])
    inner = pd.concat(rows) if rows else fit.iloc[0:0]
    e = inner["error_volpts"].to_numpy()
    e = e[np.isfinite(e)]
    return {"rms_volpts": float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan"),
            "max_volpts": float(np.max(np.abs(e))) if e.size else float("nan")}


def _quality(rms: float) -> str:
    """good / fair / poor from the rms of the fit (vol points)."""
    for bound, label in QUALITY:
        if rms <= bound:
            return label
    return "poor"


def check_calibration(model, equity: Equity) -> dict:
    """Errors of stored parameters on the market of *equity* (e.g. a newer snapshot): ``rms_volpts``, ``max_volpts``
    and ``stale`` (rms above :data:`STALE_VOLPTS`)."""
    errors = _fit_errors(model_fit(model, equity))
    return {**errors, "stale": bool(errors["rms_volpts"] > STALE_VOLPTS)}


# ----------------------------------------------------------------------------------------------------------------------
# calibration sets
# ----------------------------------------------------------------------------------------------------------------------

def _snapshot_market(snapshot):
    from ..markethandles.marketenvironment import MarketEnvironment

    snapshot = Path(snapshot)
    day = _dt.datetime.strptime(snapshot.parent.name, "%Y%m%d").date()
    return MarketEnvironment.from_folder(snapshot.parent.parent, day, snapshot.name), day


def _equity_tickers(mkt) -> list[tuple[str, str]]:
    keys = mkt.risk_factors().keys()
    return sorted({(k.split(":")[2], k.split(":")[1]) for k in keys if k.startswith("EQ:") and k.endswith(":VOL")})


def calibrate_snapshot_models(snapshot, root, models=("Heston", "DisplacedDiffusion"), tickers=None, slv=(),
                              now: Optional[_dt.datetime] = None, progress=None, plan: Optional[dict] = None) -> Path:
    """Calibrate the equity models on a market snapshot and write a calibration set.

    Args:
        snapshot: Snapshot folder ``<yyyymmdd>/<set>/`` (with the equity vol surfaces).
        root: Root of the calibration sets (``<root>/<yyyymmdd>/<hhmm>/`` is written).
        models: Model names of :data:`MODEL_CLASSES` for every underlying (Heston SLV is slow: use *slv*).
        tickers: Underlyings (default: every equity with a vol surface).
        slv: Underlyings also calibrated with Heston SLV.
        now (datetime, optional): Time of the set (its folder name); default now.
        progress (callable, optional): ``progress(done, total, label)`` after each model.
        plan (dict, optional): ``{ticker: [model names]}``, the models of each underlying (e.g. the flags of a
            configuration); replaces *models*, *tickers* and *slv*. Unknown names are reported as skipped.

    Returns:
        Path: The calibration set folder.
    """
    from ..timehandles.utils import Settings

    mkt, day = _snapshot_market(snapshot)
    now = now or _dt.datetime.now()
    folder = Path(root) / day.strftime("%Y%m%d") / now.strftime("%H%M")
    folder.mkdir(parents=True, exist_ok=True)
    if plan is not None:
        pairs = [(t, c) for t, c in _equity_tickers(mkt) if plan.get(t)]
        jobs = [(t, c, m) for t, c in pairs for m in plan[t]]
        models = sorted({m for t, _ in pairs for m in plan[t] if m != "HestonSLV"})
        slv = [t for t, _ in pairs if "HestonSLV" in plan[t]]
    else:
        pairs = [(t, c) for t, c in _equity_tickers(mkt) if tickers is None or t in tickers]
        jobs = [(t, c, m) for t, c in pairs for m in models] + [(t, c, "HestonSLV") for t, c in pairs if t in slv]
    previous, rows, t_start = Settings.evaluation_date, [], time.perf_counter()
    Settings.evaluation_date = day
    try:
        for n, (ticker, ccy, name) in enumerate(jobs, start=1):
            row = {"ticker": ticker, "ccy": ccy, "model": name}
            t0 = time.perf_counter()
            if name not in MODEL_CLASSES:                 # e.g. a flag of a model not in the sets yet
                rows.append({**row, "status": "skipped", "error": f"{name}: not supported in the calibration sets"})
                continue
            try:
                equity = mkt.get_equity(ticker, ccy=Currency[ccy])
                model = MODEL_CLASSES[name].calibrate(equity)
                seconds = time.perf_counter() - t0
                fit = model_fit(model, equity)
                out = folder / ticker
                out.mkdir(exist_ok=True)
                for table, frame in model.to_frames().items():
                    frame.to_csv(out / f"{name}_{table}.csv", index=False)
                fit.to_csv(out / f"{name}_fit.csv", index=False)
                errors = _fit_errors(fit)
                row.update(status="ok", quality=_quality(errors["rms_volpts"]), seconds=seconds, **errors,
                           parameters=json.dumps({k: np.round(np.ravel(v.numpy()), 6).tolist()
                                                  for k, v in model.parameters.items()}),
                           jacobian=bool(getattr(model, "jacobian", None)))
            except Exception as exc:                      # one model failing does not stop the set
                row.update(status="error", seconds=time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            if progress is not None:
                progress(n, len(jobs), f"{ticker} {name}")
    finally:
        Settings.evaluation_date = previous
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / "summary.csv", index=False)
    info = {"created": now.isoformat(timespec="seconds"), "snapshot": {"date": day.isoformat(), "set": Path(snapshot).name},
            "models": list(models), "slv": list(slv), "tickers": [t for t, _ in pairs],
            "seconds": round(time.perf_counter() - t_start, 1),
            "counts": summary["status"].value_counts().to_dict() if len(summary) else {}}
    (folder / "calibration_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return folder


def list_calibration_sets(root) -> list[dict]:
    """Calibration sets under *root*, the most recently calibrated first (``created``; the folder date is the
    snapshot's): ``calibration_info.json`` plus ``date``, ``set`` (the folder), ``path``."""
    out = []
    root = Path(root)
    if not root.is_dir():
        return out
    for info_path in root.glob("*/*/calibration_info.json"):
        info = json.loads(info_path.read_text(encoding="utf-8"))
        folder = info_path.parent
        out.append({**info, "date": _dt.datetime.strptime(folder.parent.name, "%Y%m%d").date().isoformat(),
                    "set": folder.name, "path": str(folder)})
    return sorted(out, key=lambda r: (str(r.get("created", "")), r["date"], r["set"]), reverse=True)


def load_calibration_set(folder) -> dict:
    """``info`` (calibration_info.json) and ``summary`` (DataFrame) of a calibration set."""
    folder = Path(folder)
    return {"info": json.loads((folder / "calibration_info.json").read_text(encoding="utf-8")),
            "summary": pd.read_csv(folder / "summary.csv")}


def load_model(folder, ticker: str, name: str):
    """The model *name* of *ticker* stored in the calibration set *folder* (``from_frames``)."""
    base = Path(folder) / ticker
    frames = {}
    for path in base.glob(f"{name}_*.csv"):            # every table of to_frames (nested ones: heston.*, leverage_*)
        table = path.stem[len(name) + 1:]
        if table == "fit":
            continue
        settings = table.endswith("settings")          # JSON text, read as is
        frames[table] = pd.read_csv(path, dtype=str if settings else None, keep_default_na=not settings)
    if "model" not in frames:
        raise FileNotFoundError(f"no {name} model for {ticker} in {folder}")
    return MODEL_CLASSES[name].from_frames(frames)


# ----------------------------------------------------------------------------------------------------------------------
# model test: European products on shared paths
# ----------------------------------------------------------------------------------------------------------------------

def _simulate(model, equity: Equity, t_obs, n_paths: int, seed: int, step: float = 1.0 / 52.0) -> np.ndarray:
    """Spots at the times *t_obs* ``[n_paths, n_obs]`` (one simulation: daily steps over the first three months,
    weekly after, as the autocallable pricer)."""
    t_obs = np.asarray(t_obs, dtype=float)
    grid = np.unique(np.concatenate([simulation_grid(t_obs[-1], step), t_obs]))
    process = model.process(equity, float(t_obs[-1])) if hasattr(model, "process") else model
    k = process.noise_size()
    dw = tf.random.stateless_normal([n_paths, len(grid)] + ([k] if k > 1 else []), seed=[seed, 0], dtype=DTYPE)
    return process.evolve(tf.constant(grid, DTYPE), dw, observation_times=t_obs).numpy()


TEST_VANILLA = {"moneyness": (0.8, 0.9, 1.0, 1.1, 1.2), "maturities": (0.5, 1.0, 2.0, 3.0)}
TEST_BARRIER = {"barriers": (0.6, 0.7, 0.8), "strike": 1.0, "maturities": (1.0, 3.0)}
TEST_DIGITAL = {"moneyness": (0.9, 1.0, 1.1), "maturities": (1.0, 3.0)}


def _test_products(s0: float) -> list[dict]:
    """The battery: vanillas (call above the spot, put below and at), down-and-in puts observed at expiry, cash
    digitals (call and put)."""
    out = []
    for t in TEST_VANILLA["maturities"]:
        for m in TEST_VANILLA["moneyness"]:
            out.append({"product": "Vanilla", "option_type": "Call" if m > 1.0 else "Put", "t": t, "strike_pct": 100 * m,
                        "barrier_pct": None, "strike": m * s0})
    for t in TEST_BARRIER["maturities"]:
        for h in TEST_BARRIER["barriers"]:
            out.append({"product": "Down-and-in put (at expiry)", "option_type": "Put", "t": t,
                        "strike_pct": 100 * TEST_BARRIER["strike"], "barrier_pct": 100 * h,
                        "strike": TEST_BARRIER["strike"] * s0, "barrier": h * s0})
    for t in TEST_DIGITAL["maturities"]:
        for m in TEST_DIGITAL["moneyness"]:
            for cp in ("Call", "Put"):
                out.append({"product": "Digital (cash)", "option_type": cp, "t": t, "strike_pct": 100 * m,
                            "barrier_pct": None, "strike": m * s0})
    return out


def _payoff(p: dict, s: np.ndarray) -> np.ndarray:
    k = p["strike"]
    if p["product"] == "Vanilla":
        return np.maximum(s - k, 0.0) if p["option_type"] == "Call" else np.maximum(k - s, 0.0)
    if p["product"].startswith("Down-and-in"):
        return np.maximum(k - s, 0.0) * (s <= p["barrier"])
    return (s > k).astype(float) if p["option_type"] == "Call" else (s < k).astype(float)


def _black_price(p: dict, equity: Equity, mkt, expiry: _dt.date) -> float:
    """Black-Scholes with the surface vol at the strike; cash digitals as a call spread on the smile."""
    from ..instruments.barrier import BarrierOption, BarrierType
    from ..instruments.digital import DigitalOption
    from ..instruments.option import VanillaOption
    from ..pricers.black import BlackScholesPricer
    from ..timehandles.utils import Settings

    today, ot = Settings.evaluation_date, OptionType[p["option_type"]]
    if p["product"] == "Vanilla":
        product, pricer = VanillaOption(equity.currency, today, expiry, ot, p["strike"], underlying=equity.ticker), \
            BlackScholesPricer()
    elif p["product"].startswith("Down-and-in"):
        product = BarrierOption(equity.currency, today, expiry, ot, p["strike"], p["barrier"], BarrierType.DownIn,
                                monitoring_dates=[expiry], underlying=equity.ticker)
        pricer = BlackScholesPricer()
    else:
        product = DigitalOption(equity.currency, today, expiry, ot, p["strike"], underlying=equity.ticker)
        pricer = BlackScholesPricer(digital_spread=0.005)
    pricer.price(product, mkt, autodiff=False)
    return float(product.price)


def model_test(models: dict, equity: Equity, mkt, n_paths: int = 20_000, seed: int = 11) -> pd.DataFrame:
    """Price the battery with each model on shared paths (one simulation per model) against Black.

    Args:
        models (dict): ``{name: calibrated model}`` (the local vol model is added as ``LocalVol``).
        equity (Equity): The underlying.
        mkt (MarketEnvironment): Its market (for the Black prices).
        n_paths (int): Monte Carlo paths.
        seed (int): Seed of the draws (the same for every model).

    Returns:
        pd.DataFrame: One row per product: ``product``, ``option_type``, ``t``, ``expiry``, ``strike_pct``,
        ``barrier_pct``, ``Black`` and per model its price and standard error (``<name>_se``); vanillas also the
        implied vols (``<name>_vol``, ``Black_vol``). Prices in % of the spot (vanillas, barriers) or of the cash
        amount (digitals, paying 1).
    """
    from ..timehandles.utils import Settings

    today = Settings.evaluation_date
    s0 = float(equity.spot)
    products = _test_products(s0)
    t_obs = sorted({p["t"] for p in products})
    expiries = {t: today + _dt.timedelta(days=int(round(365 * t))) for t in t_obs}
    year = {t: equity.year_fraction(expiries[t]) for t in t_obs}
    for p in products:
        p["expiry"] = expiries[p["t"]]
    engines = {"LocalVol": LocalVolatilityModel(), **models}
    df = {t: float(equity.discount_curve.discount(expiries[t])) for t in t_obs}
    fwd = {t: float(equity.forward(expiries[t])) for t in t_obs}
    rows = [{k: v for k, v in p.items() if k not in ("strike", "barrier")} for p in products]
    unit = {id(p): 1.0 if p["product"].startswith("Digital") else s0 for p in products}   # % of cash or spot
    for row, p in zip(rows, products):
        row["expiry"] = p["expiry"].isoformat()
        row["Black"] = 100 * _black_price(p, equity, mkt, p["expiry"]) / unit[id(p)]
    for name, model in engines.items():
        start = time.perf_counter()
        terminal = _simulate(model, equity, [year[t] for t in t_obs], n_paths, seed)
        for row, p in zip(rows, products):
            i = t_obs.index(p["t"])
            discounted = df[p["t"]] * _payoff(p, terminal[:, i])
            row[name] = 100 * float(discounted.mean()) / unit[id(p)]
            row[f"{name}_se"] = 100 * float(discounted.std(ddof=1) / np.sqrt(n_paths)) / unit[id(p)]
        rows[0].setdefault("_seconds", {})[name] = time.perf_counter() - start
    for row, p in zip(rows, products):                    # implied vols of the vanillas
        if p["product"] == "Vanilla":
            t, cp = p["t"], p["option_type"][0]
            for name in ["Black", *engines]:
                undiscounted = row[name] * s0 / 100 / df[t]
                row[f"{name}_vol"] = float(black_implied_vol([undiscounted], fwd[t], [p["strike"]], year[t], cp)[0])
    seconds = rows[0].pop("_seconds", {})
    frame = pd.DataFrame(rows)
    frame.attrs["seconds"] = seconds
    return frame
