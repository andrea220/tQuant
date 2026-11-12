"""Build a snapshot: the tables of ``MarketEnvironment.from_folder`` (``eq_spot.csv``, ``eq_dividends.csv``,
``fx_spot.csv``, ``correlations.csv``) plus ``ir_fixings.csv`` and ``snapshot_info.json``, from a source and the
universe. Curves and vols come later from the internal service (``internal.py``)."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .history import correlations, update_history
from .options import equity_option_quotes
from .universe import Curve, Instrument


DECLARED_WINDOW = 30          # days: a projection this close to a declared dividend is that dividend
# not regular cash dividends: stock events, and special (extraordinary) cash dividends, for which the exchanges adjust
# the listed options (strike, multiplier) and OTC terms usually do the same: they do not lower the forward
NON_CASH = ("split", "stock", "spinoff", "spin-off", "rights", "special")


def _to_date(value) -> date:
    return pd.Timestamp(value).date()


def _months(tenor: str) -> int:
    n, unit = int(tenor[:-1]), tenor[-1].upper()
    return {"W": 0, "M": n, "Y": 12 * n}[unit] if unit != "W" else 0


def curve_quotes(source, curves: list[Curve], field: str) -> pd.DataFrame:
    """Quotes of the curve members (mid, decimal), one row per instrument: ``ccy, curve, tenor, instrument, quote,
    bloomberg, last_update``. Instruments: the index fixing (``fixing``, a deposit on its tenor: overnight on OIS
    curves, the Ibor tenor on Ibor curves), then OIS curves ``ois``, Ibor curves the FRAs up to 18M (``fra``, ending at
    the tenor, starting one index tenor earlier) and ``swap``."""
    rows = []
    for c in curves:
        members = source.bulk(c.bbg, field)
        if members.empty:
            raise ValueError(f"curve {c.ccy} {c.name}: no members from {c.bbg}")
        for _, m in members.iterrows():
            tenor, ticker = str(m["Tenor"]).strip(), str(m["Tenor Ticker"]).strip()
            if ticker.endswith("Index"):                  # the index fixing (overnight or Ibor): a deposit
                instrument = "fixing"
            elif c.kind == "ois":
                instrument = "ois"
            elif tenor[-1].upper() == "M" and _months(tenor) <= 18 or tenor.upper() == "18M":
                instrument = "fra"
            else:
                instrument = "swap"
            rows.append({"ccy": c.ccy, "curve": c.name, "tenor": tenor, "instrument": instrument,
                         "quote": round(float(m["Mid Yield"]) / 100.0, 10), "bloomberg": ticker,
                         "last_update": str(m.get("Last Update", ""))})
    return pd.DataFrame(rows, columns=["ccy", "curve", "tenor", "instrument", "quote", "bloomberg", "last_update"])


def build_tables(source, universe: list[Instrument], fields: dict, today: date, history: pd.DataFrame,
                 window: int, curves: list[Curve] = (), options: dict = None,
                 cache_dir: Path = None) -> dict[str, pd.DataFrame]:
    """Snapshot tables of *universe* on *today* (``{file name: frame}``); with *options* (the ``options`` block of
    the settings) also ``eq_option_quotes.csv`` (chains cached in *cache_dir*)."""
    tickers = [i.bbg for i in universe]
    ref = source.reference(tickers, [fields["price"], fields["currency"], fields["name"], fields["last_update"]])
    tables: dict[str, pd.DataFrame] = {}

    equities = [i for i in universe if i.kind == "equity"]
    currency = {i.bbg: str(ref.loc[i.bbg, fields["currency"]]) for i in universe}
    tables["eq_spot.csv"] = pd.DataFrame(
        [{"ticker": i.code, "ccy": currency[i.bbg], "spot": float(ref.loc[i.bbg, fields["price"]]), "repo": None,
          "div_yield": None} for i in equities], columns=["ticker", "ccy", "spot", "repo", "div_yield"])

    rows = []
    for i in equities:
        # declared dividends with a future ex-date (pay date known), then the projections (pay date = ex-date)
        # that are not within DECLARED_WINDOW days of a declared one (the same dividend, now announced)
        declared = []
        if fields.get("declared_dividends"):
            for r in source.bulk(i.bbg, fields["declared_dividends"]).to_dict("records"):
                ex, amount, kind = r.get("Ex-Date"), r.get("Dividend Amount"), str(r.get("Dividend Type") or "")
                if ex in ("", None) or pd.isna(ex) or amount is None or pd.isna(amount) \
                        or any(w in kind.lower() for w in NON_CASH):
                    continue
                ex = _to_date(ex)
                if ex > today and float(amount) > 0:
                    pay = r.get("Payable Date")
                    pay = _to_date(pay) if pay not in ("", None) and not pd.isna(pay) else ex
                    declared.append(ex)
                    rows.append({"ticker": i.code, "ccy": currency[i.bbg], "ex_date": ex.isoformat(),
                                 "pay_date": pay.isoformat(), "amount": float(amount)})
        div = source.bulk(i.bbg, fields["dividends"])
        if div.empty:
            continue
        ex_col = next(c for c in div.columns if c.lower().startswith("ex"))
        amount_col = next(c for c in div.columns if "amount" in c.lower())
        for _, r in div.iterrows():
            if pd.isna(r[ex_col]) or r[ex_col] in ("", None) or pd.isna(r[amount_col]):
                continue
            ex = _to_date(r[ex_col])
            if any(abs((ex - d).days) <= DECLARED_WINDOW for d in declared):
                continue
            if ex > today and float(r[amount_col]) > 0:
                rows.append({"ticker": i.code, "ccy": currency[i.bbg], "ex_date": ex.isoformat(),
                             "pay_date": ex.isoformat(), "amount": float(r[amount_col])})
    rows.sort(key=lambda r: (r["ticker"], r["ex_date"]))
    tables["eq_dividends.csv"] = pd.DataFrame(rows, columns=["ticker", "ccy", "ex_date", "pay_date", "amount"])

    tables["fx_spot.csv"] = pd.DataFrame(
        [{"base": i.fx_base, "quote": i.fx_quote, "spot": float(ref.loc[i.bbg, fields["price"]])}
         for i in universe if i.kind == "fx"], columns=["base", "quote", "spot"])

    tables["ir_fixings.csv"] = pd.DataFrame(
        [{"index": i.code, "ccy": currency[i.bbg], "fixing_date": _to_date(ref.loc[i.bbg, fields["last_update"]]).isoformat(),
          "value": round(float(ref.loc[i.bbg, fields["price"]]) / 100.0, 10), "bloomberg": i.bbg}
         for i in universe if i.kind == "fixing"], columns=["index", "ccy", "fixing_date", "value", "bloomberg"])

    if curves:
        tables["ir_quotes.csv"] = curve_quotes(source, list(curves), fields["curve_rates"])

    if options and equities:
        rates = {ccy: float(ref.loc[bbg, fields["price"]]) / 100.0 for ccy, bbg in (options.get("rates") or {}).items()
                 if bbg in ref.index}                    # overnight fixings: forward estimates for the strikes
        spots = {i.bbg: float(ref.loc[i.bbg, fields["price"]]) for i in equities}
        tables["eq_option_quotes.csv"] = equity_option_quotes(source, equities, spots, currency,
                                                              tables["eq_dividends.csv"], rates, options,
                                                              Path(cache_dir), today, history)

    names = {i.bbg: f"EQ:{currency[i.bbg]}:{i.code}" for i in equities}
    names.update({i.bbg: f"FX:{i.fx_quote}:{i.code}" for i in universe if i.kind == "fx"})
    tables["correlations.csv"] = correlations(history, names, window)
    return tables


def write_snapshot(root: Path, tables: dict[str, pd.DataFrame], info: dict, now: datetime) -> Path:
    """Write the tables under ``root/<yyyymmdd>/<hhmm>/`` with ``snapshot_info.json``; returns the folder."""
    folder = Path(root) / now.strftime("%Y%m%d") / now.strftime("%H%M")
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(folder / name, index=False)
    (folder / "snapshot_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return folder


def run(source, universe: list[Instrument], settings: dict, base: Path, now: datetime,
        curves: list[Curve] = ()) -> Path:
    """Update the history, build the tables and write the snapshot (the whole manual run)."""
    fields, out = settings["fields"], settings["output"]
    priced = [i.bbg for i in universe if i.kind in ("equity", "fx")]
    history = update_history(Path(base) / out["history"] / "prices.parquet", source, priced, fields["price"],
                             settings["history"]["years"], now.date())
    tables = build_tables(source, universe, fields, now.date(), history, settings["correlation"]["window_days"],
                          curves, settings.get("options"), Path(base) / out.get("cache", "cache") / "option_chains")
    info = {"created": now.isoformat(timespec="seconds"), "source": type(source).__name__,
            "universe": [i.bbg for i in universe], "fields": fields,
            "history": {"rows": int(len(history)), "first": str(history["date"].min().date()) if len(history) else None},
            "rows": {name: int(len(frame)) for name, frame in tables.items()},
            "curves": {f"{c.ccy} {c.name}": c.bbg for c in curves},
            "missing": "ir_vol, fx_vol: from the internal service (not yet); curves: quotes only (ir_quotes.csv), "
                       "bootstrapped by tQuant; eq_vol / eq_repo: from the option quotes (eq_option_quotes.csv), "
                       "built by tQuant"}
    return write_snapshot(Path(base) / out["snapshots"], tables, info, now)
