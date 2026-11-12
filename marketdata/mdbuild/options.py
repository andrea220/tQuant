"""Listed equity options for the implied vol surfaces, with few Bloomberg data.

- The chain (``OPT_CHAIN``: one request per underlying) is kept in ``cache/option_chains/<code>.csv`` and refreshed
  every ``chain_max_age_days``: its tickers already carry root, exchange, expiry, put / call and strike
  (``ENI IM 12/17/27 P20 Equity``), so none of them is requested as a field. The exercise style is read once per
  chain on one option.
- Only the main root (default: the underlying's ticker and exchange, ``ENI IM`` / ``AAPL US``; ``option_roots`` in
  ``universe.yaml`` overrides it).
- Expiries: the first ``front_expiries``, then the quarterly ones (Mar / Jun / Sep / Dec), up to ``max_years``.
- Strikes per expiry, around a forward estimate ``F = (S - dividends to T) exp(r T)`` (BDVD projections, overnight
  fixing of the currency): for each target ``moneyness`` of ``F`` the nearest listed out-of-the-money option (puts
  below ``F``, calls above), plus calls and puts at the ``parity_strikes`` strikes on each side of ``F`` (put-call
  parity for the implied forward).
- Four fields (``PX_BID``, ``PX_ASK``, ``PX_LAST`` = settlement, ``LAST_UPDATE_BID_RT``). Session check on the
  time of the last bid update (``LAST_UPDATE_BID_RT``: a time when it is of today, a date otherwise; not
  ``LAST_UPDATE_DT`` / ``TIME``, which are of the last trade, often days old on live quotes): the quote date is the
  most frequent one of the two-sided quotes; quotes older than ``stale_days`` before it are dropped; if it is not
  today (market not open yet) the reference spot is the close of that day from the history, so that spot and
  options are of the same session.
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

CHAIN = re.compile(r"^(\S+) (\S+) (\d\d/\d\d/\d\d) ([CP])([\d.]+) Equity$")
QUOTE_COLUMNS = ["ticker", "ccy", "expiry", "strike", "option_type", "exercise", "bid", "ask", "last", "last_update",
                 "quote_date", "ref_spot", "spot_source", "bloomberg"]


def parse_chain(tickers) -> pd.DataFrame:
    """Chain tickers -> ``root, exch, expiry, option_type, strike, bloomberg`` (unparsable ones dropped)."""
    rows = []
    for t in tickers:
        m = CHAIN.match(str(t).strip())
        if m:
            root, exch, expiry, cp, strike = m.groups()
            rows.append({"root": root, "exch": exch, "expiry": pd.Timestamp(expiry).date(), "option_type": cp,
                         "strike": float(strike), "bloomberg": str(t).strip()})
    return pd.DataFrame(rows, columns=["root", "exch", "expiry", "option_type", "strike", "bloomberg"])


def main_root(underlying: str, overrides: Optional[dict] = None) -> tuple[str, str]:
    """``(root, exchange)`` of the options of *underlying*: override from ``option_roots`` or ticker and exchange."""
    if overrides and underlying in overrides:
        root, exch = str(overrides[underlying]).split()[:2]
        return root, exch
    parts = underlying.split()
    return parts[0], parts[1]


def load_chain(source, underlying: str, code: str, cache_dir: Path, today: date, max_age_days: int,
               overrides: Optional[dict] = None) -> pd.DataFrame:
    """Chain of the main root of *underlying*, from the cache if younger than *max_age_days* (else downloaded and
    cached), with the ``exercise`` style (read on one option at download)."""
    path = Path(cache_dir) / f"{code}.csv"
    if path.is_file():
        cached = pd.read_csv(path, parse_dates=["expiry", "downloaded"])
        if len(cached) and (today - cached["downloaded"].iloc[0].date()).days < max_age_days:
            return cached.assign(expiry=cached["expiry"].dt.date, downloaded=cached["downloaded"].dt.date)
    raw = source.bulk(underlying, "OPT_CHAIN")
    column = raw.columns[0] if len(raw.columns) else None
    chain = parse_chain(raw[column] if column is not None else [])
    root, exch = main_root(underlying, overrides)
    chain = chain[(chain["root"] == root) & (chain["exch"] == exch)].reset_index(drop=True)
    exercise = "American"
    if len(chain):
        style = source.reference([chain["bloomberg"].iloc[0]], ["OPT_EXER_TYP"]).iloc[0, 0]
        exercise = "European" if str(style).lower().startswith("euro") else "American"
    chain = chain.assign(exercise=exercise, downloaded=today)
    path.parent.mkdir(parents=True, exist_ok=True)
    chain.to_csv(path, index=False)
    return chain


def select_expiries(expiries, today: date, front: int, max_years: float) -> list[date]:
    """The first *front* expiries after today, then the quarterly ones (Mar / Jun / Sep / Dec), up to *max_years*."""
    horizon = today + timedelta(days=int(round(max_years * 365.25)))
    listed = sorted(e for e in set(expiries) if today < e <= horizon)
    front_set = listed[:front]
    return front_set + [e for e in listed[front:] if e.month in (3, 6, 9, 12)]


def forward_estimate(spot: float, dividends: pd.DataFrame, rate: float, today: date, expiry: date) -> float:
    """``(S - sum of the dividends with ex-date in (today, expiry]) exp(r T)``: only to choose the strikes."""
    t = (expiry - today).days / 365.0
    divs = 0.0
    if dividends is not None and len(dividends):
        ex = pd.to_datetime(dividends["ex_date"]).dt.date
        divs = float(dividends.loc[(ex > today) & (ex <= expiry), "amount"].sum())
    return (spot - divs) * math.exp(rate * t)


def select_options(chain: pd.DataFrame, forwards: dict, moneyness, parity_strikes: int) -> pd.DataFrame:
    """Options to request: per expiry (keys of *forwards*), the nearest out-of-the-money option to each target
    ``moneyness x F`` and both types at the *parity_strikes* strikes on each side of ``F``."""
    picked = []
    for expiry, f in forwards.items():
        grp = chain[chain["expiry"] == expiry]
        puts, calls = grp[(grp["option_type"] == "P") & (grp["strike"] < f)], grp[(grp["option_type"] == "C") & (grp["strike"] >= f)]
        for m in moneyness:
            side = puts if m * f < f else calls
            if len(side):
                picked.append(side.loc[(side["strike"] - m * f).abs().idxmin()])
        both = sorted(set(grp.loc[grp["option_type"] == "C", "strike"]) & set(grp.loc[grp["option_type"] == "P", "strike"]))
        below = [k for k in both if k < f][-parity_strikes:]
        above = [k for k in both if k >= f][:parity_strikes]
        picked += [r for _, r in grp[grp["strike"].isin(below + above)].iterrows()]
    if not picked:
        return chain.iloc[0:0]
    out = pd.DataFrame(picked).drop_duplicates("bloomberg")
    return out.sort_values(["expiry", "option_type", "strike"]).reset_index(drop=True)


def quote_day(value, today: date):
    """Date of a ``LAST_UPDATE_BID_RT`` value: a time (``16:29:32``) is today, a date is that date; ``None`` if
    missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if ":" in text and "-" not in text and "/" not in text:
        return today
    stamp = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(stamp) else stamp.date()


def option_quotes(source, selection: pd.DataFrame, code: str, ccy: str, underlying: str, spot: float, today: date,
                  history: Optional[pd.DataFrame], stale_days: int) -> pd.DataFrame:
    """Quotes of the selected options (four fields) with the session check (``quote_date``, ``ref_spot``)."""
    if selection.empty:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    fields = ["PX_BID", "PX_ASK", "PX_LAST", "LAST_UPDATE_BID_RT"]
    ref = source.reference(list(selection["bloomberg"]), fields)
    q = selection.set_index("bloomberg").join(ref)
    for f in fields[:3]:
        q[f] = pd.to_numeric(q[f], errors="coerce")
    q["last_update"] = [quote_day(v, today) for v in q["LAST_UPDATE_BID_RT"]]
    two_sided = q["PX_BID"].notna() & q["PX_ASK"].notna()
    dates = q.loc[two_sided, "last_update"].dropna()
    dates = dates if len(dates) else q["last_update"].dropna()
    quote_date = dates.mode().iloc[0] if len(dates) else today
    q = q[q["last_update"].notna() & (q["last_update"] >= quote_date - timedelta(days=stale_days))]
    ref_spot, spot_source = spot, "snapshot"
    if quote_date != today:
        closes = None if history is None else history[(history["ticker"] == underlying)
                                                       & (history["date"].dt.date == quote_date)]
        if closes is not None and len(closes):
            ref_spot, spot_source = float(closes["value"].iloc[-1]), f"close {quote_date.isoformat()}"
        else:
            spot_source = f"snapshot (no close for {quote_date.isoformat()})"
    out = pd.DataFrame({
        "ticker": code, "ccy": ccy, "expiry": [e.isoformat() for e in q["expiry"]], "strike": q["strike"],
        "option_type": q["option_type"], "exercise": q["exercise"], "bid": q["PX_BID"], "ask": q["PX_ASK"],
        "last": q["PX_LAST"], "last_update": [d.isoformat() for d in q["last_update"]],
        "quote_date": quote_date.isoformat(), "ref_spot": ref_spot, "spot_source": spot_source,
        "bloomberg": q.index})
    return out[QUOTE_COLUMNS].reset_index(drop=True)


def equity_option_quotes(source, equities, spots: dict, currencies: dict, dividends: pd.DataFrame, rates: dict,
                         settings: dict, cache_dir: Path, today: date,
                         history: Optional[pd.DataFrame]) -> pd.DataFrame:
    """``eq_option_quotes.csv`` for the equities of the universe (``settings``: the ``options`` block)."""
    frames = []
    for inst in equities:
        chain = load_chain(source, inst.bbg, inst.code, cache_dir, today, settings["chain_max_age_days"],
                           settings.get("roots"))
        if chain.empty:
            continue
        divs = dividends[dividends["ticker"] == inst.code] if dividends is not None else None
        rate = rates.get(currencies[inst.bbg], 0.0)
        expiries = select_expiries(chain["expiry"], today, settings["front_expiries"], settings["max_years"])
        forwards = {e: forward_estimate(spots[inst.bbg], divs, rate, today, e) for e in expiries}
        selection = select_options(chain, forwards, settings["moneyness"], settings["parity_strikes"])
        frames.append(option_quotes(source, selection, inst.code, currencies[inst.bbg], inst.bbg, spots[inst.bbg],
                                    today, history, settings["stale_days"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=QUOTE_COLUMNS)
