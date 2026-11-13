"""Listed equity options for the implied vol surfaces, with few Bloomberg data.

- The chain, as Bloomberg advises for large chains (``OPT_CHAIN`` is truncated: SPX at 8000 options, up to 4
  expiries): ``CHAIN_EXPIRATION_DATES`` (all the expiries, one request), the selection below, then ``CHAIN_TICKERS``
  of each selected expiry for calls and for puts (two requests), filtered by strikes (``strike_range`` of the spot),
  unadjusted contracts only and the main exchange; Bloomberg's default exercise filter keeps the American equity
  options and the European index options. Kept in ``cache/option_chains/<code>.csv``, refreshed every
  ``chain_max_age_days`` or when its first expiry has passed. The tickers carry root, exchange, expiry, put / call and
  strike (``ENI IM 12/17/27 P20 Equity``), so none of them is requested as a field.
- The main root (default: the underlying's ticker and exchange, ``ENI IM`` / ``AAPL US``; ``option_roots`` in
  ``universe.yaml`` overrides it) and its weekly roots on the same exchange (a digit and the main root: ``2ENI``,
  ``4ENI`` on IDEM; US weeklies are in the main root). On the same expiry the main root wins.
- Expiries: the first ``weekly_expiries`` listed, the monthlies (third Friday) to ``monthly_months``, the quarterly
  monthlies (Mar / Jun / Sep / Dec) to ``quarterly_months``, then one a year (the nearest to mid-December) to
  ``max_years``.
- Strikes per expiry, around a forward estimate ``F = (S - dividends to T) exp(r T)`` (BDVD projections, overnight
  fixing of the currency): for each target ``moneyness`` of ``F`` the nearest listed out-of-the-money option (puts
  below ``F``, calls above), plus calls and puts at the ``parity_strikes`` strikes on each side of ``F`` (put-call
  parity for the implied forward).
- Four fields (``PX_BID``, ``PX_ASK``, ``PX_LAST`` = settlement, ``LAST_UPDATE_BID_RT``). Session check on the
  time of the last bid update (``LAST_UPDATE_BID_RT``: a time when it is of today, a date otherwise; not
  ``LAST_UPDATE_DT`` / ``TIME``, which are of the last trade, often days old on live quotes): the quote date is the
  most frequent one of the two-sided quotes; bid / ask older than ``stale_days`` before it are dropped (the option
  stays on its last price, as a settlement, if it has one); if it is not
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

# equity options "ENI IM 12/17/27 P20 Equity", index options "SPX US 12/18/26 C5000 Index", "SX5E 12/18/26 C5000 Index"
CHAIN = re.compile(r"^(\S+)(?: (\S+))? (\d\d/\d\d/\d\d) ([CP])([\d.]+) (?:Equity|Index)$")
QUOTE_COLUMNS = ["ticker", "ccy", "expiry", "strike", "option_type", "exercise", "bid", "ask", "last", "last_update",
                 "quote_date", "ref_spot", "spot_source", "bloomberg"]


def parse_chain(tickers) -> pd.DataFrame:
    """Chain tickers -> ``root, exch, expiry, option_type, strike, bloomberg`` (unparsable ones dropped)."""
    rows = []
    for t in tickers:
        m = CHAIN.match(str(t).strip())
        if m:
            root, exch, expiry, cp, strike = m.groups()
            rows.append({"root": root, "exch": exch or "", "expiry": pd.Timestamp(expiry).date(), "option_type": cp,
                         "strike": float(strike), "bloomberg": str(t).strip()})
    return pd.DataFrame(rows, columns=["root", "exch", "expiry", "option_type", "strike", "bloomberg"])


def main_root(underlying: str, overrides: Optional[dict] = None) -> tuple[str, Optional[str]]:
    """``(root, exchange)`` of the options of *underlying*: override from ``option_roots`` or ticker and exchange
    (an index: its name, any exchange, ``None``)."""
    if overrides and underlying in overrides:
        parts = str(overrides[underlying]).split()
        return parts[0], parts[1] if len(parts) > 1 else None
    parts = underlying.split()
    if parts[-1] == "Index":
        return parts[0], None
    return parts[0], parts[1]


def is_weekly_root(candidate: str, root: str) -> bool:
    """Weekly root of *root*: a digit and the root (IDEM ``2ENI``, ``4ENI``) or the root and ``W`` (``SPXW``)."""
    return (len(candidate) > len(root) and candidate[0].isdigit() and candidate[1:] == root) \
        or candidate == root + "W"


CHAIN_CACHE_VERSION = 3          # 3: CHAIN_TICKERS of the selected expiries (2: OPT_CHAIN, main and weekly roots)


def chain_overrides(underlying: str, expiry: str, option_type: str, strike_range: str) -> dict:
    """``CHAIN_TICKERS`` filters of one expiry (``YYYYMMDD``) and type: strikes in *strike_range* of the spot
    (``"40%-200%"``), unadjusted contracts, all periodicities, the exchange of an equity (an index has none)."""
    out = {"CHAIN_EXP_DT_OVRD": expiry, "CHAIN_EXP_MATCH_OVRD": "Exact", "CHAIN_PUT_CALL_TYPE_OVRD": option_type,
           "CHAIN_STRIKE_PX_OVRD": strike_range, "CHAIN_POINTS_OVRD": "1000", "CHAIN_PERIODICITY_OVRD": "ALL",
           "CHAIN_ADJ_CONTRACT_OVRD": "N"}
    parts = underlying.split()
    if parts[-1] != "Index":
        out["CHAIN_EXCH_OVRD"] = parts[1]
    return out


def load_chain(source, underlying: str, code: str, cache_dir: Path, today: date, settings: dict) -> pd.DataFrame:
    """Options of the selected expiries of *underlying* (main and weekly roots, ``weekly`` flag, ``exercise``), from
    the cache if younger than ``chain_max_age_days`` and its first expiry still to come, else downloaded (expiries,
    then two ``CHAIN_TICKERS`` requests per selected expiry) and cached. *settings*: the ``options`` block."""
    path = Path(cache_dir) / f"{code}.csv"
    if path.is_file():
        cached = pd.read_csv(path, parse_dates=["expiry", "downloaded"])
        if len(cached) and "version" in cached and int(cached["version"].iloc[0]) == CHAIN_CACHE_VERSION \
                and (today - cached["downloaded"].iloc[0].date()).days < settings["chain_max_age_days"] \
                and cached["expiry"].min().date() > today:
            return cached.assign(expiry=cached["expiry"].dt.date, downloaded=cached["downloaded"].dt.date)
    index = underlying.split()[-1] == "Index"
    dates_filter = {"CHAIN_EXP_DT_OVRD": "ALL", "CHAIN_PERIODICITY_OVRD": "ALL"}
    if not index:
        dates_filter["CHAIN_EXCH_OVRD"] = underlying.split()[1]
    listed = source.bulk(underlying, "CHAIN_EXPIRATION_DATES", dates_filter)
    expiries = [pd.Timestamp(d).date() for d in (listed.iloc[:, 0] if len(listed.columns) else [])]
    selected = select_expiries(expiries, today, settings["weekly_expiries"], settings["monthly_months"],
                               settings["quarterly_months"], settings["max_years"])
    tickers = []
    for expiry in selected:
        for cp in ("C", "P"):
            got = source.bulk(underlying, "CHAIN_TICKERS",
                              chain_overrides(underlying, expiry.strftime("%Y%m%d"), cp, settings["strike_range"]))
            if len(got.columns):
                tickers += [f"{t} {underlying.split()[-1]}" for t in got.iloc[:, 0].astype(str)]
    chain = parse_chain(tickers)
    root, exch = main_root(underlying, settings.get("roots"))
    weekly = chain["root"].map(lambda r: is_weekly_root(r, root)).astype(bool) if len(chain) else chain["root"]
    chain = chain[(chain["root"] == root) | weekly].reset_index(drop=True)
    chain["weekly"] = chain["root"] != root
    chain = chain.assign(exercise="European" if index else "American", downloaded=today,
                         version=CHAIN_CACHE_VERSION)
    path.parent.mkdir(parents=True, exist_ok=True)
    chain.to_csv(path, index=False)
    return chain


def _monthly(d: date) -> bool:
    """Standard monthly expiry: the third Friday (the Thursday before when the Friday is a holiday)."""
    return 15 <= d.day <= 21 and d.weekday() in (3, 4)


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    return date(d.year + y, m + 1, min(d.day, 28))


def select_expiries(expiries, today: date, weekly: int = 2, monthly_months: int = 3, quarterly_months: int = 18,
                    max_years: float = 6.2) -> list[date]:
    """Expiries to request: the first *weekly* listed, the monthlies to *monthly_months*, the quarterly monthlies
    (Mar / Jun / Sep / Dec) to *quarterly_months*, then one a year (the nearest to mid-December) to *max_years*."""
    horizon = today + timedelta(days=int(round(max_years * 365.25)))
    listed = sorted(e for e in set(expiries) if today < e <= horizon)
    out = set(listed[:weekly])
    monthly_end, quarterly_end = _add_months(today, monthly_months), _add_months(today, quarterly_months)
    out |= {e for e in listed if _monthly(e) and e <= monthly_end}
    out |= {e for e in listed if _monthly(e) and e.month in (3, 6, 9, 12) and e <= quarterly_end}
    for year in sorted({e.year for e in listed if e > quarterly_end}):
        later = [e for e in listed if e.year == year and e > quarterly_end]
        out.add(min(later, key=lambda e: abs((e - date(year, 12, 15)).days)))
    return sorted(out)


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
        if "weekly" in grp and (~grp["weekly"].astype(bool)).any():     # main root wins on a shared expiry
            grp = grp[~grp["weekly"].astype(bool)]
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
    """Quotes of the selected options (four fields) with the session check (``quote_date``, ``ref_spot``). The
    underlying is in the same request: its ``PX_LAST`` is the spot at the time of the quotes (the snapshot spot is
    read minutes earlier, a fast index moves in between)."""
    if selection.empty:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    fields = ["PX_BID", "PX_ASK", "PX_LAST", "LAST_UPDATE_BID_RT"]
    ref = source.reference(list(selection["bloomberg"]) + [underlying], fields)
    live = pd.to_numeric(ref.loc[underlying, "PX_LAST"], errors="coerce") if underlying in ref.index else float("nan")
    ref = ref.drop(index=underlying, errors="ignore")
    q = selection.set_index("bloomberg").join(ref)
    for f in fields[:3]:
        q[f] = pd.to_numeric(q[f], errors="coerce")
    q["last_update"] = [quote_day(v, today) for v in q["LAST_UPDATE_BID_RT"]]
    two_sided = q["PX_BID"].notna() & q["PX_ASK"].notna()
    dates = q.loc[two_sided, "last_update"].dropna()
    dates = dates if len(dates) else q["last_update"].dropna()
    quote_date = dates.mode().iloc[0] if len(dates) else today
    # stale bid / ask (older than stale_days before the quote date): dropped, the option kept on its last price
    # (settlement, used by tQuant with a lower weight) if it has one; without a valid quote at all it is dropped
    stale = q["last_update"].isna() | q["last_update"].map(
        lambda d: d is not None and d < quote_date - timedelta(days=stale_days)).astype(bool)
    q.loc[stale, ["PX_BID", "PX_ASK"]] = float("nan")
    q = q[(~stale & q["PX_BID"].notna() & q["PX_ASK"].notna()) | (q["PX_LAST"] > 0)]
    q["last_update"] = q["last_update"].map(lambda d: d if d is not None else quote_date)
    ref_spot, spot_source = spot, "snapshot"
    if quote_date == today and live == live and live > 0:          # live spot read with the quotes
        ref_spot, spot_source = float(live), "with the quotes"
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
        chain = load_chain(source, inst.bbg, inst.code, cache_dir, today, settings)
        if chain.empty:
            continue
        divs = dividends[dividends["ticker"] == inst.code] if dividends is not None else None
        rate = rates.get(currencies[inst.bbg], 0.0)
        expiries = select_expiries(chain["expiry"], today, settings["weekly_expiries"], settings["monthly_months"],
                                   settings["quarterly_months"], settings["max_years"])
        forwards = {e: forward_estimate(spots[inst.bbg], divs, rate, today, e) for e in expiries}
        selection = select_options(chain, forwards, settings["moneyness"], settings["parity_strikes"])
        quotes = option_quotes(source, selection, inst.code, currencies[inst.bbg], inst.bbg, spots[inst.bbg],
                               today, history, settings["stale_days"])
        if inst.bbg in (settings.get("implied_spot") or ()) and len(quotes):
            quotes["spot_source"] = "implied from the options"          # delayed price: tQuant implies the spot
        frames.append(quotes)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=QUOTE_COLUMNS)
