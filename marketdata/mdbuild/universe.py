"""The universe (``config/universe.yaml``): the instruments with their attributes and an ``active`` flag (downloaded by
a full download), the rate curves. Bloomberg tickers only, the type read from the yellow key.

Format::

    instruments:
      - ticker: ENI IM Equity
        active: true
      - ticker: SPX Index
        active: true
        equity_index: true          # Index yellow key, but an equity underlying (spot, history, listed options)
        implied_spot: true          # price delayed against its options: the spot is implied by the options
        models: {HestonSLV: true}   # calibration flags (the defaults of calibration.yaml when absent)
      - ticker: ENEL IM Equity
        active: false
        option_root: ENE IM         # root of its listed options, when not the underlying's ticker and exchange
    curves:
      - name: EUR ESTR              # "<CCY> <tQuant curve name>"
        ticker: YCSW0514 Index      # Bloomberg ICVS curve
        active: true

The former format (``tickers`` / ``equity_indices`` lists, excluded tickers commented out, ``curves`` mapping,
``option_roots``, ``implied_spot``) is still read; :func:`convert_universe` rewrites a file in the new one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

KINDS = {"Equity": "equity", "Curncy": "fx", "Index": "fixing"}
ATTRIBUTES = ("equity_index", "implied_spot", "option_root", "models")      # optional keys of an instrument


@dataclass(frozen=True)
class Instrument:
    """One Bloomberg ticker of the universe.

    Attributes:
        bbg: Bloomberg ticker with its yellow key (``"ENI IM Equity"``).
        kind: ``"equity"``, ``"fx"`` or ``"fixing"``.
        code: Ticker in the snapshot tables: equities ``"ENI_IM"`` (spaces to ``_``, yellow key dropped), FX the pair
            (``"EURUSD"``), fixings the Bloomberg name (``"EUR006M"``).
    """

    bbg: str
    kind: str
    code: str

    @property
    def fx_base(self) -> str:
        return self.code[:3]

    @property
    def fx_quote(self) -> str:
        return self.code[3:6]


def parse(ticker: str) -> Instrument:
    """Instrument of a Bloomberg ticker.

    Raises:
        ValueError: If the yellow key is not Equity, Curncy or Index, or an FX ticker is not a 6-letter pair.
    """
    parts = ticker.strip().split()
    if len(parts) < 2 or parts[-1] not in KINDS:
        raise ValueError(f"'{ticker}': expected a Bloomberg ticker ending with one of {sorted(KINDS)}")
    kind = KINDS[parts[-1]]
    body = parts[:-1]
    if kind == "fx":
        pair = "".join(body)
        if len(pair) != 6 or not pair.isalpha():
            raise ValueError(f"'{ticker}': an FX ticker is a 6-letter pair (EURUSD Curncy)")
        return Instrument(ticker.strip(), kind, pair.upper())
    return Instrument(ticker.strip(), kind, "_".join(body).replace("/", ""))


def instrument(entry: dict) -> Instrument:
    """Instrument of an entry of the universe (an equity index: an ``Index`` ticker flagged ``equity_index``)."""
    bbg = str(entry["ticker"]).strip()
    if entry.get("equity_index"):
        return Instrument(bbg, "equity", "_".join(bbg.split()[:-1]))
    return parse(bbg)


@dataclass(frozen=True)
class Curve:
    """A rate curve of the universe: tQuant name (``ccy``, ``name``: ``EUR``, ``6M``) and Bloomberg ICVS ticker.
    ``kind`` is ``"ibor"`` for names like ``6M`` / ``3M`` (fixing, FRAs, swaps), ``"ois"`` otherwise."""

    ccy: str
    name: str
    bbg: str

    @property
    def kind(self) -> str:
        return "ibor" if self.name[:-1].isdigit() and self.name[-1] in "MY" else "ois"

    @property
    def label(self) -> str:
        return f"{self.ccy} {self.name}"


def _curve(name: str, ticker: str) -> Curve:
    parts = str(name).split()
    if len(parts) != 2 or len(parts[0]) != 3:
        raise ValueError(f"curve '{name}': expected '<CCY> <name>' (e.g. 'EUR ESTR')")
    return Curve(parts[0].upper(), parts[1], str(ticker).strip())


# ------------------------------------------------------------------------------------------------- reading


_EXCLUDED = re.compile(r"^\s*#\s*-\s*(\S.*\s(?:Equity|Curncy|Index))\s*$")


def _former(text: str) -> tuple[list[dict], list[dict]]:
    """Entries of the former format: listed tickers active, commented ones (``# - <ticker>`` in ``tickers`` /
    ``equity_indices``) inactive, in the order of the file."""
    data = yaml.safe_load(text) or {}
    indices = {str(t).strip() for t in data.get("equity_indices") or []}
    roots = {str(k).strip(): str(v).strip() for k, v in (data.get("option_roots") or {}).items()}
    implied = {str(t).strip() for t in data.get("implied_spot") or []}
    entries, seen, section = [], set(), None
    for line in text.splitlines():
        if re.match(r"^[A-Za-z_]+:", line):
            section = line.split(":")[0]
        if section not in ("tickers", "equity_indices"):
            continue
        active = re.match(r"^\s+-\s*(\S.*?)\s*$", line)
        excluded = _EXCLUDED.match(line)
        if not (active or excluded):
            continue
        ticker = (active or excluded).group(1).strip()
        if ticker in seen:
            continue
        seen.add(ticker)
        entry = {"ticker": ticker, "active": bool(active)}
        if section == "equity_indices" or ticker in indices:
            entry["equity_index"] = True
        if ticker in implied:
            entry["implied_spot"] = True
        if ticker in roots:
            entry["option_root"] = roots[ticker]
        entries.append(entry)
    curves = [{"name": str(k), "ticker": str(v).strip(), "active": True} for k, v in (data.get("curves") or {}).items()]
    return entries, curves


def read_entries(path: Path) -> tuple[list[dict], list[dict]]:
    """All the instruments and curves of the universe file (active or not), in the new format's entries.

    Returns:
        tuple: ``(instruments, curves)``: lists of dicts (``ticker``, ``active`` and the optional attributes;
        curves ``name``, ``ticker``, ``active``).

    Raises:
        ValueError: An entry without a ticker, or a repeated ticker.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if "instruments" not in data:
        return _former(text)
    instruments, seen = [], set()
    for raw in data.get("instruments") or []:
        if not isinstance(raw, dict) or not raw.get("ticker"):
            raise ValueError(f"universe entry without a ticker: {raw}")
        entry = {"ticker": " ".join(str(raw["ticker"]).split()), "active": bool(raw.get("active", True))}
        for key in ATTRIBUTES:
            if raw.get(key) not in (None, False, "", {}):
                entry[key] = raw[key]
        if entry["ticker"] in seen:
            raise ValueError(f"'{entry['ticker']}' twice in the universe")
        seen.add(entry["ticker"])
        instruments.append(entry)
    curves = [{"name": str(c["name"]), "ticker": str(c["ticker"]).strip(), "active": bool(c.get("active", True))}
              for c in data.get("curves") or []]
    return instruments, curves


def _chosen(entries: list[dict], select, key: str) -> list[dict]:
    """The active entries, or those named in *select* (whatever their flag)."""
    if select is None:
        return [e for e in entries if e["active"]]
    wanted = {" ".join(str(s).split()).upper() for s in select}
    return [e for e in entries if e[key].upper() in wanted]


def load_universe(path: Path, select=None) -> list[Instrument]:
    """Instruments to download: the active ones, or the ones named in *select* (Bloomberg tickers)."""
    return [instrument(e) for e in _chosen(read_entries(path)[0], select, "ticker")]


def load_curves(path: Path, select=None) -> list[Curve]:
    """Curves to download: the active ones, or the ones named in *select* (``"<CCY> <name>"``)."""
    return [_curve(c["name"], c["ticker"]) for c in _chosen(read_entries(path)[1], select, "name")]


def load_option_roots(path: Path) -> dict:
    """Underlying -> ``"<root> <exchange>"`` of its listed options (only to override the default, the underlying's
    own ticker and exchange)."""
    return {e["ticker"]: str(e["option_root"]).strip() for e in read_entries(path)[0] if e.get("option_root")}


def load_implied_spot(path: Path) -> list[str]:
    """Underlyings whose price is delayed against their options (the spot is implied by the options in tQuant)."""
    return [e["ticker"] for e in read_entries(path)[0] if e.get("implied_spot")]


# ------------------------------------------------------------------------------------------------- writing


HEADER = """# Universe of the market data snapshots (edited from tqstudio > Market Data > Universe).
# instruments: Bloomberg tickers, the yellow key decides the type (Equity -> equity spot, dividends, history, listed
#   options; Curncy -> FX spot, history; Index -> interest-rate fixing, or an equity index with equity_index: true).
#   active: downloaded by a full download (an inactive one can still be downloaded on its own).
#   Optional: equity_index, implied_spot (price delayed against its options: spot implied by the options),
#   option_root ("<root> <exchange>" of the listed options when not the underlying's own), models (calibration flags,
#   the defaults of calibration.yaml when absent).
# curves: "<CCY> <tQuant curve name>" and the Bloomberg ICVS curve (names like 6M are Ibor curves, the others OIS).
"""


def write_entries(path: Path, instruments: list[dict], curves: list[dict]) -> None:
    """Write the universe file in the new format (one entry per instrument, the file's line endings kept)."""
    path = Path(path)
    newline = "\r\n" if path.is_file() and b"\r\n" in path.read_bytes() else "\n"
    clean = []
    for e in instruments:
        entry = {"ticker": e["ticker"], "active": bool(e.get("active", True))}
        entry.update({k: e[k] for k in ATTRIBUTES if e.get(k) not in (None, False, "", {})})
        clean.append(entry)
    body = yaml.safe_dump({"instruments": clean, "curves": [
        {"name": c["name"], "ticker": c["ticker"], "active": bool(c.get("active", True))} for c in curves]},
        sort_keys=False, default_flow_style=None, allow_unicode=True, width=120)
    text = HEADER + "\n" + body
    path.write_bytes(newline.join(text.splitlines()).encode("utf-8") + newline.encode())


def convert_universe(path: Path) -> bool:
    """Rewrite a universe file of the former format in the new one (commented tickers become inactive).

    Returns:
        bool: True if the file was converted, False if it was already in the new format.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "instruments" in data:
        return False
    write_entries(path, *read_entries(path))
    return True
