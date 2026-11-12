"""The universe: Bloomberg tickers only, the type read from the yellow key."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

KINDS = {"Equity": "equity", "Curncy": "fx", "Index": "fixing"}


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


def load_curves(path: Path) -> list[Curve]:
    """Curves of ``universe.yaml`` (``curves:`` mapping ``"<CCY> <name>": <ICVS ticker>``)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out = []
    for key, ticker in (data.get("curves") or {}).items():
        parts = str(key).split()
        if len(parts) != 2 or len(parts[0]) != 3:
            raise ValueError(f"curve '{key}': expected '<CCY> <name>' (e.g. 'EUR ESTR')")
        out.append(Curve(parts[0].upper(), parts[1], str(ticker).strip()))
    return out


def load_option_roots(path: Path) -> dict:
    """``option_roots`` of ``universe.yaml``: underlying -> ``"<root> <exchange>"`` of its listed options (only to
    override the default, the underlying's own ticker and exchange)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k).strip(): str(v).strip() for k, v in (data.get("option_roots") or {}).items()}


def load_universe(path: Path) -> list[Instrument]:
    """Instruments of ``universe.yaml`` (``tickers:`` list), duplicates removed in order."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    seen, out = set(), []
    for ticker in data.get("tickers") or []:
        inst = parse(ticker)
        if inst.bbg not in seen:
            seen.add(inst.bbg)
            out.append(inst)
    return out
