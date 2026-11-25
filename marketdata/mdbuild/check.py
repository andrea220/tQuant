"""``python -m mdbuild.check "<Bloomberg ticker>"`` (from the ``marketdata`` folder): whether the ticker exists on
Bloomberg, with its name, currency and security type, as one JSON line (the Bloomberg host and port of
``config/settings.yaml``). Used by tqstudio before adding a ticker to the universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

FIELDS = ["NAME", "CRNCY", "SECURITY_TYP"]


def check(source, ticker: str) -> dict:
    """``{"ticker", "found", "name", "ccy", "type", "error"}`` of *ticker* from a source (:class:`BloombergSource`)."""
    try:
        ref = source.reference([ticker], FIELDS)
    except ValueError as exc:                          # unknown security
        return {"ticker": ticker, "found": False, "error": str(exc)}
    row = ref.loc[ticker] if ticker in ref.index else {}
    value = lambda f: None if f not in row or row[f] is None or row[f] != row[f] else str(row[f])
    found = value("NAME") is not None
    return {"ticker": ticker, "found": found, "name": value("NAME"), "ccy": value("CRNCY"), "type": value("SECURITY_TYP"),
            "error": None if found else f"Bloomberg: no data for '{ticker}'"}


def main() -> None:
    parser = argparse.ArgumentParser(prog="mdbuild.check", description=__doc__.splitlines()[0])
    parser.add_argument("ticker")
    parser.add_argument("--settings", default=str(Path(__file__).resolve().parents[1] / "config" / "settings.yaml"))
    args = parser.parse_args()
    bbg = yaml.safe_load(Path(args.settings).read_text(encoding="utf-8"))["bloomberg"]
    from .sources import BloombergSource
    try:
        with BloombergSource(bbg["host"], bbg["port"], min(int(bbg.get("timeout_ms", 30000)), 15000)) as source:
            out = check(source, " ".join(args.ticker.split()))
    except ConnectionError as exc:
        out = {"ticker": args.ticker, "found": None, "error": str(exc)}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
