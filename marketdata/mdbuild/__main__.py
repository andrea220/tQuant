"""``python -m mdbuild`` (from the ``marketdata`` folder): update the price history from Bloomberg and write a
snapshot under ``live/<yyyymmdd>/<hhmm>/`` with the active instruments of the universe.

``python -m mdbuild --select "ENI IM Equity,SX5E Index"``: those equities only (active or not), with every active
rate fixing, FX pair and curve (selected curves and fixings too, active or not): a complete snapshot of its own."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import yaml

from .snapshot import run
from .sources import BloombergSource
from .universe import load_curves, load_implied_spot, load_option_roots, load_universe, read_entries

BASE = Path(__file__).resolve().parents[1]          # marketdata/


def bootstrap_curves(folder: Path) -> None:
    """Bootstrap the curves of the snapshot with tQuant, if installed (optional: marketdata does not depend on it):
    writes ir_curves.csv, ir_curves_jacobian.csv, ir_calibration.csv next to ir_quotes.csv."""
    try:
        from tensorquant.markethandles.bootstrapping import bootstrap_snapshot
    except ImportError:
        print("tensorquant not installed: curves left as quotes (ir_quotes.csv), bootstrapped by tQuant on load")
        return
    t0 = time.perf_counter()
    written = bootstrap_snapshot(folder)
    print(f"curves bootstrapped in {time.perf_counter() - t0:.0f}s: {', '.join(written)}")


def build_vols(folder: Path) -> None:
    """Equity vol surfaces and repo curves of the snapshot with tQuant, if installed: eq_vol.csv, eq_repo.csv,
    eq_vol_calibration.csv from eq_option_quotes.csv (after the curves)."""
    try:
        from tensorquant.calibration.equityvol import calibrate_snapshot_vols
    except ImportError:
        print("tensorquant not installed: equity vols left as option quotes (eq_option_quotes.csv)")
        return
    t0 = time.perf_counter()
    written = calibrate_snapshot_vols(folder)
    print(f"equity vols built in {time.perf_counter() - t0:.0f}s: {', '.join(written)}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mdbuild", description=__doc__)
    parser.add_argument("--root", default=str(BASE),
                        help="market data folder: config, snapshots, history, caches (default: this marketdata folder)")
    parser.add_argument("--universe", help="default <root>/config/universe.yaml")
    parser.add_argument("--settings", help="default <root>/config/settings.yaml")
    parser.add_argument("--select", help="comma-separated Bloomberg tickers (and curve names, 'EUR ESTR'): those "
                                         "equities, active or not, with every active rate, FX pair and curve "
                                         "(default: every active instrument)")
    args = parser.parse_args()
    root = Path(args.root)
    args.universe = args.universe or str(root / "config" / "universe.yaml")
    args.settings = args.settings or str(root / "config" / "settings.yaml")
    print(f"market data folder {root}", flush=True)
    settings = yaml.safe_load(Path(args.settings).read_text(encoding="utf-8"))
    path = Path(args.universe)
    if settings.get("options"):
        settings["options"]["roots"] = load_option_roots(path)
        settings["options"]["implied_spot"] = load_implied_spot(path)
    now = datetime.now()
    names = None
    if args.select:                                    # the selection, with the whole market base
        names = [" ".join(n.split()) for n in args.select.split(",") if n.strip()]
        labels = {c["name"].upper() for c in read_entries(path)[1]}
        universe = load_universe(path, select=[n for n in names if n.upper() not in labels])
        curves = load_curves(path, select=[n for n in names if n.upper() in labels])
        found = {i.bbg.upper() for i in universe} | {c.label.upper() for c in curves}
        missing = [n for n in names if n.upper() not in found]
        if missing:
            raise SystemExit(f"not in the universe: {', '.join(missing)}")
        universe += [i for i in load_universe(path) if i.kind != "equity" and i not in universe]
        curves += [c for c in load_curves(path) if c not in curves]
        print(f"selected: {', '.join(names)} (with the active rates, FX, fixings and curves)", flush=True)
    else:
        universe, curves = load_universe(path), load_curves(path)
    bbg = settings["bloomberg"]
    hist = settings["history"]
    with BloombergSource(bbg["host"], bbg["port"], bbg["timeout_ms"], hist.get("adjust_splits", True),
                         hist.get("adjust_dividends", False)) as source:
        folder = run(source, universe, settings, root, now, curves, selected=names)
    if (folder / "ir_quotes.csv").is_file():
        bootstrap_curves(folder)
    if (folder / "eq_option_quotes.csv").is_file():
        build_vols(folder)
    print(f"snapshot -> {folder}")
    for f in sorted(folder.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
