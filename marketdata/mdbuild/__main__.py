"""``python -m mdbuild`` (from the ``marketdata`` folder): update the price history from Bloomberg and write a
snapshot under ``live/<yyyymmdd>/<hhmm>/``."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import yaml

from .snapshot import run
from .sources import BloombergSource
from .universe import load_curves, load_implied_spot, load_option_roots, load_universe

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
    parser.add_argument("--universe", default=str(BASE / "config" / "universe.yaml"))
    parser.add_argument("--settings", default=str(BASE / "config" / "settings.yaml"))
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.settings).read_text(encoding="utf-8"))
    universe = load_universe(Path(args.universe))
    if settings.get("options"):
        settings["options"]["roots"] = load_option_roots(Path(args.universe))
        settings["options"]["implied_spot"] = load_implied_spot(Path(args.universe))
    curves = load_curves(Path(args.universe))
    bbg = settings["bloomberg"]
    now = datetime.now()
    hist = settings["history"]
    with BloombergSource(bbg["host"], bbg["port"], bbg["timeout_ms"], hist.get("adjust_splits", True),
                         hist.get("adjust_dividends", False)) as source:
        folder = run(source, universe, settings, BASE, now, curves)
    if (folder / "ir_quotes.csv").is_file():
        bootstrap_curves(folder)
    if (folder / "eq_option_quotes.csv").is_file():
        build_vols(folder)
    print(f"snapshot -> {folder}")
    for f in sorted(folder.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
