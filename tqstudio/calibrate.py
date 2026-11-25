"""``python -m tqstudio.calibrate``: calibrate the equity models on a market snapshot and write a calibration set.

Examples (from the repository root, in tqenv)::

    python -m tqstudio.calibrate                              # latest snapshot, the models flagged in the universe
    python -m tqstudio.calibrate --models Heston --tickers SX5E,ENI_IM --slv SX5E    # explicit choice
    python -m tqstudio.calibrate --date 2026-11-13 --set 1619 --models Heston

Without ``--models`` / ``--tickers`` / ``--slv`` the models of each underlying are the flags of
the universe (models of each instrument, defaults in ``marketdata/config/calibration.yaml``). The sets go to ``marketdata/calibrations/<source>/<yyyymmdd>/<hhmm>/`` (git-ignored); the web app lists them in the
Calibration tab and starts this command for its "New calibration set" button (a separate process: the calibrations
take minutes and set the global evaluation date).
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_SOURCE = os.environ.get("TQSTUDIO_DATA", "live")
from .paths import MARKETDATA  # noqa: E402  (the shared market data folder)

MARKET_DATA = MARKETDATA / ("live" if DATA_SOURCE == "live" else "")
CALIBRATIONS = MARKETDATA / "calibrations" / DATA_SOURCE


def snapshot_folder(day: str = None, calibration_set: str = None) -> Path:
    """The snapshot folder of a date and set (latest when omitted)."""
    days = sorted(d.name for d in MARKET_DATA.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8)
    if not days:
        raise FileNotFoundError(f"no snapshot under {MARKET_DATA}")
    day_name = date.fromisoformat(day).strftime("%Y%m%d") if day else days[-1]
    sets = sorted(s.name for s in (MARKET_DATA / day_name).iterdir() if s.is_dir() and not s.name.startswith("_"))
    return MARKET_DATA / day_name / (calibration_set or sets[-1])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="tqstudio.calibrate", description=__doc__.splitlines()[0])
    parser.add_argument("--date", help="snapshot date (YYYY-MM-DD), default the latest")
    parser.add_argument("--set", help="snapshot set (HHMM), default the latest of the date")
    parser.add_argument("--models", help="comma-separated models for every underlying (default: the universe flags)")
    parser.add_argument("--tickers", help="comma-separated underlyings, default all with a vol surface")
    parser.add_argument("--slv", default="", help="comma-separated underlyings also calibrated with Heston SLV")
    args = parser.parse_args(argv)

    from tensorquant.calibration.models import calibrate_snapshot_models

    snapshot = snapshot_folder(args.date, args.set)
    print(f"snapshot {snapshot}", flush=True)
    explicit = args.models or args.tickers or args.slv
    models = [m for m in (args.models or "Heston,DisplacedDiffusion").split(",") if m]
    tickers = [t for t in args.tickers.split(",") if t] if args.tickers else None
    slv = [t for t in args.slv.split(",") if t]
    plan = None
    if not explicit:                                   # the flags of the universe
        from datetime import datetime

        from tensorquant.markethandles.marketenvironment import MarketEnvironment

        from .views.calibration import plan as universe_plan, read_config

        day = datetime.strptime(snapshot.parent.name, "%Y%m%d").date()
        plan = universe_plan(read_config(), MarketEnvironment.from_folder(snapshot.parent.parent, day, snapshot.name))
        print("plan " + ", ".join(t + ": " + ("/".join(m) or "-") for t, m in plan.items()), flush=True)


    def progress(done, total, label):
        print(f"progress {done}/{total} {label}", flush=True)

    folder = calibrate_snapshot_models(snapshot, CALIBRATIONS, models=models, tickers=tickers, slv=slv,
                                       progress=progress, plan=plan)
    print(f"calibration set -> {folder}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
