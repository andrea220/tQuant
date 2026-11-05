"""Write a synthetic market snapshot to ``marketdata/<yyyymmdd>/<HHMM>/`` (date and time of generation).

The data come from ``tensorquant.markethandles.synthetic.generate_market_data`` (seed 42): 50 equities
(TICKER1-30 EUR, TICKER31-40 GBP, TICKER41-50 USD) with SSVI vol surfaces and discrete dividends,
Nelson-Siegel OIS curves (ESTR, SONIA, SOFR), EURUSD/GBPUSD spots and SSVI vol surfaces, equity
correlations. Same values on each run; all dates are relative to the evaluation date.

Usage: python marketdata/generate_synthetic.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensorquant.markethandles.synthetic import generate_market_data

SEED = 42


def main() -> None:
    now = datetime.now()
    out_dir = Path(__file__).resolve().parent / now.strftime("%Y%m%d") / now.strftime("%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in generate_market_data(now.date(), SEED).items():
        frame.to_csv(out_dir / name, index=False)
        print(f"{name:22s} {len(frame):5d} rows")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
