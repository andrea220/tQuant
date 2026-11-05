"""
Local Volatility Model (Dupire) on a marketdata/ snapshot — calibration check by Monte Carlo.

Same analysis as ``local_vol.py`` (``analyse_local_vol``): for each vol-surface maturity the
model is built with ``LocalVolatilityModel.from_equity`` (Gatheral local vol in forward
moneyness, market forward), the MC forward is checked against ``Equity.forward`` and the MC implied vols (OTM
options) against the market vols at every strike node. Plots in examples/out/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorquant as tq
from local_vol import analyse_local_vol

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
MARKET_DATA      = Path(__file__).resolve().parents[1] / "marketdata"
SNAPSHOT_DATE    = None        # date(yyyy, m, d) | None = latest snapshot
CALIBRATION_SET  = None        # e.g. "1516"      | None = latest of the day
TICKER           = "TICKER1"   # TICKER1-30 EUR | TICKER31-40 GBP | TICKER41-50 USD
OUT_DIR          = Path(__file__).resolve().parent / "out"
# ----------------------------------------------------------------


def main() -> None:
    market_env = tq.MarketEnvironment.from_folder(MARKET_DATA, SNAPSHOT_DATE, CALIBRATION_SET)
    tq.Settings.evaluation_date = market_env.evaluation_date
    print(f"snapshot marketdata/{market_env.evaluation_date:%Y%m%d}/{market_env.calibration_set}")
    analyse_local_vol(market_env, TICKER, OUT_DIR)


if __name__ == "__main__":
    main()
