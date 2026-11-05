"""Generate the synthetic market snapshot in ``marketdata/<yyyymmdd>/<HHMM>/`` (date and time of generation).

50 equities (TICKER1-30 EUR, TICKER31-40 GBP, TICKER41-50 USD), each with spot, repo,
vol surface and discrete dividends; one OIS zero curve per currency; FX spots; equity
correlations. Fixed seed: same values on each run; all dates are relative to the evaluation date.

Usage: python marketdata/generate_synthetic.py
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
GENERATED_AT = datetime.now()
EVALUATION_DATE = GENERATED_AT.date()             # snapshot date = generation date
CALIBRATION_SET = GENERATED_AT.strftime("%H%M")   # snapshot time = generation time (HHMM)
OUT_DIR = Path(__file__).resolve().parent / EVALUATION_DATE.strftime("%Y%m%d") / CALIBRATION_SET

CURRENCIES = ["EUR"] * 30 + ["GBP"] * 10 + ["USD"] * 10
TICKERS = [f"TICKER{i}" for i in range(1, len(CURRENCIES) + 1)]

# One OIS zero curve per currency: (name, short rate, long rate)
CURVES = {"EUR": ("ESTR", 0.020, 0.028), "GBP": ("SONIA", 0.040, 0.042), "USD": ("SOFR", 0.042, 0.039)}
CURVE_TENORS_M = [1, 3, 6, 12, 24, 36, 60, 84, 120, 180, 240, 360]

FX = [("EUR", "USD", 1.0850), ("GBP", "USD", 1.2700)]   # 1 base = spot quote

VOL_TENORS_M = [1, 3, 6, 12, 18, 24, 36, 60]
VOL_MONEYNESS = np.round(np.arange(0.6, 1.401, 0.1), 2)
DIV_FREQUENCY = {"EUR": 1, "GBP": 2, "USD": 4}            # payments per year
DIV_HORIZON_Y = 5
CORR_RANGE = (0.30, 0.80)


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    return date(d.year + y, m + 1, min(d.day, 28))


def ir_curves(rng) -> pd.DataFrame:
    rows = []
    for ccy, (name, short, long) in CURVES.items():
        for m in CURVE_TENORS_M:
            t = m / 12
            rate = long + (short - long) * np.exp(-t / 3) + rng.normal(0, 0.0005)
            rows.append({"ccy": ccy, "curve": name, "pillar": add_months(EVALUATION_DATE, m),
                         "value": round(rate, 6), "quote": "zero",
                         "daycount": "Actual365", "interp": "LINEAR"})
    return pd.DataFrame(rows)


def equities(rng):
    spot_rows, vol_rows, div_rows = [], [], []
    for ticker, ccy in zip(TICKERS, CURRENCIES):
        spot = round(float(rng.uniform(10, 500)), 2)
        spot_rows.append({"ticker": ticker, "ccy": ccy, "spot": spot,
                          "repo": round(float(rng.uniform(0.0, 0.01)), 5), "div_yield": None})

        # IV(T, m) = atm + ts (sqrt(T) - 1) + skew m / sqrt(max(T, 0.25)) + curv m^2,  m = ln(K/S)
        atm, ts = rng.uniform(0.15, 0.40), rng.uniform(-0.04, 0.02)
        skew, curv = rng.uniform(-0.20, -0.05), rng.uniform(0.10, 0.40)
        for tm in VOL_TENORS_M:
            t = tm / 12
            for mny in VOL_MONEYNESS:
                k = round(spot * mny, 2)
                m = np.log(k / spot)
                iv = atm + ts * (np.sqrt(t) - 1) + skew * m / np.sqrt(max(t, 0.25)) + curv * m ** 2
                vol_rows.append({"ticker": ticker, "ccy": ccy, "expiry": add_months(EVALUATION_DATE, tm),
                                 "strike": k, "vol": round(float(np.clip(iv, 0.05, 1.0)), 6)})

        # Discrete dividends: yearly yield split in equal payments, growing 3% a year
        freq = DIV_FREQUENCY[ccy]
        amount = spot * rng.uniform(0.0, 0.06) / freq
        ex_date = EVALUATION_DATE + timedelta(days=int(rng.integers(5, 365 // freq)))
        while ex_date <= add_months(EVALUATION_DATE, 12 * DIV_HORIZON_Y):
            div_rows.append({"ticker": ticker, "ccy": ccy, "ex_date": ex_date,
                             "pay_date": ex_date + timedelta(days=14), "amount": round(amount, 4)})
            ex_date = add_months(ex_date, 12 // freq)
            amount *= 1.03 ** (1 / freq)
    return pd.DataFrame(spot_rows), pd.DataFrame(vol_rows), pd.DataFrame(div_rows)


def correlations(rng) -> pd.DataFrame:
    """One-factor model rho_ij = b_i b_j: every pair in CORR_RANGE and the matrix is PSD by construction."""
    lo, hi = CORR_RANGE
    b = rng.uniform(np.sqrt(lo), np.sqrt(hi), len(TICKERS))
    corr = np.round(np.outer(b, b), 4)
    np.fill_diagonal(corr, 1.0)
    assert np.linalg.eigvalsh(corr).min() > 0, "correlation matrix not positive definite"
    names = [f"EQ:{ccy}:{ticker}" for ticker, ccy in zip(TICKERS, CURRENCIES)]
    rows = [{"factor_1": names[i], "factor_2": names[j], "corr": corr[i, j]}
            for i in range(len(names)) for j in range(i + 1, len(names))]
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spot, vol, div = equities(rng)
    files = {
        "ir_curves.csv": ir_curves(rng),
        "fx_spot.csv": pd.DataFrame(FX, columns=["base", "quote", "spot"]),
        "eq_spot.csv": spot,
        "eq_vol.csv": vol,
        "eq_dividends.csv": div,
        "correlations.csv": correlations(rng),
    }
    for name, df in files.items():
        df.to_csv(OUT_DIR / name, index=False)
        print(f"{name:22s} {len(df):5d} rows")


if __name__ == "__main__":
    main()
