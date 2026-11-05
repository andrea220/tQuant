"""Generate the synthetic market snapshot in ``marketdata/<yyyymmdd>/<HHMM>/`` (date and time of generation).

50 equities (TICKER1-30 EUR, TICKER31-40 GBP, TICKER41-50 USD), each with spot, repo,
vol surface and discrete dividends; one OIS zero curve per currency; FX spots; equity
correlations. Fixed seed: same values on each run; all dates are relative to the evaluation date.

Vol surfaces are SSVI (Gatheral-Jacquier) in forward log-moneyness ``k = ln(K / F(T))``:
``w(k, T) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))``, ``theta(T) = atm(T)^2 T``
increasing, ``phi(theta) = eta / sqrt(theta (1 + theta))`` with ``eta (1 + |rho|) <= 2``: free of
static (butterfly and calendar) arbitrage by construction, so a local vol model can reprice them.

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
    """Spots, repos, SSVI parameters and dividends (the vols need the curves: see ``vol_surfaces``)."""
    spot_rows, div_rows, ssvi = [], [], {}
    for ticker, ccy in zip(TICKERS, CURRENCIES):
        spot = round(float(rng.uniform(10, 500)), 2)
        spot_rows.append({"ticker": ticker, "ccy": ccy, "spot": spot,
                          "repo": round(float(rng.uniform(0.0, 0.01)), 5), "div_yield": None})

        # SSVI: ATM vol term structure atm + ts (sqrt(T) - 1); skew rho; eta = u * 2 / (1 + |rho|), u < 1
        atm, ts = rng.uniform(0.15, 0.40), rng.uniform(-0.04, 0.02)
        rho, u = rng.uniform(-0.80, -0.30), rng.uniform(0.50, 0.95)
        ssvi[ticker] = dict(atm=atm, ts=ts, rho=rho, eta=u * 2.0 / (1.0 + abs(rho)))

        # Discrete dividends: yearly yield split in equal payments, growing 3% a year
        freq = DIV_FREQUENCY[ccy]
        amount = spot * rng.uniform(0.0, 0.06) / freq
        ex_date = EVALUATION_DATE + timedelta(days=int(rng.integers(5, 365 // freq)))
        while ex_date <= add_months(EVALUATION_DATE, 12 * DIV_HORIZON_Y):
            div_rows.append({"ticker": ticker, "ccy": ccy, "ex_date": ex_date,
                             "pay_date": ex_date + timedelta(days=14), "amount": round(amount, 4)})
            ex_date = add_months(ex_date, 12 // freq)
            amount *= 1.03 ** (1 / freq)
    return pd.DataFrame(spot_rows), ssvi, pd.DataFrame(div_rows)


def _yf(d: date) -> float:
    return (d - EVALUATION_DATE).days / 365.0


def _discount(curve: pd.DataFrame, d: date) -> float:
    """Discount factor as RateCurve: zero rates linear in the Actual365 pillar times, flat outside."""
    t = _yf(d)
    times = [_yf(x) for x in curve["pillar"]]
    return float(np.exp(-np.interp(t, times, curve["value"]) * t))


def forward(spot: float, repo: float, divs: pd.DataFrame, curve: pd.DataFrame, d: date) -> float:
    """Equity forward as ``Equity.forward`` (discrete dividends): ``(S - PV(div)) e^{-repo T} / P(0, T)``."""
    pv = sum(a * _discount(curve, x) for x, a in zip(divs["ex_date"], divs["amount"]) if EVALUATION_DATE < x <= d)
    return (spot - pv) * np.exp(-repo * _yf(d)) / _discount(curve, d)


def vol_surfaces(spots: pd.DataFrame, ssvi: dict, divs: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """SSVI implied vols on the strike grid (``VOL_MONEYNESS`` x spot) for ``VOL_TENORS_M``."""
    rows = []
    for row in spots.itertuples(index=False):
        p = ssvi[row.ticker]
        curve = curves[curves.ccy == row.ccy]
        tdivs = divs[divs.ticker == row.ticker]
        for tm in VOL_TENORS_M:
            expiry = add_months(EVALUATION_DATE, tm)
            t = _yf(expiry)
            F = forward(row.spot, row.repo, tdivs, curve, expiry)
            theta = (p["atm"] + p["ts"] * (np.sqrt(t) - 1.0)) ** 2 * t
            phi = p["eta"] / np.sqrt(theta * (1.0 + theta))
            for mny in VOL_MONEYNESS:
                strike = round(row.spot * mny, 2)
                k = np.log(strike / F)
                w = 0.5 * theta * (1 + p["rho"] * phi * k + np.sqrt((phi * k + p["rho"]) ** 2 + 1 - p["rho"] ** 2))
                rows.append({"ticker": row.ticker, "ccy": row.ccy, "expiry": expiry,
                             "strike": strike, "vol": round(float(np.sqrt(w / t)), 6)})
    vol = pd.DataFrame(rows)
    assert vol["vol"].between(0.05, 1.5).all(), "SSVI vol out of range"
    return vol


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
    spot, ssvi, div = equities(rng)
    curves = ir_curves(rng)
    vol = vol_surfaces(spot, ssvi, div, curves)
    files = {
        "ir_curves.csv": curves,
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
