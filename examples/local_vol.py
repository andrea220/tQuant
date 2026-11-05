"""
Local Volatility Model (Dupire) — calibration check by Monte Carlo, static synthetic market.

For each maturity of the vol surface:
  1. Build ``LocalVolatilityModel.from_equity`` (market vol nodes, re-gridded before Dupire).
  2. Simulate to that maturity (stateless seed).
  3. Check the MC forward against ``Equity.forward`` and the MC implied vols (OTM options:
     puts below the forward, calls above) against the market vols at every strike node
     (errors in vol points, with their standard error).

``analyse_local_vol`` is reused by ``local_vol_mdm.py`` on the marketdata/ snapshot.
"""

import io
import os
import sys
from datetime import date, timedelta
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

import tensorquant as tq
from tensorquant.models.localvolatility import LocalVolatilityModel

N_PATHS = 50_000
STEPS_PER_YEAR = 50
SEED = 42


def black_call(F: float, K: float, sigma: float, T: float, DF: float) -> float:
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    return DF * (F * norm.cdf(d1) - K * norm.cdf(d1 - sigma * np.sqrt(T)))


def black_otm(F: float, K: float, sigma: float, T: float, DF: float) -> float:
    """Out-of-the-money option: put below the forward, call above (put via parity)."""
    call = black_call(F, K, sigma, T, DF)
    return call - DF * (F - K) if K < F else call


def implied_vol(price: float, F: float, K: float, T: float, DF: float) -> float:
    """Implied vol of an OTM price (ITM calls have almost no vega: tiny price errors explode)."""
    try:
        return brentq(lambda s: black_otm(F, K, s, T, DF) - price, 1e-4, 3.0)
    except ValueError:
        return float("nan")


def analyse_local_vol(market_env: tq.MarketEnvironment, ticker: str, out_dir: Path) -> None:
    """Monte Carlo calibration check of the local vol model on *ticker*; plots in *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_date = tq.Settings.evaluation_date
    eq = market_env.get_equity(ticker)
    vs = eq.vol_surface
    S0 = float(eq.spot)
    K = np.array(vs.strike)
    print(f"\n{ticker} ({eq.currency.name})  spot={S0:.4f}  dividends={eq.dividend_model}  "
          f"repo={float(eq.repo):.4%}")
    print(f"strike nodes {K[0]:.4f}..{K[-1]:.4f} ({K[0] / S0:.0%}..{K[-1] / S0:.0%} of spot), "
          f"maturities {[round(t, 2) for t in vs.maturity]}")

    rows, paths_last, t_last, model_last = [], None, None, None
    for t_node in vs.maturity:
        expiry = eval_date + timedelta(days=max(1, int(round(t_node * 365))))
        T = vs.daycounter.year_fraction(eval_date, expiry)
        model = LocalVolatilityModel.from_equity(eq, expiry)
        n_steps = max(5, int(round(STEPS_PER_YEAR * T)))
        t_grid = tf.linspace(tf.constant(0.0, tf.float64), tf.constant(T, tf.float64), n_steps + 1)[1:]
        dw = tf.random.stateless_normal([N_PATHS, n_steps], seed=[SEED, 0], dtype=tf.float64)
        paths = model.evolve(t_grid, dw).numpy()
        s_t = paths[:, -1]
        DF, F = float(eq.discount_curve.discount(expiry)), float(eq.forward(expiry))
        iv_mkt = np.array([float(vs.volatility(k, T)) for k in K])
        iv_mc, iv_se = [], []
        for k, iv in zip(K, iv_mkt):
            payoff = DF * (np.maximum(k - s_t, 0.0) if k < F else np.maximum(s_t - k, 0.0))  # OTM
            vega = DF * F * norm.pdf((np.log(F / k) + 0.5 * iv * iv * T) / (iv * np.sqrt(T))) * np.sqrt(T)
            iv_mc.append(implied_vol(payoff.mean(), F, k, T, DF))
            iv_se.append(payoff.std() / np.sqrt(N_PATHS) / vega)
        rows.append(dict(T=T, F=F, F_mc=s_t.mean(), F_se=s_t.std() / np.sqrt(N_PATHS),
                         iv_mkt=iv_mkt, iv_mc=np.array(iv_mc), iv_se=np.array(iv_se),
                         outside=np.mean((s_t < K[0]) | (s_t > K[-1]))))
        paths_last, t_last, model_last = paths, t_grid.numpy(), model

    # ---- tables ----
    print("\nForward: MC vs Equity.forward")
    for r in rows:
        print(f"  T={r['T']:5.2f}  MC {r['F_mc']:10.4f} +/- {r['F_se']:.4f}   F {r['F']:10.4f}   "
              f"diff {(r['F_mc'] - r['F']) / r['F_se']:+5.1f} SE   paths outside the strike nodes {r['outside']:.0%}")
    hdr = " ".join(f"{k / S0:7.0%}" for k in K)
    print(f"\nImplied vol error MC - market (vol points), K/S0:\n  {'T':>5s}  {hdr}   max|err|/SE")
    for r in rows:
        err = (r["iv_mc"] - r["iv_mkt"]) * 100
        print(f"  {r['T']:5.2f}  " + " ".join(f"{e:+7.2f}" for e in err)
              + f"   {np.nanmax(np.abs(err) / (r['iv_se'] * 100)):5.1f}")
    tt, kk = np.meshgrid(vs.maturity, K, indexing="ij")
    loc = model_last.sigma_loc(tf.constant(tt.ravel(), tf.float64), tf.constant(kk.ravel(), tf.float64))
    loc = loc.numpy().reshape(tt.shape)
    print(f"\nLocal vol at the nodes (model of the last maturity), K/S0:\n  {'T':>5s}  {hdr}")
    for t, row in zip(vs.maturity, loc):
        print(f"  {t:5.2f}  " + " ".join(f"{v:7.4f}" for v in row))

    # ---- plot 1: smiles (market IV, MC IV +/- 2 SE, local vol) ----
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    fig.suptitle(f"Local vol — {ticker}: market vs MC implied vols (OTM options)", fontweight="bold")
    for ax, r, lv in zip(axes[0], rows, loc):
        ax.plot(K / S0, r["iv_mkt"], "o-", color="steelblue", label="market IV")
        ax.errorbar(K / S0, r["iv_mc"], yerr=2 * r["iv_se"], fmt="s", color="tomato", ms=4, label="MC IV ±2SE")
        ax.plot(K / S0, lv, "--", color="grey", label="local vol")
        lo, hi = r["iv_mkt"].min(), r["iv_mkt"].max()
        ax.set_ylim(lo - 0.5 * (hi - lo) - 0.02, hi + 0.5 * (hi - lo) + 0.02)   # far-OTM short-dated SE bars are huge
        ax.set_title(f"T={r['T']:.2f}Y", fontsize=10)
        ax.set_xlabel("K / S0")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "local_vol_smiles.png", dpi=120)
    plt.close()

    # ---- plot 2: paths of the last maturity, MC forward vs Equity.forward ----
    fwd_eq = [float(eq.forward(eval_date + timedelta(days=max(1, int(round(t * 365)))))) for t in t_last]
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    for path in paths_last[np.random.default_rng(0).choice(N_PATHS, 200, replace=False)]:
        ax2.plot(t_last, path, color="steelblue", alpha=0.05, lw=0.6)
    ax2.plot(t_last, paths_last.mean(axis=0), color="tomato", lw=2, label="MC E[S_t]")
    ax2.plot(t_last, fwd_eq, "k--", lw=1.5, label="Equity.forward(t)")
    lo, hi = np.percentile(paths_last[:, -1], [10, 90])
    ax2.set_ylim(0.8 * lo, 1.1 * hi)
    ax2.set_title(f"Local vol paths — {ticker} (model of T={t_last[-1]:.2f}Y: flat r, q)", fontsize=11)
    ax2.set_xlabel("t (years)")
    ax2.legend()
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "local_vol_paths.png", dpi=120)
    plt.close()

    # ---- plot 3: implied vol errors with the MC band ----
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    for r, color in zip(rows, plt.cm.viridis(np.linspace(0, 0.9, n))):
        err = (r["iv_mc"] - r["iv_mkt"]) * 100
        ax3.plot(K / S0, err, "o-", color=color, ms=3, label=f"T={r['T']:.2f}")
        ax3.fill_between(K / S0, -2 * r["iv_se"] * 100, 2 * r["iv_se"] * 100, color=color, alpha=0.06)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_ylim(-5, 5)   # vol points; far-OTM short-dated options have very wide bands
    ax3.set_title(f"MC implied vol - market (vol points), band = ±2 SE — {ticker}", fontsize=11)
    ax3.set_xlabel("K / S0")
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "local_vol_mc_calls.png", dpi=120)
    plt.close()
    print(f"\nPlots saved to {out_dir}: local_vol_smiles.png, local_vol_paths.png, local_vol_mc_calls.png")


def build_static_market_environment(evaluation_date: date) -> tq.MarketEnvironment:
    """Wide strike grid (30%-170%) with a smooth quadratic smile, continuous carry (repo -3.5%)."""
    daycounter = tq.DayCounter(tq.DayCounterConvention.Actual365)
    spot = 5.174
    moneyness_grid = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70]
    strike_grid = [round(spot * m, 4) for m in moneyness_grid]
    maturity_grid = [0.25, 0.75, 1.50, 3.00]
    # Smooth quadratic smile: (m-1)^2 is C-inf, no kink at ATM (Dupire needs a differentiable surface).
    volatility_matrix = [[round(base_vol + 0.15 * (m - 1.0) ** 2, 5) for m in moneyness_grid]
                         for base_vol in [0.34, 0.31, 0.29, 0.27]]
    estr_curve = tq.RateCurve(
        reference_date=evaluation_date,
        pillars=[0.25, 0.5, 1.0, 2.0, 5.0],
        rates=[0.0200, 0.0210, 0.0220, 0.0230, 0.0240],
        interp="LINEAR",
        daycounter_convention=tq.DayCounterConvention.Actual365,
    )
    vol_surface = tq.VolatilitySurface(
        reference_date=evaluation_date,
        calendar=None,
        daycounter=daycounter,
        strike=strike_grid,
        maturity=maturity_grid,
        volatility_matrix=volatility_matrix,
    )
    return tq.MarketEnvironment(market={
        "IR:EUR:ESTR:SPOT": estr_curve,
        "EQ:EUR:TESTCTP:SPOT": spot,
        "EQ:EUR:TESTCTP:VOL": vol_surface,
        "EQ:EUR:TESTCTP:REPO": -0.035,
        "EQ:EUR:TESTCTP:DIVYIELD": 0.0,
    })


def main() -> None:
    evaluation_date = date(2026, 3, 31)
    tq.Settings.evaluation_date = evaluation_date
    analyse_local_vol(build_static_market_environment(evaluation_date), "TESTCTP", Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
