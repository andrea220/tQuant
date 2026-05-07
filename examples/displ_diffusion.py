"""
Displaced Diffusion Model — calibration and Monte Carlo example.

Steps:
  1. Build a static MarketEnvironment with a quadratic smile surface.
  2. Calibrate DisplacedDiffusionModel (spot + forward betas).
  3. Print calibration summary table.
  4. Plot: market IV vs DD IV (one panel per maturity).
  5. MC sanity checks: forward match, put-call parity, ATM vol proxy.
  6. Paths plot: simulated paths + analytical vs simulated forward.
"""

from datetime import date
from pathlib import Path
import sys
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tensorquant as tq
from tensorquant.models.displaceddiffusion import (
    DisplacedDiffusionModel,
    _bs_call_np,
)


# ----------------------------------------------------------------
# Static market environment  (skewed smile: quadratic + slope)
# ----------------------------------------------------------------

def build_market_env(evaluation_date: date) -> tq.MarketEnvironment:
    daycounter    = tq.DayCounter(tq.DayCounterConvention.Actual365)
    spot          = 5.174
    moneyness     = [0.40, 0.55, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50, 1.65]
    strike_grid   = [round(spot * m, 4) for m in moneyness]
    maturity_grid = [0.25, 0.75, 1.50, 3.00]

    # Skewed smile: put-side higher than call-side (realistic equity skew)
    volatility_matrix = []
    for base_vol, skew in [(0.33, -0.06), (0.30, -0.05), (0.28, -0.04), (0.26, -0.03)]:
        row = [round(base_vol + 0.08 * (m - 1.0) ** 2 + skew * (m - 1.0), 5)
               for m in moneyness]
        volatility_matrix.append(row)

    estr = tq.RateCurve(
        reference_date=evaluation_date,
        pillars=[0.25, 0.5, 1.0, 2.0, 5.0],
        rates=[0.0200, 0.0210, 0.0220, 0.0230, 0.0240],
        interp="LINEAR",
        daycounter_convention=tq.DayCounterConvention.Actual365,
    )
    vol_surf = tq.VolatilitySurface(
        reference_date=evaluation_date,
        calendar=None,
        daycounter=daycounter,
        strike=strike_grid,
        maturity=maturity_grid,
        volatility_matrix=volatility_matrix,
    )
    market = {
        "IR:EUR:ESTR:SPOT": estr,
        "EQ:EUR:TESTCTP:SPOT": spot,
        "EQ:EUR:TESTCTP:VOL": vol_surf,
        "EQ:EUR:TESTCTP:REPO": -0.035,
        "EQ:EUR:TESTCTP:DIVYIELD": 0.0,
    }
    return tq.MarketEnvironment(market=market)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    evaluation_date = date(2026, 3, 31)
    tq.Settings.evaluation_date = evaluation_date
    market_env = build_market_env(evaluation_date)

    ticker = "TESTCTP"
    ccy    = tq.Currency.EUR

    spot      = market_env.get_eq_spot(ticker, ccy=ccy)
    vol_surf  = market_env.get_eq_vol_surface(ticker, ccy=ccy)
    disc_curve = market_env.get_ir_curve(ccy)
    q_val     = float(market_env.get_eq_div_yield(ticker, ccy=ccy) or 0.0)

    T_grid = np.array(vol_surf.maturity)
    K_grid = np.array(vol_surf.strike)
    iv_mat = np.array(vol_surf.volatility_matrix)
    nT, nK = iv_mat.shape
    S0     = float(spot)

    print("\n" + "=" * 60)
    print("Displaced Diffusion Calibration — TESTCTP")
    print("=" * 60)
    print(f"S0 = {S0:.4f}   q = {q_val:.4f}")

    # ---- Calibrate ----
    model = DisplacedDiffusionModel.from_implied_vol(
        spot=spot,
        vol_surface=vol_surf,
        disc_curve=disc_curve,
        q=q_val,
        forward_skew=True,
    )
    model.print_calibration_summary()

    # ---- DD implied vol surface ----
    iv_dd = model.implied_vol_surface()    # [nT, nK]

    # ---- Market call prices for comparison ----
    r_vec = np.array([float(disc_curve.zero_rate(T)) for T in T_grid])
    mkt_calls = np.array([
        [_bs_call_np(S0, K_grid[k], T_grid[i], iv_mat[i, k], r_vec[i], q_val)
         for k in range(nK)]
        for i in range(nT)
    ])
    dd_calls = np.array([
        [float(model.dd_call(float(K_grid[k]), T_idx=i).numpy())
         for k in range(nK)]
        for i in range(nT)
    ])
    price_err = dd_calls - mkt_calls

    # ---- Print smile comparison ----
    hdr = "  ".join(f"K={K_grid[k]:.2f}" for k in range(nK))
    for i, T in enumerate(T_grid):
        print(f"  T={T:.2f}  mkt IV  : " +
              "  ".join(f"{iv_mat[i, k]:.4f}" for k in range(nK)))
        dd_row = [iv_dd[i, k] if not np.isnan(iv_dd[i, k]) else float("nan")
                  for k in range(nK)]
        print(f"  T={T:.2f}  DD  IV  : " +
              "  ".join(f"{v:.4f}" if not np.isnan(v) else "  nan " for v in dd_row))
        print(f"  T={T:.2f}  price err: " +
              "  ".join(f"{price_err[i, k]:+.5f}" for k in range(nK)))
        print()

    max_err = np.nanmax(np.abs(price_err))
    print(f"  Max abs price error (all T, K): {max_err:.6f}")

    # ----------------------------------------------------------------
    # Monte Carlo sanity checks
    # ----------------------------------------------------------------
    n_mc     = 50_000
    n_steps  = 120
    T_max    = float(T_grid[-1])
    tf.random.set_seed(42)

    t_fine   = tf.cast(tf.linspace(0.0, T_max, n_steps + 1), tf.float32)[1:]
    dw_mc    = tf.random.normal([n_mc, n_steps], dtype=tf.float32)
    paths_mc = model.evolve(t_fine, dw_mc)                     # [n_mc, n_steps]

    t_fine_np    = t_fine.numpy()
    paths_np     = paths_mc.numpy()
    mat_idx      = [int(np.argmin(np.abs(t_fine_np - T))) for T in T_grid]
    terminal     = [paths_mc[:, idx] for idx in mat_idx]

    r_mc  = float(model._r.numpy())
    q_mc  = float(model._q.numpy())

    print("\n" + "=" * 60)
    print("Monte Carlo Sanity Checks")
    print("=" * 60)
    for i, T in enumerate(T_grid):
        S_T     = terminal[i]
        log_ret = tf.math.log(S_T / tf.constant(S0, tf.float32))

        mc_fwd   = float(tf.reduce_mean(S_T).numpy())
        theo_fwd = S0 * np.exp((r_mc - q_mc) * T)
        fwd_err  = (mc_fwd - theo_fwd) / theo_fwd * 100

        df_T     = np.exp(-r_mc * T)
        mc_calls = df_T * tf.reduce_mean(
            tf.maximum(S_T[:, None] - tf.constant(K_grid, tf.float32)[None, :], 0.0),
            axis=0,
        ).numpy()
        mc_puts  = df_T * tf.reduce_mean(
            tf.maximum(tf.constant(K_grid, tf.float32)[None, :] - S_T[:, None], 0.0),
            axis=0,
        ).numpy()

        # Put-call parity
        pcp_lhs = mc_calls - mc_puts
        pcp_rhs = S0 * np.exp(-q_mc * T) - K_grid * df_T
        pcp_err = np.abs(pcp_lhs - pcp_rhs)

        # ATM vol proxy
        atm_k   = int(np.argmin(np.abs(K_grid - S0)))
        mc_vol  = float(tf.math.reduce_std(log_ret).numpy()) / np.sqrt(T)
        mkt_atm = float(iv_mat[i, atm_k])

        # MC call vs market
        call_err = np.abs(mc_calls - mkt_calls[i])

        print(f"\n  T = {T:.2f}Y")
        print(f"    [1] Forward  — MC: {mc_fwd:.4f}  Theo: {theo_fwd:.4f}  Err: {fwd_err:+.3f}%")
        print(f"    [2] ATM vol  — MC: {mc_vol:.4f}  Mkt ATM IV: {mkt_atm:.4f}  Diff: {mc_vol - mkt_atm:+.4f}")
        print(f"    [3] PCP      — max: {pcp_err.max():.6f}  mean: {pcp_err.mean():.6f}")
        print(f"    [4] Call err — max: {call_err.max():.4f}  mean: {call_err.mean():.4f}")

    # ----------------------------------------------------------------
    # Plot 1: IV smile — market vs DD (one panel per maturity)
    # ----------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, nT, figsize=(4 * nT, 4), sharey=False)
    fig1.suptitle("Displaced Diffusion — IV Smile: Market vs DD", fontsize=13, fontweight="bold")

    for i, T in enumerate(T_grid):
        ax = axes1[i]
        ax.plot(K_grid, iv_mat[i], color="steelblue", marker="o", label="Market IV")
        valid = ~np.isnan(iv_dd[i])
        ax.plot(K_grid[valid], iv_dd[i][valid], color="tomato", marker="s",
                linestyle="--", label="DD IV")
        ax.axvline(S0, color="grey", linestyle=":", linewidth=1)
        ax.set_title(f"T = {T:.2f}Y", fontsize=10)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied vol" if i == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/dd_smiles.png", dpi=150)
    plt.show()
    print("\nSmile plot saved to examples/dd_smiles.png")

    # ----------------------------------------------------------------
    # Plot 2: MC paths + analytical vs simulated forward
    # ----------------------------------------------------------------
    fwd_theo = S0 * np.exp((r_mc - q_mc) * t_fine_np)
    fwd_mc_t = paths_np.mean(axis=0)

    p20 = float(np.percentile(paths_np[:, -1], 20))
    p80 = float(np.percentile(paths_np[:, -1], 80))
    y_lo = max(0.0, p20 - (p80 - p20) * 0.15)
    y_hi = p80 + (p80 - p20) * 0.15

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    fig2.suptitle("MC Paths — Displaced Diffusion (TESTCTP)", fontsize=13, fontweight="bold")

    rng   = np.random.default_rng(0)
    idx   = rng.choice(paths_np.shape[0], min(300, n_mc), replace=False)
    for path in paths_np[idx]:
        ax2.plot(t_fine_np, path, color="steelblue", alpha=0.04, linewidth=0.6)

    ax2.plot(t_fine_np, fwd_mc_t,  color="tomato", linewidth=2.2,
             label="Simulated forward  E[S_t]", zorder=3)
    ax2.plot(t_fine_np, fwd_theo,  color="black",  linewidth=1.8,
             linestyle="--", label=f"Analytical forward  S\u2080\u00b7e^{{(r-q)t}}", zorder=4)

    for T in T_grid:
        ax2.axvline(T, color="grey", linestyle=":", linewidth=0.9, alpha=0.6)
        ax2.text(T + 0.02, y_hi * 0.97, f"T={T:.2f}", fontsize=7, color="grey", va="top")

    ax2.axhline(S0, color="dimgrey", linestyle=":", linewidth=0.9, alpha=0.5,
                label=f"S\u2080 = {S0:.2f}")
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Spot")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("examples/dd_paths.png", dpi=150)
    plt.show()
    print("Paths plot saved to examples/dd_paths.png")


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    main()
