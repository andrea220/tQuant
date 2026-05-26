"""
Local Volatility Model — real market data from MDM datastore.

Steps:
  1. Load MarketEnvironment from a MDM datastore folder.
  2. Extract spot, implied vol surface, discount curve, div yield.
  3. Calibrate LocalVolatilityModel (Dupire).
  4. Print calibration summary + MC sanity checks.
  5. Save plots to out/.
"""

import io
import os
import sys
from datetime import date
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorquant as tq
from tensorquant.models.localvolatility import LocalVolatilityModel

# ----------------------------------------------------------------
# Configuration — edit these to point to your datastore snapshot
# ----------------------------------------------------------------
EVALUATION_DATE  = date(2026, 5, 7)
DATA_PATH        = "W:/Derivative/EquityDerivatives/AndreaC/dev/mdm/datastore/"
CALIBRATION_SET  = "152933"
TICKER           = "SX5E"
CCY              = tq.Currency.EUR
OUT_DIR          = Path(__file__).resolve().parent / "out"
# ----------------------------------------------------------------


def bs_call_price(
    spot: tf.Tensor,
    strike: tf.Tensor,
    maturity: tf.Tensor,
    rate: tf.Tensor,
    dividend_yield: tf.Tensor,
    volatility: tf.Tensor,
) -> tf.Tensor:
    maturity   = tf.maximum(tf.cast(maturity,   tf.float32), tf.constant(1e-10))
    volatility = tf.maximum(tf.cast(volatility, tf.float32), tf.constant(1e-10))
    sqrt_t = tf.sqrt(maturity)
    d1 = (
        tf.math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * tf.square(volatility)) * maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    N  = lambda x: 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(tf.constant(2.0))))
    return (
        spot   * tf.exp(-dividend_yield * maturity) * N(d1)
        - strike * tf.exp(-rate          * maturity) * N(d2)
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load market environment ----
    tq.Settings.evaluation_date = EVALUATION_DATE
    market_env = tq.MarketEnvironment.from_data_path(
        evaluation_date=EVALUATION_DATE,
        data_path=DATA_PATH,
        calibration_set=CALIBRATION_SET,
    )

    # ---- Extract market data ----
    spot       = tf.constant(market_env.get_eq_spot(TICKER, ccy=CCY), dtype=tf.float32)
    vol_surface = market_env.get_eq_vol_surface(TICKER, ccy=CCY)
    disc_curve  = market_env.get_ir_curve(CCY)
    q_raw       = market_env.get_eq_div_yield(TICKER, ccy=CCY)

    T_grid    = tf.constant(vol_surface.maturity,          dtype=tf.float32)
    K_grid    = tf.constant(vol_surface.strike,            dtype=tf.float32)
    iv_matrix = tf.Variable(vol_surface.volatility_matrix, dtype=tf.float32)

    T_max = float(T_grid[-1].numpy())
    r     = tf.cast(disc_curve.zero_rate(T_max), tf.float32)
    q     = tf.constant(float(q_raw or 0.0),                 dtype=tf.float32)
    S0    = float(spot.numpy())

    nT, nK = T_grid.shape[0], K_grid.shape[0]

    print(f"\nTicker : {TICKER}")
    print(f"S0     : {S0:.4f}")
    print(f"r      : {float(r.numpy()):.4f}")
    print(f"q      : {float(q.numpy()):.4f}")
    print(f"Maturities ({nT}): {T_grid.numpy().tolist()}")
    print(f"Strikes    ({nK}): {K_grid.numpy().tolist()}")

    # ---- Calibrate local vol (Dupire) ----
    t0 = time.perf_counter()
    local_vol_model = LocalVolatilityModel.from_implied_vol(
        iv_matrix=iv_matrix,
        T_grid=T_grid,
        K_grid=K_grid,
        S0=spot,
        r=r,
        q=q,
    )
    print(f"\nCalibration elapsed: {time.perf_counter() - t0:.3f}s")

    # ---- Market call prices ----
    market_call_prices = [
        bs_call_price(spot, K_grid, T_grid[i], r, q, iv_matrix[i, :])
        for i in range(nT)
    ]
    market_call_matrix = tf.stack(market_call_prices, axis=0)   # [nT, nK]

    # ---- Local vol surface ----
    TT, KK          = tf.meshgrid(T_grid, K_grid, indexing="ij")
    sigma_loc_flat   = local_vol_model.sigma_loc(tf.reshape(TT, [-1]), tf.reshape(KK, [-1]))
    sigma_loc_matrix = tf.reshape(sigma_loc_flat, tf.shape(TT))  # [nT, nK]

    # ---- Reprice with local vol ----
    model_call_prices = [
        bs_call_price(spot, K_grid, T_grid[i], r, q, sigma_loc_matrix[i, :])
        for i in range(nT)
    ]
    model_call_matrix = tf.stack(model_call_prices, axis=0)
    price_error       = model_call_matrix - market_call_matrix
    abs_price_error   = tf.abs(price_error)

    # ---- Print calibration summary ----
    col_w  = 10
    fmt    = lambda vals: "  ".join(f"{v:>{col_w}.5f}" for v in vals)
    K_hdr  = "  ".join(f"K={k:>8.2f}" for k in K_grid.numpy())

    print("\n=== Local Vol Calibration (Dupire) ===")
    print(f"\nImplied vol surface (market)\n  {K_hdr}")
    for i in range(nT):
        print(f"  T={T_grid[i].numpy():.2f}  {fmt(iv_matrix[i].numpy())}")

    print(f"\nLocal vol surface (Dupire)\n  {K_hdr}")
    for i in range(nT):
        print(f"  T={T_grid[i].numpy():.2f}  {fmt(sigma_loc_matrix[i].numpy())}")

    print(f"\nReprice error BS(sigma_loc) - market call\n  {K_hdr}")
    for i in range(nT):
        print(f"  T={T_grid[i].numpy():.2f}  {fmt(price_error[i].numpy())}")

    print(f"\nMax abs reprice error per maturity:")
    for i in range(nT):
        print(f"  T={T_grid[i].numpy():.2f}  {float(tf.reduce_max(abs_price_error[i]).numpy()):.6f}")

    # ----------------------------------------------------------------
    # Monte Carlo sanity checks
    # ----------------------------------------------------------------
    n_mc    = 50_000
    n_steps = 100
    tf.random.set_seed(42)

    t_fine   = tf.linspace(0.0, T_max, n_steps + 1)[1:]
    dw_fine  = tf.random.normal([n_mc, n_steps], dtype=tf.float32)
    paths_mc = local_vol_model.evolve(t_fine, dw_fine)           # [n_mc, n_steps]

    t_fine_np    = t_fine.numpy()
    paths_np     = paths_mc.numpy()
    maturity_idx = [int(np.argmin(np.abs(t_fine_np - float(T_grid[i].numpy())))) for i in range(nT)]
    terminal     = [paths_mc[:, idx] for idx in maturity_idx]

    print("\n" + "=" * 70)
    print(f"Monte Carlo Sanity Checks — {TICKER}")
    print("=" * 70)

    for i in range(nT):
        Ti      = float(T_grid[i].numpy())
        S_T     = terminal[i]
        log_S_T = tf.math.log(S_T / spot)

        mc_fwd      = float(tf.reduce_mean(S_T).numpy())
        theo_fwd    = S0 * float(tf.exp((r - q) * Ti).numpy())
        fwd_err_pct = (mc_fwd - theo_fwd) / theo_fwd * 100

        df_T    = float(tf.exp(-r * Ti).numpy())
        mc_calls = df_T * tf.reduce_mean(
            tf.maximum(S_T[:, None] - K_grid[None, :], 0.0), axis=0
        ).numpy()
        mc_puts  = df_T * tf.reduce_mean(
            tf.maximum(K_grid[None, :] - S_T[:, None], 0.0), axis=0
        ).numpy()

        pcp_err     = np.abs((mc_calls - mc_puts) - (S0 * float(tf.exp(-q * Ti).numpy()) - K_grid.numpy() * df_T))
        atm_idx     = int(np.argmin(np.abs(K_grid.numpy() - S0)))
        mc_vol      = float(tf.math.reduce_std(log_S_T).numpy()) / np.sqrt(Ti)
        mkt_atm     = float(iv_matrix[i, atm_idx].numpy())
        mc_call_err = np.abs(mc_calls - market_call_matrix[i].numpy())

        print(f"\n  T = {Ti:.2f}Y")
        print(f"    [1] Forward  — MC: {mc_fwd:.4f}  Theo: {theo_fwd:.4f}  Err: {fwd_err_pct:+.3f}%")
        print(f"    [2] ATM vol  — MC: {mc_vol:.4f}  Mkt: {mkt_atm:.4f}  Diff: {mc_vol - mkt_atm:+.4f}")
        print(f"    [3] PCP      — max: {pcp_err.max():.6f}  mean: {pcp_err.mean():.6f}")
        print(f"    [4] Call err — max: {mc_call_err.max():.4f}  mean: {mc_call_err.mean():.4f}")

    # ----------------------------------------------------------------
    # Plot 1: IV vs Local Vol smile (one panel per maturity)
    # ----------------------------------------------------------------
    K_np   = K_grid.numpy()
    T_np   = T_grid.numpy()
    iv_np  = iv_matrix.numpy()
    lv_np  = sigma_loc_matrix.numpy()
    mc_np  = market_call_matrix.numpy()
    ml_np  = model_call_matrix.numpy()
    err_np = price_error.numpy()

    fig, axes = plt.subplots(3, nT, figsize=(5 * nT, 12), sharex="col")
    if nT == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(f"Local Vol Calibration — {TICKER}", fontsize=14, fontweight="bold")

    for i in range(nT):
        T_label = f"T = {T_np[i]:.2f}Y"

        ax = axes[0, i]
        ax.plot(K_np, iv_np[i], color="steelblue", marker="o", label="Implied vol (market)")
        ax.plot(K_np, lv_np[i], color="tomato",    marker="s", linestyle="--", label="Local vol (Dupire)")
        ax.axvline(S0, color="grey", linestyle=":", linewidth=1, label="ATM")
        ax.set_title(T_label, fontsize=11)
        ax.set_ylabel("Volatility" if i == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, i]
        ax.plot(K_np, mc_np[i], color="steelblue", marker="o", label="Market call")
        ax.plot(K_np, ml_np[i], color="tomato",    marker="s", linestyle="--", label="LV reprice")
        ax.axvline(S0, color="grey", linestyle=":", linewidth=1)
        ax.set_ylabel("Call price" if i == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2, i]
        bar_w = (K_np[1] - K_np[0]) * 0.5 if nK > 1 else 1.0
        ax.bar(K_np, err_np[i], width=bar_w,
               color=["tomato" if e < 0 else "steelblue" for e in err_np[i]], alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(S0, color="grey", linestyle=":", linewidth=1)
        ax.set_ylabel("Price error" if i == 0 else "")
        ax.set_xlabel("Strike")
        ax.grid(True, alpha=0.3, axis="y")

    for row, title in enumerate([
        "Implied vol vs Local vol",
        "Call prices: market vs LV reprice",
        "Reprice error (LV - market)",
    ]):
        axes[row, 0].annotate(
            title, xy=(-0.3, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", va="center", rotation=90,
        )

    plt.tight_layout()
    out1 = OUT_DIR / "local_vol_smiles.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"\nSmile plot saved to {out1}")

    # ----------------------------------------------------------------
    # Plot 2: MC paths + forward
    # ----------------------------------------------------------------
    fwd_theo = S0 * np.exp((float(r.numpy()) - float(q.numpy())) * t_fine_np)
    fwd_mc_t = paths_np.mean(axis=0)
    p20      = float(np.percentile(paths_np[:, -1], 20))
    p80      = float(np.percentile(paths_np[:, -1], 80))
    margin   = (p80 - p20) * 0.10
    y_lo, y_hi = max(0.0, p20 - margin), p80 + margin

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    fig2.suptitle(f"MC Paths — Local Vol  [{TICKER}  {EVALUATION_DATE}]",
                  fontsize=13, fontweight="bold")

    rng = np.random.default_rng(0)
    idx = rng.choice(paths_np.shape[0], min(300, n_mc), replace=False)
    for path in paths_np[idx]:
        ax2.plot(t_fine_np, path, color="steelblue", alpha=0.04, linewidth=0.6)

    ax2.plot(t_fine_np, fwd_mc_t, color="tomato", linewidth=2.2,
             label="Simulated forward  E[S_t]", zorder=3)
    ax2.plot(t_fine_np, fwd_theo, color="black",  linewidth=1.8,
             linestyle="--", label="Analytical forward  S\u2080\u00b7e^{(r-q)t}", zorder=4)

    for Ti in T_np:
        ax2.axvline(Ti, color="grey", linestyle=":", linewidth=0.9, alpha=0.6)
        ax2.text(Ti + 0.02, y_hi * 0.98, f"T={Ti:.2f}", fontsize=7, color="grey", va="top")

    ax2.axhline(S0, color="dimgrey", linestyle=":", linewidth=0.9, alpha=0.5,
                label=f"S\u2080 = {S0:.2f}")
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Spot")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    out2 = OUT_DIR / "local_vol_paths.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Paths plot saved to {out2}")

    # ----------------------------------------------------------------
    # Plot 3: MC call prices vs market (one panel per maturity)
    # ----------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, nT, figsize=(5 * nT, 4), sharey=False)
    if nT == 1:
        axes3 = [axes3]
    fig3.suptitle(f"MC Call Prices vs Market — {TICKER}", fontsize=13, fontweight="bold")

    for i in range(nT):
        Ti   = float(T_grid[i].numpy())
        df_T = float(tf.exp(-r * Ti).numpy())
        mc_c = df_T * tf.reduce_mean(
            tf.maximum(terminal[i][:, None] - K_grid[None, :], 0.0), axis=0
        ).numpy()

        ax = axes3[i]
        ax.plot(K_np, mc_np[i], color="steelblue", marker="o", label="Market (BS)")
        ax.plot(K_np, mc_c,     color="tomato",    marker="s", linestyle="--", label="MC Local Vol")
        ax.axvline(S0, color="grey", linestyle=":", linewidth=1, label="ATM")
        ax.set_title(f"T = {Ti:.2f}Y", fontsize=11)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Call price" if i == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out3 = OUT_DIR / "local_vol_mc_calls.png"
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"MC calls plot saved to {out3}")


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    main()
