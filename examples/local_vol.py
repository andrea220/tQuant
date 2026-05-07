from datetime import date
from pathlib import Path
import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Ensure local package is imported when running `python examples/...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorquant as tq
from tensorquant.models.localvolatility import LocalVolatilityModel


def bs_call_price(
    spot: tf.Tensor,
    strike: tf.Tensor,
    maturity: tf.Tensor,
    rate: tf.Tensor,
    dividend_yield: tf.Tensor,
    volatility: tf.Tensor,
) -> tf.Tensor:
    maturity = tf.maximum(tf.cast(maturity, tf.float32), tf.constant(1e-10))
    volatility = tf.maximum(tf.cast(volatility, tf.float32), tf.constant(1e-10))
    sqrt_t = tf.sqrt(maturity)
    d1 = (
        tf.math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * tf.square(volatility)) * maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    normal_cdf = lambda x: 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(tf.constant(2.0))))
    return (
        spot * tf.exp(-dividend_yield * maturity) * normal_cdf(d1)
        - strike * tf.exp(-rate * maturity) * normal_cdf(d2)
    )


def build_static_market_environment(evaluation_date: date) -> tq.MarketEnvironment:
    daycounter = tq.DayCounter(tq.DayCounterConvention.Actual365)
    spot = 5.174
    moneyness_grid = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70]
    strike_grid = [round(spot * m, 4) for m in moneyness_grid]
    maturity_grid = [0.25, 0.75, 1.50, 3.00]

    # Smooth quadratic smile: (m-1)^2 is C-inf, no kink at ATM.
    # Dupire requires a differentiable IV surface; abs(m-1) creates a delta-like
    # spike in d²C/dK² at ATM that artificially deflates local vol there.
    volatility_matrix = []
    for base_vol in [0.34, 0.31, 0.29, 0.27]:
        row = []
        for m in moneyness_grid:
            smile_bump = 0.15 * (m - 1.0) ** 2
            row.append(round(base_vol + smile_bump, 5))
        volatility_matrix.append(row)

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

    market = {
        "IR:EUR:ESTR:SPOT": estr_curve,
        "EQ:EUR:TESTCTP:SPOT": spot,
        "EQ:EUR:TESTCTP:VOL": vol_surface,
        "EQ:EUR:TESTCTP:REPO": -0.035,
        "EQ:EUR:TESTCTP:DIVYIELD": 0.0,
    }
    return tq.MarketEnvironment(market=market)


def main() -> None:
    evaluation_date = date(2026, 3, 31)
    tq.Settings.evaluation_date = evaluation_date
    market_env = build_static_market_environment(evaluation_date)

    ticker = "TESTCTP"
    ccy = tq.Currency.EUR

    spot = tf.constant(market_env.get_eq_spot(ticker, ccy=ccy), dtype=tf.float32)
    vol_surface = market_env.get_eq_vol_surface(ticker, ccy=ccy)
    disc_curve = market_env.get_ir_curve(ccy)

    T_grid = tf.constant(vol_surface.maturity, dtype=tf.float32)
    K_grid = tf.constant(vol_surface.strike, dtype=tf.float32)
    iv_matrix = tf.Variable(vol_surface.volatility_matrix, dtype=tf.float32)

    # Flat-rate approximation: extract r from the ESTR curve at the longest
    # grid maturity. For Dupire on sparse grids this is a standard convention.
    T_max = float(T_grid[-1].numpy())
    r = tf.cast(disc_curve.zero_rate(T_max), tf.float32)
    q = tf.constant(market_env.get_eq_div_yield(ticker, ccy=ccy), dtype=tf.float32)

    local_vol_model = LocalVolatilityModel.from_implied_vol(
        iv_matrix=iv_matrix,
        T_grid=T_grid,
        K_grid=K_grid,
        S0=spot,
        r=r,
        q=q,
    )

    # --- Compute market call prices (BS with full r, q) ---
    market_call_prices = []
    for i in range(T_grid.shape[0]):
        row = bs_call_price(
            spot=spot,
            strike=K_grid,
            maturity=T_grid[i],
            rate=r,
            dividend_yield=q,
            volatility=iv_matrix[i, :],
        )
        market_call_prices.append(row)
    market_call_matrix = tf.stack(market_call_prices, axis=0)   # [nT, nK]

    # --- Call price diffs (butterfly proxy for d²C/dK²) ---
    call_price_diffs     = market_call_matrix[:, 1:] - market_call_matrix[:, :-1]

    # --- Calibrate local vol surface from the same call prices ---
    TT, KK = tf.meshgrid(T_grid, K_grid, indexing="ij")
    sigma_loc_flat  = local_vol_model.sigma_loc(tf.reshape(TT, [-1]), tf.reshape(KK, [-1]))
    sigma_loc_matrix = tf.reshape(sigma_loc_flat, tf.shape(TT))   # [nT, nK]

    # --- Sanity check: reprice with local vol and compare to market ---
    # Approximation: use σ_loc(T_i, K_j) as a flat vol in Black formula
    # and compare to market IV. For a well-calibrated LV surface the
    # discrepancy should be small around ATM and larger at the wings
    # (finite-difference noise on sparse grids is expected).
    model_call_prices = []
    for i in range(T_grid.shape[0]):
        row = bs_call_price(
            spot=spot,
            strike=K_grid,
            maturity=T_grid[i],
            rate=r,
            dividend_yield=q,
            volatility=sigma_loc_matrix[i, :],
        )
        model_call_prices.append(row)
    model_call_matrix = tf.stack(model_call_prices, axis=0)   # [nT, nK]
    price_error       = model_call_matrix - market_call_matrix   # [nT, nK]
    abs_price_error   = tf.abs(price_error)

    # --- Quick path simulation ---
    n_paths    = 2000
    n_steps    = 60
    t_grid_sim = tf.linspace(tf.constant(0.0), tf.constant(T_max), n_steps + 1)[1:]
    dw         = tf.random.normal([n_paths, n_steps], dtype=tf.float32)
    paths      = local_vol_model.evolve(t_grid_sim, dw)

    # --- Output ---
    nT, nK = T_grid.shape[0], K_grid.shape[0]
    col_w = 10

    def fmt_row(values):
        return "  ".join(f"{v:>{col_w}.5f}" for v in values)

    header_k = "  ".join(f"K={k:>6.3f}" for k in K_grid.numpy())

    print("\n=== Local Vol Calibration (full Dupire: r, q, non-uniform grid) ===")
    print(f"spot={float(spot.numpy()):.4f}  r={float(r.numpy()):.4f}  q={float(q.numpy()):.4f}")
    print(f"T grid: {T_grid.numpy().tolist()}")
    print(f"K grid: {K_grid.numpy().tolist()}")

    print(f"\n{'Implied vol surface (market)':}")
    print("  " + header_k)
    for i in range(nT):
        print(f"  T={T_grid[i].numpy():.2f}  " + fmt_row(iv_matrix[i].numpy()))

    print(f"\n{'Local vol surface (Dupire)':}")
    print("  " + header_k)
    for i in range(nT):
        print(f"  T={T_grid[i].numpy():.2f}  " + fmt_row(sigma_loc_matrix[i].numpy()))

    print(f"\n{'Market call prices (BS)':}")
    print("  " + header_k)
    for i in range(nT):
        print(f"  T={T_grid[i].numpy():.2f}  " + fmt_row(market_call_matrix[i].numpy()))

    print(f"\n{'Call price diffs by strike (dC/dK proxy)':}")
    for i in range(nT):
        print(f"  T={T_grid[i].numpy():.2f}  " + fmt_row(call_price_diffs[i].numpy()))

    print(f"\n{'Reprice error: BS(sigma_loc) - market call (model vs market)':}")
    print("  " + header_k)
    for i in range(nT):
        print(f"  T={T_grid[i].numpy():.2f}  " + fmt_row(price_error[i].numpy()))

    print(f"\n{'Max abs reprice error per maturity':}")
    for i in range(nT):
        print(f"  T={T_grid[i].numpy():.2f}  max_abs={float(tf.reduce_max(abs_price_error[i]).numpy()):.6f}")

    print(f"\nSimulated paths shape: {paths.shape}")
    print(f"Mean terminal spot (T={T_max:.2f}): {float(tf.reduce_mean(paths[:, -1]).numpy()):.4f}")

    # ----------------------------------------------------------------
    # Monte Carlo sanity checks
    # Simulate paths at the exact maturity dates in T_grid so that
    # each terminal slice S[:,i] corresponds to T_grid[i].
    # ----------------------------------------------------------------
    n_mc    = 50_000
    n_steps = 100
    tf.random.set_seed(42)

    # evolve() returns [n_paths, n_steps] at each t in t_grid.
    # We pass T_grid directly so that step i ends exactly at T_grid[i].
    dw_mc   = tf.random.normal([n_mc, T_grid.shape[0]], dtype=tf.float32)
    # One step per maturity: evolve with t_grid = T_grid (single-step per maturity).
    # For smoother paths use sub-steps and pick the last of each segment.
    # Here we use a fine grid and sub-sample at the maturity indices.
    t_fine      = tf.linspace(0.0, T_max, n_steps + 1)[1:]               # [n_steps]
    dw_fine     = tf.random.normal([n_mc, n_steps], dtype=tf.float32)
    paths_mc    = local_vol_model.evolve(t_fine, dw_fine)                 # [n_mc, n_steps]

    # Indices of t_fine closest to each T_grid[i]
    t_fine_np   = t_fine.numpy()
    maturity_idx = [int(np.argmin(np.abs(t_fine_np - float(T_grid[i].numpy())))) for i in range(nT)]
    terminal     = [paths_mc[:, idx] for idx in maturity_idx]            # list of [n_mc]

    print("\n" + "=" * 70)
    print("Monte Carlo Sanity Checks")
    print("=" * 70)

    for i in range(nT):
        Ti       = float(T_grid[i].numpy())
        S_T      = terminal[i]                                            # [n_mc]
        log_S_T  = tf.math.log(S_T / spot)

        # 1) Forward match: E[S_T] vs theoretical forward
        mc_fwd      = float(tf.reduce_mean(S_T).numpy())
        theo_fwd    = float(spot.numpy()) * float(tf.exp((r - q) * Ti).numpy())
        fwd_err_pct = (mc_fwd - theo_fwd) / theo_fwd * 100

        # 2) MC call and put prices at each strike
        df_T     = float(tf.exp(-r * Ti).numpy())
        mc_calls = df_T * tf.reduce_mean(
            tf.maximum(S_T[:, None] - K_grid[None, :], 0.0), axis=0
        ).numpy()                                                          # [nK]
        mc_puts  = df_T * tf.reduce_mean(
            tf.maximum(K_grid[None, :] - S_T[:, None], 0.0), axis=0
        ).numpy()                                                          # [nK]

        # 3) Put-call parity error: C - P - (S0*e^{-qT} - K*e^{-rT})
        pcp_lhs  = mc_calls - mc_puts
        pcp_rhs  = (float(spot.numpy()) * float(tf.exp(-q * Ti).numpy())
                    - K_grid.numpy() * df_T)
        pcp_err  = np.abs(pcp_lhs - pcp_rhs)

        # 4) Vol proxy: std of log-returns vs ATM implied vol
        atm_idx  = int(np.argmin(np.abs(K_grid.numpy() - float(spot.numpy()))))
        mc_vol   = float(tf.math.reduce_std(log_S_T).numpy()) / np.sqrt(Ti)
        mkt_atm  = float(iv_matrix[i, atm_idx].numpy())

        # 5) MC call error vs market
        mc_call_err = np.abs(mc_calls - market_call_matrix[i].numpy())

        print(f"\n  T = {Ti:.2f}Y")
        print(f"    [1] Forward — MC: {mc_fwd:.4f}  |  Theo: {theo_fwd:.4f}  |  Error: {fwd_err_pct:+.3f}%")
        print(f"    [2] ATM vol proxy — MC std/sqrt(T): {mc_vol:.4f}  |  Market ATM IV: {mkt_atm:.4f}  |  Diff: {mc_vol - mkt_atm:+.4f}")
        print(f"    [3] Put-call parity — max abs error: {pcp_err.max():.6f}  |  mean: {pcp_err.mean():.6f}")
        print(f"    [4] MC call vs market — max abs: {mc_call_err.max():.4f}  |  mean: {mc_call_err.mean():.4f}")
        print(f"        (strikes: " + "  ".join(f"K={k:.2f}: {e:.4f}" for k, e in
              zip(K_grid.numpy()[::3], mc_call_err[::3])) + ")")

    # ----------------------------------------------------------------
    # Plot: simulated paths + analytical vs simulated forward
    # ----------------------------------------------------------------
    t_np       = t_fine.numpy()                                       # [n_steps]
    paths_np   = paths_mc.numpy()                                     # [n_mc, n_steps]
    S0_val     = float(spot.numpy())
    r_val      = float(r.numpy())
    q_val      = float(q.numpy())

    # Analytical forward at each time step
    fwd_theo   = S0_val * np.exp((r_val - q_val) * t_np)             # [n_steps]
    # Simulated forward (cross-sectional mean)
    fwd_mc     = paths_np.mean(axis=0)                                # [n_steps]

    fig_paths, ax_p = plt.subplots(figsize=(11, 5))
    fig_paths.suptitle("MC Paths — Local Vol Model (TESTCTP)", fontsize=13, fontweight="bold")

    # Clip y-axis to [1%, 99%] percentile of terminal values to avoid
    # extreme paths squashing the view.
    p01 = float(np.percentile(paths_np[:, -1], 20))
    p99 = float(np.percentile(paths_np[:, -1], 80))
    margin = (p99 - p01) * 0.10
    y_lo, y_hi = max(0.0, p01 - margin), p99 + margin

    # Background paths: plot a random subset, heavily faded
    n_show = min(300, paths_np.shape[0])
    rng    = np.random.default_rng(0)
    idx    = rng.choice(paths_np.shape[0], n_show, replace=False)
    for path in paths_np[idx]:
        ax_p.plot(t_np, path, color="steelblue", alpha=0.04, linewidth=0.6)

    # Simulated forward (mean path)
    ax_p.plot(t_np, fwd_mc,   color="tomato",  linewidth=2.2,
              label="Simulated forward  E[S_t]", zorder=3)
    # Analytical forward
    ax_p.plot(t_np, fwd_theo, color="black",   linewidth=1.8,
              linestyle="--", label="Analytical forward  S\u2080\u00b7e^{(r-q)t}", zorder=4)

    # Maturity markers
    for Ti in T_grid.numpy():
        ax_p.axvline(Ti, color="grey", linestyle=":", linewidth=0.9, alpha=0.6)
        ax_p.text(Ti + 0.02, y_hi * 0.98, f"T={Ti:.2f}", fontsize=7,
                  color="grey", va="top")

    ax_p.axhline(S0_val, color="dimgrey", linestyle=":", linewidth=0.9, alpha=0.5,
                 label=f"S\u2080 = {S0_val:.2f}")
    ax_p.set_ylim(y_lo, y_hi)
    ax_p.set_xlabel("Time (years)")
    ax_p.set_ylabel("Spot")
    ax_p.legend(fontsize=9)
    ax_p.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("examples/local_vol_paths.png", dpi=150)
    plt.show()
    print("Paths plot saved to examples/local_vol_paths.png")

    # ----------------------------------------------------------------
    # Plot: MC call prices vs market (one panel per maturity)
    # ----------------------------------------------------------------
    fig_mc, axes_mc = plt.subplots(1, nT, figsize=(5 * nT, 4), sharey=False)
    fig_mc.suptitle("MC Call Prices vs Market — TESTCTP", fontsize=13, fontweight="bold")

    for i in range(nT):
        Ti   = float(T_grid[i].numpy())
        df_T = float(tf.exp(-r * Ti).numpy())
        mc_c = df_T * tf.reduce_mean(
            tf.maximum(terminal[i][:, None] - K_grid[None, :], 0.0), axis=0
        ).numpy()

        ax = axes_mc[i]
        ax.plot(K_grid.numpy(), market_call_matrix[i].numpy(),
                color="steelblue", marker="o", label="Market (BS)")
        ax.plot(K_grid.numpy(), mc_c,
                color="tomato", marker="s", linestyle="--", label="MC Local Vol")
        ax.axvline(float(spot.numpy()), color="grey", linestyle=":", linewidth=1, label="ATM")
        ax.set_title(f"T = {Ti:.2f}Y", fontsize=11)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Call price" if i == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/local_vol_mc_calls.png", dpi=150)
    plt.show()
    print("MC plot saved to examples/local_vol_mc_calls.png")

    # --- Plots: one column per maturity, three rows ---
    K_np   = K_grid.numpy()
    T_np   = T_grid.numpy()
    iv_np  = iv_matrix.numpy()
    lv_np  = sigma_loc_matrix.numpy()
    mc_np  = market_call_matrix.numpy()
    ml_np  = model_call_matrix.numpy()
    err_np = price_error.numpy()
    S0     = float(spot.numpy())

    fig, axes = plt.subplots(3, nT, figsize=(5 * nT, 12), sharex="col")
    fig.suptitle("Local Vol Calibration — TESTCTP", fontsize=14, fontweight="bold")

    row_titles = [
        "Implied vol vs Local vol",
        "Call prices: market vs LV reprice",
        "Reprice error (LV - market)",
    ]

    for i in range(nT):
        T_label = f"T = {T_np[i]:.2f}Y"

        # Row 0 — smile: IV vs LV
        ax = axes[0, i]
        ax.plot(K_np, iv_np[i], color="steelblue", marker="o", label="Implied vol (market)")
        ax.plot(K_np, lv_np[i], color="tomato",    marker="s", linestyle="--", label="Local vol (Dupire)")
        ax.axvline(S0, color="grey", linestyle=":", linewidth=1, label="ATM")
        ax.set_title(T_label, fontsize=11)
        ax.set_ylabel("Volatility" if i == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 1 — call prices: market vs LV
        ax = axes[1, i]
        ax.plot(K_np, mc_np[i], color="steelblue", marker="o", label="Market call")
        ax.plot(K_np, ml_np[i], color="tomato",    marker="s", linestyle="--", label="LV reprice")
        ax.axvline(S0, color="grey", linestyle=":", linewidth=1)
        ax.set_ylabel("Call price" if i == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 2 — reprice error
        ax = axes[2, i]
        ax.bar(K_np, err_np[i], width=(K_np[1] - K_np[0]) * 0.5, color=[
            "tomato" if e < 0 else "steelblue" for e in err_np[i]
        ], alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(S0, color="grey", linestyle=":", linewidth=1)
        ax.set_ylabel("Price error" if i == 0 else "")
        ax.set_xlabel("Strike")
        ax.grid(True, alpha=0.3, axis="y")

    for row, title in enumerate(row_titles):
        axes[row, 0].annotate(
            title, xy=(-0.3, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", va="center", rotation=90,
        )

    plt.tight_layout()
    plt.savefig("examples/local_vol_smiles.png", dpi=150)
    plt.show()
    print("Plot saved to examples/local_vol_smiles.png")


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    main()
