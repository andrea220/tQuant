"""
Displaced Diffusion Model on a marketdata/ snapshot: calibration, closed form vs market, Monte Carlo, greeks.

Steps:
  1. ``DisplacedDiffusionModel.from_equity``: per vol-surface maturity (beta_T, sigma_T) of the closed form
     fitted to the market smile (vega-weighted), forward pieces by skew averaging; calibration report.
  2. Plot: market smile vs displaced diffusion (closed form) per maturity -> examples/dd_smiles.png.
  3. Monte Carlo (exact shifted-lognormal steps on the market forward): forward check and MC implied vols
     vs the closed form at the nodes; a few paths -> examples/dd_paths.png.
  4. ``VanillaMCPricer(model=DisplacedDiffusionModel)``: 1Y ATM call, delta and vega by AAD through the
     calibration (implicit function theorem) vs bump-and-reprice.

The synthetic equity smiles (SSVI) are steeper than the normal skew (the displaced diffusion limit,
d sigma / d ln K ~ -sigma / 2): beta ends on its lower bound, ATM is fitted, the wings are not.
"""

import io
import os
import sys
from datetime import timedelta
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorquant as tq
from tensorquant.models.displaceddiffusion import DisplacedDiffusionModel, _black_implied_vol

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
MARKET_DATA      = Path(__file__).resolve().parents[1] / "marketdata"
SNAPSHOT_DATE    = None        # date(yyyy, m, d) | None = latest snapshot
CALIBRATION_SET  = None        # e.g. "1516"      | None = latest of the day
TICKER           = "TICKER1"   # TICKER1-30 EUR | TICKER31-40 GBP | TICKER41-50 USD
N_PATHS          = 100_000
SEED             = 42
OUT_DIR          = Path(__file__).resolve().parent
# ----------------------------------------------------------------


def main() -> None:
    market_env = tq.MarketEnvironment.from_folder(MARKET_DATA, SNAPSHOT_DATE, CALIBRATION_SET)
    eval_date = market_env.evaluation_date
    tq.Settings.evaluation_date = eval_date
    print(f"snapshot marketdata/{eval_date:%Y%m%d}/{market_env.calibration_set}, {TICKER}")
    eq = market_env.get_equity(TICKER)
    surface = eq.vol_surface
    T = np.asarray(surface.maturity, dtype=float)
    K = np.asarray(surface.strike, dtype=float)
    spot = float(eq.spot.numpy())

    # 1. calibration
    model = DisplacedDiffusionModel.from_equity(eq)
    pd.set_option("display.width", 200)
    print("\nCalibration (slice beta/sigma, forward pieces, model - market vol errors in vol points)")
    print(model.calibration.round(4).to_string(index=False))

    # 2. smiles: market vs closed form
    fig, axes = plt.subplots(2, (len(T) + 1) // 2, figsize=(4 * ((len(T) + 1) // 2), 7))
    k_fine = np.linspace(K[0], K[-1], 60)
    for ax, i in zip(axes.ravel(), range(len(T))):
        ax.plot(K / spot, 100 * surface.volatility_matrix.numpy()[i], "o", label="market")
        ax.plot(k_fine / spot, [100 * model.implied_vol(k, T[i]) for k in k_fine], "-", label="DD")
        ax.set_title(f"T = {T[i]:.2f}y  beta = {model.calibration.beta[i]:.2f}")
        ax.set_xlabel("K / S0")
        ax.set_ylabel("vol (%)")
    axes.ravel()[0].legend()
    fig.suptitle(f"Displaced diffusion vs market smile ({TICKER})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dd_smiles.png", dpi=120)
    plt.close(fig)

    # 3. Monte Carlo at the nodes (exact steps: the grid only needs the nodes)
    dw = tf.random.stateless_normal([N_PATHS, len(T)], seed=[SEED, 0], dtype=tf.float64)
    paths = DisplacedDiffusionModel.from_equity(eq, martingale_correction=False).evolve(T, dw).numpy()
    print("\nMonte Carlo vs closed form at the nodes (vol points; forward error in standard errors)")
    for i, t in enumerate(T):
        F = float(eq.forward(float(t)))
        z = (paths[:, i].mean() - F) / (paths[:, i].std() / np.sqrt(N_PATHS))
        errs = []
        for k in K[2:-2]:
            mc = _black_implied_vol(float(np.mean(np.maximum(paths[:, i] / F - k / F, 0.0))), k / F, t)
            errs.append(100 * (mc - model.implied_vol(k, t)))
        print(f"  T={t:5.2f}  forward z={z:+.2f}  MC - CF: {np.round(errs, 2)}")
    grid = np.linspace(T[-1] / 250, T[-1], 250)
    sample = model.evolve(grid, tf.random.stateless_normal([20, len(grid)], seed=[SEED, 1], dtype=tf.float64)).numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grid, sample.T, lw=0.8, alpha=0.7)
    ax.plot(grid, [float(eq.forward(float(t))) for t in grid], "k--", lw=2, label="market forward")
    ax.set_xlabel("t (years)")
    ax.set_ylabel("S")
    ax.set_title(f"Displaced diffusion paths ({TICKER})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dd_paths.png", dpi=120)
    plt.close(fig)

    # 4. VanillaMCPricer with AAD greeks through the calibration
    expiry = eval_date + timedelta(days=365)
    strike = round(spot, 2)

    def option():
        return tq.VanillaOption(tq.Currency.EUR, start_date=eval_date, end_date=expiry, strike=strike,
                                option_type=tq.OptionType.Call, exercise_type=tq.ExerciseType.European,
                                underlying=TICKER)

    pricer = tq.VanillaMCPricer(model=DisplacedDiffusionModel, n_paths=N_PATHS, n_steps=12, seed=SEED)
    opt = option()
    pricer.price(opt, market_env, autodiff=True)
    d_spot, d_vol = pricer.tape.gradient(opt.price, [eq.spot, surface.volatility_matrix])
    closed = float(model.vanilla_price(strike, expiry))

    def reprice():
        o = option()
        tq.VanillaMCPricer(model=DisplacedDiffusionModel, n_paths=N_PATHS, n_steps=12, seed=SEED).price(o, market_env)
        return float(o.price)

    h, out = 0.01, []
    for sgn in (1, -1):
        eq.spot.assign(spot + sgn * h)
        out.append(reprice())
    eq.spot.assign(spot)
    base, vega_bump = surface.volatility_matrix.numpy(), []
    for sgn in (1, -1):
        surface.volatility_matrix.assign(base + sgn * 1e-4)
        vega_bump.append(reprice())
    surface.volatility_matrix.assign(base)
    print(f"\n1Y ATM call (K = {strike}): MC {float(opt.price):.4f} +- {float(opt.standard_error):.4f}, closed form {closed:.4f}")
    print(f"  delta: AAD {float(d_spot):.5f}  bump {(out[0] - out[1]) / (2 * h):.5f}")
    print(f"  vega (parallel, per 1.00 vol): AAD {float(tf.reduce_sum(d_vol)):.4f}  bump {(vega_bump[0] - vega_bump[1]) / 2e-4:.4f}")
    print(f"\nPlots: {OUT_DIR / 'dd_smiles.png'}, {OUT_DIR / 'dd_paths.png'}")


if __name__ == "__main__":
    main()
