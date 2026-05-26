"""
4_montecarlo.py -- Dinamiche Monte Carlo del forward price (GBM con dividendi discreti).

Confronta il forward price analitico con la stima Monte Carlo via GBM:

    S*_0  = S - PV_div(T)               (net-of-dividends spot)
    dS*/S* = (r - repo) dt + sigma dW   (GBM risk-neutral)
    E[S*_T] = S*_0 * exp((r - repo)*T) = F_analitico(T)

Il modello GBM usa:
    x0    = S - PV_div(MATURITY_YEARS)   (solo dividendi entro l'orizzonte)
    mu    = r(T) - repo                  (drift risk-neutral)
    sigma = vol implicita ATM a T        (dalla vol surface)

SOURCE_MKT_DATA = 'Real'      -> MDM datastore
SOURCE_MKT_DATA = 'Generated' -> market_data.py (dati sintetici)
"""

import io
import os
import sys
from datetime import date, timedelta
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorquant as tq
from tensorquant.models.brownian import GeometricBrownianMotion
from market_data import EVALUATION_DATE as _EVAL_DATE_GEN, build_market_environment

# ══════════════════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════════════════

SOURCE_MKT_DATA = "Real"     # 'Real' | 'Generated'
TICKER          = "SX5E"     # Real: ISP | SX5E | AMZN | BABA
                              # Generated: TICKER1 | TICKER2

MATURITY_YEARS  = 1.0        # orizzonte della simulazione (anni)
USE_IMPL_REPO   = True       # True: usa il repo dal market | False: repo = 0

N_PATHS     = 50_000         # numero di path MC
N_STEPS     = 252            # passi temporali (circa 1 giorno per anno)
N_PLOT      = 80             # path da visualizzare nel grafico
SEED        = 42

# Parametri per SOURCE_MKT_DATA = 'Real'
EVALUATION_DATE_REAL = date(2026, 5, 26)
DATA_PATH            = "W:/Derivative/EquityDerivatives/AndreaC/dev/mdm/datastore/"
CALIBRATION_SET      = "142910"

# ══════════════════════════════════════════════════════════════════════════════

OUT_DIR = Path(__file__).resolve().parent / "out"
SEP = "=" * 68
sep = "-" * 68


def load_market() -> tuple[tq.MarketEnvironment, date]:
    if SOURCE_MKT_DATA == "Real":
        eval_date = EVALUATION_DATE_REAL
        tq.Settings.evaluation_date = eval_date
        mkt = tq.MarketEnvironment.from_data_path(
            evaluation_date=eval_date,
            data_path=DATA_PATH,
            calibration_set=CALIBRATION_SET,
        )
    elif SOURCE_MKT_DATA == "Generated":
        eval_date = _EVAL_DATE_GEN
        tq.Settings.evaluation_date = eval_date
        mkt = build_market_environment(eval_date)
    else:
        raise ValueError(f"SOURCE_MKT_DATA non valido: '{SOURCE_MKT_DATA}'")
    return mkt, eval_date


def analytical_forward_curve(
    t_fine: np.ndarray,
    spot: float,
    repo: float,
    disc_curve,
    div_curve,
    eval_date: date,
) -> np.ndarray:
    """Calcola la curva del forward analitico F(t) su griglia fine."""
    fwds = []
    for T in t_fine:
        exp_d = eval_date + timedelta(days=int(round(T * 365)))
        df_t  = float(disc_curve.discount(T))
        r_t   = -np.log(df_t) / T
        try:
            pv_t = sum(
                amt * float(disc_curve.discount(ex))
                for ex, amt in zip(div_curve.ex_dates, div_curve.amounts)
                if eval_date < ex <= exp_d
            )
        except Exception:
            pv_t = 0.0
        fwds.append((spot - pv_t) * np.exp((r_t - repo) * T))
    return np.array(fwds)


def main() -> None:
    tf.keras.backend.set_floatx("float64")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    market_env, EVALUATION_DATE = load_market()

    ccy        = market_env.get_currency(TICKER)
    spot       = float(market_env.get_eq_spot(TICKER, ccy=ccy))
    repo       = float(market_env.get_eq_repo(TICKER, ccy=ccy)) if USE_IMPL_REPO else 0.0
    disc_curve = market_env.get_ir_curve(tq.Currency.EUR)

    horizon = EVALUATION_DATE + timedelta(days=int(round(MATURITY_YEARS * 365)))
    df_T    = float(disc_curve.discount(MATURITY_YEARS))
    r_T     = -np.log(df_T) / MATURITY_YEARS

    # ── Dividendi nell'orizzonte ──────────────────────────────────────────────
    try:
        div_curve = market_env.get_eq_dividends(TICKER, ccy=ccy)
        pv_div = sum(
            amt * float(disc_curve.discount(ex))
            for ex, amt in zip(div_curve.ex_dates, div_curve.amounts)
            if EVALUATION_DATE < ex <= horizon
        )
        div_dates = [
            ex for ex in div_curve.ex_dates
            if EVALUATION_DATE < ex <= horizon
        ]
    except ValueError:
        pv_div    = 0.0
        div_dates = []
        div_curve = None

    # ── Vol ATM alla scadenza ─────────────────────────────────────────────────
    try:
        vol_surface = market_env.get_eq_vol_surface(TICKER, ccy=ccy)
        sigma_atm   = float(vol_surface.volatility(strike=spot, tenor=MATURITY_YEARS))
    except Exception:
        sigma_atm = 0.20
        vol_surface = None

    # ── Forward analitico a T ─────────────────────────────────────────────────
    s_net      = spot - pv_div
    f_analytic = s_net * np.exp((r_T - repo) * MATURITY_YEARS)

    # ── Stampa inputs ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  INPUTS")
    print(f"{SEP}")
    print(f"  Source      : {SOURCE_MKT_DATA}")
    print(f"  Ticker      : {TICKER} ({ccy.name})")
    print(f"  Eval date   : {EVALUATION_DATE}")
    print(f"  Horizon     : {MATURITY_YEARS:.2f}Y  ({horizon})")
    print(f"  Spot        : {spot:.4f}")
    print(f"  Repo        : {repo*100:.4f}%  ({'implied' if USE_IMPL_REPO else 'forced = 0'})")
    print(f"  r(T)        : {r_T*100:.4f}%  (zero rate cont. a {MATURITY_YEARS:.2f}Y)")
    print(f"  df(T)       : {df_T:.6f}")
    print(f"  PV_div(T)   : {pv_div:.4f}  ({pv_div/spot*100:.3f}% di spot)")
    print(f"  S* = S-PVdiv: {s_net:.4f}")
    print(f"  sigma ATM   : {sigma_atm*100:.2f}%  (vol implicita ATM a {MATURITY_YEARS:.2f}Y)")
    print(f"  F analitico : {f_analytic:.4f}")

    print(f"\n{SEP}")
    print(f"  MONTE CARLO  (GBM)")
    print(f"{SEP}")
    print(f"  N_PATHS = {N_PATHS:,}    N_STEPS = {N_STEPS}    seed = {SEED}")
    print(f"  x0 = S* = {s_net:.4f}   mu = r-repo = {(r_T-repo)*100:.4f}%   sigma = {sigma_atm*100:.2f}%")

    # ── Simulazione GBM ───────────────────────────────────────────────────────
    gbm = GeometricBrownianMotion(
        mu    = r_T - repo,
        sigma = sigma_atm,
        x0    = s_net,
    )

    t_grid = tf.constant(
        np.linspace(0.0, MATURITY_YEARS, N_STEPS + 1)[1:], dtype=tf.float64
    )

    tf.random.set_seed(SEED)
    dw    = tf.random.normal([N_PATHS, N_STEPS], dtype=tf.float64)
    paths = gbm.evolve(t_grid, dw).numpy()   # [N_PATHS, N_STEPS]

    t_np = t_grid.numpy()

    # media dei path a ogni step
    mean_path = paths.mean(axis=0)

    # forward MC a T_max
    f_mc = float(paths[:, -1].mean())

    err_abs = f_mc - f_analytic
    err_rel = err_abs / f_analytic * 100

    print(f"\n  F MC   (media dei path a T): {f_mc:.4f}")
    print(f"  F analitico               : {f_analytic:.4f}")
    print(f"  Errore  abs / rel         : {err_abs:+.4f}  /  {err_rel:+.4f}%")

    # percentili distribuzione terminale
    S_T = paths[:, -1]
    pct = np.percentile(S_T, [5, 25, 50, 75, 95])
    print(f"\n  Distribuzione S_T a {MATURITY_YEARS:.2f}Y:")
    print(f"  {'P5':>8s}  {'P25':>8s}  {'P50':>8s}  {'P75':>8s}  {'P95':>8s}")
    print(f"  {pct[0]:>8.2f}  {pct[1]:>8.2f}  {pct[2]:>8.2f}  {pct[3]:>8.2f}  {pct[4]:>8.2f}")

    # ── Curva forward analitica su griglia fine ───────────────────────────────
    t_fine   = np.linspace(0.01, MATURITY_YEARS, 300)
    fwd_anal = (
        analytical_forward_curve(t_fine, spot, repo, disc_curve, div_curve, EVALUATION_DATE)
        if div_curve is not None
        else s_net * np.exp((r_T - repo) * t_fine)
    )
    # media MC su griglia temporale (ogni step)
    # ricampiona t_np -> t_fine con interpolazione
    mean_path_fine = np.interp(t_fine, t_np, mean_path)

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Monte Carlo Forward — {TICKER} ({ccy.name})  spot={spot:.2f}  "
        f"sigma={sigma_atm*100:.1f}%  [{EVALUATION_DATE}]  [{SOURCE_MKT_DATA}]",
        fontsize=12, fontweight="bold",
    )

    # ── subplot 1: sample paths + media MC + forward analitico ───────────────
    ax1 = axes[0]
    rng = np.random.default_rng(SEED)
    idx_plot = rng.choice(N_PATHS, size=min(N_PLOT, N_PATHS), replace=False)

    t_with0 = np.concatenate([[0.0], t_np])
    for i in idx_plot:
        path_with0 = np.concatenate([[s_net], paths[i]])
        ax1.plot(t_with0, path_with0, color="steelblue", lw=0.4, alpha=0.25)

    # media MC (step-by-step)
    mean_with0 = np.concatenate([[s_net], mean_path])
    ax1.plot(t_with0, mean_with0, color="darkorange", lw=2.0, label="Media MC  E[S*_t]")

    # forward analitico
    ax1.plot(
        np.concatenate([[0.0], t_fine]),
        np.concatenate([[s_net], fwd_anal]),
        color="crimson", lw=1.8, linestyle="--", label="F analitico",
    )

    # linee dividendi
    for ex in div_dates:
        tau_div = float(disc_curve.daycounter.year_fraction(EVALUATION_DATE, ex))
        ax1.axvline(tau_div, color="seagreen", lw=0.7, alpha=0.5)

    ax1.axhline(spot,  color="grey", lw=0.8, linestyle=":", label=f"Spot = {spot:.2f}")
    ax1.axhline(s_net, color="grey", lw=0.8, linestyle="-.", alpha=0.6,
                label=f"S* = {s_net:.2f}")
    ax1.set_xlabel("T (anni)")
    ax1.set_ylabel("Prezzo")
    ax1.set_title(f"Path GBM  (N={N_PLOT} sample, N_tot={N_PATHS:,})", fontsize=10)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # ── subplot 2: distribuzione terminale ────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(S_T, bins=120, density=True, color="steelblue", alpha=0.55, label="MC dist.")

    ax2.axvline(spot,       color="grey",       lw=1.2, linestyle=":",  label=f"Spot = {spot:.2f}")
    ax2.axvline(s_net,      color="dimgrey",    lw=1.0, linestyle="-.", label=f"S* = {s_net:.2f}")
    ax2.axvline(f_analytic, color="crimson",    lw=1.8, linestyle="--", label=f"F analitico = {f_analytic:.2f}")
    ax2.axvline(f_mc,       color="darkorange", lw=1.8, linestyle="-",  label=f"F MC = {f_mc:.2f}")
    ax2.axvline(pct[0],     color="steelblue",  lw=0.8, linestyle=":",  alpha=0.6, label="P5 / P95")
    ax2.axvline(pct[4],     color="steelblue",  lw=0.8, linestyle=":",  alpha=0.6)

    ax2.set_xlabel("S_T")
    ax2.set_ylabel("Densità")
    ax2.set_title(f"Distribuzione terminale S_T a T={MATURITY_YEARS:.2f}Y", fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── didascalia ────────────────────────────────────────────────────────────
    caption = (
        f"Spot={spot:.4f}  |  S*={s_net:.4f}  |  "
        f"F analitico={f_analytic:.4f}  |  F MC={f_mc:.4f}  |  "
        f"err={err_rel:+.3f}%  |  "
        f"sigma={sigma_atm*100:.2f}%  |  repo={repo*100:.4f}%  |  "
        f"N={N_PATHS:,}  T={MATURITY_YEARS:.2f}Y ({horizon})"
    )
    fig.text(
        0.5, 0.01, caption,
        ha="center", va="bottom", fontsize=7.5,
        color="dimgrey",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="lightgrey", alpha=0.7),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUT_DIR / f"4_montecarlo_{TICKER}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot salvato -> {out}")


if __name__ == "__main__":
    main()
