"""
2_forwards.py -- Analisi dividendi e calcolo del forward price.

SOURCE_MKT_DATA = 'Real'      -> MDM datastore
SOURCE_MKT_DATA = 'Generated' -> market_data.py (dati sintetici)

Formula forward (modello dividendi discreti, coerente con BlackScholesPricer):
    F(T) = (S - PV_div(T)) * exp((r(T) - repo) * T)

dove:
    PV_div(T) = somma degli importi dei dividendi con ex-date in (t0, T]
                scontati con la curva ESTR
    r(T)      = -ln(P(0,T)) / T   (zero rate continuo dalla curva discount)
    repo      = margine repo dal market (market_env.get_eq_repo)
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
from market_data import EVALUATION_DATE as _EVAL_DATE_GEN, build_market_environment

# ══════════════════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════════════════

SOURCE_MKT_DATA = "Real"     # 'Real' | 'Generated'
TICKER          = "ISP"     # Real: ISP | SX5E | AMZN | BABA
                              # Generated: TICKER1 | TICKER2

MATURITY_YEARS  = 1.0        # orizzonte dell'analisi (anni)
USE_IMPL_REPO   = True       # True: usa il repo dal market | False: repo = 0

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


def main() -> None:
    tf.keras.backend.set_floatx("float64")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    market_env, EVALUATION_DATE = load_market()

    ccy        = market_env.get_currency(TICKER)
    spot       = float(market_env.get_eq_spot(TICKER, ccy=ccy))
    repo       = float(market_env.get_eq_repo(TICKER, ccy=ccy)) if USE_IMPL_REPO else 0.0
    disc_curve = market_env.get_ir_curve(tq.Currency.EUR)

    horizon    = EVALUATION_DATE + timedelta(days=int(round(MATURITY_YEARS * 365)))
    df_T       = float(disc_curve.discount(MATURITY_YEARS))
    r_T        = -np.log(df_T) / MATURITY_YEARS

    # ── Riepilogo inputs ──────────────────────────────────────────────────────
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

    # ── Analisi dividendi nell'orizzonte ──────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  DIVIDENDI in ({EVALUATION_DATE}, {horizon}]")
    print(f"{SEP}")

    try:
        div_curve = market_env.get_eq_dividends(TICKER, ccy=ccy)
        divs_in_horizon = [
            (ex, amt)
            for ex, amt in zip(div_curve.ex_dates, div_curve.amounts)
            if EVALUATION_DATE < ex <= horizon
        ]
    except ValueError:
        divs_in_horizon = []

    div_details = []   # (ex, amt, tau, df_ex, pv, cum_gross, cum_pv)
    if divs_in_horizon:
        cum_gross = 0.0
        cum_pv    = 0.0
        print(f"\n  {'Ex date':<14s}  {'Gross':>10s}  {'% spot':>7s}  "
              f"{'df(τ)':>9s}  {'PV':>10s}  {'CumGross':>10s}  {'CumPV':>10s}")
        print(f"  {sep}")
        for ex, amt in divs_in_horizon:
            tau      = float(disc_curve.daycounter.year_fraction(EVALUATION_DATE, ex))
            df_ex    = float(disc_curve.discount(ex))
            pv       = amt * df_ex
            cum_gross += amt
            cum_pv   += pv
            div_details.append((ex, amt, tau, df_ex, pv, cum_gross, cum_pv))
            print(f"  {str(ex):<14s}  {amt:>10.4f}  {amt/spot*100:>6.3f}%  "
                  f"{df_ex:>9.6f}  {pv:>10.4f}  {cum_gross:>10.4f}  {cum_pv:>10.4f}")
        pv_div = cum_pv
    else:
        print(f"\n  Nessun dividendo nell'orizzonte.")
        pv_div = 0.0

    # ── Calcolo forward ───────────────────────────────────────────────────────
    s_net      = spot - pv_div
    f_discrete = s_net * np.exp((r_T - repo) * MATURITY_YEARS)
    f_no_div   = spot  * np.exp((r_T - repo) * MATURITY_YEARS)

    print(f"\n{SEP}")
    print(f"  FORWARD  a {MATURITY_YEARS:.2f}Y  ({horizon})")
    print(f"{SEP}")
    print(f"\n  PV dividendi          : {pv_div:.4f}  ({pv_div/spot*100:.3f}% di spot)")
    print(f"  Spot net  S* = S-PVdiv: {s_net:.4f}")
    print(f"\n  {'Metodo':<35s}  {'Forward':>10s}  {'vs Spot':>9s}")
    print(f"  {'-'*58}")
    for label, fwd in [
        ("Discreto  F=(S−PVdiv)·exp((r−repo)T)", f_discrete),
        ("No div    F=S·exp((r−repo)T)",          f_no_div),
    ]:
        print(f"  {label:<35s}  {fwd:>10.4f}  {(fwd/spot-1)*100:>+8.3f}%")

    # ── Dati per il plot: forward su griglia fine ────────────────────────────
    t_fine    = np.linspace(0.01, MATURITY_YEARS, 300)
    fwd_arr   = []
    fwd_nodiv = []
    for T in t_fine:
        exp_d = EVALUATION_DATE + timedelta(days=int(round(T * 365)))
        df_t  = float(disc_curve.discount(T))
        r_t   = -np.log(df_t) / T
        try:
            pv_t = sum(
                amt * float(disc_curve.discount(ex))
                for ex, amt in zip(div_curve.ex_dates, div_curve.amounts)
                if EVALUATION_DATE < ex <= exp_d
            )
        except Exception:
            pv_t = 0.0
        fwd_arr.append((spot - pv_t) * np.exp((r_t - repo) * T))
        fwd_nodiv.append(spot * np.exp((r_t - repo) * T))
    fwd_arr   = np.array(fwd_arr)
    fwd_nodiv = np.array(fwd_nodiv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Forward Analysis — {TICKER} ({ccy.name})  spot={spot:.2f}  "
        f"[{EVALUATION_DATE}]  [{SOURCE_MKT_DATA}]",
        fontsize=12, fontweight="bold",
    )

    # ── subplot 1: forward curve ──────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(t_fine, fwd_arr,   color="steelblue", lw=1.8,
             label="F discreto  (S−PVdiv)·e^{(r−repo)T}")
    ax1.plot(t_fine, fwd_nodiv, color="tomato",    lw=1.4, linestyle="--",
             label="F no-div  S·e^{(r−repo)T}")
    ax1.axhline(spot, color="grey", lw=0.8, linestyle=":", label=f"Spot = {spot:.2f}")

    # barre verticali per ogni ex-date
    for ex, amt, tau, *_ in div_details:
        ax1.axvline(tau, color="seagreen", lw=0.6, alpha=0.4)

    ax1.set_xlabel("T (anni)")
    ax1.set_ylabel("Forward price")
    ax1.set_title("Forward curve", fontsize=10)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # ── subplot 2: dividendi singoli (gross e PV) + CumPV ─────────────────────
    ax2  = axes[1]
    ax2b = ax2.twinx()

    if div_details:
        taus     = np.array([d[2] for d in div_details])
        grosses  = np.array([d[1] for d in div_details])
        pvs      = np.array([d[4] for d in div_details])
        cum_pvs  = np.array([d[6] for d in div_details])

        w = min(np.diff(taus).min() * 0.6 if len(taus) > 1 else 0.04, 0.06)
        ax2.bar(taus, grosses, width=w, color="steelblue", alpha=0.55, label="Gross")
        ax2.bar(taus, pvs,     width=w, color="tomato",    alpha=0.70, label="PV")
        ax2b.plot(taus, cum_pvs, "o-", color="darkgreen", lw=1.6, ms=3, label="CumPV")
        ax2b.set_ylabel("CumPV dividendi", color="darkgreen")
        ax2b.tick_params(axis="y", labelcolor="darkgreen")

        total_gross = grosses.sum()
        total_pv    = float(cum_pvs[-1])
    else:
        total_gross = 0.0
        total_pv    = 0.0

    ax2.set_xlabel("T (anni)")
    ax2.set_ylabel("Importo per ex-date")
    ax2.set_title("Dividendi: Gross vs PV + CumPV", fontsize=10)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── didascalia sotto la figura ────────────────────────────────────────────
    caption = (
        f"Spot = {spot:.4f}   |   "
        f"Forward ({MATURITY_YEARS:.2f}Y) = {f_discrete:.4f}   |   "
        f"Maturity = {horizon}   |   "
        f"Repo = {repo*100:.4f}%   |   "
        f"Σ Gross div = {total_gross:.2f} ({total_gross/spot*100:.2f}% spot)   |   "
        f"Σ NPV div = {total_pv:.2f} ({total_pv/spot*100:.2f}% spot)"
    )
    fig.text(
        0.5, 0.01, caption,
        ha="center", va="bottom", fontsize=7.5,
        color="dimgrey",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="lightgrey", alpha=0.7),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUT_DIR / f"2_forwards_{TICKER}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot salvato -> {out}")


if __name__ == "__main__":
    main()
