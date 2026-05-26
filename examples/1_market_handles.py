"""
1_market_handles.py -- MarketEnvironment: step by step.

SOURCE_MKT_DATA = 'Real'      -> MDM datastore
SOURCE_MKT_DATA = 'Generated' -> market_data.py (dati sintetici)
"""

import io
import os
import sys
from datetime import date
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorquant as tq
from market_data import EVALUATION_DATE as _EVAL_DATE_GEN, build_market_environment

# ── Configurazione ─────────────────────────────────────────────────────────────

SOURCE_MKT_DATA = "Real"   # 'Real' | 'Generated'

EVALUATION_DATE_REAL = date(2026, 5, 26)
DATA_PATH            = "W:/Derivative/EquityDerivatives/AndreaC/dev/mdm/datastore/"
CALIBRATION_SET      = "140226"

OUT_DIR = Path(__file__).resolve().parent / "out"

SEP = "=" * 64
sep = "-" * 64


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ── Caricamento market ─────────────────────────────────────────────────────────

def load_market() -> tuple[tq.MarketEnvironment, date]:
    if SOURCE_MKT_DATA == "Real":
        eval_date  = EVALUATION_DATE_REAL
        tq.Settings.evaluation_date = eval_date
        mkt = tq.MarketEnvironment.from_data_path(
            evaluation_date=eval_date,
            data_path=DATA_PATH,
            calibration_set=CALIBRATION_SET,
        )
        print(f"[Real] {DATA_PATH}{eval_date.strftime('%Y%m%d')}/{CALIBRATION_SET}")
    elif SOURCE_MKT_DATA == "Generated":
        eval_date  = _EVAL_DATE_GEN
        tq.Settings.evaluation_date = eval_date
        mkt = build_market_environment(eval_date)
        print(f"[Generated] market_data.py  (eval date: {eval_date})")
    else:
        raise ValueError(f"SOURCE_MKT_DATA deve essere 'Real' o 'Generated', ricevuto: '{SOURCE_MKT_DATA}'")
    return mkt, eval_date


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    market_env, EVALUATION_DATE = load_market()

    print(f"Evaluation date : {EVALUATION_DATE}")
    print(f"Market keys     : {len(market_env._market)}")
    for k in sorted(market_env._market):
        print(f"  {k:<38s}  {type(market_env._market[k]).__name__}")

    # =========================================================================
    # Rate Curves
    # =========================================================================
    section("RATE CURVES")

    query_pillars = [0.25, 0.50, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    ir_tickers = [
        k.split(":")[2]
        for k in sorted(market_env._market)
        if k.startswith("IR:EUR:") and k.endswith(":SPOT")
    ]

    for ticker in ir_tickers:
        curve = market_env.get_ir_curve(tq.Currency.EUR, ticker=ticker)
        print(f"\n  IR:EUR:{ticker}  (ref: {curve.reference_date}  "
              f"convention: {curve.daycounter_convention.name}  "
              f"nodi: {len(curve.nodes)})")

        print(f"\n  {'T (Y)':>7s}  {'r_cc':>8s}  {'Discount':>10s}  "
              f"{'Fwd 1Y':>10s}  {'Inst fwd':>10s}")
        print(f"  {sep}")
        for T in query_pillars:
            df   = float(curve.discount(T))
            r_cc = -np.log(df) / T
            fr   = float(curve.forward_rate(T, T + 1.0))
            ifr  = float(curve.inst_fwd(T))
            print(f"  {T:>7.2f}  {r_cc:>8.4f}  {df:>10.6f}  {fr:>10.4f}  {ifr:>10.4f}")

    # =========================================================================
    # Plot rate curves
    # =========================================================================
    t_plt  = np.linspace(0.25, 10.0, 200)
    t_fwd  = np.linspace(0.25, 9.0,  180)
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(ir_tickers)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Rate Curves  [{EVALUATION_DATE}]  [{SOURCE_MKT_DATA}]",
                 fontsize=13, fontweight="bold")

    ax_zr, ax_df, ax_fwd = axes

    for ticker, color in zip(ir_tickers, colors):
        c  = market_env.get_ir_curve(tq.Currency.EUR, ticker=ticker)
        zr = [-np.log(float(c.discount(t))) / t * 100 for t in t_plt]
        df = [float(c.discount(t)) for t in t_plt]
        fr = [float(c.forward_rate(t, t + 1.0)) * 100 for t in t_fwd]

        ax_zr.plot(t_plt, zr, color=color, lw=1.8, label=f"EUR {ticker}")
        ax_df.plot(t_plt, df, color=color, lw=1.8, label=f"EUR {ticker}")
        ax_fwd.plot(t_fwd, fr, color=color, lw=1.8, label=f"EUR {ticker}")

        # scatter pillar nodes
        for d, r in c.nodes:
            tau = float(c.daycounter.year_fraction(c.reference_date, d))
            if tau <= 10.0:
                ax_zr.scatter(tau, r * 100, color=color, s=20, zorder=5)

    for ax, title, ylabel in [
        (ax_zr,  "Zero rates (cont. comp.)",  "r_cc (%)"),
        (ax_df,  "Discount factors",           "P(0,T)"),
        (ax_fwd, "Forward rates (1Y tenor)",   "Fwd rate (%)"),
    ]:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("T (Y)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "1_rate_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot salvato -> {out}")

    # =========================================================================
    # Dividend Curves
    # =========================================================================
    section("DISCRETE DIVIDEND CURVES")

    # (ccy_str, ticker) per tutti gli EQ che hanno una DividendCurve
    div_instruments = [
        (k.split(":")[1], k.split(":")[2])
        for k in sorted(market_env._market)
        if k.startswith("EQ:") and k.endswith(":DIV")
    ]

    for ccy_str, ticker in div_instruments:
        ccy       = tq.Currency[ccy_str]
        spot      = float(market_env.get_eq_spot(ticker, ccy=ccy))
        div_curve = market_env.get_eq_dividends(ticker, ccy=ccy)
        print(f"\n  {ticker} ({ccy_str})  spot={spot:.4f}  —  {len(div_curve.ex_dates)} dividendi")
        print(f"  {'Ex date':<14s}  {'Importo':>10s}  {'% spot':>8s}")
        print(f"  {'-'*36}")
        for ex, amt in zip(div_curve.ex_dates, div_curve.amounts):
            print(f"  {str(ex):<14s}  {amt:>10.4f}  {amt / spot * 100:>7.3f}%")

    # =========================================================================
    # Plot dividend curves  (tutti su un unico grafico, in % spot)
    # =========================================================================
    palette = plt.cm.tab10(np.linspace(0, 1, len(div_instruments)))

    fig2, ax_div = plt.subplots(figsize=(12, 4))
    fig2.suptitle(f"Discrete Dividends (% spot)  [{EVALUATION_DATE}]  [{SOURCE_MKT_DATA}]",
                  fontsize=13, fontweight="bold")

    for (ccy_str, ticker), color in zip(div_instruments, palette):
        ccy       = tq.Currency[ccy_str]
        spot      = float(market_env.get_eq_spot(ticker, ccy=ccy))
        div_curve = market_env.get_eq_dividends(ticker, ccy=ccy)
        dc        = div_curve._daycounter
        ts        = [float(dc.year_fraction(EVALUATION_DATE, ex)) for ex in div_curve.ex_dates]
        pcts      = [amt / spot * 100 for amt in div_curve.amounts]
        ax_div.plot(ts, pcts, "o-", color=color, lw=1.4, ms=4,
                    label=f"{ticker} ({ccy_str})")

    ax_div.set_xlabel("T dall'eval date (anni)")
    ax_div.set_ylabel("Dividendo (% spot)")
    ax_div.legend(fontsize=9)
    ax_div.grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = OUT_DIR / "1_dividends.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot salvato -> {out2}")

    # =========================================================================
    # Volatility Surfaces
    # =========================================================================
    section("VOLATILITY SURFACES")

    vol_instruments = [
        (k.split(":")[1], k.split(":")[2])
        for k in sorted(market_env._market)
        if k.startswith("EQ:") and k.endswith(":VOL")
    ]

    for ccy_str, ticker in vol_instruments:
        ccy  = tq.Currency[ccy_str]
        vs   = market_env.get_eq_vol_surface(ticker, ccy=ccy)
        spot = float(market_env.get_eq_spot(ticker, ccy=ccy))
        # strikes più vicini a 90%, 100%, 110% dello spot
        qs   = [
            min(vs.strike, key=lambda k: abs(k / spot - 0.90)),
            min(vs.strike, key=lambda k: abs(k / spot - 1.00)),
            min(vs.strike, key=lambda k: abs(k / spot - 1.10)),
        ]
        print(f"\n  {ticker} ({ccy_str})  spot={spot:.2f}  "
              f"grid: {len(vs.maturity)}T x {len(vs.strike)}K  "
              f"[T: {vs.maturity[0]:.2f}..{vs.maturity[-1]:.2f}Y  "
              f"K: {vs.strike[0]:.2f}..{vs.strike[-1]:.2f}]")
        print(f"  {'T (Y)':>7s}  " + "  ".join(f"K={K:.2f}" for K in qs))
        print(f"  {sep}")
        for T in vs.maturity:
            row = f"  {T:>7.2f}  "
            row += "  ".join(f"{vs.volatility(strike=K, tenor=T)*100:>8.2f}%" for K in qs)
            print(row)

    # =========================================================================
    # Plot volatility surfaces  (smile per maturity, un subplot per ticker)
    # =========================================================================
    n     = len(vol_instruments)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                squeeze=False)
    fig3.suptitle(f"Implied Volatility Smiles  [{EVALUATION_DATE}]  [{SOURCE_MKT_DATA}]",
                  fontsize=13, fontweight="bold")

    for idx, (ccy_str, ticker) in enumerate(vol_instruments):
        ax   = axes3[idx // ncols][idx % ncols]
        ccy  = tq.Currency[ccy_str]
        vs   = market_env.get_eq_vol_surface(ticker, ccy=ccy)
        spot = float(market_env.get_eq_spot(ticker, ccy=ccy))
        K_np = np.array(vs.strike)
        cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(vs.maturity)))

        for i, T in enumerate(vs.maturity):
            iv = [vs.volatility(strike=K, tenor=T) * 100 for K in K_np]
            ax.plot(K_np / spot, iv, "o-", color=cmap[i], lw=1.3, ms=3,
                    label=f"T={T:.2f}Y")

        ax.axvline(1.0, color="grey", lw=0.8, linestyle=":")
        ax.set_title(f"{ticker} ({ccy_str})  spot={spot:.2f}", fontsize=10)
        ax.set_xlabel("K / spot")
        ax.set_ylabel("IV (%)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes3[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    out3 = OUT_DIR / "1_vol_surfaces.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot salvato -> {out3}")


if __name__ == "__main__":
    main()
