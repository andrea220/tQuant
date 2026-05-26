"""
Local Volatility vs GBM -- European vanilla option pricing.

Mostra come usare VanillaMCPricer con due modelli diversi:
  - GeometricBrownianMotion  (Black-Scholes MC)
  - LocalVolatilityModel     (Dupire MC)

e li confronta con il benchmark analitico Black-Scholes.

Struttura:
  1. Costruire un MarketEnvironment sintetico.
  2. Definire un VanillaOption (call e put).
  3. Prezzare con BlackScholesPricer (analitico, benchmark).
  4. Prezzare con VanillaMCPricer + GeometricBrownianMotion.
  5. Prezzare con VanillaMCPricer + LocalVolatilityModel (Dupire).
  6. Stampare il confronto e salvare i plot in out/.

Run:
    python examples/local_vol_vanilla.py
"""

import io
import os
import sys
from datetime import date
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorquant as tq
from tensorquant.markethandles.dividendcurve import DividendCurve
from tensorquant.models.brownian import GeometricBrownianMotion
from tensorquant.models.localvolatility import LocalVolatilityModel
from tensorquant.pricers.vanillamc import VanillaMCPricer

OUT_DIR = Path(__file__).resolve().parent / "out"

# ── Parametri ─────────────────────────────────────────────────────────────────

EVALUATION_DATE = date(2026, 5, 15)
UNDERLYING      = "TESTLV"
CCY             = tq.Currency.EUR
STRIKE          = 105.0
END_DATE        = date(2027, 5, 17)   # ~1Y

N_PATHS = 100_000
N_STEPS = 252
SEED    = 42

# ── MarketEnvironment sintetico ───────────────────────────────────────────────

def build_market(evaluation_date: date) -> tq.MarketEnvironment:
    """Costruisce un MarketEnvironment con dati sintetici."""
    daycounter = tq.DayCounter(tq.DayCounterConvention.Actual365)

    rate_curve = tq.RateCurve(
        reference_date=evaluation_date,
        pillars=[0.25, 0.50, 1.0, 1.5, 2.0],
        rates=[0.028, 0.029, 0.030, 0.031, 0.032],
        interp="LINEAR",
        daycounter_convention=tq.DayCounterConvention.Actual365,
    )

    # superficie di vol implicita con skew negativo e curvatura (smile)
    #   IV(T, K) = 0.20 + 0.02*sqrt(T) - 0.10*m + 0.30*m^2   (m = log(K/S0))
    S0_ref     = 100.0
    strikes    = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]
    maturities = [0.25, 0.50, 0.75, 1.00, 1.50]
    vol_matrix = [
        [
            float(np.clip(
                0.20 + 0.02 * np.sqrt(T)
                - 0.10 * np.log(K / S0_ref)
                + 0.30 * np.log(K / S0_ref) ** 2,
                0.05, 0.80,
            ))
            for K in strikes
        ]
        for T in maturities
    ]

    vol_surface = tq.VolatilitySurface(
        reference_date=evaluation_date,
        calendar=None,
        daycounter=daycounter,
        strike=strikes,
        maturity=maturities,
        volatility_matrix=vol_matrix,
    )

    dividend_curve = DividendCurve(
        reference_date=evaluation_date,
        ex_dates=[],
        amounts=[],
        currency=CCY,
        daycounter_convention=tq.DayCounterConvention.Actual365,
    )

    market = {
        f"IR:{CCY.value}:ESTR:SPOT":             rate_curve,
        f"EQ:{CCY.value}:{UNDERLYING}:SPOT":     S0_ref,
        f"EQ:{CCY.value}:{UNDERLYING}:VOL":      vol_surface,
        f"EQ:{CCY.value}:{UNDERLYING}:REPO":     0.0,
        f"EQ:{CCY.value}:{UNDERLYING}:DIVYIELD": 0.01,
        f"EQ:{CCY.value}:{UNDERLYING}:DIV":      dividend_curve,
    }
    return tq.MarketEnvironment(market=market)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_option(option_type: tq.OptionType) -> tq.VanillaOption:
    return tq.VanillaOption(
        CCY,
        start_date=EVALUATION_DATE,
        end_date=END_DATE,
        option_type=option_type,
        strike=STRIKE,
        underlying=UNDERLYING,
    )


def build_gbm(market_env: tq.MarketEnvironment, T: float) -> GeometricBrownianMotion:
    """Calibra un GBM con la vol ATM@STRIKE e il drift r-q dal mercato."""
    vol_surface = market_env.get_eq_vol_surface(UNDERLYING, ccy=CCY)
    disc_curve  = market_env.get_ir_curve(CCY)
    S0          = float(market_env.get_eq_spot(UNDERLYING, ccy=CCY))
    q           = float(market_env.get_eq_div_yield(UNDERLYING, ccy=CCY) or 0.0)

    # zero rate per la scadenza (approssimazione lineare sulla curva)
    end_date_obj = END_DATE
    df  = float(disc_curve.discount(end_date_obj))
    r   = -np.log(df) / T

    # vol implicita interpolata alla scadenza e allo strike
    sigma = float(vol_surface.volatility(strike=STRIKE, tenor=T))

    return GeometricBrownianMotion(mu=r - q, sigma=sigma, x0=S0)


def build_local_vol(market_env: tq.MarketEnvironment) -> LocalVolatilityModel:
    """Calibra un LocalVolatilityModel (Dupire) dalla superficie di vol."""
    vol_surface = market_env.get_eq_vol_surface(UNDERLYING, ccy=CCY)
    disc_curve  = market_env.get_ir_curve(CCY)
    S0          = float(market_env.get_eq_spot(UNDERLYING, ccy=CCY))
    q           = float(market_env.get_eq_div_yield(UNDERLYING, ccy=CCY) or 0.0)

    T_grid    = tf.constant(vol_surface.maturity,          dtype=tf.float32)
    K_grid    = tf.constant(vol_surface.strike,            dtype=tf.float32)
    iv_matrix = tf.constant(vol_surface.volatility_matrix, dtype=tf.float32)

    T_max = float(T_grid[-1].numpy())
    df    = float(disc_curve.discount(END_DATE))
    T_ref = float(tq.DayCounter(tq.DayCounterConvention.Actual365).year_fraction(
        EVALUATION_DATE, END_DATE
    ))
    r = -np.log(df) / T_ref

    return LocalVolatilityModel.from_implied_vol(
        iv_matrix=iv_matrix,
        T_grid=T_grid,
        K_grid=K_grid,
        S0=tf.constant(S0, dtype=tf.float32),
        r=tf.constant(r, dtype=tf.float32),
        q=tf.constant(q, dtype=tf.float32),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tq.Settings.evaluation_date = EVALUATION_DATE
    market_env = build_market(EVALUATION_DATE)

    daycounter = tq.DayCounter(tq.DayCounterConvention.Actual365)
    T = daycounter.year_fraction(EVALUATION_DATE, END_DATE)

    # ---- 1. Black-Scholes analitico (benchmark) ----------------------------
    bs_pricer = tq.BlackScholesPricer(dividend_model="continuous", use_implied_repo=True)

    call_bs = make_option(tq.OptionType.Call)
    put_bs  = make_option(tq.OptionType.Put)
    bs_pricer.price(call_bs, market_env)
    bs_pricer.price(put_bs,  market_env)
    bs_call = float(call_bs.price.numpy())
    bs_put  = float(put_bs.price.numpy())

    # r e q per la parita' put-call
    r_val = float(call_bs.risk_free_rate.numpy())
    q_val = 0.01
    df_val = float(call_bs.discount_factor.numpy())

    # ---- 2. GBM Monte Carlo ------------------------------------------------
    gbm_model = build_gbm(market_env, T)
    gbm_pricer = VanillaMCPricer(model=gbm_model, n_paths=N_PATHS, n_steps=N_STEPS, seed=SEED)

    call_gbm = make_option(tq.OptionType.Call)
    put_gbm  = make_option(tq.OptionType.Put)
    gbm_pricer.price(call_gbm, market_env)
    gbm_pricer.price(put_gbm,  market_env)
    gbm_call = float(call_gbm.price.numpy())
    gbm_put  = float(put_gbm.price.numpy())

    # ---- 3. Local Volatility Monte Carlo (Dupire) --------------------------
    lv_model  = build_local_vol(market_env)
    lv_pricer = VanillaMCPricer(model=lv_model, n_paths=N_PATHS, n_steps=N_STEPS, seed=SEED)

    call_lv = make_option(tq.OptionType.Call)
    put_lv  = make_option(tq.OptionType.Put)
    lv_pricer.price(call_lv, market_env)
    lv_pricer.price(put_lv,  market_env)
    lv_call = float(call_lv.price.numpy())
    lv_put  = float(put_lv.price.numpy())

    # ---- Confronto ---------------------------------------------------------
    print()
    print("=" * 64)
    print(f"  European option  K={STRIKE}  T={END_DATE}  S0=100  r={r_val:.3f}  q={q_val}")
    print("=" * 64)
    print(f"  {'Pricer / Modello':28s}  {'Call':>10s}  {'Put':>10s}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}")
    print(f"  {'Black-Scholes (analitico)':28s}  {bs_call:>10.4f}  {bs_put:>10.4f}")
    print(f"  {'GBM MC':28s}  {gbm_call:>10.4f}  {gbm_put:>10.4f}")
    print(f"  {'Local Vol MC (Dupire)':28s}  {lv_call:>10.4f}  {lv_put:>10.4f}")
    print("=" * 64)

    pcp_th = 100.0 * np.exp(-q_val * T) - STRIKE * np.exp(-r_val * T)
    for label, c, p in [("GBM MC", gbm_call, gbm_put), ("LV MC", lv_call, lv_put)]:
        err = (c - p) - pcp_th
        print(f"  Put-call parity [{label}]  C-P={c-p:+.4f}  teorico={pcp_th:+.4f}  err={err:+.6f}")

    # ── Sezione plot ─────────────────────────────────────────────────────────

    vol_surface = market_env.get_eq_vol_surface(UNDERLYING, ccy=CCY)
    strikes_np  = np.array(vol_surface.strike)
    T_idx  = int(np.argmin(np.abs(np.array(vol_surface.maturity) - T)))
    iv_row = np.array(vol_surface.volatility_matrix[T_idx])

    S0_tf = tf.constant(100.0, dtype=tf.float32)
    r_tf  = tf.constant(r_val, dtype=tf.float32)
    q_tf  = tf.constant(q_val, dtype=tf.float32)

    from tensorquant.pricers.black import blackscholes_calc

    # BS analitico su ogni strike
    bs_calls_arr = np.array([
        float(blackscholes_calc(
            S0_tf, tf.constant(K, tf.float32), r_tf,
            tf.constant(float(iv_row[i]), tf.float32),
            tf.constant(T, tf.float32), q_tf, tq.OptionType.Call,
        ).numpy())
        for i, K in enumerate(strikes_np)
    ])
    bs_puts_arr = np.array([
        float(blackscholes_calc(
            S0_tf, tf.constant(K, tf.float32), r_tf,
            tf.constant(float(iv_row[i]), tf.float32),
            tf.constant(T, tf.float32), q_tf, tq.OptionType.Put,
        ).numpy())
        for i, K in enumerate(strikes_np)
    ])

    # GBM MC su ogni strike (stesso seed, stessi path)
    tf.random.set_seed(SEED)
    T_max_gbm = T
    t_grid_gbm = tf.cast(tf.linspace(0.0, T_max_gbm, N_STEPS + 1)[1:], tf.float64)
    dw_gbm     = tf.cast(tf.random.normal([N_PATHS, N_STEPS]), tf.float64)
    paths_gbm  = gbm_model.evolve(t_grid_gbm, dw_gbm)
    t_np_gbm   = t_grid_gbm.numpy()
    idx_gbm    = int(np.argmin(np.abs(t_np_gbm - T)))
    S_T_gbm    = tf.cast(paths_gbm[:, idx_gbm], tf.float32)

    gbm_calls_arr = np.array([
        float(df_val * tf.reduce_mean(tf.maximum(S_T_gbm - K, 0.0)).numpy())
        for K in strikes_np
    ])
    gbm_puts_arr = np.array([
        float(df_val * tf.reduce_mean(tf.maximum(K - S_T_gbm, 0.0)).numpy())
        for K in strikes_np
    ])

    # LV MC su ogni strike (stesso seed, stessi path)
    T_max_lv  = float(lv_model._T_grid[-1].numpy())
    tf.random.set_seed(SEED)
    t_grid_lv = tf.linspace(0.0, T_max_lv, N_STEPS + 1)[1:]
    dw_lv     = tf.random.normal([N_PATHS, N_STEPS], dtype=tf.float32)
    paths_lv  = lv_model.evolve(t_grid_lv, dw_lv)
    t_np_lv   = t_grid_lv.numpy()
    idx_lv    = int(np.argmin(np.abs(t_np_lv - T)))
    S_T_lv    = paths_lv[:, idx_lv]

    lv_calls_arr = np.array([
        float(df_val * tf.reduce_mean(tf.maximum(S_T_lv - K, 0.0)).numpy())
        for K in strikes_np
    ])
    lv_puts_arr = np.array([
        float(df_val * tf.reduce_mean(tf.maximum(K - S_T_lv, 0.0)).numpy())
        for K in strikes_np
    ])

    # ── Figure 1: prezzi call e put (3 modelli) ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"European Vanilla -- K={STRIKE}, T~1Y, S0=100",
        fontsize=12, fontweight="bold",
    )
    colors = {"BS": "steelblue", "GBM": "darkorange", "LV": "tomato"}
    for ax, (mc_bs, mc_gbm, mc_lv, label) in zip(
        axes,
        [
            (bs_calls_arr, gbm_calls_arr, lv_calls_arr, "Call"),
            (bs_puts_arr,  gbm_puts_arr,  lv_puts_arr,  "Put"),
        ],
    ):
        ax.plot(strikes_np, mc_bs,  "o-",  color=colors["BS"],  label="BS (analitico)", lw=1.8)
        ax.plot(strikes_np, mc_gbm, "^--", color=colors["GBM"], label="GBM MC",         lw=1.8)
        ax.plot(strikes_np, mc_lv,  "s--", color=colors["LV"],  label="Local Vol MC",   lw=1.8)
        ax.axvline(100.0,   color="grey",  ls=":",  lw=1,         label="ATM")
        ax.axvline(STRIKE,  color="green", ls="--", lw=1, alpha=0.7, label=f"K={STRIKE}")
        ax.set_title(f"European {label}", fontsize=11)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Prezzo")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = OUT_DIR / "lv_vanilla_prices.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"\nPlot prezzi  -> {out1}")

    # ── Figure 2: smile IV back-out da MC (GBM vs LV vs mercato) ─────────
    def implied_vol_nr(S, K, T_opt, price, r, q, is_call=True):
        import math
        phi = 1.0 if is_call else -1.0
        sig = 0.20
        S, K, r, q = float(S), float(K), float(r), float(q)
        for _ in range(60):
            sqT = math.sqrt(T_opt)
            d1  = (math.log(S / K) + (r - q + 0.5 * sig**2) * T_opt) / (sig * sqT)
            d2  = d1 - sig * sqT
            nc  = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            c   = phi * (S * math.exp(-q * T_opt) * nc(phi * d1)
                         - K * math.exp(-r * T_opt) * nc(phi * d2))
            v   = S * math.exp(-q * T_opt) * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) * sqT
            diff = c - price
            if abs(diff) < 1e-8:
                break
            sig = max(sig - diff / (v + 1e-12), 1e-4)
        return sig

    gbm_iv = np.array([
        implied_vol_nr(100.0, K, T, gbm_calls_arr[i], r_val, q_val)
        for i, K in enumerate(strikes_np)
    ])
    lv_iv = np.array([
        implied_vol_nr(100.0, K, T, lv_calls_arr[i],  r_val, q_val)
        for i, K in enumerate(strikes_np)
    ])

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    fig2.suptitle("Smile IV -- mercato vs GBM MC vs Local Vol MC  (T~1Y)",
                  fontsize=12, fontweight="bold")
    ax2.plot(strikes_np, iv_row * 100,  "o-",  color=colors["BS"],  label="IV mercato (input)", lw=1.8)
    ax2.plot(strikes_np, gbm_iv  * 100, "^--", color=colors["GBM"], label="IV da GBM MC (piatta)", lw=1.8)
    ax2.plot(strikes_np, lv_iv   * 100, "s--", color=colors["LV"],  label="IV da Local Vol MC",    lw=1.8)
    ax2.axvline(100.0,  color="grey",  ls=":",  lw=1, label="ATM")
    ax2.axvline(STRIKE, color="green", ls="--", lw=1, alpha=0.7, label=f"K={STRIKE}")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Vol implicita (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = OUT_DIR / "lv_vanilla_smile.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Plot smile   -> {out2}")

    # ── Figure 3: distribuzione terminale a confronto ─────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.suptitle("Distribuzione terminale S_T  (T~1Y)", fontsize=12, fontweight="bold")

    ax3.hist(S_T_gbm.numpy(), bins=100, density=True, alpha=0.45,
             color=colors["GBM"], label="GBM")
    ax3.hist(S_T_lv.numpy(),  bins=100, density=True, alpha=0.45,
             color=colors["LV"],  label="Local Vol (Dupire)")
    ax3.axvline(100.0,  color="dimgrey", ls=":",  lw=1.5, label="S0=100")
    ax3.axvline(STRIKE, color="green",   ls="--", lw=1.5, label=f"K={STRIKE}")
    ax3.set_xlabel("S_T")
    ax3.set_ylabel("Densita'")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.25)
    plt.tight_layout()
    out3 = OUT_DIR / "lv_vanilla_terminal_dist.png"
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"Plot dist    -> {out3}")


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float32")
    main()
