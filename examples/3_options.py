"""
2_options.py -- Prezzatura di vanilla options (European BS + American BS93).

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

import tensorflow as tf

import tensorquant as tq
from market_data import EVALUATION_DATE as _EVAL_DATE_GEN, build_market_environment

# ══════════════════════════════════════════════════════════════════════════════
# INPUT DI PRICING
# ══════════════════════════════════════════════════════════════════════════════

SOURCE_MKT_DATA = "Real"      # 'Real' | 'Generated'
TICKER          = "SX5E"      # Real: ISP | SX5E | AMZN | BABA
                               # Generated: TICKER1 | TICKER2

MONEYNESS       = 1.00        # K = MONEYNESS * spot  (es. 0.90 OTM put, 1.10 OTM call)
MATURITY_YEARS  = 1.0         # scadenza in anni

# Parametri per SOURCE_MKT_DATA = 'Real'
EVALUATION_DATE_REAL = date(2026, 5, 26)
DATA_PATH            = "W:/Derivative/EquityDerivatives/AndreaC/dev/mdm/datastore/"
CALIBRATION_SET      = "142910"

# ══════════════════════════════════════════════════════════════════════════════

OUT_DIR = Path(__file__).resolve().parent / "out"
SEP = "=" * 60
sep = "-" * 60


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

    market_env, EVALUATION_DATE = load_market()

    ccy    = market_env.get_currency(TICKER)
    spot   = float(market_env.get_eq_spot(TICKER, ccy=ccy))
    strike = round(MONEYNESS * spot, 4)
    expiry = EVALUATION_DATE + timedelta(days=int(round(MATURITY_YEARS * 365)))

    # ── Riepilogo inputs ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  INPUTS")
    print(f"{SEP}")
    print(f"  Source      : {SOURCE_MKT_DATA}")
    print(f"  Ticker      : {TICKER} ({ccy.name})")
    print(f"  Eval date   : {EVALUATION_DATE}")
    print(f"  Spot        : {spot:.4f}")
    print(f"  Moneyness   : {MONEYNESS:.2f}  →  Strike = {strike:.4f}")
    print(f"  Maturity    : {MATURITY_YEARS:.2f}Y  →  Expiry = {expiry}")

    # ── Pricer ────────────────────────────────────────────────────────────────
    pricer = tq.BlackScholesPricer(dividend_model="discrete", use_implied_repo=True)

    def make_option(opt_type, exercise=tq.ExerciseType.European):
        return tq.VanillaOption(
            ccy,
            start_date=EVALUATION_DATE,
            end_date=expiry,
            option_type=opt_type,
            strike=strike,
            underlying=TICKER,
            exercise_type=exercise,
        )

    # chiave spot nel market dict
    spot_key = f"EQ:{ccy.name}:{TICKER}:SPOT"

    def get_greeks(opt) -> dict:
        """Vega / Rho / Theta: autodiff dal tape (pricer._sigma, _r, _t sono tf.Variable).
           Delta / Gamma: bump ±0.1% sullo spot nel market dict (pricer._s per il
           modello discreto è s_net = s - PV_div, un Tensor senza .assign)."""
        pricer.price(opt, market_env, autodiff=True)
        tape = pricer.tape

        # Vega, Rho, Theta dal tape in una sola chiamata
        grads = tape.gradient(opt.price, [pricer._sigma, pricer._r, pricer._t])
        vega  = float(grads[0].numpy()) * 0.01   # per +1% vol
        rho   = float(grads[1].numpy()) * 0.0001  # per +1bp rate
        theta = float(grads[2].numpy()) / 365.0  # per-day

        # Delta e Gamma: bump spot nel market
        s0 = float(market_env._market[spot_key])
        dS = s0 * 0.001   # 0.1% bump

        def _reprice(s_val):
            market_env._market[spot_key] = s_val
            o = make_option(opt.option_type, opt.exercise_type)
            pricer.price(o, market_env)
            return float(o.price.numpy())

        px_mid = float(opt.price.numpy())
        px_up  = _reprice(s0 + dS)
        px_dn  = _reprice(s0 - dS)
        market_env._market[spot_key] = s0   # ripristina

        delta = (px_up - px_dn) / (2.0 * dS)
        gamma = (px_up - 2.0 * px_mid + px_dn) / dS ** 2

        return {
            "price": px_mid,
            "delta": delta,
            "gamma": gamma,
            "vega":  vega,
            "theta": theta,
            "rho":   rho,
        }

    eu_call = make_option(tq.OptionType.Call)
    eu_put  = make_option(tq.OptionType.Put)
    am_call = make_option(tq.OptionType.Call, tq.ExerciseType.American)
    am_put  = make_option(tq.OptionType.Put,  tq.ExerciseType.American)

    gk_ec = get_greeks(eu_call)
    gk_ep = get_greeks(eu_put)
    gk_ac = get_greeks(am_call)
    gk_ap = get_greeks(am_put)

    # ── Market inputs estratti dal pricer ─────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  MARKET INPUTS  (da BlackScholesPricer)")
    print(f"{SEP}")
    print(f"  Spot net S* : {float(eu_call.spot_net.numpy()):.6f}  "
          f"(PV div = {float(eu_call.pv_discrete_dividends.numpy()):.6f})")
    print(f"  Repo margin : {float(eu_call.repo_margin.numpy()):.6f}")
    print(f"  Risk-free r : {float(eu_call.risk_free_rate.numpy()):.6f}")
    print(f"  Discount df : {float(eu_call.discount_factor.numpy()):.6f}")
    print(f"  Implied vol : {float(eu_call.volatility.numpy()):.4f}  "
          f"({float(eu_call.volatility.numpy())*100:.2f}%)")
    print(f"  T to expiry : {float(eu_call.time_to_maturity.numpy()):.4f}Y  (expiry: {expiry})")
    print(f"  Forward     : {float(eu_call.forward.numpy()):.6f}")

    # ── Prezzi e greche ───────────────────────────────────────────────────────
    fwd     = float(eu_call.forward.numpy())
    df      = float(eu_call.discount_factor.numpy())
    pcp_err = gk_ec["price"] - gk_ep["price"] - (fwd - strike) * df

    hdr = f"  {'':25s}  {'Price':>10s}  {'% spot':>7s}  {'Delta':>8s}  {'Gamma':>8s}  {'Vega':>8s}  {'Theta':>8s}  {'Rho':>8s}"
    row_sep = f"  {'-'*95}"

    def _fmt(v, w=8, d=4):
        return f"{'n/a':>{w}}" if (v is None or v != v) else f"{v:>{w}.{d}f}"

    def fmt_row(label, gk):
        return (
            f"  {label:25s}  {gk['price']:>10.4f}  {gk['price']/spot*100:>6.3f}%"
            f"  {_fmt(gk['delta'])}  {_fmt(gk['gamma'])}"
            f"  {_fmt(gk['vega'])}  {_fmt(gk['theta'])}  {_fmt(gk['rho'])}"
        )

    print(f"\n{SEP}")
    print(f"  PREZZI E GRECHE  (Vega per +1% vol | Theta per-day | Rho per +1bp rate)")
    print(f"{SEP}")
    print(hdr)
    print(row_sep)
    print(fmt_row("European Call (BS)",  gk_ec))
    print(fmt_row("European Put  (BS)",  gk_ep))
    print(fmt_row("American Call (BS93)", gk_ac))
    print(fmt_row("American Put  (BS93)", gk_ap))
    print(f"\n  Put-Call Parity error : {pcp_err:.2e}")


if __name__ == "__main__":
    main()
