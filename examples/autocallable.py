"""
autocallable.py -- Phoenix autocallable on TICKER1 (marketdata/ snapshot), investor's side.

Local vol model built from the market (default of ``AutocallableMCPricer``), digitals smoothed
with eps = 1% of the strike so the AAD greeks see the barriers. Prints price (coupons - put),
greeks by AAD and the smoothing bias (eps = 1% vs hard payoff).
"""

import io
import os
import sys
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf

import tensorquant as tq

MARKET_DATA     = Path(__file__).resolve().parents[1] / "marketdata"
SNAPSHOT_DATE   = None        # date(yyyy, m, d) | None = latest snapshot
CALIBRATION_SET = None        # e.g. "2130"      | None = latest of the day
TICKER          = "TICKER1"   # TICKER1-30 EUR | TICKER31-40 GBP | TICKER41-50 USD
NOTIONAL        = 1e6
N_PATHS         = 50_000
SEED            = 42


def build_autocallable(ref_date, ccy: tq.Currency, strike: float) -> tq.AutocallableOption:
    """5Y phoenix: semi-annual 4% coupons (barrier 80%), autocall from 1Y at 100%, put barrier 80%."""
    calendar = tq.TARGET()
    mf = tq.BusinessDayConvention.ModifiedFollowing
    schedule_gen = tq.ScheduleGenerator(calendar, mf)
    end_date = calendar.advance(ref_date, 5, tq.TimeUnit.Years, mf)
    coupon_fixing_dates = schedule_gen.generate(ref_date, end_date, 6, tq.TimeUnit.Months)[1:]
    autocall_fixing_dates = schedule_gen.generate(
        calendar.advance(ref_date, 1, tq.TimeUnit.Years, mf), end_date, 6, tq.TimeUnit.Months
    )
    return tq.AutocallableOption(
        ccy=ccy,
        notional=NOTIONAL,
        start_date=ref_date,
        end_date=end_date,
        strike=strike,
        coupon_fixing_dates=coupon_fixing_dates,
        coupon_payment_dates=coupon_fixing_dates,
        coupon_rates=[4.0],
        coupon_barriers=[80.0],
        memory=False,
        autocall_fixing_dates=autocall_fixing_dates,
        autocall_payment_dates=autocall_fixing_dates,
        autocall_barrier=[100.0],
        payoff_barrier=80.0,
        payoff_participation=1.0,
        payoff_type="put",
        underlying=TICKER,
    )


def main() -> None:
    market_env = tq.MarketEnvironment.from_folder(MARKET_DATA, SNAPSHOT_DATE, CALIBRATION_SET)
    tq.Settings.evaluation_date = market_env.evaluation_date
    eq = market_env.get_equity(TICKER)
    strike = round(float(eq.spot), 2)   # initial fixing = spot
    print(f"snapshot marketdata/{market_env.evaluation_date:%Y%m%d}/{market_env.calibration_set}   "
          f"{TICKER} spot {float(eq.spot):.2f} ({eq.currency.name}), strike {strike:.2f}")

    product = build_autocallable(market_env.evaluation_date, eq.currency, strike)
    pricer = tq.AutocallableMCPricer(n_paths=N_PATHS, seed=SEED)        # local vol, eps = 1%
    pricer.price(product, market_env, autodiff=True)
    factors = eq.risk_factors()
    grads = pricer.tape.gradient(product.price, factors)
    ccy = eq.currency.name

    print(f"\n--- Autocallable phoenix, investor's side (local vol, {N_PATHS:,} paths) ---")
    print(f"  NPV          : {float(product.price):>12,.2f}  ({float(product.price) / NOTIONAL:.4%} of notional)"
          f"  +/- {float(product.standard_error):,.2f}")
    print(f"  coupons      : {float(product.coupon_pv) * NOTIONAL:>12,.2f}")
    print(f"  put (short)  : {-float(product.put_pv) * NOTIONAL:>12,.2f}")

    delta = float(grads[f"EQ:{ccy}:{TICKER}:SPOT"][0])
    vega = float(tf.reduce_sum(grads[f"EQ:{ccy}:{TICKER}:VOL"][0])) / 100
    rho = sum(float(g) for g in grads[eq.discount_curve.name] if g is not None) / 1e4
    div = float(tf.reduce_sum(grads[f"EQ:{ccy}:{TICKER}:DIV"][0]))
    print("\n--- Greeks (AAD) ---")
    print(f"  delta        : {delta:>12,.2f}  per 1 unit of spot  ({delta * float(eq.spot) / 100:,.2f} per 1%)")
    print(f"  vega         : {vega:>12,.2f}  per 1 vol point, parallel on all nodes")
    print(f"  rho          : {rho:>12,.2f}  per 1bp parallel on the discount curve")
    print(f"  dividends    : {div:>12,.2f}  per +1 on every dividend amount")

    hard = build_autocallable(market_env.evaluation_date, eq.currency, strike)
    tq.AutocallableMCPricer(n_paths=N_PATHS, seed=SEED, barrier_smoothing=0.0).price(hard, market_env)
    print(f"\n  hard payoff (eps = 0): {float(hard.price):,.2f}   smoothing bias (eps = 1%): "
          f"{float(product.price) - float(hard.price):+,.2f}")


if __name__ == "__main__":
    main()
