from pathlib import Path
import sys

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorquant as tq

# --- market parameters (flat curves + constant vol) -------------------------
DISC_RATE  = 0.021   # ESTR flat rate
EUR6M_RATE = 0.026   # EUR 6M flat rate
VOL        = 0.25    # Black constant volatility
SPOT       = 100.0   # initial underlying spot


def build_market_environment(ref_date) -> tq.MarketEnvironment:
    disc_curve  = tq.FlatCurve(ref_date, DISC_RATE,  tq.DayCounterConvention.Actual360)
    eur6m_curve = tq.FlatCurve(ref_date, EUR6M_RATE, tq.DayCounterConvention.Actual360)
    vol         = tq.BlackConstantVolatility(ref_date, volatility=VOL)

    market = {
        "IR:EUR:ESTR:SPOT":   disc_curve,
        "IR:EUR:6M:SPOT":     eur6m_curve,
        "EQ:EUR:DEFAULT:VOL": vol,
    }
    return tq.MarketEnvironment(market)


def build_autocallable(ref_date) -> tq.AutocallableOption:
    calendar     = tq.TARGET()
    schedule_gen = tq.ScheduleGenerator(
        calendar, tq.BusinessDayConvention.ModifiedFollowing
    )

    currency   = tq.Currency.EUR
    notional   = 100e6
    start_date = ref_date
    end_date   = calendar.advance(
        ref_date, 5, tq.TimeUnit.Years, tq.BusinessDayConvention.ModifiedFollowing
    )
    strike = 100

    coupon_fixing_dates  = schedule_gen.generate(ref_date, end_date, 6, tq.TimeUnit.Months)[1:]
    coupon_payment_dates = coupon_fixing_dates
    coupon_rates         = [4.0] * len(coupon_fixing_dates)
    coupon_barriers      = [80]  * len(coupon_fixing_dates)
    memory               = False

    first_autocall_date   = calendar.advance(
        ref_date, 1, tq.TimeUnit.Years, tq.BusinessDayConvention.ModifiedFollowing
    )
    autocall_fixing_dates  = schedule_gen.generate(
        first_autocall_date, end_date, 6, tq.TimeUnit.Months
    )
    autocall_payment_dates = autocall_fixing_dates
    autocall_barrier       = [100] * len(autocall_fixing_dates)

    payoff_barrier       = 80.0
    payoff_participation = 1.0
    payoff_type          = "put"

    return tq.AutocallableOption(
        ccy=currency,
        notional=notional,
        start_date=start_date,
        end_date=end_date,
        strike=strike,
        coupon_fixing_dates=coupon_fixing_dates,
        coupon_payment_dates=coupon_payment_dates,
        coupon_rates=coupon_rates,
        coupon_barriers=coupon_barriers,
        memory=memory,
        autocall_fixing_dates=autocall_fixing_dates,
        autocall_payment_dates=autocall_payment_dates,
        autocall_barrier=autocall_barrier,
        payoff_barrier=payoff_barrier,
        payoff_participation=payoff_participation,
        payoff_type=payoff_type,
    )


def build_model() -> tq.GeometricBrownianMotion:
    """GBM under risk-neutral measure: drift = disc_rate, diffusion = vol."""
    return tq.GeometricBrownianMotion(mu=DISC_RATE, sigma=VOL, x0=SPOT)


def main() -> None:
    ref_date   = tq.Settings.evaluation_date
    market_env = build_market_environment(ref_date)
    option_leg = build_autocallable(ref_date)

    model  = build_model()
    pricer = tq.AutocallableMCPricer(
        model=model,
        n_paths=20_000,
        seed=12,
        daycounter_convention=tq.DayCounterConvention.Actual360,
        discretization_months=1,
    )

    pricer.price(option_leg, market_env)

    print("--- Autocallable Phoenix (Monte Carlo) ---")
    print(f"  option leg PV : {option_leg.price.numpy():,.0f}")


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    main()
