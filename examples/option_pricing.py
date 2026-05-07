from datetime import date
from pathlib import Path
import sys

import tensorflow as tf

# Ensure local package is imported when running `python examples/...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorquant as tq
from tensorquant.markethandles.dividendcurve import DividendCurve


def build_static_market_environment(evaluation_date: date) -> tq.MarketEnvironment:
    daycounter = tq.DayCounter(tq.DayCounterConvention.Actual365)

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
        strike=[4.0, 5.0, 6.0],
        maturity=[0.25, 0.75, 1.50],
        volatility_matrix=[
            [0.34, 0.32, 0.31],
            [0.33, 0.31, 0.30],
            [0.32, 0.30, 0.29],
        ],
    )

    dividend_curve = DividendCurve(
        reference_date=evaluation_date,
        ex_dates=[date(2026, 6, 15), date(2026, 9, 15)],
        amounts=[0.10, 0.11],
        currency=tq.Currency.EUR,
        daycounter_convention=tq.DayCounterConvention.Actual365,
    )

    market = {
        "IR:EUR:ESTR:SPOT": estr_curve,
        "EQ:EUR:TESTCTP:SPOT": 5.174,
        "EQ:EUR:TESTCTP:VOL": vol_surface,
        "EQ:EUR:TESTCTP:REPO": -0.035,
        "EQ:EUR:TESTCTP:DIVYIELD": 0.0,
        "EQ:EUR:TESTCTP:DIV": dividend_curve,
    }
    return tq.MarketEnvironment(market=market)


def main() -> None:
    evaluation_date = date(2026, 3, 31)
    tq.Settings.evaluation_date = evaluation_date
    market_env = build_static_market_environment(evaluation_date)

    option = tq.VanillaOption(
        tq.Currency.EUR,
        start_date=evaluation_date,
        end_date=date(2026, 12, 18),
        option_type=tq.OptionType.Call,
        strike=5.0,
        underlying="TESTCTP",
    )

    black_pricer = tq.BlackScholesPricer(dividend_model="discrete", use_implied_repo=True)
    black_pricer.price(option, market_env)

    print("--- European Call (Black-Scholes) ---")
    print("spot:", option.spot.numpy())
    print("spot_net:", option.spot_net.numpy())
    print("volatility:", option.volatility.numpy())
    print("forward:", option.forward.numpy())
    print("price:", option.price.numpy())

    american_put = tq.VanillaOption(
        tq.Currency.EUR,
        start_date=evaluation_date,
        end_date=date(2026, 12, 18),
        option_type=tq.OptionType.Put,
        strike=5.0,
        underlying="TESTCTP",
        exercise_type=tq.ExerciseType.American,
    )
    black_pricer.price(american_put, market_env)

    european_put = tq.VanillaOption(
        tq.Currency.EUR,
        start_date=evaluation_date,
        end_date=date(2026, 12, 18),
        option_type=tq.OptionType.Put,
        strike=5.0,
        underlying="TESTCTP",
    )
    black_pricer.price(european_put, market_env)

    print("\n--- American Put (Bjerksund-Stensland 1993) ---")
    print("price:", american_put.price.numpy())
    print("--- European Put (Black-Scholes) ---")
    print("price:", european_put.price.numpy())
    print("early exercise premium:", (american_put.price - european_put.price).numpy())


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    main()
