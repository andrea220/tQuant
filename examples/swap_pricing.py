from datetime import date

import tensorflow as tf

import tensorquant as tq


def build_static_market_environment(evaluation_date: date) -> tq.MarketEnvironment:
    estr_curve = tq.RateCurve(
        reference_date=evaluation_date,
        pillars=[0.5, 1.0, 2.0, 5.0, 10.0],
        rates=[0.0180, 0.0190, 0.0205, 0.0220, 0.0230],
        interp="LINEAR",
        daycounter_convention=tq.DayCounterConvention.Actual360,
    )

    eur6m_curve = tq.RateCurve(
        reference_date=evaluation_date,
        pillars=[0.5, 1.0, 2.0, 5.0, 10.0],
        rates=[0.0200, 0.0210, 0.0225, 0.0240, 0.0250],
        interp="LINEAR",
        daycounter_convention=tq.DayCounterConvention.Actual360,
    )

    market = {
        "IR:EUR:ESTR:SPOT": estr_curve,
        "IR:EUR:6M:SPOT": eur6m_curve,
    }
    return tq.MarketEnvironment(market=market)


def main() -> None:
    evaluation_date = date(2026, 1, 5)
    market_env = build_static_market_environment(evaluation_date)

    settlement_delay = 3
    period_fixed_leg = "1Y"
    period_float_leg = "6M"
    notional = 100e6
    calendar = tq.TARGET()

    eur6m_index = tq.IborIndex(
        calendar, 6, tq.TimeUnit.Months, tq.Currency.EUR, fixing_days=2
    )
    eur6m_index.add_fixing(date(2026, 1, 6), 0.0215)

    irs_eur6m_generator = tq.SwapGenerator(
        tq.Currency.EUR,
        settlement_delay,
        period_fixed_leg,
        period_float_leg,
        tq.BusinessDayConvention.ModifiedFollowing,
        notional,
        tq.DayCounterConvention.Actual360,
        tq.DayCounterConvention.Actual360,
        calendar,
        eur6m_index,
    )

    swap = irs_eur6m_generator.build(
        trade_date=evaluation_date,
        quote=0.0220,
        term="5Y",
    )

    swap_engine = tq.SwapPricer()
    swap_engine.price(swap, market_env, autodiff=True)

    gradients = swap_engine.tape.gradient(
        swap.price, market_env.get_ir_curve(tq.Currency.EUR, "6M")._rates
    )
    gradient_values = [float(g.numpy()) if g is not None else None for g in gradients]

    print(f"Swap NPV: {float(swap.price.numpy()):,.2f}")
    print("dNPV/d(6M curve nodes):", gradient_values)


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    main()
