import datetime
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .product import Product
from ..markethandles.utils import Currency


class AutocallableOption(Product):
    """
    Equity autocallable option instrument.

    The class holds only the **trade terms** (schedules and payoff parameters); the pricing is done by a pricer
    (:class:`AutocallableMCPricer`).

    Conventions:
    - All the levels are in **percent of the strike (initial fixing)**:
      - ``autocall_barrier = 100`` -> 100% of the strike;
      - ``coupon_barriers[i] = 70`` -> 70% of the strike;
      - ``coupon_rates[i] = 5`` -> a 5% coupon.
    - Step-down autocall: ``autocall_step`` lowers the autocall barrier at each autocall observation and
      ``autocall_floor`` bounds it from below: at the observation ``i`` (from 0)
      ``max(autocall_barrier[i] - i * autocall_step, autocall_floor)`` (``autocall_barrier = 100``, ``autocall_step = 1``,
      ``autocall_floor = 90`` -> 100, 99, 98, ..., 90, 90).
    - The simulation grid is built by the pricer (fixing dates merged with the model's time steps).
    """

    def __init__(
        self,
        ccy: Currency,
        notional: float,
        start_date: datetime.date,
        end_date: datetime.date,
        strike: Optional[float],
        # --- COUPON LEG ---
        coupon_fixing_dates: Sequence[datetime.date],
        coupon_payment_dates: Sequence[datetime.date],
        coupon_rates: Sequence[float],
        coupon_barriers: Sequence[float],
        memory: bool,
        # --- AUTOCALL LEG ---
        autocall_fixing_dates: Sequence[datetime.date],
        autocall_payment_dates: Sequence[datetime.date],
        autocall_barrier: Sequence[float],
        # --- FINAL REDEMPTION (OPTION ONLY) ---
        payoff_barrier: float,
        payoff_participation: float = 1.0,
        payoff_type: str = "put",
        underlying: Optional[str] = None,
        put_strike: float = 100.0,
        strike_date: Optional[datetime.date] = None,
        autocall_step: float = 0.0,
        autocall_floor: Optional[float] = None,
        payment_ccy: Optional[Currency] = None,
        fx_mode: str = "quanto",
    ) -> None:
        """
        Initialize an AutocallableOption.

        All the schedules (coupon / autocall fixing and payment dates) are given by the caller, built for instance
        with ``ScheduleGenerator``.

        Forward start: with a ``strike_date`` after ``start_date`` the strike (initial level) is the spot fixed on that
        date, so ``strike`` is ``None``; every barrier and the put strike are in % of it, and the coupon and autocall
        fixings must follow it. Otherwise ``strike`` is the initial level, already known.

        Step-down: ``autocall_step`` (% of the strike, >= 0) is subtracted from the autocall barrier at each autocall
        observation after the first, ``autocall_floor`` (% of the strike, optional) is its lowest level; the
        resulting barriers are ``autocall_barrier`` (one per observation).

        Quanto: with a ``payment_ccy`` other than ``ccy`` (the underlying's currency) the coupons and the put, in % of
        the notional, are paid in ``payment_ccy`` (a fixed rate): see :class:`AutocallableMCPricer`. Composite
        (``fx_mode="composite"``): the underlying is ``S X`` (``X`` the value of one unit of ``ccy`` in
        ``payment_ccy``), the strike its initial level and every barrier in % of it.

        Raises:
            ValueError: If the strike is missing for a trade struck on or before ``start_date``, given for a future
                strike date, or a fixing is not after the strike date; if ``autocall_step`` is negative or
                ``autocall_floor`` is above the first autocall barrier.
        """
        super().__init__(ccy=ccy, start_date=start_date, end_date=end_date, payment_ccy=payment_ccy, fx_mode=fx_mode)

        forward_start = strike_date is not None and strike_date > start_date
        if forward_start and strike is not None:
            raise ValueError("forward start: the strike is the spot fixed on the strike date, do not give a level")
        if strike is None and strike_date is None:
            raise ValueError("strike missing: give the initial level or a strike date")
        if strike_date is not None:
            early = [d for d in list(coupon_fixing_dates) + list(autocall_fixing_dates) if d <= strike_date]
            if early:
                raise ValueError(f"coupon / autocall fixings on or before the strike date {strike_date}: {early[0]}")
        self.notional = notional
        self.strike = None if strike is None else float(strike)
        self.strike_date = strike_date
        self.underlying = underlying

        # schedules
        self.coupon_fixing_dates: List[datetime.date] = list(coupon_fixing_dates)
        self.coupon_payment_dates: List[datetime.date] = list(coupon_payment_dates)
        self.autocall_fixing_dates: List[datetime.date] = list(autocall_fixing_dates)
        self.autocall_payment_dates: List[datetime.date] = list(autocall_payment_dates)

        # consistent lengths
        n_coupons = len(self.coupon_fixing_dates)
        if len(self.coupon_payment_dates) != n_coupons:
            raise ValueError(
                "coupon_fixing_dates and coupon_payment_dates must have the same length "
                f"({n_coupons} vs {len(self.coupon_payment_dates)})"
            )

        if len(self.autocall_fixing_dates) != len(self.autocall_payment_dates):
            raise ValueError(
                "autocall_fixing_dates and autocall_payment_dates must have the same length "
                f"({len(self.autocall_fixing_dates)} vs {len(self.autocall_payment_dates)})"
            )

        # scalar payoff parameters
        self.payoff_barrier = float(payoff_barrier)
        self.payoff_participation = float(payoff_participation)
        self.put_strike = float(put_strike)       # strike of the redemption put, % of the strike (initial level)
        self.memory = memory

        # coupon and barrier vectors (a single value is broadcast)
        def _broadcast(x: Sequence[float], name: str, n: int) -> List[float]:
            if len(x) == 1 and n > 1:
                return [float(x[0])] * n
            if len(x) != n:
                raise ValueError(
                    f"{name} length ({len(x)}) must be 1 or equal to target length ({n})"
                )
            return [float(v) for v in x]

        n_coupons = len(self.coupon_fixing_dates)
        n_autocall = len(self.autocall_fixing_dates)

        self.coupon_rates: List[float] = _broadcast(coupon_rates, "coupon_rates", n_coupons)
        self.coupon_barriers: List[float] = _broadcast(coupon_barriers, "coupon_barriers", n_coupons)
        barriers = _broadcast(autocall_barrier, "autocall_barrier", n_autocall)
        if autocall_step < 0:
            raise ValueError(f"autocall_step must be >= 0, got {autocall_step}")
        if autocall_floor is not None and barriers and autocall_floor > barriers[0]:
            raise ValueError(f"autocall_floor {autocall_floor} above the first autocall barrier {barriers[0]}")
        self.autocall_step = float(autocall_step)
        self.autocall_floor = None if autocall_floor is None else float(autocall_floor)
        stepped = [b - i * self.autocall_step for i, b in enumerate(barriers)]          # step-down
        self.autocall_barrier: List[float] = (stepped if self.autocall_floor is None
                                              else [max(b, self.autocall_floor) for b in stepped])

        # final payoff (derivative only, no capital redemption)
        self.redemption_payoff: Optional[Callable[..., float]] = None
        if payoff_type.lower() == "put":
            self.redemption_payoff = self.put_redemption

    @property
    def maturity_date(self) -> datetime.date:
        """Alias of ``end_date``, clearer for equity products."""
        return self.end_date

    def schedule_dataframe(self) -> "pd.DataFrame":
        """
        The coupon and autocall schedules **side by side**: a date that is both a coupon and an autocall date is on
        one row.

        Columns:
            - fixing_date, payment_date: the common key
            - coupon_rate_pct, coupon_barrier_pct
            - autocall_barrier_pct
            - memory (the same for the whole product)
            - payoff_barrier_pct
        """
        rows: List[dict] = []

        # (fixing, payment) keys of each leg
        coupon_keys = list(zip(self.coupon_fixing_dates, self.coupon_payment_dates))
        autocall_keys = list(zip(self.autocall_fixing_dates, self.autocall_payment_dates))

        all_keys = sorted(set(coupon_keys) | set(autocall_keys))

        for key in all_keys:
            fix, pay = key

            try:
                i_c = coupon_keys.index(key)
            except ValueError:
                i_c = None

            try:
                j_a = autocall_keys.index(key)
            except ValueError:
                j_a = None

            rows.append(
                {
                    "fixing_date": fix,
                    "payment_date": pay,
                    "coupon_rate_pct": self.coupon_rates[i_c] if i_c is not None else None,
                    "coupon_barrier_pct": self.coupon_barriers[i_c] if i_c is not None else None,
                    "autocall_barrier_pct": self.autocall_barrier[j_a] if j_a is not None else None,
                    "memory": self.memory,
                }
            )

        df = pd.DataFrame(rows)

        # payoff_barrier applies to the last row (maturity) only
        if not df.empty:
            df["payoff_barrier_pct"] = None
            df.loc[df.index[-1], "payoff_barrier_pct"] = self.payoff_barrier

        return df

    @staticmethod
    def put_redemption(S_T, strike: float, partecipation: float):
        """Capital-at-risk put as a fraction of notional: ``participation * max(K - S_T, 0) / K``.

        Positive value; the investor is short it (``AutocallableMCPricer`` subtracts it).
        """
        return np.maximum(strike - S_T, 0) / strike * partecipation
