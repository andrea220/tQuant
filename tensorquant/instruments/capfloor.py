from __future__ import annotations

import datetime

from .product import Product
from ..flows.floatingcoupon import FloatingRateLeg
from ..index.curverateindex import IborIndex
from ..markethandles.utils import Currency, OptionType
from ..timehandles.daycounter import DayCounter


class CapFloor(Product):
    """Cap (``OptionType.Call``) or floor (``OptionType.Put``) on an Ibor index: a strip of caplets / floorlets.

    Each period of the underlying :class:`FloatingRateLeg` pays at its payment date
    ``N * tau * max(phi (L - K), 0)``, ``L`` the index fixing for the period (``phi = +1`` cap,
    ``-1`` floor) and ``tau`` the accrual year fraction.

    Args:
        ccy (Currency): Currency.
        start_dates, end_dates, pay_dates (list[date]): Caplet accrual periods and payment dates.
        strike (float): Strike rate (decimal).
        notional (float): Notional.
        day_counter (DayCounter): Accrual day count.
        index (IborIndex): Underlying index (name ``CCY:TENOR``, fixings).
        option_type (OptionType): ``Call`` = cap, ``Put`` = floor.
    """

    def __init__(
        self,
        ccy: Currency,
        start_dates: list[datetime.date],
        end_dates: list[datetime.date],
        pay_dates: list[datetime.date],
        strike: float,
        notional: float,
        day_counter: DayCounter,
        index: IborIndex,
        option_type: OptionType = OptionType.Call,
    ) -> None:
        if not start_dates:
            raise ValueError("A cap/floor needs at least one period")
        super().__init__(ccy, start_dates[0], end_dates[-1])
        n = len(pay_dates)
        self._leg = FloatingRateLeg(pay_dates, start_dates, end_dates, [notional] * n, [1] * n, [0] * n,
                                    index, day_counter)
        self._strike = float(strike)
        self._notional = float(notional)
        self._index = index
        self._option_type = option_type

    @property
    def leg(self) -> FloatingRateLeg:
        return self._leg

    @property
    def caplets(self) -> list:
        """Floating coupons of the periods (fixing date, accrual, payment date)."""
        return self._leg.leg_flows

    @property
    def strike(self) -> float:
        return self._strike

    @property
    def notional(self) -> float:
        return self._notional

    @property
    def index(self) -> IborIndex:
        return self._index

    @property
    def option_type(self) -> OptionType:
        return self._option_type

    @property
    def is_cap(self) -> bool:
        return self._option_type == OptionType.Call


class Collar(Product):
    """Long cap at ``cap_strike``, short floor at ``floor_strike`` on the same schedule (``cap - floor``).

    Args:
        cap (CapFloor): The cap (``OptionType.Call``).
        floor (CapFloor): The floor (``OptionType.Put``), same index and periods as the cap.
    """

    def __init__(self, cap: CapFloor, floor: CapFloor) -> None:
        if not cap.is_cap or floor.is_cap:
            raise ValueError("Collar needs a cap and a floor")
        if [c.date for c in cap.caplets] != [c.date for c in floor.caplets]:
            raise ValueError("Cap and floor of a collar must have the same periods")
        super().__init__(cap.ccy, cap.start_date, cap.end_date)
        self.cap = cap
        self.floor = floor
