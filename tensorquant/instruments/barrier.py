from datetime import date
from typing import Optional, Sequence

from .option import Option
from ..markethandles.utils import BarrierType, Currency, ExerciseType, OptionType


class BarrierOption(Option):
    """European single-barrier option (knock-in / knock-out call or put, no rebate).

    Payoff at expiry ``max(phi (S_T - K), 0)`` times ``1{hit}`` (knock-in) or ``1{not hit}`` (knock-out), with
    ``hit`` = the spot reaches the barrier ``H`` from above (``Down``) or below (``Up``) during the monitoring:

    - continuous monitoring (``monitoring_dates=None``, default): any time in ``(0, T]``;
    - discrete monitoring: only on ``monitoring_dates`` (``[end_date]`` = barrier observed at expiry only).

    Knock-in + knock-out = vanilla. Monitoring dates up to the evaluation date are ignored (no fixing history:
    the barrier is assumed not hit so far). Priced by :class:`BlackScholesPricer` (Reiner-Rubinstein closed
    form, Broadie-Glasserman-Kou shift for discrete monitoring) and :class:`VanillaMCPricer`.

    Args:
        ccy (Currency): Currency.
        start_date (date): Trade date.
        end_date (date): Expiry (and payment) date.
        option_type (OptionType): ``Call`` or ``Put``.
        strike (float): Strike.
        barrier (float): Barrier level (absolute).
        barrier_type (BarrierType): ``DownIn``, ``DownOut``, ``UpIn`` or ``UpOut``.
        monitoring_dates (Sequence[date], optional): Discrete monitoring dates (``None`` = continuous).
        underlying (str): Ticker of the underlying.

    Raises:
        ValueError: If the barrier is not positive or a monitoring date is after the expiry.
    """

    def __init__(
        self,
        ccy: Currency,
        start_date: date,
        end_date: date,
        option_type: OptionType,
        strike: float,
        barrier: float,
        barrier_type: BarrierType,
        monitoring_dates: Optional[Sequence[date]] = None,
        underlying: str = "DEFAULT",
    ):
        if barrier <= 0:
            raise ValueError(f"barrier must be positive, got {barrier}")
        if not isinstance(barrier_type, BarrierType):
            raise ValueError(f"barrier_type must be a BarrierType, got {barrier_type!r}")
        if monitoring_dates is not None:
            monitoring_dates = sorted(set(monitoring_dates))
            if not monitoring_dates:
                raise ValueError("monitoring_dates must not be empty (None = continuous monitoring)")
            if monitoring_dates[-1] > end_date:
                raise ValueError("monitoring dates must not be after the expiry")
        super().__init__(ccy, start_date, end_date, option_type, strike, ExerciseType.European, underlying)
        self._barrier = float(barrier)
        self._barrier_type = barrier_type
        self._monitoring_dates = monitoring_dates

    @property
    def barrier(self) -> float:
        return self._barrier

    @property
    def barrier_type(self) -> BarrierType:
        return self._barrier_type

    @property
    def monitoring_dates(self) -> Optional[list]:
        """Discrete monitoring dates, ``None`` for continuous monitoring."""
        return self._monitoring_dates

    @property
    def continuous(self) -> bool:
        return self._monitoring_dates is None
