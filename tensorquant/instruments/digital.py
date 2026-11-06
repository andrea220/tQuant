from datetime import date

from .option import Option
from ..markethandles.utils import Currency, ExerciseType, OptionType, PayoffType


class DigitalOption(Option):
    """European digital (binary) option.

    Payoff at expiry, with ``phi = +1`` (call) or ``-1`` (put):

    - cash-or-nothing: ``cash_amount * 1{phi (S_T - K) > 0}``;
    - asset-or-nothing: ``S_T * 1{phi (S_T - K) > 0}``.

    Priced by :class:`BlackScholesPricer` (closed form) and :class:`VanillaMCPricer` (Monte Carlo, optionally
    smoothed indicator for pathwise greeks).

    Args:
        ccy (Currency): Currency.
        start_date (date): Trade date.
        end_date (date): Expiry (and payment) date.
        option_type (OptionType): ``Call`` (pays above the strike) or ``Put`` (below).
        strike (float): Strike.
        payoff_type (PayoffType): ``CashOrNothing`` (default) or ``AssetOrNothing``.
        cash_amount (float): Amount paid by a cash-or-nothing digital (default 1).
        underlying (str): Ticker of the underlying.

    Raises:
        ValueError: If ``payoff_type`` is ``PlainVanilla`` (use :class:`VanillaOption`).
    """

    def __init__(
        self,
        ccy: Currency,
        start_date: date,
        end_date: date,
        option_type: OptionType,
        strike: float,
        payoff_type: PayoffType = PayoffType.CashOrNothing,
        cash_amount: float = 1.0,
        underlying: str = "DEFAULT",
    ):
        if payoff_type not in (PayoffType.CashOrNothing, PayoffType.AssetOrNothing):
            raise ValueError(f"payoff_type must be CashOrNothing or AssetOrNothing, got {payoff_type}")
        super().__init__(ccy, start_date, end_date, option_type, strike, ExerciseType.European, underlying)
        self._payoff_type = payoff_type
        self._cash_amount = float(cash_amount)

    @property
    def payoff_type(self) -> PayoffType:
        return self._payoff_type

    @property
    def cash_amount(self) -> float:
        return self._cash_amount
