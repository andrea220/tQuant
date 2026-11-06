from __future__ import annotations

import datetime

from .product import Product
from .swap import Swap


class Swaption(Product):
    """European swaption, physically settled: the right, at *expiry*, to enter the underlying swap.

    ``payer=True``: right to pay the fixed rate (call on the swap rate); ``False``: receiver (put).
    The strike is the fixed rate of the underlying swap.

    Args:
        swap (Swap): Underlying (forward-starting) swap; its fixed rate is the strike.
        expiry (date): Exercise date.
        payer (bool): Payer (True) or receiver swaption.
        tenor_years (float): Underlying swap tenor in years (vol cube lookup).
    """

    def __init__(self, swap: Swap, expiry: datetime.date, payer: bool, tenor_years: float) -> None:
        if expiry > swap.start_date:
            raise ValueError("swaption expiry must not be after the swap start")
        super().__init__(swap.ccy, expiry, swap.end_date)
        self.swap = swap
        self.expiry = expiry
        self.payer = payer
        self.tenor_years = float(tenor_years)

    @property
    def strike(self) -> float:
        return float(self.swap.fixed_rate)
