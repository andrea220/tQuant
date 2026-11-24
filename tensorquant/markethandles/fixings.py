"""Historical fixings of an equity underlying (closes), read by the pricers of live trades."""

from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

import pandas as pd

__all__ = ["Fixings"]


class Fixings:
    """Past values of one underlying by date (the official close, Bloomberg ``PX_LAST``).

    Not a risk factor: a fixing already observed does not move with the market.
    """

    def __init__(self, dates: Iterable[date], values: Iterable[float]):
        """
        Args:
            dates: Fixing dates.
            values: Values on those dates (same length).

        Raises:
            ValueError: If the lengths differ or a date is repeated.
        """
        dates, values = list(dates), [float(v) for v in values]
        if len(dates) != len(values):
            raise ValueError(f"{len(dates)} fixing dates for {len(values)} values")
        if len(set(dates)) != len(dates):
            raise ValueError("repeated fixing date")
        order = sorted(range(len(dates)), key=lambda i: dates[i])
        self._values = {dates[i]: values[i] for i in order}

    @property
    def dates(self) -> list[date]:
        return list(self._values)

    @property
    def values(self) -> list[float]:
        return list(self._values.values())

    def get(self, fixing_date: date) -> Optional[float]:
        """The value on *fixing_date*, ``None`` when there is none."""
        return self._values.get(fixing_date)

    def __len__(self) -> int:
        return len(self._values)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"date": self.dates, "value": self.values})

    def __repr__(self) -> str:
        if not self._values:
            return "Fixings(empty)"
        return f"Fixings({len(self)} dates, {self.dates[0]} to {self.dates[-1]})"
