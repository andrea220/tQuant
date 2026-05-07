from __future__ import annotations

import bisect
from datetime import date
from typing import Optional, Union

import tensorflow as tf

from ..timehandles.daycounter import DayCounter, DayCounterConvention
from .utils import Currency


class DividendCurve:
    """Schedule of discrete cash dividends for a single equity.

    A pure market-data container: stores ex-dates, declared dates and per-share
    amounts without any reference to a discount curve.  The discount curve must
    be supplied explicitly to the helper :meth:`pv_dividends` when a present
    value is required, keeping this class decoupled from interest-rate objects
    (same pattern as :class:`DefaultCurve`).

    The class is aware of the ``reference_date`` so that dividends with an
    ex-date on or before that date are automatically excluded from present-value
    calculations (they have already detached from the stock price).

    Attributes:
        _reference_date (date): Evaluation / curve anchor date.
        _currency (Currency): Currency of the dividend amounts.
        _ex_dates (list[date]): Ex-dividend dates, sorted ascending.
        _amounts (list[float]): Cash amount per share corresponding to each
            ex-date.
        _declared_dates (list[date] | None): Announcement dates (informational
            only).
        _payment_dates (list[date] | None): Actual cash-payment dates
            (informational only; pricing uses ex-dates).
        _daycounter (DayCounter): Used to convert dates to year fractions for
            discount-factor lookups.
    """

    def __init__(
        self,
        reference_date: date,
        ex_dates: list[date],
        amounts: list[float],
        currency: Currency,
        declared_dates: Optional[list[date]] = None,
        payment_dates: Optional[list[date]] = None,
        daycounter_convention: DayCounterConvention = DayCounterConvention.Actual365,
    ) -> None:
        """Initialise the DividendCurve.

        Args:
            reference_date (date): Evaluation date (anchor for all time
                computations).  Dividends with ``ex_date <= reference_date``
                are kept in the object but skipped in :meth:`pv_dividends`.
            ex_dates (list[date]): Ex-dividend dates (the date from which the
                stock price trades without the dividend).  Must be the same
                length as ``amounts``.  Used as the pricing date for each
                dividend — the stock price drops by the dividend amount on
                this date.
            amounts (list[float]): Cash dividend per share for each ex-date.
                Must be non-negative.
            currency (Currency): Currency of the dividend payments.
            declared_dates (list[date], optional): Declaration / announcement
                dates.  If provided, must be the same length as ``ex_dates``.
                Stored for informational purposes only.
            payment_dates (list[date], optional): Actual cash settlement dates
                (typically a few days after the ex-date).  If provided, must
                be the same length as ``ex_dates``.  Stored for informational
                purposes; pricing always uses ``ex_dates``.
            daycounter_convention (DayCounterConvention): Day-count convention
                used when converting dates to year fractions.  Should match
                the convention of the discount curve passed to
                :meth:`pv_dividends`.

        Raises:
            ValueError: If ``ex_dates`` and ``amounts`` have different lengths,
                if any optional date list has a different length, or if any
                amount is negative.
        """
        n = len(ex_dates)
        if len(amounts) != n:
            raise ValueError(
                f"ex_dates and amounts must have the same length "
                f"(got {n} and {len(amounts)})"
            )
        for name, lst in (("declared_dates", declared_dates), ("payment_dates", payment_dates)):
            if lst is not None and len(lst) != n:
                raise ValueError(
                    f"{name} must have the same length as ex_dates "
                    f"(got {len(lst)} and {n})"
                )
        if any(a < 0 for a in amounts):
            raise ValueError("All dividend amounts must be non-negative")

        self._reference_date = reference_date
        self._currency = currency
        self._daycounter = DayCounter(daycounter_convention)
        self._daycounter_convention = daycounter_convention

        # Sort everything by ex_date ascending
        order = sorted(range(n), key=lambda i: ex_dates[i])
        self._ex_dates: list[date] = [ex_dates[i] for i in order]
        self._amounts: list[float] = [amounts[i] for i in order]
        self._declared_dates: Optional[list[date]] = (
            [declared_dates[i] for i in order] if declared_dates is not None else None
        )
        self._payment_dates: Optional[list[date]] = (
            [payment_dates[i] for i in order] if payment_dates is not None else None
        )

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        reference_date: date,
        df,
        ex_date_col: str = "Ex Date",
        amount_col: str = "Amount Per Share",
        declared_date_col: Optional[str] = "Declared Date",
        payment_date_col: Optional[str] = "Payment Date",
        currency_col: Optional[str] = "currency",
        currency: Optional[Currency] = None,
        daycounter_convention: DayCounterConvention = DayCounterConvention.Actual365,
    ) -> "DividendCurve":
        """Build a DividendCurve from a pandas DataFrame row-set.

        The DataFrame is expected to contain at least an ex-date column and an
        amount column (as in ``dividend_df`` from the standard MDM data store).
        Rows with missing or NaN amounts are silently dropped.

        Args:
            reference_date (date): Evaluation date.
            df: pandas DataFrame (or compatible) with dividend data.
            ex_date_col (str): Column name for ex-dividend dates.
            amount_col (str): Column name for dividend amounts per share.
            declared_date_col (str, optional): Column name for declared dates.
                Pass ``None`` to ignore.  Silently skipped if the column is
                not present in the DataFrame.
            payment_date_col (str, optional): Column name for payment / cash
                settlement dates.  Pass ``None`` to ignore.  Silently skipped
                if the column is not present in the DataFrame.
            currency_col (str, optional): Column name for the currency string.
                The first non-null value in the column is used.  Ignored when
                ``currency`` is provided explicitly.
            currency (Currency, optional): Override the currency instead of
                reading it from the DataFrame.
            daycounter_convention (DayCounterConvention): Day-count convention.

        Returns:
            DividendCurve: A new instance built from the DataFrame.

        Raises:
            ValueError: If the currency cannot be determined.
        """
        import pandas as pd

        def _parse_dates(raw: list) -> list[date]:
            return [d if isinstance(d, date) else pd.to_datetime(d).date() for d in raw]

        df = df.copy()
        df = df.dropna(subset=[amount_col])
        df = df[df[amount_col] > 0]

        ex_dates = _parse_dates(df[ex_date_col].tolist())
        amounts = [float(a) for a in df[amount_col].tolist()]

        declared_dates = None
        if declared_date_col is not None and declared_date_col in df.columns:
            declared_dates = _parse_dates(df[declared_date_col].tolist())

        payment_dates = None
        if payment_date_col is not None and payment_date_col in df.columns:
            payment_dates = _parse_dates(df[payment_date_col].tolist())

        if currency is None:
            if currency_col is not None and currency_col in df.columns:
                ccy_str = df[currency_col].dropna().iloc[0]
                try:
                    currency = Currency(ccy_str.strip().upper())
                except ValueError:
                    raise ValueError(
                        f"Cannot map currency string '{ccy_str}' to a Currency enum value"
                    )
            else:
                raise ValueError(
                    "Currency must be supplied either via the 'currency' parameter "
                    "or via 'currency_col' referencing a column in the DataFrame"
                )

        return cls(
            reference_date=reference_date,
            ex_dates=ex_dates,
            amounts=amounts,
            currency=currency,
            declared_dates=declared_dates,
            payment_dates=payment_dates,
            daycounter_convention=daycounter_convention,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def reference_date(self) -> date:
        """Evaluation date of the curve."""
        return self._reference_date

    @property
    def currency(self) -> Currency:
        """Currency of the dividend amounts."""
        return self._currency

    @property
    def ex_dates(self) -> list[date]:
        """All ex-dividend dates, sorted ascending."""
        return self._ex_dates

    @property
    def amounts(self) -> list[float]:
        """Cash amounts per share corresponding to each ex-date."""
        return self._amounts

    @property
    def declared_dates(self) -> Optional[list[date]]:
        """Announcement dates (informational only), or None if not provided."""
        return self._declared_dates

    @property
    def payment_dates(self) -> Optional[list[date]]:
        """Cash settlement dates, or None if not provided.

        Informational only: pricing always uses ``ex_dates``.
        """
        return self._payment_dates

    @property
    def daycounter_convention(self) -> DayCounterConvention:
        """Day-count convention used by this curve."""
        return self._daycounter_convention

    def to_dataframe(self):
        """Return the full dividend schedule as a pandas DataFrame.

        The DataFrame always contains the columns ``ex_date``, ``amount`` and
        ``currency``.  Optional columns ``declared_date`` and ``payment_date``
        are included when the corresponding data was provided at construction
        time.

        Returns:
            pandas.DataFrame: One row per dividend event, sorted by
                ``ex_date`` ascending.
        """
        import pandas as pd

        data: dict = {
            "ex_date": self._ex_dates,
            "amount":  self._amounts,
        }
        if self._declared_dates is not None:
            data["declared_date"] = self._declared_dates
        if self._payment_dates is not None:
            data["payment_date"] = self._payment_dates
        data["currency"] = self._currency.value

        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Core query methods
    # ------------------------------------------------------------------

    def _to_date(self, t: Union[date, float]) -> date:
        """Convert a year-fraction to an approximate calendar date."""
        if isinstance(t, date):
            return t
        from datetime import timedelta
        days = int(t * self._daycounter.year_fraction(
            self._reference_date,
            self._reference_date.replace(year=self._reference_date.year + 1)
        ) * 365)
        return self._reference_date + timedelta(days=int(t * 365))

    def dividends_before(
        self, t: Union[date, float]
    ) -> list[tuple[date, float]]:
        """Return (ex_date, amount) pairs with ``reference_date < ex_date <= t``.

        Dividends that have already detached (``ex_date <= reference_date``) are
        excluded; this matches the forward-price convention where only future
        dividends reduce the stock price.

        Args:
            t (Union[date, float]): Upper bound (date or year fraction from
                ``reference_date``).

        Returns:
            list[tuple[date, float]]: List of ``(ex_date, amount)`` tuples,
                sorted by ex_date ascending.
        """
        cutoff = self._to_date(t) if isinstance(t, float) else t
        return [
            (d, a)
            for d, a in zip(self._ex_dates, self._amounts)
            if self._reference_date < d <= cutoff
        ]

    def dividends_between(
        self,
        t1: Union[date, float],
        t2: Union[date, float],
    ) -> list[tuple[date, float]]:
        """Return (ex_date, amount) pairs with ``t1 < ex_date <= t2``.

        Useful for computing the dividend impact over a specific sub-period
        (e.g., inside a numerical scheme step).

        Args:
            t1 (Union[date, float]): Lower bound (exclusive).
            t2 (Union[date, float]): Upper bound (inclusive).

        Returns:
            list[tuple[date, float]]: Filtered and sorted list of dividend
                tuples.
        """
        d1 = self._to_date(t1) if isinstance(t1, float) else t1
        d2 = self._to_date(t2) if isinstance(t2, float) else t2
        return [
            (d, a)
            for d, a in zip(self._ex_dates, self._amounts)
            if d1 < d <= d2
        ]

    def pv_dividends(
        self,
        t: Union[date, float],
        discount_curve,
    ) -> tf.Tensor:
        """Present value of all future discrete dividends up to time *t*.

        Computes

        .. math::

            PV(t) = \\sum_{\\substack{i \\\\ t_0 < ex_i \\le t}} D_i \\cdot P(0, ex_i)

        where :math:`P(0, ex_i)` is the discount factor from ``discount_curve``
        and :math:`t_0` is the ``reference_date``.

        The discount curve is passed as a parameter (not stored), keeping
        ``DividendCurve`` decoupled from interest-rate objects.

        Args:
            t (Union[date, float]): Horizon date or year fraction.  Only
                dividends with ``ex_date <= t`` are included.
            discount_curve: A :class:`~tensorquant.markethandles.ircurve.RateCurve`
                (or any object exposing a ``discount(date) -> tf.Tensor``
                method) used to compute :math:`P(0, ex_i)`.

        Returns:
            tf.Tensor: Scalar tensor with the present value of dividends.
                Returns zero (``tf.constant(0.0, dtype=tf.float64)``) if there
                are no dividends in the range.
        """
        divs = self.dividends_before(t)
        if not divs:
            return tf.constant(0.0, dtype=tf.float64)

        total = tf.constant(0.0, dtype=tf.float64)
        for ex_date, amount in divs:
            df = tf.cast(discount_curve.discount(ex_date), tf.float64)
            total = total + tf.constant(amount, dtype=tf.float64) * df
        return total

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of dividend events in the schedule."""
        return len(self._ex_dates)

    def __repr__(self) -> str:
        future = sum(
            1 for d in self._ex_dates if d > self._reference_date
        )
        return (
            f"DividendCurve("
            f"reference_date={self._reference_date}, "
            f"currency={self._currency.value}, "
            f"n_dividends={len(self._ex_dates)}, "
            f"n_future={future})"
        )
