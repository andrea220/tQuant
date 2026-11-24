import datetime

from abc import ABC
from ..markethandles.utils import Currency


class Product(ABC):
    """
    Abstract base class for financial products.

    Attributes:
        ccy (Currency): The currency of the product.
        start_date (datetime.date): The start date of the product.
        end_date (datetime.date): The end date or maturity date of the product.
        price (float): The price or interest rate associated with the product.
    """

    def __init__(
        self,
        ccy: Currency,
        start_date: datetime.date,
        end_date: datetime.date,
        payment_ccy: Currency = None,
        fx_mode: str = "quanto",
    ):
        """
        Initializes the base product class with common attributes.

        Args:
            ccy (Currency): The currency of the product.
            start_date (datetime.date): The start date of the product.
            end_date (datetime.date): The end or maturity date of the product.
            payment_ccy (Currency, optional): Currency of the payments when it differs from ``ccy`` (the currency of
                the underlying). Default ``ccy``.
            fx_mode (str): With a ``payment_ccy`` other than ``ccy``: ``"quanto"`` (the payoff on the underlying in
                its currency, paid in ``payment_ccy`` at a fixed rate) or ``"composite"`` (the payoff on the
                underlying converted into ``payment_ccy``, ``S X``: strikes and levels in ``payment_ccy``).

        Raises:
            ValueError: If ``fx_mode`` is neither ``"quanto"`` nor ``"composite"``.
        """
        if fx_mode not in ("quanto", "composite"):
            raise ValueError(f"fx_mode must be 'quanto' or 'composite', got {fx_mode!r}")
        self._fx_mode = fx_mode
        self._ccy = ccy
        self._payment_ccy = payment_ccy if payment_ccy is not None else ccy
        self._start_date = start_date
        self._end_date = end_date
        self._price = None

    @property
    def ccy(self) -> Currency:
        """
        Get the currency of the product.

        Returns:
            Currency: The currency in which the product is denominated.
        """
        return self._ccy

    @property
    def payment_ccy(self) -> Currency:
        """Currency of the payments (``ccy`` unless quanto)."""
        return getattr(self, "_payment_ccy", None) or self._ccy

    @property
    def is_quanto(self) -> bool:
        """Paid in a currency other than the underlying's at a fixed rate."""
        return self.payment_ccy != self._ccy and getattr(self, "_fx_mode", "quanto") == "quanto"

    @property
    def is_composite(self) -> bool:
        """The underlying converted into the payment currency (``S X``), the payoff on it."""
        return self.payment_ccy != self._ccy and getattr(self, "_fx_mode", "quanto") == "composite"

    @property
    def start_date(self) -> datetime.date:
        """
        Get the start date of the product.

        Returns:
            datetime.date: The start date.
        """
        return self._start_date

    @property
    def end_date(self) -> datetime.date:
        """
        Get the end or maturity date of the product.

        Returns:
            datetime.date: The end date.
        """
        return self._end_date

    @property
    def price(self) -> float:
        """
        Get the price associated with the product.

        Returns:
            float: The price of the product.
        """
        if self._price is None:
            raise ValueError("you must define a pricer")
        return self._price

    @price.setter
    def price(self, value):
        self._price = value
