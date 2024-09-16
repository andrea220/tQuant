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
        quote (float): The price or interest rate associated with the product.
    """

    def __init__(
        self,
        ccy: Currency,
        start_date: datetime.date,
        end_date: datetime.date,
        quote: float,
    ):
        """
        Initializes the base product class with common attributes.

        Args:
            ccy (Currency): The currency of the product.
            start_date (datetime.date): The start date of the product.
            end_date (datetime.date): The end or maturity date of the product.
            quote (float): The price or rate of the product.
        """
        self._ccy = ccy
        self._start_date = start_date
        self._end_date = end_date
        self._quote = quote

    @property
    def ccy(self) -> Currency:
        """
        Get the currency of the product.

        Returns:
            Currency: The currency in which the product is denominated.
        """
        return self._ccy

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
    def quote(self) -> float:
        """
        Get the interest rate associated with the product.

        Returns:
            float: The rate of the product.
        """
        return self._quote