from abc import ABC, abstractmethod
import datetime


class CashFlow(ABC):
    """
    Abstract base class representing a generic cash flow.

    This class provides a blueprint for creating various cash flow types,
    ensuring that subclasses implement the core methods for retrieving
    the payment date and amount.
    """

    @property
    @abstractmethod
    def date(self) -> datetime.date:
        """
        Get the payment date of the cash flow.

        Subclasses must implement this property to return the specific date
        on which the cash flow occurs.

        Returns:
            datetime.date: The payment date of the cash flow.
        """
        return

    @property
    @abstractmethod
    def amount(self) -> float:
        """
        Get the future value (non-discounted) of the cash flow.

        Subclasses must implement this method to specify the exact amount
        associated with the cash flow, not taking into account any discounting.

        Returns:
            float: The future amount of the cash flow.
        """
        return

    def has_occurred(self, ref_date: datetime.date) -> bool:
        """
        Determine if the cash flow has occurred relative to a reference date.

        This method checks if the cash flow's payment date is on or before
        the provided reference date.

        Args:
            ref_date (datetime.date): The reference date to compare against
            the cash flow's payment date.

        Returns:
            bool: True if the cash flow's date is on or before the reference
            date, otherwise False.
        """
        cf = self.date
        if cf <= ref_date:
            return True
        return False
