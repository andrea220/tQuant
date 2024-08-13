from abc import ABC, abstractmethod
from datetime import date


class CashFlow(ABC):
    ''' 
    The abstract class for CashFlow implementations.
    '''
    @abstractmethod
    def date(self) -> date:
        ''' 
        Payment date of the cashflow
        '''
        pass
        
    @abstractmethod
    def amount(self) -> float:
        ''' 
        Future amount (not discounted) of the cashflow.
        '''
        pass

    def has_occurred(self,
                     ref_date: date) -> bool:
        ''' 
        Check if a given date cashflow is in the past.

        Parameters:
        -------
            ref_date: date
                The date to check.

        Returns:
        -------
            bool: True if the given cashflow has occurred, False otherwise.
        '''
        cf = self.date
        if cf <= ref_date:
            return True
        if cf > ref_date:
            return False



class CashFlow(ABC):
    """
    An abstract base class that defines the interface for CashFlow implementations.

    This class serves as a blueprint for creating different types of cash flow objects,
    enforcing the implementation of core methods such as `date` and `amount`.
    """
    @abstractmethod
    def date(self) -> date:
        """
        Returns the payment date of the cash flow.

        This method must be implemented by any subclass to specify the date on which 
        the cash flow occurs.

        Returns:
        -------
        date
            The payment date of the cash flow.
        """
        pass
        
    @abstractmethod
    def amount(self) -> float:
        """
        Returns the future value (non-discounted) of the cash flow.

        This method must be implemented by any subclass to specify the exact amount 
        associated with the cash flow, not taking into account any discounting.

        Returns:
        -------
        float
            The future amount of the cash flow.
        """
        pass

    def has_occurred(self,
                     ref_date: date) -> bool:
        """
        Determines whether the cash flow has occurred relative to a reference date.

        This method checks if the cash flow's payment date is in the past compared to the 
        provided reference date.

        Parameters:
        -------
        ref_date: date
            The reference date against which to check the cash flow's occurrence.

        Returns:
        -------
        bool
            True if the cash flow's date is on or before the reference date, False otherwise.
        """
        cf = self.date
        if cf <= ref_date:
            return True
        if cf > ref_date:
            return False
