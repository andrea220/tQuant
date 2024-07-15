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

