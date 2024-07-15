from abc import ABC, abstractmethod
import datetime 


class CashFlow(ABC):
    ''' 
    Abstract CashFlow object
    '''
    @abstractmethod
    def date(self) -> datetime.date:
        ''' 
        Payment Date of the cashflow
        '''
        pass
        
    @abstractmethod
    def amount(self) -> float:
        ''' 
        Future amount (not discounted) of the cashflow
        '''
        pass

    def has_occurred(self, ref_date: datetime.date) -> bool:
        ''' 
        True if the cashflow date is before (or the same day) of refDate
        False otherwise
        '''
        cf = self.date
        if cf <= ref_date:
            return True
        if cf > ref_date:
            return False

