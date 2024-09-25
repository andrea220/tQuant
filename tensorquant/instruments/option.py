from abc import ABC, abstractmethod
from ..markethandles.utils import Currency, OptionType, ExerciseType
from .product import Product

from datetime import date 
from tensorflow import Variable, float64

class Option(Product, ABC):

    def __init__(self, ccy: Currency, start_date: date, end_date: date, 
                 option_type: OptionType, underlying: str, strike: float | list[float], exercise_type: ExerciseType ):
        super().__init__(ccy, start_date, end_date)
        self._option_type = option_type
        self._strike = Variable(strike, dtype=float64) 
        self._underlying = underlying
        self._exercise_type = exercise_type

    @property
    def option_type(self):
        return self._option_type
    
    @property
    def strike(self):
        return self._strike
    
    @property
    def underlying(self):
        return self._underlying
        
    @property
    def exercise_type(self):
        return self._exercise_type

    @abstractmethod
    def implied_volatility(self,
                           price: float,
                           dividends: float,
                           accuracy: float,
                           max_evaluations: float,
                           min_vol: float,
                           max_vol: float):
        return
    
class VanillaOption(Option):

    def __init__(self, ccy: Currency, start_date: date, end_date: date, option_type, underlying: str, strike):
        super().__init__(ccy, start_date, end_date, 
                         option_type, underlying, strike, ExerciseType.European)
        self._delta = None
        self._gamma = None 
        self._theta = None
        self._vega = None 
        self._rho = None 
    
    def implied_volatility(self,price: float, dividends, accuracy: float = 1e-4, max_evaluations: float = 100, min_vol: float = 1e-7, max_vol: float=4):
        return
    