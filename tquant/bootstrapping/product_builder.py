import datetime
from abc import ABC, abstractmethod


class ProductBuilder(ABC):
    def __init__(self, name: str, ccy: str, notional: float):
        self.name = name
        self.ccy = ccy
        self.notional = notional

    @abstractmethod
    def build(self, trade_date: datetime, quote: float, term: str):
        pass



    
