from abc import ABC, abstractmethod

class Pricer(ABC):

    @abstractmethod
    def price(self):
        return 


class AbstractPricerAP:
    @abstractmethod
    def price(self,
              product,
              trade_date,
              curves):
        pass