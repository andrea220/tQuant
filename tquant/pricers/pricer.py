from abc import ABC, abstractmethod

class Pricer(ABC):

    @abstractmethod
    def price(self):
        return 

    @abstractmethod
    def price_aad(self):
        return
    
