from abc import ABC, abstractmethod

class Trade(ABC):
    
    @abstractmethod
    def __init__(self) -> None:
        self._trade_id = None
        self._currency = None
        self._product_type = None
        self._counterparty = None
        self._netting_set = None
    
    @property
    def trade_id(self):
        if self._trade_id is not None:
            return self._trade_id
        else:
            raise ValueError("Trade ID not assigned")
    @trade_id.setter
    def trade_id(self, value):
        self._trade_id = value
        
    @property
    def currency(self):
        if self._currency is not None:
            return self._currency
        else:
            raise ValueError("Currency not assigned")
    @currency.setter
    def currency(self, value):
        self._currency = value
        
    @property
    def product_type(self):
        if self._product_type is not None:
            return self._product_type
        else:
            raise ValueError("not assigned")
    @product_type.setter
    def product_type(self, value):
        self._product_type = value
        
    @property
    def counterparty(self):
        if self._counterparty is not None:
            return self._counterparty
        else:
            raise ValueError("not assigned")
    @counterparty.setter
    def counterparty(self, value):
        self._counterparty = value
        
    @property
    def netting_set(self):
        if self._netting_set is not None:
            return self._netting_set
        else:
            raise ValueError("not assigned")
    @netting_set.setter
    def netting_set(self, value):
        self._netting_set = value