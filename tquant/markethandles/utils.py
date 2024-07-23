from enum import Enum

class Position(Enum):
    Long = 1
    Short = -1

class SwapType(Enum):
    """
    Enumeration of time-units.
    """
    Payer = "Payer"
    Receiver = "Receiver"

    def __str__(self):
        return self.value
    
class Currency(Enum):
    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    CHF = "CHF"
    AUD = "AUD"

    def __str__(self):
        return self.value
