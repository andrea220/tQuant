from enum import Enum
from tensorflow import Tensor

class Position(Enum):
    Long = 1
    Short = -1

class SwapType(Enum):
    Payer = 1
    Receiver = -1
    
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

def extract_value(x):
    if isinstance(x, Tensor):
        return x.numpy()  # Extract value from TensorFlow tensor
    else:
        return x
    

curve_map = {
            "EUR": {
                "ON": "EUR:ESTR",
                "3M": "EUR:3M",
                "6M": "EUR:6M"
            },
            "USD": {
                "ON": "USD:SOFR",
                "3M": "USD:3M",
                "6M": "USD:6M"
            }
        }