from enum import Enum
from tensorflow import Tensor


class Position(Enum):
    """Enum to represent trading position types.

    Attributes:
        Long (int): Represents a long position with a value of 1.
        Short (int): Represents a short position with a value of -1.
    """

    Long = 1
    Short = -1


class SwapType(Enum):
    """Enum to represent types of swaps in a financial context.

    Attributes:
        Payer (int): Represents a payer swap with a value of 1.
        Receiver (int): Represents a receiver swap with a value of -1.
    """

    Payer = 1
    Receiver = -1


class Currency(Enum):
    """Enum to represent currency codes.

    Attributes:
        EUR (str): Euro currency code.
        USD (str): US Dollar currency code.
        GBP (str): British Pound currency code.
        JPY (str): Japanese Yen currency code.
        CAD (str): Canadian Dollar currency code.
        CHF (str): Swiss Franc currency code.
        AUD (str): Australian Dollar currency code.
    """

    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    CHF = "CHF"
    AUD = "AUD"

    def __str__(self):
        """Return the string representation of the currency.

        Returns:
            str: The currency code (e.g., 'EUR', 'USD').
        """
        return self.value


def extract_value(x):
    """Extracts the underlying value from a TensorFlow Tensor or returns the input if it is not a Tensor.

    Args:
        x (Union[Tensor, Any]): Input value, which can be a TensorFlow `Tensor` or any other type.

    Returns:
        Any: The extracted value from the Tensor if `x` is a Tensor, otherwise returns `x` as is.
    """
    if isinstance(x, Tensor):
        return x.numpy()  # Extract value from TensorFlow tensor
    else:
        return x

class OptionType(Enum):
    Call = 1
    Put = -1

    def __str__(self):
        return self.value
    
class PayoffType(Enum):
    PlainVanilla = "PlainVanilla"
    AssetOrNothing = "AssetOrNothing"
    CashOrNothing = "CashOrNothing"

    def __str__(self):
        return self.value
    
class ExerciseType(Enum):
    American = 0
    Bermudan = 1
    European = 2

    def __str__(self):
        return self.value
    

curve_map = {
    "EUR": {"ON": "EUR:ESTR", "3M": "EUR:3M", "6M": "EUR:6M"},
    "USD": {"ON": "USD:SOFR", "3M": "USD:3M", "6M": "USD:6M"},
}


ir_eur_crv_map = {"ON": "EUR:ESTR", "3M": "EUR:3M", "6M": "EUR:6M"}
ir_usd_crv_map = {"ON": "USD:SOFR", "3M": "USD:3M", "6M": "USD:6M"}
eq_eur_vol_map = {"DEFAULT": "DEFAULT"}

market_map = {
    "IR:EUR": ir_eur_crv_map,
    "IR:USD": ir_usd_crv_map,
    "VOLEQ:EUR": eq_eur_vol_map
}