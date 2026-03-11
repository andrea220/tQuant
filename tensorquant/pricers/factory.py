from ..instruments.product import Product

from ..instruments.deposit import Deposit
from .deposit import DepositPricer

from ..instruments.ois import Ois
from .swapdiscounting import OisPricer

from ..instruments.forward import Fra
from .fradiscounting import FraPricer

from ..instruments.swap import Swap
from .swapdiscounting import SwapPricer


class PricerAssignment:

    @staticmethod
    def create(product: Product):
        if type(product) == Deposit:
            return DepositPricer()
        elif type(product) == Ois:
            return OisPricer()
        elif type(product) == Fra:
            return FraPricer()
        elif type(product) == Swap:
            return SwapPricer()
        else:
            raise TypeError(f"{product} is a wrong product type")
