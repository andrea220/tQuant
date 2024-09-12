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
    def create(product: Product, curve_map: dict):
        if type(product) == Deposit:
            return DepositPricer(curve_map)
        elif type(product) == Ois:
            return OisPricer(curve_map)
        elif type(product) == Fra:
            return FraPricer(curve_map)
        elif type(product) == Swap:
            return SwapPricer(curve_map)
        else:
            raise TypeError(f"{product} is a wrong product type")
