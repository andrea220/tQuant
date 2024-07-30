
from ..instruments.product import Product

from ..instruments.deposit import Deposit
from .deposit import DepositEngine

from ..instruments.ois import Ois
from .swapdiscounting import SwapAnalyticEngine, OisAnalyticEngine


class PricerAssignment:
    @staticmethod
    def create(product: Product):
        if type(product) == Deposit:
            return DepositEngine(product)
        elif type(product) == Ois:
            return OisAnalyticEngine(product)
        else:
            raise TypeError("Wrong product type")