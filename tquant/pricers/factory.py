
from ..instruments.product import Product

from ..instruments.deposit import Deposit
from .deposit import DepositPricer

from ..instruments.ois import Ois
from .swapdiscounting import OisPricer


class PricerAssignment:
    """ 
    passa il curve_name
    
    """
    @staticmethod
    def create(product: Product, curve_name: str):
        if type(product) == Deposit:
            return DepositPricer(curve_name)
        elif type(product) == Ois:
            return OisPricer(curve_name)
        else:
            raise TypeError(f"{product} is a wrong product type")