from tquant import Product, Deposit
from tquant.bootstrapping.basis_swap import BasisSwap
from tquant.bootstrapping.basis_swap_pricer import BasisSwapPricer
from tquant.bootstrapping.curve_assignment import CurveAssignment
from tquant.bootstrapping.deposit_pricer import DepositPricer
from tquant.bootstrapping.fra import Fra
from tquant.bootstrapping.fra_pricer import FraPricer
from tquant.bootstrapping.future import Future
from tquant.bootstrapping.future_pricer import FuturePricer
from tquant.bootstrapping.ois import Ois
from tquant.bootstrapping.ois_pricer import OisPricer
from tquant.bootstrapping.swap import Swap
from tquant.bootstrapping.swap_pricer import SwapPricer


class PricerFactory:
    @staticmethod
    def create(product: Product, curve_assignment: CurveAssignment):
        if type(product) == Deposit:
            return DepositPricer(curve_assignment)
        elif type(product) == Fra:
            return FraPricer(curve_assignment)
        elif type(product) == Swap:
            return SwapPricer(curve_assignment)
        elif type(product) == BasisSwap:
            return BasisSwapPricer(curve_assignment)
        elif type(product) == Ois:
            return OisPricer(curve_assignment)
        elif type(product) == Future:
            return FuturePricer(curve_assignment)
        else:
            raise TypeError("Wrong product type")
