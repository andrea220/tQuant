from ..instruments.product import Product

from ..instruments.deposit import Deposit
from .deposit import DepositPricer

from ..instruments.ois import Ois
from .swapdiscounting import OisPricer

from ..instruments.forward import Fra
from .fradiscounting import FraPricer

from ..instruments.swap import Swap
from .swapdiscounting import SwapPricer

from ..instruments.option import VanillaOption
from ..instruments.digital import DigitalOption
from ..instruments.barrier import BarrierOption
from .black import BlackScholesPricer

from ..instruments.capfloor import CapFloor, Collar
from .capfloor import CapFloorPricer

from ..instruments.swaption import Swaption
from .swaption import SwaptionPricer


class PricerAssignment:
    """Default pricer for a product type."""

    _PRICERS = (
        (Deposit, DepositPricer),
        (Ois, OisPricer),
        (Fra, FraPricer),
        (Swap, SwapPricer),
        (VanillaOption, BlackScholesPricer),
        (DigitalOption, BlackScholesPricer),
        (BarrierOption, BlackScholesPricer),
        (CapFloor, CapFloorPricer),
        (Collar, CapFloorPricer),
        (Swaption, SwaptionPricer),
    )

    @staticmethod
    def create(product: Product):
        """Return a new default pricer for *product* (``isinstance``, so subclasses are covered).

        Args:
            product (Product): Deposit, Ois, Fra, Swap, VanillaOption, DigitalOption, BarrierOption, CapFloor, Collar or Swaption.

        Returns:
            Pricer: ``DepositPricer``, ``OisPricer``, ``FraPricer``, ``SwapPricer`` or
                ``BlackScholesPricer()`` (default arguments) ``CapFloorPricer`` or ``SwaptionPricer``.

        Raises:
            TypeError: If no default pricer exists for the product type.
        """
        for product_type, pricer_type in PricerAssignment._PRICERS:
            if isinstance(product, product_type):
                return pricer_type()
        raise TypeError(f"No default pricer for {type(product).__name__}")
