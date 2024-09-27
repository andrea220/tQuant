from abc import ABC, abstractmethod
from tensorflow import GradientTape
from ..instruments.product import Product


class Pricer(ABC):
    """Abstract base class for pricing financial products.

    This abstract class defines the interface for pricing financial products. Concrete implementations
    must provide a method for calculating the price of a product based on the trade date and market curves.

    Methods:
        calculate_price: Abstract method to be implemented by subclasses to calculate the price of a product.
        price: Calculates the price of a product and optionally returns the gradient if autodiff is enabled.
    """

    def __init__(self) -> None:
        self._tape = None

    @property
    def tape(self):
        if self._tape is None:
            raise ValueError("autodiff must be enabled")
        return self._tape

    @abstractmethod
    def calculate_price(self, product, market):
        """Abstract method to calculate the price of a financial product.

        Args:
            product (Product): The financial product to be priced.
            curves (dict): A dictionary containing market curves needed for pricing.

        Returns:
            float: The calculated price of the product.

        Notes:
            This method must be implemented by any subclass of Pricer.
        """
        return

    def price(self, product: Product, market: dict, autodiff: bool = False):
        """Calculates the price of a financial product, with optional automatic differentiation.

        Args:
            product (Product): The financial product to be priced.
            market (dict): A dictionary containing market curves needed for pricing.
            autodiff (bool, optional): Whether to compute gradients using TensorFlow's autodiff. Defaults to False.

        """
        if autodiff:
            with GradientTape() as tape:
                npv = self.calculate_price(product, market)
            product.price = npv
            self._tape = tape
        else:
            product.price = self.calculate_price(product, market)
