from abc import ABC, abstractmethod
import tensorflow as tf
from datetime import date
from ..instruments.product import Product


class Pricer(ABC):
    """Abstract base class for pricing financial products.

    This abstract class defines the interface for pricing financial products. Concrete implementations
    must provide a method for calculating the price of a product based on the trade date and market curves.

    Methods:
        calculate_price: Abstract method to be implemented by subclasses to calculate the price of a product.
        price: Calculates the price of a product and optionally returns the gradient if autodiff is enabled.
    """

    @abstractmethod
    def calculate_price(self, product, trade_date, curves):
        """Abstract method to calculate the price of a financial product.

        Args:
            product (Product): The financial product to be priced.
            trade_date (date): The trade date for which the price is to be calculated.
            curves (dict): A dictionary containing market curves needed for pricing.

        Returns:
            float: The calculated price of the product.

        Notes:
            This method must be implemented by any subclass of Pricer.
        """
        return

    def price(
        self, product: Product, trade_date: date, market: dict, autodiff: bool = False
    ):
        """Calculates the price of a financial product, with optional automatic differentiation.

        Args:
            product (Product): The financial product to be priced.
            trade_date (date): The trade date for which the price is to be calculated.
            market (dict): A dictionary containing market curves needed for pricing.
            autodiff (bool, optional): Whether to compute gradients using TensorFlow's autodiff. Defaults to False.

        Returns:
            tuple or float: If autodiff is True, returns a tuple containing the calculated price and the gradient tape.
                            If autodiff is False, returns only the calculated price.
        """
        if autodiff:
            with tf.GradientTape() as tape:
                npv = self.calculate_price(product, trade_date, market)
            return npv, tape
        else:
            return self.calculate_price(product, trade_date, market)
