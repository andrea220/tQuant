from abc import ABC, abstractmethod
from tensorflow import GradientTape
from ..instruments.product import Product
from ..markethandles.marketenvironment import MarketEnvironment


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
    def calculate_price(self, product, market_env: MarketEnvironment):
        """Abstract method to calculate the price of a financial product.

        Args:
            product (Product): The financial product to be priced.
            market_env (MarketEnvironment): The market environment providing
                access to market data (curves, spots, volatilities).

        Returns:
            float: The calculated price of the product.

        Notes:
            This method must be implemented by any subclass of Pricer.
        """
        return

    def price(self, product: Product, market_env: MarketEnvironment, autodiff: bool = False, price_currency=None):
        """Calculates the price of a financial product, with optional automatic differentiation.

        Args:
            product (Product): The financial product to be priced.
            market_env (MarketEnvironment): The market environment providing
                access to market data (curves, spots, volatilities).
            autodiff (bool, optional): Whether to compute gradients using TensorFlow's autodiff. Defaults to False.
            price_currency (Currency, optional): Currency of the price: the price in the product's payment currency
                converted at the FX spot (``get_fx_spot``, a risk factor of the price: delta to the FX spot by AAD),
                ``product.standard_error`` too. Default: the payment currency.

        """
        # the market curves pinned (monotone convex coefficients once, inside the tape: gradients unchanged)
        if autodiff:
            with GradientTape() as tape:
                with market_env.pinned():
                    npv = self._in_currency(self.calculate_price(product, market_env), product, market_env,
                                            price_currency)
            product.price = npv
            self._tape = tape
        else:
            with market_env.pinned():
                product.price = self._in_currency(self.calculate_price(product, market_env), product, market_env,
                                                  price_currency)

    @staticmethod
    def _in_currency(npv, product, market_env, price_currency):
        """*npv* (in the payment currency of *product*) converted into *price_currency* at the FX spot."""
        payment = getattr(product, "payment_ccy", None) or product.ccy
        product.price_currency = payment if price_currency is None else price_currency
        if price_currency is None or price_currency == payment:
            return npv
        fx = market_env.get_fx_spot(payment, price_currency)
        if getattr(product, "standard_error", None) is not None:
            product.standard_error = product.standard_error * fx
        return npv * fx
