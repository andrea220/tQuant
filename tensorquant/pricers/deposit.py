import tensorflow as tf

from .pricer import Pricer
from ..numericalhandles.dtypes import DTYPE
from ..instruments.deposit import Deposit
from ..markethandles.marketenvironment import MarketEnvironment
from ..timehandles.utils import Settings


class DepositPricer(Pricer):
    def __init__(self):
        """Initialize the Deposit pricer.

        The pricer uses the default market_map from utils.py via MarketEnvironment.
        No market_map parameter is needed.
        """
        super().__init__()
        self.disc_curve = None

    def calculate_price(
        self, product: Deposit, market_env: MarketEnvironment
    ) -> tf.Tensor:
        """Price a Deposit using discount curve.

        ``NPV = N P(0, start) - N (1 + r tau) P(0, end)`` (the start flow only if it is
        not in the past, the end flow only before maturity), discount factors read at
        the dates so the curve uses its own day count.

        Args:
            product (Deposit): The deposit to be priced.
            market_env (MarketEnvironment): The market environment providing
                access to market data (curves).

        Returns:
            tf.Tensor: float64 NPV of the deposit.

        Raises:
            TypeError: If the product is not a Deposit.
            ValueError: If the discount curve is not found.
        """
        if not isinstance(product, Deposit):
            raise TypeError("Wrong product type")

        try:
            # Get discount curve using default ticker (ESTR for EUR, SOFR for USD)
            self.disc_curve = market_env.get_ir_curve(product.ccy)
        except ValueError as e:
            raise ValueError(f"Unknown Curve: {e}") from e

        npv = tf.constant(0.0, dtype=DTYPE)
        if product.start_date >= Settings.evaluation_date:
            npv = npv + product.notional * tf.cast(self.disc_curve.discount(product.start_date), DTYPE)
        if product.end_date > Settings.evaluation_date:
            yf = product.day_counter.year_fraction(product.start_date, product.end_date)
            npv = npv - product.notional * (1.0 + product.rate * yf) * tf.cast(
                self.disc_curve.discount(product.end_date), DTYPE
            )
        return npv
