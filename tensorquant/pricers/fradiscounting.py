import tensorflow as tf

from .pricer import Pricer
from ..numericalhandles.dtypes import DTYPE
from ..instruments.forward import Fra
from ..markethandles.marketenvironment import MarketEnvironment
from ..markethandles.utils import Currency
from ..timehandles.utils import Settings


class FraPricer(Pricer):
    def __init__(self):
        """Initialize the FRA pricer.

        The pricer uses the default market_map from utils.py via MarketEnvironment.
        No market_map parameter is needed.
        """
        super().__init__()
        self.disc_curve = None
        self.fwd_curve = None
        self.fwd = None

    def calculate_price(
        self, product: Fra, market_env: MarketEnvironment
    ) -> tf.Tensor:
        """Price a Forward Rate Agreement (FRA), settled at the start date.

        ``NPV = side * N * tau * (F - K) * P_disc(0, T1) / (1 + F * tau)``, with
        ``tau`` the FRA accrual on the product dates ``T1 = start_date``,
        ``T2 = end_date`` and ``F = (P_fwd(0, T1) / P_fwd(0, T2) - 1) / tau`` from the
        index (forwarding) curve on the same dates. Zero once the period has started.

        Args:
            product (Fra): The FRA to be priced.
            market_env (MarketEnvironment): The market environment providing
                access to market data (curves).

        Returns:
            tf.Tensor: float64 NPV of the FRA.

        Raises:
            TypeError: If the product is not a Fra.
            ValueError: If the discount or forward curve is not found.
        """
        if not isinstance(product, Fra):
            raise TypeError("Wrong product type")

        try:
            # Get discount curve using default ticker (ESTR for EUR, SOFR for USD)
            self.disc_curve = market_env.get_ir_curve(product.ccy)

            # Get forward curve for the index
            # Index name format: "CCY:TENOR" (e.g., "EUR:6M")
            index_name_parts = product._index.name.split(":")
            if len(index_name_parts) != 2:
                raise ValueError(
                    f"Invalid index name format: '{product._index.name}'. "
                    f"Expected format: 'CCY:TENOR'"
                )
            index_ccy_str, index_ticker = index_name_parts

            # Convert currency string to Currency enum
            try:
                index_ccy = Currency[index_ccy_str]
            except KeyError:
                raise ValueError(
                    f"Unknown currency '{index_ccy_str}' in index name '{product._index.name}'"
                )

            # Map legacy ticker format (e.g., "6M" -> "6M" or handle appropriately)
            # For now, use the ticker as-is from the index name
            self.fwd_curve = market_env.get_ir_curve(index_ccy, ticker=index_ticker)

        except ValueError as e:
            raise ValueError(f"Unknown Curve: {e}") from e

        if product.start_date <= Settings.evaluation_date:
            self.fwd = None
            return tf.constant(0.0, dtype=DTYPE)

        t1, t2 = product.start_date, product.end_date
        tau = product.day_counter.year_fraction(t1, t2)
        fwd = (tf.cast(self.fwd_curve.discount(t1), DTYPE) / tf.cast(self.fwd_curve.discount(t2), DTYPE) - 1.0) / tau
        self.fwd = fwd
        df_settle = tf.cast(self.disc_curve.discount(t1), DTYPE)
        return (
            product.side.value * product.notional * tau * (fwd - product.fixed_rate)
            * df_settle / (1.0 + fwd * tau)
        )
