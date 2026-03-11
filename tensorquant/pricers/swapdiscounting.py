from .pricer import Pricer

from ..instruments.ois import Ois
from ..instruments.swap import Swap
from ..markethandles.utils import SwapType, Currency
from ..markethandles.marketenvironment import MarketEnvironment
from .floatingflow import (
    FloatingLegDiscounting,
    OisLegDiscounting,
)
from .fixedflow import FixedLegDiscounting  # , FixedCouponDiscounting
from datetime import date


class OisPricer(Pricer):
    def __init__(self):
        """Initialize the OIS pricer.

        The pricer uses the default market_map from utils.py via MarketEnvironment.
        No market_map parameter is needed.
        """
        super().__init__()
        self.disc_curve = None

    def calculate_price(self, product: Ois, market_env: MarketEnvironment):
        """Price an Overnight Indexed Swap (OIS).

        Args:
            product (Ois): The OIS to be priced.
            market_env (MarketEnvironment): The market environment providing
                access to market data (curves).

        Returns:
            float: The NPV of the OIS.

        Raises:
            TypeError: If the product is not an Ois.
            ValueError: If the discount curve is not found.
        """
        if not isinstance(product, Ois):
            raise TypeError("Wrong product type")

        try:
            # Overnight index name format: "CCY:ON" (e.g., "EUR:ON")
            index_name_parts = product._index.name.split(":")
            if len(index_name_parts) != 2:
                raise ValueError(
                    f"Invalid index name format: '{product._index.name}'. "
                    f"Expected format: 'CCY:ON'"
                )
            index_ccy_str, index_ticker = index_name_parts

            # Convert currency string to Currency enum
            try:
                index_ccy = Currency[index_ccy_str]
            except KeyError:
                raise ValueError(
                    f"Unknown currency '{index_ccy_str}' in index name '{product._index.name}'"
                )

            # Get single curve for OIS valuation and discounting
            # Legacy ticker "ON" is mapped inside MarketEnvironment.get_ir_curve
            self.disc_curve = market_env.get_ir_curve(index_ccy, ticker=index_ticker)

            # Store curve identifiers on the product for transparency
            product.discount_curve = self.disc_curve.name
            product.estimation_curve = product.discount_curve

        except ValueError as e:
            raise ValueError(f"Unknown Curve: {e}") from e

        floating_leg_pricer = OisLegDiscounting(product.floating_leg)
        fixed_leg_pricer = FixedLegDiscounting(product.fixed_leg)

        if product.swap_type == SwapType.Payer:
            product.floating_leg.price = floating_leg_pricer.calculate_price(
                self.disc_curve
            )
            product.fixed_leg.price = -fixed_leg_pricer.calculate_price(
                self.disc_curve
            )
        else:
            product.floating_leg.price = -floating_leg_pricer.calculate_price(
                self.disc_curve
            )
            product.fixed_leg.price = fixed_leg_pricer.calculate_price(
                self.disc_curve
            )
        return product.floating_leg.price + product.fixed_leg.price


class SwapPricer(Pricer):
    def __init__(self):
        """Initialize the plain vanilla interest-rate swap pricer.

        The pricer uses the default market_map from utils.py via MarketEnvironment.
        No market_map parameter is needed.
        """
        super().__init__()
        self.disc_curve = None
        self.fwd_curve = None

    def calculate_price(self, product: Swap, market_env: MarketEnvironment):
        """Price a plain vanilla interest-rate swap.

        Args:
            product (Swap): The swap to be priced.
            market_env (MarketEnvironment): The market environment providing
                access to market data (curves).

        Returns:
            float: The NPV of the swap.

        Raises:
            TypeError: If the product is not a Swap.
            ValueError: If the discount or estimation curve is not found.
        """
        if not isinstance(product, Swap):
            raise TypeError("Wrong product type")

        try:
            # Discount curve: use default overnight curve for the swap currency
            self.disc_curve = market_env.get_ir_curve(product.ccy)
            product.discount_curve = self.disc_curve.name

            # Forward (estimation) curve for the index
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

            self.fwd_curve = market_env.get_ir_curve(index_ccy, ticker=index_ticker)
            product.estimation_curve = self.fwd_curve.name

        except ValueError as e:
            raise ValueError(f"Unknown Curve: {e}") from e

        floating_leg_pricer = FloatingLegDiscounting(product.floating_leg)
        fixed_leg_pricer = FixedLegDiscounting(product.fixed_leg)

        if product.swap_type == SwapType.Payer:
            product.floating_leg.price = floating_leg_pricer.calculate_price(
                self.disc_curve, self.fwd_curve
            )
            product.fixed_leg.price = -fixed_leg_pricer.calculate_price(
                self.disc_curve
            )
        else:
            product.floating_leg.price = -floating_leg_pricer.calculate_price(
                self.disc_curve, self.fwd_curve
            )
            product.fixed_leg.price = fixed_leg_pricer.calculate_price(
                self.disc_curve
            )

        return product.floating_leg.price + product.fixed_leg.price
