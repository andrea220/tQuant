from .pricer import Pricer
from ..instruments.forward import Fra
from ..markethandles.marketenvironment import MarketEnvironment
from ..markethandles.utils import Currency
from ..timehandles.utils import TimeUnit, BusinessDayConvention, Settings


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
    ) -> float:
        """Price a Forward Rate Agreement (FRA).

        Args:
            product (Fra): The FRA to be priced.
            market_env (MarketEnvironment): The market environment providing
                access to market data (curves).

        Returns:
            float: The NPV of the FRA.

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

        pv = 0.0
        fwd = 0.0
        accrual = 0.0
        
        if product.start_date > Settings.evaluation_date:
            accrual = product.day_counter.year_fraction(
                product.start_date, product.end_date
            )
            fixing_d = product.fixing_date
            d1 = product._index.fixing_calendar.advance(
                fixing_d, 2, TimeUnit.Days, BusinessDayConvention.ModifiedFollowing
            )  # valuedate-start date
            d2 = product._index.fixing_maturity(d1)
            t = product._index.daycounter.year_fraction(d1, d2)
            disc1 = self.fwd_curve.discount(d1)
            disc2 = self.fwd_curve.discount(d2)
            fwd = (disc1 / disc2 - 1) / t
            self.fwd = fwd
            pv += (
                product.notional
                * accrual
                * (fwd - product.fixed_rate)
                * product.side.value
                * self.disc_curve.discount(
                    product.day_counter.year_fraction(
                        Settings.evaluation_date, product.end_date
                    )
                )
            )
        
        if accrual > 0.0:
            return pv / (1 + fwd * accrual)
        else:
            return pv
