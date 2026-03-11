from .pricer import Pricer
from ..instruments.deposit import Deposit
from ..markethandles.marketenvironment import MarketEnvironment
from ..timehandles.utils import DayCounterConvention, Settings
from ..timehandles.daycounter import DayCounter


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
    ) -> float:
        """Price a Deposit using discount curve.

        Args:
            product (Deposit): The deposit to be priced.
            market_env (MarketEnvironment): The market environment providing
                access to market data (curves).

        Returns:
            float: The NPV of the deposit.

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

        day_counter = DayCounter(DayCounterConvention.Actual365)
        ts = day_counter.year_fraction(Settings.evaluation_date, product.start_date)
        te = day_counter.year_fraction(Settings.evaluation_date, product.end_date)
        df_s = self.disc_curve.discount(ts)
        df_e = self.disc_curve.discount(te)

        start_cashflow = 0.0
        if ts >= 0.0:
            start_cashflow = 1.0
        end_cashflow = 0.0
        if te > 0.0:
            yf = product.day_counter.year_fraction(
                product.start_date, product.end_date
            )
            end_cashflow = 1.0 + product.rate * yf
        start_cashflow *= product.notional
        end_cashflow *= product.notional

        return start_cashflow * df_s - end_cashflow * df_e
