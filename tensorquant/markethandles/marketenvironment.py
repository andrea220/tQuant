from typing import Any, Optional
from enum import Enum
import re

from .utils import Currency, market_map as default_market_map
from .ircurve import RateCurve
from .volatilitysurface import VolatilitySurface


class RiskFactor(Enum):
    """Enum for risk factor types in market data mapping.

    Attributes:
        IR (str): Interest Rate risk factor.
        EQ (str): Equity risk factor.
    """
    IR = "IR"
    EQ = "EQ"


class MarketDataType(Enum):
    """Enum for market data types.

    Attributes:
        SPOT (str): Spot/current value (for curves or equity spots).
        VOL (str): Volatility data (for volatility surfaces).
    """
    SPOT = "SPOT"
    VOL = "VOL"


class MarketMapValidator:
    """Validates the structure of a market_map dictionary.

    The market_map should follow the structure:
        {
            "RiskFactor:CCY:TICKER": {
                "SPOT": "RiskFactor:CCY:TICKER:SPOT",
                "VOL": "RiskFactor:CCY:TICKER:VOL"  # optional
            },
            ...
        }

    Examples:
        - "IR:EUR:ESTR" -> {"SPOT": "IR:EUR:ESTR:SPOT"}
        - "EQ:EUR:SX5E" -> {"SPOT": "EQ:EUR:SX5E:SPOT", "VOL": "EQ:EUR:SX5E:VOL"}
    """

    KEY_PATTERN = re.compile(r"^(IR|EQ):([A-Z]{3}):([A-Z0-9_]+)$")
    VALUE_PATTERN = re.compile(r"^(IR|EQ):([A-Z]{3}):([A-Z0-9_]+):(SPOT|VOL)$")

    @classmethod
    def validate_key(cls, key: str) -> tuple[str, str, str]:
        """Validate and parse a market_map key.

        Args:
            key (str): The key to validate (format: "RiskFactor:CCY:TICKER").

        Returns:
            tuple: (risk_factor, ccy, ticker) if valid.

        Raises:
            ValueError: If the key format is invalid.
        """
        match = cls.KEY_PATTERN.match(key)
        if not match:
            raise ValueError(
                f"Invalid market_map key format: '{key}'. "
                f"Expected format: 'RiskFactor:CCY:TICKER' (e.g., 'IR:EUR:ESTR', 'EQ:EUR:SX5E')"
            )
        risk_factor, ccy, ticker = match.groups()
        if risk_factor not in [rf.value for rf in RiskFactor]:
            raise ValueError(
                f"Invalid risk factor '{risk_factor}' in key '{key}'. "
                f"Must be one of: {[rf.value for rf in RiskFactor]}"
            )
        return risk_factor, ccy, ticker

    @classmethod
    def validate_value(cls, value: str, expected_key: str) -> tuple[str, str, str, str]:
        """Validate and parse a market_map value.

        Args:
            value (str): The value to validate (format: "RiskFactor:CCY:TICKER:TYPE").
            expected_key (str): The key this value belongs to.

        Returns:
            tuple: (risk_factor, ccy, ticker, data_type) if valid.

        Raises:
            ValueError: If the value format is invalid or doesn't match the key.
        """
        match = cls.VALUE_PATTERN.match(value)
        if not match:
            raise ValueError(
                f"Invalid market_map value format: '{value}'. "
                f"Expected format: 'RiskFactor:CCY:TICKER:TYPE' "
                f"(e.g., 'IR:EUR:ESTR:SPOT', 'EQ:EUR:SX5E:VOL')"
            )
        risk_factor, ccy, ticker, data_type = match.groups()

        if data_type not in [dt.value for dt in MarketDataType]:
            raise ValueError(
                f"Invalid data type '{data_type}' in value '{value}'. "
                f"Must be one of: {[dt.value for dt in MarketDataType]}"
            )

        # Verify the value matches the key
        expected_prefix = expected_key
        if not value.startswith(expected_prefix + ":"):
            raise ValueError(
                f"Value '{value}' does not match key '{expected_key}'. "
                f"Value must start with '{expected_key}:'"
            )

        return risk_factor, ccy, ticker, data_type

    @classmethod
    def validate_market_map(cls, market_map: dict[str, Any]) -> None:
        """Validate the entire market_map structure.

        Args:
            market_map (dict): The market_map dictionary to validate.

        Raises:
            ValueError: If the market_map structure is invalid.
        """
        if not isinstance(market_map, dict):
            raise ValueError("market_map must be a dictionary")

        for key, value_dict in market_map.items():
            # Validate key format
            cls.validate_key(key)

            # Validate value_dict structure
            if not isinstance(value_dict, dict):
                raise ValueError(
                    f"Value for key '{key}' must be a dictionary, got {type(value_dict).__name__}"
                )

            # Must have at least SPOT
            if MarketDataType.SPOT.value not in value_dict:
                raise ValueError(
                    f"Key '{key}' must have at least a 'SPOT' entry"
                )

            # Validate all entries in value_dict
            for data_type_str, market_key in value_dict.items():
                if data_type_str not in [dt.value for dt in MarketDataType]:
                    raise ValueError(
                        f"Invalid data type '{data_type_str}' in key '{key}'. "
                        f"Must be one of: {[dt.value for dt in MarketDataType]}"
                    )
                cls.validate_value(market_key, key)

    @classmethod
    def validate_market(cls, market: dict[str, Any]) -> None:
        """Validate the structure and value types of a market dict.

        Each key must follow the format ``RiskFactor:CCY:TICKER:TYPE`` and each
        value must be of the expected type:

        - ``IR:*:*:SPOT``  → :class:`RateCurve`
        - ``EQ:*:*:SPOT``  → numeric scalar (``int``, ``float``,
          ``tf.Tensor``, ``tf.Variable``)
        - ``EQ:*:*:VOL``   → :class:`VolatilitySurface`

        Args:
            market (dict): The market dict to validate.

        Raises:
            ValueError: If any key has an invalid format.
            TypeError: If any value is of the wrong type for its key.
        """
        import tensorflow as tf

        if not isinstance(market, dict):
            raise ValueError("market must be a dictionary")

        for key, value in market.items():
            match = cls.VALUE_PATTERN.match(key)
            if not match:
                raise ValueError(
                    f"Invalid market key format: '{key}'. "
                    f"Expected format: 'RiskFactor:CCY:TICKER:TYPE' "
                    f"(e.g., 'IR:EUR:ESTR:SPOT', 'EQ:EUR:SX5E:VOL')"
                )
            risk_factor, _ccy, _ticker, data_type = match.groups()

            if risk_factor == RiskFactor.IR.value:
                if data_type == MarketDataType.SPOT.value:
                    if not isinstance(value, RateCurve):
                        raise TypeError(
                            f"Market key '{key}' expects a RateCurve, "
                            f"got {type(value).__name__}"
                        )
            elif risk_factor == RiskFactor.EQ.value:
                if data_type == MarketDataType.SPOT.value:
                    if not isinstance(value, (int, float, tf.Tensor, tf.Variable)):
                        raise TypeError(
                            f"Market key '{key}' expects a numeric value "
                            f"(int, float, tf.Tensor, tf.Variable), "
                            f"got {type(value).__name__}"
                        )
                elif data_type == MarketDataType.VOL.value:
                    if not isinstance(value, VolatilitySurface):
                        raise TypeError(
                            f"Market key '{key}' expects a VolatilitySurface, "
                            f"got {type(value).__name__}"
                        )

    @classmethod
    def validate_market_against_map(
        cls,
        market: dict[str, Any],
        market_map: dict[str, Any],
    ) -> None:
        """Cross-validate that all keys in *market* are referenced in *market_map*.

        The market_map is a global routing table that may define more instruments
        than those actually needed in a given session; it does NOT need to be
        fully covered by the market dict.  The reverse must hold: every key
        provided in market must be resolvable via market_map, so that no
        "orphan" or misspelled market data can silently slip through.

        Args:
            market (dict): The flat market-data dictionary.
            market_map (dict): The logical-to-key mapping dictionary.

        Raises:
            ValueError: If any key in market is not referenced in market_map.
        """
        # Build the complete set of valid market keys from market_map values
        valid_market_keys = {
            market_key
            for data_types in market_map.values()
            for market_key in data_types.values()
        }

        orphans = [key for key in market if key not in valid_market_keys]

        if orphans:
            raise ValueError(
                "The following market keys are not referenced in market_map "
                "(unknown or misspelled instruments):\n"
                + "\n".join(f"  '{k}'" for k in orphans)
            )


class MarketEnvironment:
    """Provides typed access to market data objects through a market_map.

    Encapsulates a raw market dict and the associated market_map, exposing
    named accessor methods instead of string-key lookups scattered across
    individual pricers.

    The `market_map` follows the structure:
        {
            "RiskFactor:CCY:TICKER": {
                "SPOT": "RiskFactor:CCY:TICKER:SPOT",
                "VOL": "RiskFactor:CCY:TICKER:VOL"  # optional
            }
        }

    The `market_map` is by default imported from `utils.py` (hardcoded
    configuration), but can be overridden by passing a custom `market_map`
    to the constructor. The structure is validated on initialization.

    Attributes:
        _market (dict): Flat dictionary mapping instrument keys to market
            objects (curves, spots, vol surfaces, …).
        _market_map (dict): Mapping that translates logical names (ccy,
            ticker) to the concrete keys used in `_market`.

    Methods:
        get_ir_curve: Returns the RateCurve for a given currency and ticker.
        get_eq_spot: Returns the spot value for a given equity ticker.
        get_eq_vol_surface: Returns the VolatilitySurface for a given
            currency / ticker pair.
    """

    def __init__(
        self,
        market: dict[str, Any],
        market_map: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialise the MarketEnvironment.

        Args:
            market (dict): Flat market-data dictionary. Keys must follow the
                format ``RiskFactor:CCY:TICKER:TYPE`` (e.g. ``IR:EUR:ESTR:SPOT``,
                ``EQ:EUR:SX5E:VOL``). Values must be of the expected type for
                their key (``RateCurve``, ``VolatilitySurface``, or numeric).
            market_map (dict, optional): Logical-to-key mapping dictionary.
                If not provided, uses the default ``market_map`` from ``utils.py``.

        Raises:
            ValueError: If the market or market_map structure is invalid, or if
                any key referenced in market_map is missing from market.
            TypeError: If any market value is of the wrong type for its key.
        """
        self._market_map = market_map if market_map is not None else default_market_map

        # 1. Validate market_map structure
        MarketMapValidator.validate_market_map(self._market_map)

        # 2. Validate market keys format and value types
        MarketMapValidator.validate_market(market)

        # 3. Cross-validate: all keys referenced in market_map must exist in market
        MarketMapValidator.validate_market_against_map(market, self._market_map)

        self._market = market

    # ------------------------------------------------------------------
    # Interest-rate curves
    # ------------------------------------------------------------------

    def get_ir_curve(self, ccy: Currency, ticker: Optional[str] = None) -> RateCurve:
        """Return the interest-rate curve for *ccy* with *ticker*.

        Args:
            ccy (Currency): The currency of the curve (e.g. Currency.EUR).
            ticker (str, optional): The ticker identifier (e.g. "ESTR", "SOFR", "3M").
                If None, defaults to the overnight rate ticker for the currency:
                - EUR -> "ESTR"
                - USD -> "SOFR"
                - Other -> raises ValueError

        Returns:
            RateCurve: The corresponding RateCurve object.

        Raises:
            ValueError: If the (ccy, ticker) pair is not found.
        """
        # Default ticker mapping for overnight rates
        if ticker is None:
            ticker_map = {
                Currency.EUR: "ESTR",
                Currency.USD: "SOFR",
            }
            if ccy not in ticker_map:
                raise ValueError(
                    f"No default ticker for currency {ccy.name}. "
                    f"Please specify ticker explicitly."
                )
            ticker = ticker_map[ccy]
        
        # Legacy support: map "ON" to default overnight ticker
        if ticker == "ON":
            ticker_map = {
                Currency.EUR: "ESTR",
                Currency.USD: "SOFR",
            }
            ticker = ticker_map.get(ccy, "ESTR")
        
        map_key = f"{RiskFactor.IR.value}:{ccy.name}:{ticker}"
        try:
            market_key = self._market_map[map_key][MarketDataType.SPOT.value]
            curve = self._market[market_key]
            # Set the curve name to the market key for identification
            curve.name = market_key
            return curve
        except KeyError as e:
            raise ValueError(
                f"Unknown IR curve: ccy={ccy.name}, ticker={ticker}. "
                f"Key '{map_key}' not found in market_map."
            ) from e

    # ------------------------------------------------------------------
    # Equity spots
    # ------------------------------------------------------------------

    def get_eq_spot(self, ticker: str, ccy: Optional[Currency] = None) -> Any:
        """Return the spot value for *ticker*.

        Args:
            ticker (str): Identifier of the equity ticker (e.g. "SX5E", "DEFAULT").
            ccy (Currency, optional): The currency. If None, tries EUR first, then USD.

        Returns:
            Any: The spot value (scalar or tf.Tensor/tf.Variable).

        Raises:
            ValueError: If the ticker is not found in the market.
        """
        # If currency is specified, try that first
        currencies_to_try = [ccy] if ccy is not None else [Currency.EUR, Currency.USD]
        
        for currency in currencies_to_try:
            if currency is None:
                continue
            map_key = f"{RiskFactor.EQ.value}:{currency.name}:{ticker}"
            if map_key in self._market_map:
                try:
                    market_key = self._market_map[map_key][MarketDataType.SPOT.value]
                    return self._market[market_key]
                except KeyError:
                    continue
        
        # Fallback: try direct lookup with old format for backward compatibility
        market_key = f"EQ:{ticker}"
        if market_key in self._market:
            return self._market[market_key]
        
        raise ValueError(f"Unknown equity ticker: {ticker}")

    # ------------------------------------------------------------------
    # Volatility surfaces
    # ------------------------------------------------------------------

    def get_eq_vol_surface(
        self, ccy: Currency, ticker: str
    ) -> VolatilitySurface:
        """Return the implied-volatility surface for *ticker* in *ccy*.

        Args:
            ccy (Currency): The currency of the product (e.g. Currency.EUR).
            ticker (str): Identifier of the equity ticker (e.g. "SX5E", "DEFAULT").

        Returns:
            VolatilitySurface: The corresponding volatility surface object.

        Raises:
            ValueError: If the vol surface is not found.
        """
        map_key = f"{RiskFactor.EQ.value}:{ccy.name}:{ticker}"
        try:
            market_key = self._market_map[map_key][MarketDataType.VOL.value]
            return self._market[market_key]
        except KeyError as e:
            # Fallback: try old format for backward compatibility
            old_key = f"VOLEQ:{ticker}"
            if old_key in self._market:
                return self._market[old_key]
            raise ValueError(
                f"Unknown Implied Volatility: ccy={ccy.name}, ticker={ticker}. "
                f"Key '{map_key}' not found in market_map or missing VOL entry."
            ) from e
