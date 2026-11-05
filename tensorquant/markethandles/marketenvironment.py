from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Optional
from enum import Enum
import re

import pandas as pd
import tensorflow as tf

from .utils import Currency, market_map as default_market_map
from .ircurve import RateCurve
from .volatilitysurface import VolatilitySurface
from .dividendcurve import DividendCurve
from .equity import Equity
from .correlation import CorrelationMatrix
from ..numericalhandles.dtypes import as_variable
from ..timehandles.daycounter import DayCounter, DayCounterConvention


class RiskFactor(Enum):
    """Enum for risk factor types in market data mapping.

    Attributes:
        IR (str): Interest Rate risk factor.
        EQ (str): Equity risk factor.
        FX (str): FX spot, key ``FX:<QUOTE>:<BASEQUOTE>:SPOT`` (1 BASE = value QUOTE).
        CORR (str): Correlation matrix, key ``CORR:ALL:GLOBAL:SPOT``.
    """
    IR = "IR"
    EQ = "EQ"
    FX = "FX"
    CORR = "CORR"


class MarketDataType(Enum):
    """Enum for market data types.

    Attributes:
        SPOT (str): Spot/current value (for curves or equity spots).
        VOL (str): Volatility data (for volatility surfaces).
        REPO (str): Repo rate (scalar) for an equity ticker.
        DIVYIELD (str): Continuous dividend yield (scalar) for an equity ticker.
        DIV (str): Discrete dividend schedule (DividendCurve) for an equity
            ticker.
    """
    SPOT = "SPOT"
    VOL = "VOL"
    REPO = "REPO"
    DIVYIELD = "DIVYIELD"
    DIV = "DIV"


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

    KEY_PATTERN = re.compile(r"^(IR|EQ|FX|CORR):([A-Z]{3}):([A-Z0-9_]+)$")
    VALUE_PATTERN = re.compile(r"^(IR|EQ|FX|CORR):([A-Z]{3}):([A-Z0-9_]+):(SPOT|VOL|REPO|DIVYIELD|DIV)$")

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
        - ``FX:*:*:SPOT``  → numeric scalar
        - ``CORR:*:*:SPOT`` → :class:`CorrelationMatrix`

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
                if data_type in (
                    MarketDataType.SPOT.value,
                    MarketDataType.REPO.value,
                    MarketDataType.DIVYIELD.value,
                ):
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
                elif data_type == MarketDataType.DIV.value:
                    if not isinstance(value, DividendCurve):
                        raise TypeError(
                            f"Market key '{key}' expects a DividendCurve, "
                            f"got {type(value).__name__}"
                        )
            elif risk_factor == RiskFactor.FX.value:
                if data_type != MarketDataType.SPOT.value:
                    raise ValueError(f"Market key '{key}': only FX SPOT is supported")
                if not isinstance(value, (int, float, tf.Tensor, tf.Variable)):
                    raise TypeError(
                        f"Market key '{key}' expects a numeric value, got {type(value).__name__}"
                    )
            elif risk_factor == RiskFactor.CORR.value:
                if data_type != MarketDataType.SPOT.value:
                    raise ValueError(f"Market key '{key}': only CORR SPOT is supported")
                if not isinstance(value, CorrelationMatrix):
                    raise TypeError(
                        f"Market key '{key}' expects a CorrelationMatrix, "
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

        # EQ, FX and CORR keys are open-ended (any ticker/pair is valid); they have
        # already been format-validated by validate_market so no whitelist check is needed.
        eq_pattern = re.compile(r"^(EQ|FX|CORR):[A-Z]{3}:[A-Z0-9_]+:(SPOT|VOL|REPO|DIVYIELD|DIV)$")

        orphans = [
            key for key in market
            if key not in valid_market_keys and not eq_pattern.match(key)
        ]

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

        # 4. Own copy of the dict; scalar equity/FX data become float64 tf.Variables
        #    (one per key, shared by every pricer that reads it).
        self._market = dict(market)
        for key, value in self._market.items():
            if self._is_scalar_key(key):
                self._market[key] = as_variable(
                    float(value) if isinstance(value, (int, float)) else value,
                    name=key.replace(":", "_"),
                )

    # ------------------------------------------------------------------
    # Risk factors
    # ------------------------------------------------------------------

    _EQ_SCALAR_TYPES = (
        MarketDataType.SPOT.value,
        MarketDataType.REPO.value,
        MarketDataType.DIVYIELD.value,
    )

    @classmethod
    def _is_scalar_key(cls, key: str) -> bool:
        """True for ``EQ:<CCY>:<TICKER>:SPOT|REPO|DIVYIELD`` and ``FX:<CCY>:<PAIR>:SPOT`` keys."""
        parts = key.split(":")
        if len(parts) != 4:
            return False
        if parts[0] == RiskFactor.EQ.value:
            return parts[3] in cls._EQ_SCALAR_TYPES
        return parts[0] == RiskFactor.FX.value and parts[3] == MarketDataType.SPOT.value

    def risk_factors(self) -> dict[str, list[tf.Variable]]:
        """Return the differentiable market inputs, keyed by market key.

        Use them as sources in ``tape.gradient`` to get sensitivities of a price
        (or of a whole portfolio priced under the same tape):

        - ``IR:<CCY>:<TICKER>:SPOT`` -> one variable per pillar rate (``RateCurve.variables``)
        - ``EQ:<CCY>:<TICKER>:VOL``  -> the volatility matrix ``[n_maturities, n_strikes]``
        - ``EQ:<CCY>:<TICKER>:SPOT|REPO|DIVYIELD`` -> one scalar variable
        - ``FX:<QUOTE>:<PAIR>:SPOT`` -> one scalar variable (stored pairs only)
        - ``CORR:ALL:GLOBAL:SPOT``   -> the correlation matrix ``[n, n]``

        Discrete dividends (``DIV``) are not risk factors yet.

        Returns:
            dict[str, list[tf.Variable]]: Market key -> list of variables.
        """
        factors: dict[str, list[tf.Variable]] = {}
        for key in sorted(self._market):
            value = self._market[key]
            if isinstance(value, (RateCurve, VolatilitySurface, CorrelationMatrix)):
                factors[key] = value.variables
            elif self._is_scalar_key(key):
                factors[key] = [value]
        return factors

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_data_path(
        cls,
        evaluation_date: date,
        data_path: str,
        calibration_set: str,
    ) -> "MarketEnvironment":
        """Build a :class:`MarketEnvironment` from a standardised data folder.

        Reads the four CSV files produced by the MDM data store
        (``curve.csv``, ``spot.csv``, ``vol.csv``, ``dividends.csv``) and
        constructs all market objects automatically.

        Expected CSV schemas
        --------------------
        ``curve.csv``
            Columns: ``CURVE_NAME`` (str), ``CURVE_PILLAR`` (date, format
            ``%d-%m-%Y``), ``CURVE_DATA`` (float, zero-coupon bond prices).
            Optional column: ``CURRENCY`` (3-letter ISO code; defaults to
            ``"EUR"`` when absent).

        ``spot.csv``
            Columns: ``ticker`` (e.g. ``"ISP IM Equity"``), ``currency``,
            ``spot_price``, ``repo``, ``div_yield``.

        ``vol.csv``
            Columns: ``ticker``, ``currency``, ``maturity`` (ISO date),
            ``strike`` (float), ``volatility`` (percent, e.g. 25.0 → 0.25).

        ``dividends.csv``
            Columns: ``ticker``, ``currency``, ``Ex Date``, ``Declared Date``,
            ``Amount Per Share``.  ``Payment Date`` is used when present.

        The market keys produced are:
        - ``IR:<CCY>:<CURVE_NAME>:SPOT`` for each unique curve in ``curve.csv``
        - ``EQ:<CCY>:<TICKER>:SPOT|REPO|DIVYIELD|VOL|DIV`` for equities

        Args:
            evaluation_date (date): Pricing / reference date.  The date is
                also used to derive the dated sub-folder
                (``YYYYMMDD`` appended to ``data_path``).
            data_path (str): Base data directory.  The string
                ``evaluation_date.strftime("%Y%m%d")`` is appended to obtain
                the dated folder, which must contain ``calibration_set``.
            calibration_set (str): Sub-directory name inside the dated folder
                that contains the four CSV files.

        Returns:
            MarketEnvironment: A fully populated environment ready for pricing.

        Raises:
            FileNotFoundError: If any of the expected CSV files is missing.
            ValueError: If the market or market_map structure is invalid.
        """
        _dc = DayCounter(DayCounterConvention.Actual365)

        data_path = data_path + evaluation_date.strftime('%Y%m%d')
        folder    = data_path + "/" + calibration_set

        ir_curve_df = pd.read_csv(f"{folder}/curve.csv")
        ir_curve_df['CURVE_PILLAR'] = pd.to_datetime(ir_curve_df['CURVE_PILLAR'], format='%d-%m-%Y').dt.date
        ir_curve_df['CURVE_DATA']   = pd.to_numeric(ir_curve_df['CURVE_DATA'], errors='raise').astype(float)
        estr_df  = ir_curve_df[ir_curve_df['CURVE_NAME'] == 'ESTR'].copy()
        eur6m_df = ir_curve_df[ir_curve_df['CURVE_NAME'] == 'EUR6M'].copy()

        eur_estr_curve = RateCurve.from_zcb(
            evaluation_date,
            estr_df['CURVE_PILLAR'].to_list(), estr_df['CURVE_DATA'].to_list(),
            interp='LINEAR',
            daycounter_convention=DayCounterConvention.Actual365,
        )
        eur_6m_curve = RateCurve.from_zcb(
            evaluation_date,
            eur6m_df['CURVE_PILLAR'].to_list(), eur6m_df['CURVE_DATA'].to_list(),
            interp='LINEAR',
            daycounter_convention=DayCounterConvention.Actual365,
        )

        spot_df = pd.read_csv(f"{folder}/spot.csv")
        spot_df.fillna(0, inplace=True)

        vol_df = pd.read_csv(f"{folder}/vol.csv")
        vol_df['maturity'] = pd.to_datetime(vol_df['maturity']).dt.date

        dividend_df = pd.read_csv(f"{folder}/dividends.csv")
        dividend_df['Ex Date']       = pd.to_datetime(dividend_df['Ex Date']).dt.date
        dividend_df['Declared Date'] = pd.to_datetime(dividend_df['Declared Date']).dt.date

        market: dict[str, Any] = {
            "IR:EUR:ESTR:SPOT": eur_estr_curve,
            "IR:EUR:6M:SPOT":   eur_6m_curve,
        }

        for _, row in spot_df.iterrows():
            ticker = row['ticker'].split()[0]
            ccy    = row['currency']
            market[f"EQ:{ccy}:{ticker}:SPOT"]     = float(row['spot_price'])
            market[f"EQ:{ccy}:{ticker}:REPO"]     = float(row['repo'])
            market[f"EQ:{ccy}:{ticker}:DIVYIELD"] = float(row['div_yield'])

        for full_ticker, grp in vol_df.groupby('ticker'):
            ticker    = full_ticker.split()[0]
            ccy       = grp['currency'].iloc[0]
            strikes   = sorted(grp['strike'].unique().tolist())
            mat_dates = sorted(grp['maturity'].unique().tolist())
            tenors    = [_dc.year_fraction(evaluation_date, m) for m in mat_dates]
            pivot = (
                grp.pivot(index='maturity', columns='strike', values='volatility') / 100.0
            ).reindex(index=mat_dates, columns=strikes)
            market[f"EQ:{ccy}:{ticker}:VOL"] = VolatilitySurface(
                reference_date=evaluation_date,
                calendar=None,
                daycounter=_dc,
                strike=strikes,
                maturity=tenors,
                volatility_matrix=pivot.values.tolist(),
            )

        for full_ticker, grp in dividend_df.groupby('ticker'):
            ticker = full_ticker.split()[0]
            ccy    = grp['currency'].iloc[0]
            market[f"EQ:{ccy}:{ticker}:DIV"] = DividendCurve.from_dataframe(
                reference_date=evaluation_date,
                df=grp,
                daycounter_convention=DayCounterConvention.Actual365,
            )

        return cls(market)

    MARKET_DATA_FILES = (
        "ir_curves.csv", "fx_spot.csv", "eq_spot.csv", "eq_vol.csv", "eq_dividends.csv", "correlations.csv",
    )

    @classmethod
    def from_folder(
        cls,
        root: str | os.PathLike,
        evaluation_date: date,
        calibration_set: Optional[str] = None,
    ) -> "MarketEnvironment":
        """Build a :class:`MarketEnvironment` from a ``marketdata/`` snapshot.

        Layout: ``<root>/<yyyymmdd>/<calibration_set>/`` with long-format CSV files
        (ISO dates, decimal units). Every file is optional; conventions come from
        the data:

        - ``ir_curves.csv``: ``ccy, curve, pillar, value, quote, daycount, interp``;
          ``quote`` is ``df`` (discount factors) or ``zero`` (continuously
          compounded zero rates) -> ``IR:<ccy>:<curve>:SPOT``.
        - ``fx_spot.csv``: ``base, quote, spot`` (1 base = spot quote) ->
          ``FX:<quote>:<base><quote>:SPOT``.
        - ``eq_spot.csv``: ``ticker, ccy, spot, repo, div_yield`` (empty cell =
          absent) -> ``EQ:<ccy>:<ticker>:SPOT|REPO|DIVYIELD``.
        - ``eq_vol.csv``: ``ticker, ccy, expiry, strike, vol``, a full expiry x
          strike grid per ticker -> ``EQ:<ccy>:<ticker>:VOL`` (Actual365 tenors).
        - ``eq_dividends.csv``: ``ticker, ccy, ex_date, pay_date, amount`` ->
          ``EQ:<ccy>:<ticker>:DIV``.
        - ``correlations.csv``: ``factor_1, factor_2, corr`` (one row per pair) ->
          ``CORR:ALL:GLOBAL:SPOT``.

        Args:
            root (str | os.PathLike): Base folder (e.g. ``"marketdata"``).
            evaluation_date (date): Snapshot date (folder ``yyyymmdd``) and curve
                reference date.
            calibration_set (str, optional): Snapshot folder inside the date
                folder (e.g. ``"1516"``); ``None`` takes the latest one.

        Returns:
            MarketEnvironment: The populated environment.

        Raises:
            FileNotFoundError: If the snapshot folder does not exist or contains
                none of the market data files.
            ValueError: If the data are inconsistent (unknown quote type, an
                incomplete vol grid, ...).
        """
        day_folder = Path(root) / evaluation_date.strftime("%Y%m%d")
        if calibration_set is None:
            sets = sorted(d.name for d in day_folder.iterdir() if d.is_dir()) if day_folder.is_dir() else []
            if not sets:
                raise FileNotFoundError(f"No snapshot found in '{day_folder}'")
            calibration_set = sets[-1]
        folder = day_folder / calibration_set
        if not folder.is_dir():
            raise FileNotFoundError(f"Snapshot folder '{folder}' does not exist")
        if not any((folder / name).is_file() for name in cls.MARKET_DATA_FILES):
            raise FileNotFoundError(f"No market data files in '{folder}'")

        def read(name: str) -> Optional[pd.DataFrame]:
            path = folder / name
            return pd.read_csv(path) if path.is_file() else None

        def to_dates(column: pd.Series) -> list[date]:
            return [d.date() for d in pd.to_datetime(column, format="%Y-%m-%d")]

        market: dict[str, Any] = {}

        curves = read("ir_curves.csv")
        if curves is not None:
            for (ccy, name), grp in curves.groupby(["ccy", "curve"], sort=False):
                quote, daycount, interp = (grp[c].unique() for c in ("quote", "daycount", "interp"))
                if len(quote) != 1 or len(daycount) != 1 or len(interp) != 1:
                    raise ValueError(f"Curve {ccy} {name}: quote/daycount/interp must be unique")
                grp = grp.sort_values("pillar")
                pillars, values = to_dates(grp["pillar"]), grp["value"].astype(float).tolist()
                convention = DayCounterConvention(daycount[0])
                if quote[0] == "df":
                    curve = RateCurve.from_zcb(evaluation_date, pillars, values, interp[0], convention)
                elif quote[0] == "zero":
                    curve = RateCurve(evaluation_date, pillars, values, interp[0], convention)
                else:
                    raise ValueError(f"Curve {ccy} {name}: unknown quote '{quote[0]}' (expected 'df' or 'zero')")
                market[f"{RiskFactor.IR.value}:{ccy}:{name}:{MarketDataType.SPOT.value}"] = curve

        fx = read("fx_spot.csv")
        if fx is not None:
            for row in fx.itertuples(index=False):
                market[f"{RiskFactor.FX.value}:{row.quote}:{row.base}{row.quote}:SPOT"] = float(row.spot)

        spots = read("eq_spot.csv")
        if spots is not None:
            for row in spots.itertuples(index=False):
                for column, data_type in (("spot", MarketDataType.SPOT), ("repo", MarketDataType.REPO),
                                          ("div_yield", MarketDataType.DIVYIELD)):
                    value = getattr(row, column)
                    if pd.notna(value):
                        market[f"{RiskFactor.EQ.value}:{row.ccy}:{row.ticker}:{data_type.value}"] = float(value)

        vols = read("eq_vol.csv")
        if vols is not None:
            dc = DayCounter(DayCounterConvention.Actual365)
            vols = vols.assign(expiry=to_dates(vols["expiry"]))
            for (ticker, ccy), grp in vols.groupby(["ticker", "ccy"], sort=False):
                grid = grp.pivot(index="expiry", columns="strike", values="vol").sort_index().sort_index(axis=1)
                if grid.isna().to_numpy().any():
                    raise ValueError(f"Vol surface {ticker}: incomplete expiry x strike grid")
                market[f"{RiskFactor.EQ.value}:{ccy}:{ticker}:{MarketDataType.VOL.value}"] = VolatilitySurface(
                    reference_date=evaluation_date,
                    calendar=None,
                    daycounter=dc,
                    strike=[float(k) for k in grid.columns],
                    maturity=[dc.year_fraction(evaluation_date, d) for d in grid.index],
                    volatility_matrix=grid.to_numpy(dtype=float).tolist(),
                )

        dividends = read("eq_dividends.csv")
        if dividends is not None:
            for (ticker, ccy), grp in dividends.groupby(["ticker", "ccy"], sort=False):
                market[f"{RiskFactor.EQ.value}:{ccy}:{ticker}:{MarketDataType.DIV.value}"] = (
                    DividendCurve.from_dataframe(
                        reference_date=evaluation_date,
                        df=grp.sort_values("ex_date"),
                        ex_date_col="ex_date",
                        amount_col="amount",
                        declared_date_col=None,
                        payment_date_col="pay_date",
                        currency=Currency(ccy),
                        daycounter_convention=DayCounterConvention.Actual365,
                    )
                )

        correlations = read("correlations.csv")
        if correlations is not None:
            market[cls.CORRELATION_KEY] = CorrelationMatrix.from_pairs(
                correlations[["factor_1", "factor_2", "corr"]].itertuples(index=False, name=None)
            )

        return cls(market)

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
                - GBP -> "SONIA"
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
                Currency.GBP: "SONIA",
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
                Currency.GBP: "SONIA",
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
    # Equity currency
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # FX
    # ------------------------------------------------------------------

    def _fx_pairs(self) -> dict[tuple[str, str], tf.Variable]:
        """Stored FX spots as ``{(base, quote): variable}``."""
        pairs = {}
        for key, value in self._market.items():
            parts = key.split(":")
            if parts[0] == RiskFactor.FX.value:
                quote, pair = parts[1], parts[2]
                pairs[(pair[:3], quote)] = value
        return pairs

    def get_fx_spot(self, base: Currency | str, quote: Currency | str) -> tf.Tensor:
        """FX spot: units of ``quote`` for 1 unit of ``base``.

        Resolved from the stored pairs only: direct, inverse (``1/x``) or cross
        through a common currency (e.g. EURGBP = EURUSD / GBPUSD). Crosses are not
        stored, so the result is differentiable w.r.t. the stored pairs.

        Args:
            base (Currency | str): Base currency.
            quote (Currency | str): Quote currency.

        Returns:
            tf.Tensor: float64 scalar (1.0 if ``base == quote``).

        Raises:
            ValueError: If the pair cannot be resolved from the stored pairs.
        """
        base = base.name if isinstance(base, Currency) else str(base)
        quote = quote.name if isinstance(quote, Currency) else str(quote)
        if base == quote:
            return tf.constant(1.0, dtype=tf.float64)
        pairs = self._fx_pairs()

        def direct(b: str, q: str):
            if (b, q) in pairs:
                return tf.convert_to_tensor(pairs[(b, q)])
            if (q, b) in pairs:
                return 1.0 / pairs[(q, b)]
            return None

        spot = direct(base, quote)
        if spot is not None:
            return spot
        pivots = sorted({c for pair in pairs for c in pair} - {base, quote})
        for pivot in pivots:
            leg_1, leg_2 = direct(base, pivot), direct(pivot, quote)
            if leg_1 is not None and leg_2 is not None:
                return leg_1 * leg_2
        raise ValueError(f"FX spot {base}{quote} cannot be resolved from the stored pairs {sorted(pairs)}")

    # ------------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------------

    CORRELATION_KEY = "CORR:ALL:GLOBAL:SPOT"

    def get_correlation_matrix(self) -> CorrelationMatrix:
        """The market correlation matrix (``CORR:ALL:GLOBAL:SPOT``).

        Raises:
            ValueError: If the market has no correlation matrix.
        """
        if self.CORRELATION_KEY not in self._market:
            raise ValueError(f"No correlation matrix in market ('{self.CORRELATION_KEY}')")
        return self._market[self.CORRELATION_KEY]

    def get_correlation(self, names: list[str]) -> tf.Tensor:
        """Correlation submatrix of the given factors (e.g. ``["EQ:EUR:TICKER1", "EQ:USD:TICKER41"]``).

        Args:
            names (list[str]): Factor names, in the order wanted.

        Returns:
            tf.Tensor: float64 ``[n, n]``, differentiable w.r.t. the market matrix.

        Raises:
            ValueError: If the market has no correlation matrix.
            KeyError: If a name is unknown.
        """
        return self.get_correlation_matrix().submatrix(names)

    def get_currency(self, ticker: str) -> Currency:
        """Return the :class:`Currency` associated with *ticker*.

        Scans the market dictionary for any key of the form
        ``EQ:<CCY>:<TICKER>:<TYPE>`` and returns the first currency found.
        No assumptions are made about which data type (SPOT, VOL, DIV, …) is
        present — the lookup succeeds as long as at least one entry for the
        ticker exists in the market.

        Args:
            ticker (str): Short equity ticker (e.g. ``"ISP"``, ``"SX5E"``).

        Returns:
            Currency: The currency enum value for the ticker.

        Raises:
            ValueError: If no market entry is found for *ticker*.
        """
        prefix = f"{RiskFactor.EQ.value}:"
        for key in self._market:
            parts = key.split(":")
            if len(parts) == 4 and parts[0] == RiskFactor.EQ.value and parts[2] == ticker:
                try:
                    return Currency(parts[1])
                except ValueError:
                    continue
        raise ValueError(
            f"No market entry found for equity ticker '{ticker}'. "
            f"Expected a key of the form 'EQ:<CCY>:{ticker}:<TYPE>'."
        )

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
            # EQ tickers are open-ended: try direct key construction without market_map
            direct_key = f"{RiskFactor.EQ.value}:{currency.name}:{ticker}:{MarketDataType.SPOT.value}"
            if direct_key in self._market:
                return self._market[direct_key]
        
        # Fallback: try direct lookup with old format for backward compatibility
        market_key = f"EQ:{ticker}"
        if market_key in self._market:
            return self._market[market_key]
        
        raise ValueError(f"Unknown equity ticker: {ticker}")

    # ------------------------------------------------------------------
    # Repo rates
    # ------------------------------------------------------------------

    def get_eq_repo(self, ticker: str, ccy: Optional[Currency] = None) -> Any:
        """Return the repo rate for *ticker*.

        Args:
            ticker (str): Identifier of the equity ticker (e.g. "ISP", "SX5E").
            ccy (Currency, optional): The currency. If None, tries EUR first, then USD.

        Returns:
            Any: The repo rate (scalar or tf.Tensor/tf.Variable).

        Raises:
            ValueError: If the ticker is not found in the market.
        """
        currencies_to_try = [ccy] if ccy is not None else [Currency.EUR, Currency.USD]

        for currency in currencies_to_try:
            if currency is None:
                continue
            direct_key = f"{RiskFactor.EQ.value}:{currency.name}:{ticker}:{MarketDataType.REPO.value}"
            if direct_key in self._market:
                return self._market[direct_key]

        raise ValueError(f"Unknown repo rate for equity ticker: {ticker}")

    # ------------------------------------------------------------------
    # Dividend yields
    # ------------------------------------------------------------------

    def get_eq_div_yield(self, ticker: str, ccy: Optional[Currency] = None) -> Any:
        """Return the continuous dividend yield for *ticker*.

        Args:
            ticker (str): Identifier of the equity ticker (e.g. "ISP", "SX5E").
            ccy (Currency, optional): The currency. If None, tries EUR first, then USD.

        Returns:
            Any: The dividend yield (scalar or tf.Tensor/tf.Variable).

        Raises:
            ValueError: If the ticker is not found in the market.
        """
        currencies_to_try = [ccy] if ccy is not None else [Currency.EUR, Currency.USD]

        for currency in currencies_to_try:
            if currency is None:
                continue
            direct_key = f"{RiskFactor.EQ.value}:{currency.name}:{ticker}:{MarketDataType.DIVYIELD.value}"
            if direct_key in self._market:
                return self._market[direct_key]

        raise ValueError(f"Unknown dividend yield for equity ticker: {ticker}")

    # ------------------------------------------------------------------
    # Discrete dividend schedules
    # ------------------------------------------------------------------

    def get_eq_dividends(
        self, ticker: str, ccy: Optional[Currency] = None
    ) -> DividendCurve:
        """Return the discrete :class:`DividendCurve` for *ticker*.

        Looks up the key ``EQ:<CCY>:<TICKER>:DIV`` in the market dictionary.

        Args:
            ticker (str): Equity ticker identifier (e.g. ``"ISP"``, ``"SX5E"``).
            ccy (Currency, optional): Currency of the equity.  If ``None``,
                tries EUR first, then USD.

        Returns:
            DividendCurve: The discrete dividend schedule for the ticker.

        Raises:
            ValueError: If no dividend curve is found for the (ticker, ccy)
                combination.
        """
        currencies_to_try = [ccy] if ccy is not None else [Currency.EUR, Currency.USD]

        for currency in currencies_to_try:
            if currency is None:
                continue
            direct_key = (
                f"{RiskFactor.EQ.value}:{currency.name}:{ticker}"
                f":{MarketDataType.DIV.value}"
            )
            if direct_key in self._market:
                return self._market[direct_key]

        raise ValueError(
            f"No DividendCurve found for equity ticker: '{ticker}'. "
            f"Expected key 'EQ:<CCY>:{ticker}:DIV' in market."
        )

    # ------------------------------------------------------------------
    # Volatility surfaces
    # ------------------------------------------------------------------

    def get_eq_vol_surface(
        self, ticker: str, ccy: Optional[Currency] = None
    ) -> VolatilitySurface:
        """Return the implied-volatility surface for *ticker*.

        Args:
            ticker (str): Identifier of the equity ticker (e.g. "SX5E", "ISP").
            ccy (Currency, optional): The currency. If ``None``, tries EUR
                first, then USD, then scans all currencies in the market.

        Returns:
            VolatilitySurface: The corresponding volatility surface object.

        Raises:
            ValueError: If the vol surface is not found.
        """
        currencies_to_try = [ccy] if ccy is not None else [Currency.EUR, Currency.USD]

        for currency in currencies_to_try:
            if currency is None:
                continue
            # Try market_map first
            map_key = f"{RiskFactor.EQ.value}:{currency.name}:{ticker}"
            if map_key in self._market_map:
                try:
                    market_key = self._market_map[map_key][MarketDataType.VOL.value]
                    return self._market[market_key]
                except KeyError:
                    pass
            # Try direct key construction (open-ended EQ tickers)
            direct_key = f"{RiskFactor.EQ.value}:{currency.name}:{ticker}:{MarketDataType.VOL.value}"
            if direct_key in self._market:
                return self._market[direct_key]

        # Fallback: scan all currencies present in the market for this ticker
        for key in self._market:
            parts = key.split(":")
            if (
                len(parts) == 4
                and parts[0] == RiskFactor.EQ.value
                and parts[2] == ticker
                and parts[3] == MarketDataType.VOL.value
            ):
                return self._market[key]

        # Legacy key format
        old_key = f"VOLEQ:{ticker}"
        if old_key in self._market:
            return self._market[old_key]

        raise ValueError(
            f"No VolatilitySurface found for equity ticker '{ticker}'. "
            f"Expected a key of the form 'EQ:<CCY>:{ticker}:VOL' in market."
        )

    # ------------------------------------------------------------------
    # Equity underlying (view)
    # ------------------------------------------------------------------

    def get_equity(self, ticker: str, ccy: Optional[Currency] = None) -> Equity:
        """Return an :class:`Equity` view of *ticker*'s pricing data.

        The view references the market objects (no copies): spot, repo,
        discrete dividends or continuous yield, vol surface and the discount
        curve of the currency (``get_ir_curve(ccy)``). Optional data that are
        missing are left as ``None`` (repo defaults to zero). ``DIVYIELD`` is
        used only when there is no ``DIV`` schedule.

        Args:
            ticker (str): Equity ticker.
            ccy (Currency, optional): Currency; resolved from the market if ``None``.

        Returns:
            Equity: The underlying view.

        Raises:
            ValueError: If the ticker has no market data or no spot.
        """
        if ccy is None:
            ccy = self.get_currency(ticker)

        def optional(getter):
            try:
                return getter(ticker, ccy=ccy)
            except ValueError:
                return None

        dividends = optional(self.get_eq_dividends)
        return Equity(
            ticker=ticker,
            currency=ccy,
            spot=self.get_eq_spot(ticker, ccy=ccy),
            discount_curve=self.get_ir_curve(ccy),
            repo=optional(self.get_eq_repo),
            dividends=dividends,
            div_yield=optional(self.get_eq_div_yield) if dividends is None else None,
            vol_surface=optional(self.get_eq_vol_surface),
        )
