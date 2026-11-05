"""Equity underlying: a read-only view on the market data needed to price it."""

from __future__ import annotations

from datetime import date
from typing import Optional, Union

import pandas as pd
import tensorflow as tf

from ..numericalhandles.dtypes import DTYPE
from .dividendcurve import DividendCurve
from .ircurve import RateCurve
from .utils import Currency
from .volatilitysurface import VolatilitySurface

__all__ = ["Equity"]


class Equity:
    """Market data of one equity underlying, as needed for pricing.

    ``Equity`` is a **view**: it holds references to the objects stored in the
    :class:`MarketEnvironment` (the same ``tf.Variable`` risk factors), never
    copies. Bumping a variable in the market is seen here, and gradients of
    :meth:`forward` flow to the market risk factors.

    Dividend model (decided by the available data):

    - ``"discrete"``: a :class:`DividendCurve` is present; ``DIVYIELD`` is ignored.
    - ``"continuous"``: no discrete schedule, ``DIVYIELD`` is used as a continuous yield.
    - ``"none"``: neither is present.

    Build it with :meth:`MarketEnvironment.get_equity`.

    Attributes:
        ticker (str): Equity ticker.
        currency (Currency): Currency of the equity.
        spot (tf.Variable): Spot (market risk factor).
        repo (tf.Tensor): Repo margin (market variable, or a float64 zero constant if absent).
        dividends (DividendCurve | None): Discrete dividend schedule.
        div_yield (tf.Variable | None): Continuous dividend yield (only in the ``"continuous"`` model).
        vol_surface (VolatilitySurface | None): Implied volatility surface.
        discount_curve (RateCurve): Discount curve of the currency.
    """

    def __init__(
        self,
        ticker: str,
        currency: Currency,
        spot: tf.Variable,
        discount_curve: RateCurve,
        repo: Optional[tf.Variable] = None,
        dividends: Optional[DividendCurve] = None,
        div_yield: Optional[tf.Variable] = None,
        vol_surface: Optional[VolatilitySurface] = None,
    ) -> None:
        """Initialise the view (prefer :meth:`MarketEnvironment.get_equity`).

        Args:
            ticker (str): Equity ticker.
            currency (Currency): Currency of the equity.
            spot (tf.Variable): Spot variable from the market.
            discount_curve (RateCurve): Discount curve of the currency.
            repo (tf.Variable, optional): Repo variable; ``None`` means zero repo.
            dividends (DividendCurve, optional): Discrete dividend schedule.
            div_yield (tf.Variable, optional): Continuous yield, used only if
                ``dividends`` is ``None``.
            vol_surface (VolatilitySurface, optional): Implied volatility surface.
        """
        self.ticker = ticker
        self.currency = currency
        self.spot = spot
        self.discount_curve = discount_curve
        self.repo = repo if repo is not None else tf.constant(0.0, dtype=DTYPE)
        self._repo_is_market = repo is not None
        self.dividends = dividends
        self.div_yield = div_yield if dividends is None else None
        self.vol_surface = vol_surface

    # ------------------------------------------------------------------
    # Dividend model
    # ------------------------------------------------------------------

    @property
    def dividend_model(self) -> str:
        """``"discrete"``, ``"continuous"`` or ``"none"`` (see class docstring)."""
        if self.dividends is not None:
            return "discrete"
        if self.div_yield is not None:
            return "continuous"
        return "none"

    # ------------------------------------------------------------------
    # Pricing quantities
    # ------------------------------------------------------------------

    def year_fraction(self, maturity: Union[date, float]) -> float:
        """Year fraction to *maturity*, with the discount curve's day counter.

        Args:
            maturity (Union[date, float]): Date, or year fraction (returned as is).

        Returns:
            float: Year fraction from the curve reference date.
        """
        if isinstance(maturity, date):
            curve = self.discount_curve
            return curve.daycounter.year_fraction(curve.reference_date, maturity)
        return float(maturity)

    def pv_dividends(self, maturity: Union[date, float]) -> tf.Tensor:
        """Present value of the discrete dividends with ex-date up to *maturity*.

        Args:
            maturity (Union[date, float]): Horizon (date or year fraction).

        Returns:
            tf.Tensor: Scalar float64 PV (zero if there is no discrete schedule).
        """
        if self.dividends is None:
            return tf.constant(0.0, dtype=DTYPE)
        return tf.cast(self.dividends.pv_dividends(maturity, self.discount_curve), DTYPE)

    def forward(self, maturity: Union[date, float]) -> tf.Tensor:
        """Forward price for *maturity*.

        - discrete:   ``F = (S - PV_div(T)) · e^{-repo·T} / P(0,T)``
        - continuous: ``F = S · e^{-(q + repo)·T} / P(0,T)``

        Args:
            maturity (Union[date, float]): Maturity (date or year fraction).

        Returns:
            tf.Tensor: Scalar float64 forward, differentiable with respect to the
                market risk factors (spot, repo, dividend yield, curve pillars).
        """
        t = self.year_fraction(maturity)
        carry = self.repo
        if self.dividend_model == "continuous":
            carry = carry + self.div_yield
        df = tf.cast(self.discount_curve.discount(maturity), DTYPE)
        return (self.spot - self.pv_dividends(maturity)) * tf.exp(-carry * t) / df

    def implied_div_yield(self, maturity: Union[date, float]) -> tf.Tensor:
        """Continuous yield equivalent to the dividends up to *maturity*.

        Discrete model: ``q_eff = -ln(1 - PV_div(T)/S) / T``, so that
        ``S·e^{-q_eff·T} = S - PV_div(T)``. Continuous model: the market yield.

        Args:
            maturity (Union[date, float]): Maturity (date or year fraction).

        Returns:
            tf.Tensor: Scalar float64 yield.

        Raises:
            ValueError: If the maturity is not after the reference date.
        """
        t = self.year_fraction(maturity)
        if t <= 0.0:
            raise ValueError(f"Maturity must be after the reference date (T = {t})")
        if self.dividend_model == "continuous":
            return self.div_yield
        return -tf.math.log(1.0 - self.pv_dividends(maturity) / self.spot) / t

    # ------------------------------------------------------------------
    # Risk factors and display
    # ------------------------------------------------------------------

    def _key(self, data_type: str) -> str:
        return f"EQ:{self.currency.name}:{self.ticker}:{data_type}"

    def risk_factors(self) -> dict[str, list[tf.Variable]]:
        """Market risk factors this underlying depends on, keyed by market key.

        Returns:
            dict[str, list[tf.Variable]]: Same structure as
                :meth:`MarketEnvironment.risk_factors`, restricted to this equity
                (spot, repo, dividend yield or discrete dividend amounts, vol
                surface, discount curve).
        """
        factors = {self._key("SPOT"): [self.spot]}
        if self._repo_is_market:
            factors[self._key("REPO")] = [self.repo]
        if self.div_yield is not None:
            factors[self._key("DIVYIELD")] = [self.div_yield]
        if self.dividends is not None:
            factors[self._key("DIV")] = self.dividends.variables
        if self.vol_surface is not None:
            factors[self._key("VOL")] = self.vol_surface.variables
        curve_key = self.discount_curve.name or f"IR:{self.currency.name}:DISCOUNT:SPOT"
        factors[curve_key] = self.discount_curve.variables
        return factors

    def summary(self, maturity: Optional[Union[date, float]] = None) -> pd.DataFrame:
        """Table of the underlying's pricing data, optionally at a maturity.

        Args:
            maturity (Union[date, float], optional): If given, adds PV of dividends,
                implied dividend yield, forward and ATM-forward volatility.

        Returns:
            pandas.DataFrame: One row per field, column ``value``.
        """
        rows = {
            "ticker": self.ticker,
            "currency": self.currency.name,
            "spot": float(self.spot),
            "repo": float(self.repo),
            "dividend model": self.dividend_model,
            "div yield": float(self.div_yield) if self.div_yield is not None else None,
            "discrete dividends": len(self.dividends) if self.dividends is not None else 0,
            "discount curve": self.discount_curve.name,
            "vol surface": self.vol_surface is not None,
        }
        if maturity is not None:
            t = self.year_fraction(maturity)
            fwd = float(self.forward(maturity))
            rows["maturity"] = maturity
            rows["T (years)"] = t
            rows["PV dividends"] = float(self.pv_dividends(maturity))
            rows["implied div yield"] = float(self.implied_div_yield(maturity)) if t > 0 else None
            rows["forward"] = fwd
            rows["ATM-forward vol"] = (
                float(self.vol_surface.volatility(strike=fwd, tenor=t))
                if self.vol_surface is not None else None
            )
        return pd.DataFrame({"value": pd.Series(rows, dtype=object)})

    def __repr__(self) -> str:
        return (
            f"Equity(ticker={self.ticker!r}, currency={self.currency.name}, "
            f"spot={float(self.spot):.4f}, dividend_model={self.dividend_model!r}, "
            f"vol_surface={self.vol_surface is not None})"
        )
