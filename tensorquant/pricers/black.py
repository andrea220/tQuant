from .pricer import Pricer
from ..instruments.option import VanillaOption
from ..markethandles.utils import ExerciseType
from ..markethandles.marketenvironment import MarketEnvironment
from ..timehandles.utils import Settings
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..markethandles.volatilitysurface import BlackConstantVolatility

import tensorflow as tf


def _norm_cdf(x: tf.Tensor) -> tf.Tensor:
    """Standard normal CDF computed via tf.math.erf (no tensorflow_probability)."""
    return 0.5 * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))


def blackscholes_calc(
    spot_price: tf.Variable,
    strike: tf.Variable,
    riskfree_rate: tf.Variable,
    volatility: tf.Variable,
    time_to_maturity: tf.Variable,
) -> tf.Tensor:
    sqrt_t = tf.sqrt(time_to_maturity)
    d1 = (
        tf.math.log(spot_price / strike)
        + (riskfree_rate + (volatility**2) / 2) * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return spot_price * _norm_cdf(d1) - strike * tf.exp(
        -riskfree_rate * time_to_maturity
    ) * _norm_cdf(d2)


class BlackScholesPricer(Pricer):
    def __init__(self):
        """Initialize the Black-Scholes pricer.

        The pricer uses the default market_map from utils.py via MarketEnvironment.
        No market_map parameter is needed.
        """
        super().__init__()
        self._s = None
        self._k = None
        self._t = None
        self._r = None
        self._sigma = None
        self._f = None

    def _build_inputs(
        self, product: VanillaOption, market_env: MarketEnvironment
    ) -> tuple:
        """Extract and validate all market inputs needed for Black-Scholes.

        Performs product / exercise-type validation, market-data lookup and
        converts all inputs to ``tf.Variable`` so the gradient tape can watch
        them.

        Args:
            product (VanillaOption): The option to be priced.
            market_env (MarketEnvironment): Typed accessor for market data.

        Returns:
            tuple: ``(s, k, r, sigma, t, f)`` as ``tf.Variable`` / ``tf.Variable``.
                Where f is the forward price F = S * exp(r * T).

        Raises:
            ValueError: If the product is not a European VanillaOption or if
                any required market datum is missing.
        """
        if not isinstance(product, VanillaOption):
            raise ValueError("product must be a VanillaOption")
        if not product.exercise_type.name == ExerciseType.European.name:
            raise ValueError("not a European option")

        vol_surface = market_env.get_eq_vol_surface(product.ccy, product.underlying)
        disc_curve = market_env.get_ir_curve(product.ccy)  # Uses default ticker (ESTR for EUR, SOFR for USD)
        spot_value = market_env.get_eq_spot(product.underlying, ccy=product.ccy)

        sigma = vol_surface.volatility(
            strike=spot_value / product.strike.numpy(),
            tenor=vol_surface.daycounter.year_fraction(
                Settings.evaluation_date, product.end_date
            ),
        )
        
        s = tf.Variable(spot_value, dtype=tf.float32)
        k = product.strike
        t = tf.Variable(
            vol_surface.daycounter.year_fraction(
                Settings.evaluation_date, product.end_date
            ),
            dtype=tf.float32,
        )
        r = tf.Variable(
            -tf.math.log(disc_curve.discount(product.end_date)).numpy() / t.numpy(),
            dtype=tf.float32,
        )
        f = s * tf.exp(r * t)

        return s, k, r, sigma, t, f

    def _formula(
        self,
        s: tf.Variable,
        k: tf.Variable,
        r: tf.Variable,
        sigma: tf.Variable,
        t: tf.Variable,
    ) -> tf.Tensor:
        """Apply the Black-Scholes closed-form formula.

        Args:
            s (tf.Variable): Spot price.
            k (tf.Variable): Strike price.
            r (tf.Variable): Risk-free rate.
            sigma (tf.Variable): Implied volatility.
            t (tf.Variable): Time to maturity (year fraction).

        Returns:
            tf.Tensor: The Black-Scholes call price.
        """
        return blackscholes_calc(
            spot_price=s,
            strike=k,
            riskfree_rate=r,
            volatility=sigma,
            time_to_maturity=t,
        )

    def calculate_price(self, product: VanillaOption, market_env: MarketEnvironment) -> tf.Tensor:
        """Price a European VanillaOption using the Black-Scholes model.

        Delegates market-data extraction to :meth:`_build_inputs` and the actual
        computation to :meth:`_formula`.  Intermediate ``tf.Variable`` inputs are
        stored on ``self`` so that an external ``GradientTape`` (managed by
        :meth:`Pricer.price`) can compute greeks.

        Args:
            product (VanillaOption): The option to be priced.
            market_env (MarketEnvironment): The market environment providing
                access to market data (curves, spots, volatilities).

        Returns:
            tf.Tensor: The NPV of the option.
        """
        self._s, self._k, self._r, self._sigma, self._t, self._f = self._build_inputs(
            product, market_env
        )
        return self._formula(self._s, self._k, self._r, self._sigma, self._t)
