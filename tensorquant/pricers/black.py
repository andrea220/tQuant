from .pricer import Pricer
from ..instruments.option import VanillaOption
from ..markethandles.utils import market_map, ExerciseType
from ..timehandles.utils import Settings
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..markethandles.volatilitysurface import BlackConstantVolatility

from tensorflow_probability.python.distributions.normal import Normal
import tensorflow as tf


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
    dist = Normal(0, 1)
    return spot_price * dist.cdf(d1) - strike * tf.exp(
        -riskfree_rate * time_to_maturity
    ) * dist.cdf(d2)


class BlackScholesPricer(Pricer):
    def __init__(self, market_map):
        super().__init__()
        self._market_map = market_map
        self._s = None
        self._k = None
        self._t = None
        self._r = None
        self._sigma = None

    def calculate_price(self, product: VanillaOption, market):
        if not isinstance(product, VanillaOption):
            raise ValueError("product must be a VanillaOption")
        try:
            vol_map = market_map[f"VOLEQ:{product.ccy.name}"][product.underlying]
            vol_surface = market[f"VOLEQ:{vol_map}"]
        except:
            raise ValueError("Unknown Implied Volatility")
        try:
            disc_curve_map = market_map[f"IR:{product.ccy.name}"]["ON"]
            disc_curve = market[disc_curve_map]
        except:
            raise ValueError("Unknown Curve")
        try:
            spot_value = market[f"EQ:{product.underlying}"]
        except:
            raise ValueError("Unknown Underlying")

        vol = vol_surface.volatility(
            strike=spot_value / product.strike.numpy(),
            tenor=vol_surface.daycounter.year_fraction(
                Settings.evaluation_date, product.end_date
            ),
        )
        black_vol_handle = BlackConstantVolatility(
            Settings.evaluation_date,
            None,
            DayCounter(DayCounterConvention.Actual365),
            vol,
        )

        if not product.exercise_type.name == ExerciseType.European.name:
            raise ValueError("not a European option")
        self._s = tf.Variable(spot_value, dtype=tf.float32)
        self._k = product.strike
        self._t = tf.Variable(
            black_vol_handle.daycounter.year_fraction(
                Settings.evaluation_date, product.end_date
            ),
            dtype=tf.float32,
        )
        self._r = tf.Variable(
            -tf.math.log(disc_curve.discount(product.end_date)).numpy()
            / self._t.numpy(),
            dtype=tf.float32,
        )
        self._sigma = black_vol_handle.volatility()
        return blackscholes_calc(
            spot_price=self._s,
            strike=self._k,
            riskfree_rate=self._r,
            volatility=self._sigma,
            time_to_maturity=self._t,
        )
