from .pricer import Pricer
from ..instruments.option import VanillaOption
from ..markethandles.utils import market_map, ExerciseType
from ..timehandles.utils import Settings
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..markethandles.volatilitysurface import BlackConstantVolatility

from math import log, sqrt
from tensorflow import float32, float64, cast
from tensorflow_probability.python.distributions.normal import Normal


def black_formula(f, k, black_variance, df):
    std_dev = sqrt(black_variance)
    if std_dev < 0:
        raise ValueError("volatility must be positive")
    if df < 0:
        raise ValueError("discount must be positive")
    dist = Normal(0,1)

    log_moneyness = cast(log(f / k), float64)

    d1 = log_moneyness/std_dev + 0.5*std_dev
    d2 = d1 - std_dev

    n_d1 = dist.cdf(cast(d1, float32))
    n_d2 = dist.cdf(cast(d2, float32))

    n_d1 = cast(n_d1, float64)
    n_d2 = cast(n_d2, float64)
    return df * (f * n_d1 -  k * n_d2)


class BlackScholesPricer(Pricer):
    def __init__(self, market_map):
        super().__init__()
        self._market_map = market_map

    def calculate_price(self, product: VanillaOption, market):
        if not isinstance(product, VanillaOption):
            raise ValueError("product must be a VanillaOption")
        try:
            vol_map = market_map[f'VOLEQ:{product.ccy.name}'][product.underlying]
            vol_surface = market[f'VOLEQ:{vol_map}']
        except:
                raise ValueError("Unknown Implied Volatility")
        try:
            disc_curve_map = market_map[f'IR:{product.ccy.name}']['ON']
            disc_curve = market[disc_curve_map]
        except:
            raise ValueError("Unknown Curve")
        try:
            spot_value = market[f'EQ:{product.underlying}']
        except:
            raise ValueError("Unknown Underlying")

        vol = vol_surface.volatility(strike=spot_value/product.strike.numpy(), 
                                maturity=vol_surface.daycounter.year_fraction(Settings.evaluation_date,
                                                                            product.end_date))
        black_vol_handle = BlackConstantVolatility(Settings.evaluation_date,
                                                    None, DayCounter(DayCounterConvention.Actual365), vol)

        if not product.exercise_type.name == ExerciseType.European.name:
            raise ValueError("not a European option")
        implied_variance = black_vol_handle.variance(product.strike, product.end_date)
        dividend_discount = 1 #bsm_process.dividendYield().discount(maturity_date_ql) #1 #TODO: implementation of dividends
        df = disc_curve.discount(product.end_date)
        forward_price = spot_value * dividend_discount / df
        return black_formula(forward_price, product.strike, implied_variance, df)
    

