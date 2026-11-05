import math

import tensorflow as tf

from .pricer import Pricer
from ..instruments.capfloor import CapFloor, Collar
from ..markethandles.marketenvironment import MarketEnvironment
from ..markethandles.utils import Currency
from ..numericalhandles.dtypes import DTYPE
from ..timehandles.utils import Settings


def _norm_cdf(x):
    return 0.5 * (1.0 + tf.math.erf(x / tf.constant(math.sqrt(2.0), dtype=x.dtype)))


def _norm_pdf(x):
    return tf.exp(-0.5 * tf.square(x)) / tf.constant(math.sqrt(2.0 * math.pi), dtype=x.dtype)


def bachelier_calc(forward, strike, vol, expiry, phi: float) -> tf.Tensor:
    """Undiscounted Bachelier (normal) option: ``phi (F - K) N(phi d) + sigma sqrt(T) n(d)``, ``d = (F - K)/(sigma sqrt T)``."""
    std = vol * tf.sqrt(expiry)
    d = (forward - strike) / std
    return phi * (forward - strike) * _norm_cdf(phi * d) + std * _norm_pdf(d)


def black_calc(forward, strike, vol, expiry, phi: float) -> tf.Tensor:
    """Undiscounted Black (lognormal) option: ``phi (F N(phi d1) - K N(phi d2))`` (needs F, K > 0)."""
    std = vol * tf.sqrt(expiry)
    d1 = (tf.math.log(forward / strike) + 0.5 * tf.square(std)) / std
    return phi * (forward * _norm_cdf(phi * d1) - strike * _norm_cdf(phi * (d1 - std)))


class CapFloorPricer(Pricer):
    """Caps, floors and collars on an Ibor index, caplet by caplet (float64).

    ``caplet = N tau P_disc(0, pay) * option(F, K, sigma, T_fix)`` with

    - ``F = (P_fwd(0, s) / P_fwd(0, e) - 1) / tau_index`` on the index curve (as :class:`SwapPricer`),
      discounting on the currency's default curve;
    - ``sigma`` from the index caplet vol surface (``market_env.get_ir_vol_surface``) at
      ``(T_fix, K)``, ``T_fix`` the year fraction to the fixing date with the surface day count;
      Bachelier for ``vol_type == "normal"``, Black for ``"lognormal"``;
    - a caplet already fixed (fixing date <= evaluation date) pays its intrinsic value on the index
      fixing; caplets paid by the evaluation date are dropped.

    Differentiable w.r.t. both curves and every vol node. Per-caplet values on ``product.caplet_values``.
    """

    def calculate_price(self, product, market_env: MarketEnvironment) -> tf.Tensor:
        """Price a :class:`CapFloor` or a :class:`Collar` (cap - floor).

        Args:
            product: The cap / floor / collar.
            market_env: Curves and the index caplet vols.

        Returns:
            tf.Tensor: float64 NPV.

        Raises:
            TypeError: If *product* is not a cap, floor or collar.
            ValueError: If a curve or the vol surface is missing.
        """
        if isinstance(product, Collar):
            cap, floor = self._strip(product.cap, market_env), self._strip(product.floor, market_env)
            product.cap.price, product.floor.price = cap, floor
            return cap - floor
        if not isinstance(product, CapFloor):
            raise TypeError("CapFloorPricer prices CapFloor and Collar")
        return self._strip(product, market_env)

    def _strip(self, product: CapFloor, market_env: MarketEnvironment) -> tf.Tensor:
        eval_date = Settings.evaluation_date
        ccy_str, tenor = product.index.name.split(":")
        index_ccy = Currency[ccy_str]
        disc = market_env.get_ir_curve(product.ccy)
        fwd = market_env.get_ir_curve(index_ccy, ticker=tenor)
        surface = market_env.get_ir_vol_surface(index_ccy, tenor)
        option = bachelier_calc if surface.vol_type == "normal" else black_calc
        phi = 1.0 if product.is_cap else -1.0
        k = tf.constant(product.strike, DTYPE)

        values = []
        for c in product.caplets:
            if c.date <= eval_date:
                continue
            df = tf.cast(disc.discount(c.date), DTYPE)
            if c.fixing_date <= eval_date:
                rate = tf.constant(c.index.fixing(c.fixing_date), DTYPE)
                payoff = tf.maximum(phi * (rate - k), 0.0)
            else:
                tau_idx = c.index.daycounter.year_fraction(c.ref_period_start, c.ref_period_end)
                forward = (tf.cast(fwd.discount(c.ref_period_start), DTYPE)
                           / tf.cast(fwd.discount(c.ref_period_end), DTYPE) - 1.0) / tau_idx
                t_fix = tf.constant(surface.daycounter.year_fraction(eval_date, c.fixing_date), DTYPE)
                vol = tf.cast(surface.volatility(strike=product.strike, tenor=float(t_fix)), DTYPE)
                payoff = option(forward, k, vol, t_fix, phi)
            values.append(c.nominal * c.accrual_period * df * payoff)
        product.caplet_values = values
        return tf.add_n(values) if values else tf.constant(0.0, DTYPE)
