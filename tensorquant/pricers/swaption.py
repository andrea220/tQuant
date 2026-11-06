import tensorflow as tf

from .pricer import Pricer
from .capfloor import bachelier_calc, black_calc
from .floatingflow import FloatingLegDiscounting
from ..instruments.swap import Swap
from ..instruments.swaption import Swaption
from ..markethandles.marketenvironment import MarketEnvironment
from ..markethandles.utils import Currency
from ..numericalhandles.dtypes import DTYPE
from ..timehandles.utils import Settings


def swap_annuity_and_forward(swap: Swap, market_env: MarketEnvironment):
    """Annuity ``A = sum_i N tau_i P(pay_i)`` of the fixed leg and forward swap rate ``F = PV(float) / A``.

    Two curves as :class:`SwapPricer`: discounting on the currency's default curve, forwards on the index
    curve. Both are float64 tensors differentiable w.r.t. the curves.
    """
    disc = market_env.get_ir_curve(swap.ccy)
    ccy_str, tenor = swap._index.name.split(":")
    fwd = market_env.get_ir_curve(Currency[ccy_str], ticker=tenor)
    annuity = tf.add_n([c.nominal * c.accrual_period * tf.cast(disc.discount(c.date), DTYPE)
                        for c in swap.fixed_leg.leg_flows])
    floating = tf.cast(FloatingLegDiscounting(swap.floating_leg).calculate_price(disc, fwd), DTYPE)
    return annuity, floating / annuity


class SwaptionPricer(Pricer):
    """European swaptions (physical settlement): ``A * option(F, K, sigma, T)``.

    ``A`` and ``F`` from :func:`swap_annuity_and_forward`; ``sigma`` from the index swaption vol cube
    (``market_env.get_swaption_vol_cube``) at (expiry year fraction, swap tenor, ``K - F``); Bachelier for
    ``vol_type == "normal"``, Black-76 for ``"lognormal"``. Differentiable w.r.t. both curves and every cube
    node. Diagnostics on the product: ``annuity``, ``forward_swap_rate``, ``volatility``.
    """

    def calculate_price(self, product: Swaption, market_env: MarketEnvironment) -> tf.Tensor:
        """Price a :class:`Swaption`.

        Raises:
            TypeError: If *product* is not a :class:`Swaption`.
            ValueError: If the expiry is not after the evaluation date or market data are missing.
        """
        if not isinstance(product, Swaption):
            raise TypeError("SwaptionPricer prices Swaption")
        eval_date = Settings.evaluation_date
        if product.expiry <= eval_date:
            raise ValueError("swaption expiry must be after the evaluation date")
        ccy_str, tenor = product.swap._index.name.split(":")
        cube = market_env.get_swaption_vol_cube(Currency[ccy_str], tenor)
        annuity, forward = swap_annuity_and_forward(product.swap, market_env)
        k = tf.constant(product.strike, DTYPE)
        t = tf.constant(cube.daycounter.year_fraction(eval_date, product.expiry), DTYPE)
        vol = cube.volatility(t, product.tenor_years, k - forward)
        option = bachelier_calc if cube.vol_type == "normal" else black_calc
        product.annuity, product.forward_swap_rate, product.volatility = annuity, forward, vol
        return annuity * option(forward, k, vol, t, 1.0 if product.payer else -1.0)
