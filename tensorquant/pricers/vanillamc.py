import tensorflow as tf

from .pricer import Pricer
from ..instruments.option import VanillaOption
from ..markethandles.marketenvironment import MarketEnvironment
from ..markethandles.utils import ExerciseType
from ..models.brownian import GeometricBrownianMotion
from ..numericalhandles.dtypes import DTYPE
from ..timehandles.utils import Settings


class VanillaMCPricer(Pricer):
    """Monte Carlo pricer for European :class:`VanillaOption` (float64).

    The model is built from the market at every pricing call:
    ``model.from_equity(market_env.get_equity(...), maturity, strike)``, so the
    simulation inputs are the market risk factors and one
    ``pricer.tape.gradient(product.price, market_env.risk_factors())`` gives
    pathwise delta, vega per vol node, rho per pillar and dividend sensitivities.

    Paths are simulated on a uniform grid ending exactly at the option maturity.
    Gaussian draws are stateless (``tf.random.stateless_normal`` with ``seed``):
    the same seed gives the same paths, so bump-and-reprice uses common random
    numbers.

    Args:
        model: Model class with a ``from_equity(equity, maturity, strike)`` factory
            (:class:`GeometricBrownianMotion`, :class:`LocalVolatilityModel`).
        n_paths: Number of Monte Carlo paths.
        n_steps: Number of time steps up to maturity.
        seed: Seed of the stateless Gaussian draws.

    Example::

        pricer = VanillaMCPricer(model=tq.LocalVolatilityModel, n_paths=100_000)
        pricer.price(option, market_env, autodiff=True)
        greeks = pricer.tape.gradient(option.price, market_env.risk_factors())
    """

    def __init__(
        self,
        model: type = GeometricBrownianMotion,
        n_paths: int = 50_000,
        n_steps: int = 252,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not hasattr(model, "from_equity"):
            raise TypeError(
                f"model must be a model class with a from_equity factory, got {model!r}"
            )
        self._model = model
        self._n_paths = n_paths
        self._n_steps = n_steps
        self._seed = seed

    def calculate_price(
        self, product: VanillaOption, market_env: MarketEnvironment
    ) -> tf.Tensor:
        """Price a European :class:`VanillaOption` by Monte Carlo.

        ``NPV = P(0, T) * mean(max(phi (S_T - K), 0))``. The Monte Carlo standard
        error is stored on ``product.standard_error``.

        Args:
            product: The vanilla option (European exercise).
            market_env: Market data (underlying via ``get_equity``).

        Returns:
            tf.Tensor: float64 NPV, differentiable w.r.t. the market risk factors.

        Raises:
            ValueError: If *product* is not a European :class:`VanillaOption`, or the
                underlying has no vol surface.
        """
        if not isinstance(product, VanillaOption):
            raise ValueError("VanillaMCPricer only supports VanillaOption")
        if product.exercise_type != ExerciseType.European:
            raise ValueError("VanillaMCPricer only supports European exercise")

        eq = market_env.get_equity(product.underlying, ccy=product.ccy)
        if eq.vol_surface is None:
            raise ValueError(f"No volatility surface for underlying '{product.underlying}'")
        model = self._model.from_equity(eq, product.end_date, float(product.strike))

        T = eq.vol_surface.daycounter.year_fraction(Settings.evaluation_date, product.end_date)
        t_grid = tf.linspace(tf.constant(0.0, DTYPE), tf.constant(T, DTYPE), self._n_steps + 1)[1:]
        dw = tf.random.stateless_normal([self._n_paths, self._n_steps], seed=[self._seed, 0], dtype=DTYPE)
        s_t = model.evolve(t_grid, dw, observation_times=[T])[:, 0]

        k = tf.cast(product.strike, DTYPE)
        phi = tf.constant(float(product.option_type.value), dtype=DTYPE)
        discount_factor = tf.cast(eq.discount_curve.discount(product.end_date), DTYPE)
        payoff = discount_factor * tf.maximum(phi * (s_t - k), 0.0)

        price = tf.reduce_mean(payoff)
        product.standard_error = tf.math.reduce_std(tf.stop_gradient(payoff)) / self._n_paths ** 0.5
        product.discount_factor = discount_factor
        product.time_to_maturity = tf.constant(T, dtype=DTYPE)
        return price
