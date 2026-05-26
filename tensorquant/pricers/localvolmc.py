import tensorflow as tf

from .pricer import Pricer
from ..instruments.option import VanillaOption
from ..markethandles.marketenvironment import MarketEnvironment
from ..markethandles.utils import OptionType, ExerciseType
from ..models.localvolatility import LocalVolatilityModel
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..timehandles.utils import Settings


class LocalVolMCPricer(Pricer):
    """Monte Carlo pricer for European :class:`VanillaOption` using a
    :class:`LocalVolatilityModel` (Dupire).

    The local-vol surface is calibrated at each :meth:`calculate_price` call
    from the implied-vol surface found in the ``MarketEnvironment``.  Paths
    are evolved with a log-Euler scheme and the option payoff is discounted
    using the risk-free curve.

    Args:
        n_paths: Number of Monte Carlo simulation paths.
        n_steps: Number of time steps in the simulation grid.
        seed: Seed for ``tf.random.normal`` (reproducibility).
        daycounter_convention: Convention used to convert dates to year
            fractions for the simulation time grid.
    """

    def __init__(
        self,
        n_paths: int = 50_000,
        n_steps: int = 252,
        seed: int = 42,
        daycounter_convention: DayCounterConvention = DayCounterConvention.Actual365,
    ) -> None:
        super().__init__()
        self._n_paths = n_paths
        self._n_steps = n_steps
        self._seed = seed
        self._daycounter = DayCounter(daycounter_convention)

    # ------------------------------------------------------------------
    # Pricer interface
    # ------------------------------------------------------------------

    def calculate_price(
        self, product: VanillaOption, market_env: MarketEnvironment
    ) -> tf.Tensor:
        """Price a European :class:`VanillaOption` via Local-Vol Monte Carlo.

        Reads all market data from *market_env* via typed accessors, calibrates
        the Dupire surface, simulates paths, and returns the discounted
        expected payoff.

        Args:
            product: The vanilla option to price.  Must have
                ``exercise_type == ExerciseType.European``.
            market_env: Market environment providing spot, vol surface,
                rate curve and dividend yield for the option's underlying.

        Returns:
            tf.Tensor (scalar, float32): The option NPV.

        Raises:
            ValueError: If *product* is not a :class:`VanillaOption` or has
                non-European exercise.
        """
        if not isinstance(product, VanillaOption):
            raise ValueError("LocalVolMCPricer only supports VanillaOption")
        if product.exercise_type != ExerciseType.European:
            raise ValueError("LocalVolMCPricer only supports European exercise")

        evaluation_date = Settings.evaluation_date

        # ---- market data via typed accessors --------------------------------
        vol_surface = market_env.get_eq_vol_surface(product.underlying, ccy=product.ccy)
        disc_curve  = market_env.get_ir_curve(product.ccy)
        spot_value  = market_env.get_eq_spot(product.underlying, ccy=product.ccy)

        T = tf.constant(
            self._daycounter.year_fraction(evaluation_date, product.end_date),
            dtype=tf.float32,
        )
        S0 = tf.constant(float(spot_value), dtype=tf.float32)

        discount_factor = tf.cast(disc_curve.discount(product.end_date), tf.float32)
        r = tf.cast(-tf.math.log(discount_factor) / T, tf.float32)

        # continuous dividend yield (DIVYIELD key, defaults to 0)
        q_raw = market_env.get_eq_div_yield(product.underlying, ccy=product.ccy)
        q = tf.constant(float(q_raw or 0.0), dtype=tf.float32)

        # ---- implied-vol surface → LocalVolatilityModel (Dupire) ------------
        T_grid = tf.constant(vol_surface.maturity, dtype=tf.float32)
        K_grid = tf.constant(vol_surface.strike,   dtype=tf.float32)
        iv_matrix = tf.constant(vol_surface.volatility_matrix, dtype=tf.float32)

        lv_model = LocalVolatilityModel.from_implied_vol(
            iv_matrix=iv_matrix,
            T_grid=T_grid,
            K_grid=K_grid,
            S0=S0,
            r=r,
            q=q,
        )

        # ---- simulation -----------------------------------------------------
        T_max   = float(T_grid[-1].numpy())
        t_grid  = tf.linspace(0.0, T_max, self._n_steps + 1)[1:]   # skip t=0

        tf.random.set_seed(self._seed)
        dw    = tf.random.normal([self._n_paths, self._n_steps], dtype=tf.float32)
        paths = lv_model.evolve(t_grid, dw)                         # [n_paths, n_steps]

        # terminal spots at product maturity
        t_np       = t_grid.numpy()
        target_idx = int(__import__("numpy").argmin(__import__("numpy").abs(t_np - float(T.numpy()))))
        S_T        = paths[:, target_idx]                           # [n_paths]

        # ---- discounted payoff ----------------------------------------------
        K   = tf.cast(product.strike, tf.float32)
        phi = tf.constant(float(product.option_type.value), dtype=tf.float32)
        payoff = tf.maximum(phi * (S_T - K), 0.0)
        price  = discount_factor * tf.reduce_mean(payoff)

        # diagnostics stored on product
        product.spot              = S0
        product.time_to_maturity  = T
        product.risk_free_rate    = r
        product.discount_factor   = discount_factor
        product.forward           = S0 * tf.exp((r - q) * T)

        return price
