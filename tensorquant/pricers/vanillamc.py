import numpy as np
import tensorflow as tf

from .pricer import Pricer
from ..instruments.option import VanillaOption
from ..markethandles.marketenvironment import MarketEnvironment
from ..markethandles.utils import ExerciseType
from ..models.stochasticprocess import StochasticProcess
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..timehandles.utils import Settings


class VanillaMCPricer(Pricer):
    """Monte Carlo pricer for European :class:`VanillaOption`.

    Delegates path simulation entirely to a pre-built :class:`StochasticProcess`
    model, making this pricer model-agnostic.  Any model that implements
    ``evolve(t_grid, dw) -> [n_paths, n_steps]`` can be plugged in
    (e.g. :class:`GeometricBrownianMotion`, :class:`LocalVolatilityModel`).

    The discount factor is always read from the ``MarketEnvironment`` via the
    rate curve; all other market data (spot, vol, dividends) must be baked into
    the model before constructing this pricer.

    Args:
        model: A calibrated :class:`StochasticProcess` whose ``evolve`` method
            accepts ``(t_grid [n_steps], dw [n_paths, n_steps])`` and returns
            simulated paths ``[n_paths, n_steps]``.
        n_paths: Number of Monte Carlo paths.
        n_steps: Number of simulation time steps.
        seed: Seed for ``tf.random.normal`` (reproducibility).
        daycounter_convention: Convention used to convert dates to year
            fractions for the time grid.

    Example::

        # --- GeometricBrownianMotion ---
        gbm = GeometricBrownianMotion(mu=r - q, sigma=atm_vol, x0=spot)
        pricer = VanillaMCPricer(model=gbm, n_paths=50_000)
        pricer.price(option, market_env)

        # --- LocalVolatilityModel (Dupire) ---
        lv = LocalVolatilityModel.from_implied_vol(iv_matrix, T_grid, K_grid, S0=spot, r=r, q=q)
        pricer = VanillaMCPricer(model=lv, n_paths=50_000)
        pricer.price(option, market_env)
    """

    def __init__(
        self,
        model: StochasticProcess,
        n_paths: int = 50_000,
        n_steps: int = 252,
        seed: int = 42,
        daycounter_convention: DayCounterConvention = DayCounterConvention.Actual365,
    ) -> None:
        super().__init__()
        self._model = model
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
        """Price a European :class:`VanillaOption` via Monte Carlo simulation.

        The method:
        1. Reads the discount factor from the rate curve in *market_env*.
        2. Builds a uniform time grid ``[dt, 2dt, ..., T_max]`` covering the
           full maturity range of the model.
        3. Draws Gaussian variates and calls ``self._model.evolve(t_grid, dw)``.
        4. Extracts terminal spots at the option maturity.
        5. Returns ``df × E[max(φ(S_T - K), 0)]``.

        The dtype of the simulation matches ``model.initial_values().dtype``
        so both float32 (e.g. :class:`LocalVolatilityModel`) and float64
        (e.g. :class:`GeometricBrownianMotion`) models are supported.

        Args:
            product: The vanilla option to price.  Must have
                ``exercise_type == ExerciseType.European``.
            market_env: Provides the rate curve for discounting.

        Returns:
            tf.Tensor (scalar, float32): The option NPV.

        Raises:
            ValueError: If *product* is not a :class:`VanillaOption` or has
                non-European exercise.
        """
        if not isinstance(product, VanillaOption):
            raise ValueError("VanillaMCPricer only supports VanillaOption")
        if product.exercise_type != ExerciseType.European:
            raise ValueError("VanillaMCPricer only supports European exercise")

        evaluation_date = Settings.evaluation_date
        disc_curve = market_env.get_ir_curve(product.ccy)

        T = self._daycounter.year_fraction(evaluation_date, product.end_date)
        discount_factor = tf.cast(disc_curve.discount(product.end_date), tf.float32)

        # ---- detect model dtype and adapt simulation inputs ----------------
        model_dtype = tf.cast(self._model.initial_values(), tf.float32).dtype
        # GBM uses float64, LV uses float32 — we cast dw to the model's dtype
        sim_dtype = self._model.initial_values().dtype

        # ---- time grid: T_max covers the full model range ------------------
        # For LV the surface extends beyond T; for GBM any T_max >= T works.
        T_max = self._T_max(T)
        t_grid = tf.cast(
            tf.linspace(0.0, T_max, self._n_steps + 1)[1:],
            sim_dtype,
        )

        # ---- simulation ----------------------------------------------------
        tf.random.set_seed(self._seed)
        dw = tf.cast(
            tf.random.normal([self._n_paths, self._n_steps]),
            sim_dtype,
        )
        paths = self._model.evolve(t_grid, dw)   # [n_paths, n_steps]

        # terminal spot at option maturity
        t_np       = t_grid.numpy()
        target_idx = int(np.argmin(np.abs(t_np - T)))
        S_T        = tf.cast(paths[:, target_idx], tf.float32)

        # ---- discounted payoff ---------------------------------------------
        K   = tf.cast(product.strike, tf.float32)
        phi = tf.constant(float(product.option_type.value), dtype=tf.float32)
        payoff = tf.maximum(phi * (S_T - K), 0.0)
        price  = discount_factor * tf.reduce_mean(payoff)

        # diagnostics stored on product
        product.discount_factor  = discount_factor
        product.time_to_maturity = tf.constant(T, dtype=tf.float32)

        return price

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _T_max(self, T: float) -> float:
        """Return the upper bound for the simulation time grid.

        For models with an explicit maturity grid (e.g. :class:`LocalVolatilityModel`)
        we extend to the last grid point; for others we use the option maturity.
        """
        if hasattr(self._model, "_T_grid"):
            return float(self._model._T_grid[-1].numpy())
        return T
