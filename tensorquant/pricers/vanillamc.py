import numpy as np
import tensorflow as tf

from .pricer import Pricer
from ..instruments.barrier import BarrierOption
from ..instruments.digital import DigitalOption
from ..instruments.option import VanillaOption
from ..markethandles.marketenvironment import MarketEnvironment
from ..markethandles.utils import ExerciseType, PayoffType
from ..models.brownian import GeometricBrownianMotion
from ..models.equitymodel import bind, check_model
from ..numericalhandles.dtypes import DTYPE
from ..timehandles.utils import Settings


class VanillaMCPricer(Pricer):
    """Monte Carlo pricer for European :class:`VanillaOption`, :class:`DigitalOption` and :class:`BarrierOption` (float64).

    The model is a **class** or an **instance** (:class:`EquityModel`):

    - class: built from the market at every pricing call, ``model.from_equity(market_env.get_equity(...), maturity,
      strike)``, calibration included, so one ``pricer.tape.gradient(product.price, market_env.risk_factors())`` gives
      pathwise delta, vega per vol node (through the calibration), rho per pillar and dividend sensitivities;
    - instance: bound to the underlying, ``model.process(equity)`` (its ticker must match): parameters held, model
      greeks ``pricer.tape.gradient(product.price, model.parameters)``, market greeks through the forwards (and the
      vol nodes for the models that read them at pricing); par greeks with :func:`par_greeks`.

    Paths are simulated on a uniform grid ending exactly at the option maturity (plus the monitoring dates of a
    discretely monitored barrier).

    Barriers: with continuous monitoring the knock-out survival on each step is the Brownian-bridge probability of not
    crossing ``H`` between the two simulated points, ``1 - exp(-2 ln(S_i/H)+ ln(S_{i+1}/H)+ / (sigma_H^2 dt))``
    (down; ``ln(H/S)`` for up), with ``sigma_H`` the surface vol at ``(H, T)``: exact for a GBM with that vol, and a
    continuous function of the path (AAD greeks without smoothing). With discrete monitoring the barrier is checked on
    the monitoring dates, the indicator smoothed like a digital (``digital_smoothing``, width as a fraction of the
    barrier). The first point is the model's initial value (the escrowed spot for ``GeometricBrownianMotion``).
    Gaussian draws are stateless (``tf.random.stateless_normal`` with ``seed``):
    the same seed gives the same paths, so bump-and-reprice uses common random
    numbers.

    Args:
        model: Model class with a ``from_equity(equity, maturity, strike)`` factory
            (:class:`GeometricBrownianMotion`, :class:`LocalVolatilityModel`, :class:`DisplacedDiffusionModel`,
            :class:`HestonModel`, :class:`HestonSLVModel`) or a model instance (:class:`EquityModel`); the process's
            ``noise_size()`` normals are drawn per path and step.
        n_paths: Number of Monte Carlo paths.
        n_steps: Number of time steps up to maturity.
        seed: Seed of the stateless Gaussian draws.
        digital_smoothing: Digitals and discrete barriers: width of the sigmoid replacing the indicator, as a fraction
            of the strike (barrier) (0 = hard indicator: unbiased price, but zero pathwise delta and vega; > 0 = smooth payoff with
            AAD greeks, at the cost of a small bias, second order in the width).

    Example::

        pricer = VanillaMCPricer(model=tq.LocalVolatilityModel, n_paths=100_000)
        pricer.price(option, market_env, autodiff=True)
        greeks = pricer.tape.gradient(option.price, market_env.risk_factors())
    """

    def __init__(
        self,
        model=GeometricBrownianMotion,
        n_paths: int = 50_000,
        n_steps: int = 252,
        seed: int = 42,
        digital_smoothing: float = 0.01,
    ) -> None:
        super().__init__()
        check_model(model)
        self._model = model
        self._n_paths = n_paths
        self._n_steps = n_steps
        self._seed = seed
        self._digital_smoothing = float(digital_smoothing)

    def calculate_price(
        self, product, market_env: MarketEnvironment
    ) -> tf.Tensor:
        """Price a European :class:`VanillaOption`, :class:`DigitalOption` or :class:`BarrierOption` by Monte Carlo.

        ``NPV = P(0, T) * mean(payoff(S_T))``: ``max(phi (S_T - K), 0)`` for a vanilla, ``A * w`` or ``S_T * w`` for a
        cash- / asset-or-nothing digital with ``w = 1{phi (S_T - K) > 0}`` or, with ``digital_smoothing = eps > 0``,
        ``w = sigmoid(phi (S_T - K) / (eps K))``. The Monte Carlo standard error is stored on
        ``product.standard_error``.

        Args:
            product: The option (European exercise).
            market_env: Market data (underlying via ``get_equity``).

        Returns:
            tf.Tensor: float64 NPV, differentiable w.r.t. the market risk factors.

        Raises:
            ValueError: If *product* is not a European :class:`VanillaOption` or a :class:`DigitalOption`, or the
                underlying has no vol surface.
        """
        if not isinstance(product, (VanillaOption, DigitalOption, BarrierOption)):
            raise ValueError("VanillaMCPricer only supports VanillaOption, DigitalOption and BarrierOption")
        if product.exercise_type != ExerciseType.European:
            raise ValueError("VanillaMCPricer only supports European exercise")

        eq = market_env.get_equity(product.underlying, ccy=product.ccy)
        if eq.vol_surface is None:
            raise ValueError(f"No volatility surface for underlying '{product.underlying}'")
        model = bind(self._model, eq, product.end_date, float(product.strike))

        daycounter = eq.vol_surface.daycounter
        T = daycounter.year_fraction(Settings.evaluation_date, product.end_date)
        times = np.linspace(0.0, T, self._n_steps + 1)[1:]
        monitoring = None
        if isinstance(product, BarrierOption) and not product.continuous:
            monitoring = np.array([daycounter.year_fraction(Settings.evaluation_date, d)
                                   for d in product.monitoring_dates if d > Settings.evaluation_date])
            times = np.unique(np.concatenate([times, monitoring]))
            times[-1] = T
        t_grid = tf.constant(times, DTYPE)
        k = model.noise_size()
        dw = tf.random.stateless_normal([self._n_paths, len(times)] + ([k] if k > 1 else []),
                                        seed=[self._seed, 0], dtype=DTYPE)
        survival = None
        if not isinstance(product, BarrierOption):
            s_t = model.evolve(t_grid, dw, observation_times=[T])[:, 0]
        elif monitoring is None:
            paths = model.evolve(t_grid, dw)
            s_t = paths[:, -1]
            survival = self._bridge_survival(model, paths, times, product, eq, T)
        else:
            obs = sorted(set(monitoring.tolist()) | {T})
            paths = model.evolve(t_grid, dw, observation_times=obs)
            s_t = paths[:, -1]
            cols = [obs.index(m) for m in monitoring]
            survival = self._discrete_survival(tf.gather(paths, cols, axis=1), product)

        k = tf.cast(product.strike, DTYPE)
        phi = tf.constant(float(product.option_type.value), dtype=DTYPE)
        discount_factor = tf.cast(eq.discount_curve.discount(product.end_date), DTYPE)
        if isinstance(product, DigitalOption):
            if self._digital_smoothing > 0.0:
                in_the_money = tf.sigmoid(phi * (s_t - k) / (self._digital_smoothing * k))
            else:
                in_the_money = tf.cast(phi * (s_t - k) > 0.0, DTYPE)
            if product.payoff_type == PayoffType.AssetOrNothing:
                amount = s_t
            else:
                amount = tf.constant(product.cash_amount, DTYPE)
            payoff = discount_factor * amount * in_the_money
        else:
            payoff = discount_factor * tf.maximum(phi * (s_t - k), 0.0)
            if survival is not None:
                payoff = payoff * ((1.0 - survival) if product.barrier_type.is_in else survival)

        price = tf.reduce_mean(payoff)
        product.standard_error = tf.math.reduce_std(tf.stop_gradient(payoff)) / self._n_paths ** 0.5
        product.discount_factor = discount_factor
        product.time_to_maturity = tf.constant(T, dtype=DTYPE)
        return price

    @staticmethod
    def _bridge_survival(model, paths, times, product, eq, T) -> tf.Tensor:
        """Probability of not crossing the barrier between the grid points (continuous monitoring), ``[n_paths]``."""
        h = tf.constant(product.barrier, DTYPE)
        s0 = tf.reshape(tf.cast(model.initial_values(), DTYPE), [-1])[0]
        full = tf.concat([tf.fill([tf.shape(paths)[0], 1], s0), paths], axis=1)
        log_dist = tf.math.log(full / h) if product.barrier_type.is_down else tf.math.log(h / full)
        dist = tf.nn.relu(log_dist)                       # 0 once on the wrong side of the barrier
        sigma = eq.vol_surface.volatility(strike=product.barrier, tenor=T)
        dt = tf.constant(np.diff(np.concatenate([[0.0], times])), DTYPE)
        cross = tf.exp(-2.0 * dist[:, :-1] * dist[:, 1:] / (sigma ** 2 * dt[None, :]))
        return tf.reduce_prod(1.0 - cross, axis=1)

    def _discrete_survival(self, s_mon: tf.Tensor, product) -> tf.Tensor:
        """Probability of not hitting the barrier on the monitoring dates (``s_mon`` ``[n_paths, n_dates]``)."""
        h = product.barrier
        dist = (s_mon - h) if product.barrier_type.is_down else (h - s_mon)
        if self._digital_smoothing > 0.0:
            alive = tf.sigmoid(dist / (self._digital_smoothing * h))
        else:
            alive = tf.cast(dist > 0.0, DTYPE)
        return tf.reduce_prod(alive, axis=1)
