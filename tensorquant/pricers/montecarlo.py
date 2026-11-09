import numpy as np
import tensorflow as tf

from .pricer import Pricer
from ..instruments.autocallable import AutocallableOption
from ..markethandles.marketenvironment import MarketEnvironment
from ..models.localvolatility import LocalVolatilityModel
from ..models.equitymodel import bind, check_model
from ..numericalhandles.dtypes import DTYPE
from ..timehandles.utils import Settings


class AutocallableMCPricer(Pricer):
    """Monte Carlo pricer for :class:`AutocallableOption` (phoenix: conditional coupons with optional
    memory, autocall, capital-at-risk put at maturity), from the **investor's side**:

    ``NPV = notional * E[ sum_i coupon_i P(0, pay_i) - put P(0, T) ]``,
    ``put = participation * max(K - S_T, 0) / K`` if ``S_T <= payoff barrier`` and not called.

    The model is a class, built from the market at every pricing call
    (``model.from_equity(market_env.get_equity(product.underlying), maturity, strike)``, greeks through the
    calibration: one ``pricer.tape.gradient(product.price, market_env.risk_factors())``), or an instance
    (:class:`EquityModel`) bound with ``model.process(equity)`` (parameters held: model greeks on
    ``model.parameters``, par greeks with :func:`par_greeks`); see :class:`VanillaMCPricer`.
    Paths are simulated (stateless seed) on the union of the future fixing dates and a uniform
    grid of step ``max_time_step``, with the vol-surface day count.

    Coupon, autocall and put barriers are digitals: their pathwise derivative is zero, so AAD would
    miss most of delta and vega. With ``barrier_smoothing = eps > 0`` every indicator
    ``1{S >= B}`` becomes ``sigmoid((S - B) / (eps K))`` (eps = 1% of the strike by default):
    a small, controlled price bias for smooth, differentiable greeks. ``eps = 0`` gives the
    exact (hard) payoff.

    Args:
        model: Model class with ``from_equity`` (default :class:`LocalVolatilityModel`) or a model instance.
        n_paths: Number of paths.
        seed: Seed of the stateless Gaussian draws.
        max_time_step: Largest simulation step (years).
        barrier_smoothing: Width of the smoothed digitals as a fraction of the strike (0 = hard).

    After pricing: ``product.coupon_pv``, ``product.put_pv`` (both per unit notional, put as a
    positive value paid by the investor), ``product.standard_error`` (currency units);
    ``pricer.paths`` ``[n_paths, n_fixings]`` at the future fixings ``pricer.date_grid`` (times in
    ``pricer.time_grid``) for diagnostics.
    """

    def __init__(
        self,
        model=LocalVolatilityModel,
        n_paths: int = 20_000,
        seed: int = 42,
        max_time_step: float = 1.0 / 52.0,
        barrier_smoothing: float = 0.01,
    ) -> None:
        super().__init__()
        check_model(model)
        if barrier_smoothing < 0:
            raise ValueError("barrier_smoothing must be >= 0")
        self._model = model
        self._n_paths = n_paths
        self._seed = seed
        self._max_time_step = max_time_step
        self._eps = barrier_smoothing
        self.paths = None
        self.time_grid = None
        self.date_grid = None

    # ------------------------------------------------------------------
    # Pricer interface
    # ------------------------------------------------------------------

    def calculate_price(self, product: AutocallableOption, market_env: MarketEnvironment) -> tf.Tensor:
        """Price an :class:`AutocallableOption` by Monte Carlo (investor's side).

        Args:
            product: The autocallable (``underlying`` must be set).
            market_env: Market data (underlying via ``get_equity``).

        Returns:
            tf.Tensor: float64 NPV in currency units, differentiable w.r.t. the market risk factors.

        Raises:
            ValueError: If *product* is not an :class:`AutocallableOption`, has no underlying,
                or the underlying has no vol surface.
        """
        if not isinstance(product, AutocallableOption):
            raise ValueError("AutocallableMCPricer only supports AutocallableOption")
        if product.underlying is None:
            raise ValueError("AutocallableOption.underlying must be set")
        eq = market_env.get_equity(product.underlying, ccy=product.ccy)
        if eq.vol_surface is None:
            raise ValueError(f"No volatility surface for underlying '{product.underlying}'")

        eval_date = Settings.evaluation_date
        daycounter = eq.vol_surface.daycounter
        fixings = sorted(d for d in set(product.coupon_fixing_dates) | set(product.autocall_fixing_dates)
                         if d > eval_date)
        if not fixings:
            return tf.constant(0.0, dtype=DTYPE)
        fixing_times = np.array([daycounter.year_fraction(eval_date, d) for d in fixings])
        uniform = np.arange(self._max_time_step, fixing_times[-1], self._max_time_step)
        time_grid = np.unique(np.concatenate([uniform, fixing_times]))
        column = {d: i for i, d in enumerate(fixings)}     # paths are returned at the fixings only

        model = bind(self._model, eq, product.end_date, float(product.strike))
        k = model.noise_size()
        dw = tf.random.stateless_normal([self._n_paths, len(time_grid)] + ([k] if k > 1 else []),
                                        seed=[self._seed, 0], dtype=DTYPE)
        paths = model.evolve(tf.constant(time_grid, DTYPE), dw, observation_times=fixing_times)
        self.paths, self.time_grid, self.date_grid = paths, fixing_times, fixings

        coupon_pv, put_pv = self._payoff(product, paths, column, fixings, eq.discount_curve)
        total = coupon_pv - put_pv
        product.coupon_pv = tf.reduce_mean(coupon_pv)
        product.put_pv = tf.reduce_mean(put_pv)
        product.standard_error = product.notional * tf.math.reduce_std(tf.stop_gradient(total)) / self._n_paths ** 0.5
        return product.notional * tf.reduce_mean(total)

    # ------------------------------------------------------------------
    # Payoff
    # ------------------------------------------------------------------

    def _above(self, s: tf.Tensor, barrier: float, strike: float) -> tf.Tensor:
        """``1{s >= barrier}``, smoothed with a sigmoid of width ``eps * strike`` when ``eps > 0``."""
        if self._eps == 0.0:
            return tf.cast(s >= barrier, DTYPE)
        return tf.sigmoid((s - barrier) / (self._eps * strike))

    def _payoff(self, product, paths, column, fixings, curve):
        """Path-wise discounted coupons and put, per unit notional (``[n_paths]`` each)."""
        strike = float(product.strike)
        coupon_index = {d: i for i, d in enumerate(product.coupon_fixing_dates)}
        autocall_index = {d: i for i, d in enumerate(product.autocall_fixing_dates)}
        last_coupon = product.coupon_fixing_dates[-1]

        n_paths = tf.shape(paths)[0]
        alive = tf.ones([n_paths], DTYPE)
        unpaid = tf.zeros([n_paths], DTYPE)
        coupon_pv = tf.zeros([n_paths], DTYPE)
        put_pv = tf.zeros([n_paths], DTYPE)

        for d in fixings:
            s = paths[:, column[d]]
            if d in coupon_index:
                i = coupon_index[d]
                above = self._above(s, product.coupon_barriers[i] / 100.0 * strike, strike)
                rate = product.coupon_rates[i] / 100.0
                amount = rate * (1.0 + unpaid) if product.memory else rate * tf.ones_like(s)
                df_pay = tf.cast(curve.discount(product.coupon_payment_dates[i]), DTYPE)
                coupon_pv = coupon_pv + alive * above * amount * df_pay
                if d == last_coupon and product.redemption_payoff is not None:
                    below = 1.0 - self._above(s, product.payoff_barrier / 100.0 * strike, strike)
                    put = product.payoff_participation * tf.maximum(strike - s, 0.0) / strike
                    put_pv = put_pv + alive * below * put * df_pay
                # memory: reset when paid, +1 when missed (alive paths)
                unpaid = alive * (1.0 - above) * (unpaid + 1.0) + (1.0 - alive) * unpaid
            if d in autocall_index:
                j = autocall_index[d]
                alive = alive * (1.0 - self._above(s, product.autocall_barrier[j] / 100.0 * strike, strike))
        return coupon_pv, put_pv
