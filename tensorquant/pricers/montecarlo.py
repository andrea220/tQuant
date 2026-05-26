import numpy as np
import tensorflow as tf

from .pricer import Pricer
from ..instruments.autocallable import AutocallableOption
from ..markethandles.marketenvironment import MarketEnvironment
from ..models.stochasticprocess import StochasticProcess
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..timehandles.schedule import ScheduleGenerator
from ..timehandles.targetcalendar import TARGET
from ..timehandles.utils import Settings, BusinessDayConvention, TimeUnit


class AutocallableMCPricer(Pricer):
    """Monte Carlo pricer for :class:`AutocallableOption` products.

    Simulates equity paths via a :class:`StochasticProcess` model and evaluates
    the path-wise payoff of the autocallable phoenix structure:
    conditional coupons, memory, autocall trigger, and capital-at-risk put at
    maturity.

    The price stored on the product via :meth:`Pricer.price` is expressed in
    **currency units** (notional-adjusted NPV).

    Args:
        model: Calibrated :class:`StochasticProcess` (e.g.
            :class:`GeometricBrownianMotion`).
        n_paths: Number of Monte Carlo simulation paths.
        seed: Seed passed to ``tf.random.normal`` for reproducibility.
        daycounter_convention: Day-count convention used to convert dates to
            year fractions for the simulation time grid.
        calendar: Business-day calendar used to build the auxiliary monthly
            discretization grid.  Defaults to TARGET when ``None``.
        discretization_months: Step (in months) of the auxiliary grid added
            on top of the product's own fixing dates.
    """

    def __init__(
        self,
        model: StochasticProcess,
        n_paths: int = 20_000,
        seed: int = 42,
        daycounter_convention: DayCounterConvention = DayCounterConvention.Actual360,
        calendar=None,
        discretization_months: int = 1,
    ) -> None:
        super().__init__()
        self._model = model
        self._n_paths = n_paths
        self._seed = seed
        self._daycounter = DayCounter(daycounter_convention)
        self._calendar = calendar if calendar is not None else TARGET()
        self._discretization_months = discretization_months

    # ------------------------------------------------------------------
    # Pricer interface
    # ------------------------------------------------------------------

    def calculate_price(
        self, product: AutocallableOption, market_env: MarketEnvironment
    ) -> tf.Tensor:
        """Price an :class:`AutocallableOption` via Monte Carlo simulation.

        Args:
            product: The autocallable option to price.
            market_env: Market environment providing the discount curve.

        Returns:
            NPV as a ``tf.Tensor`` in currency units (notional × price_pct).

        Raises:
            ValueError: If *product* is not an :class:`AutocallableOption`.
        """
        if not isinstance(product, AutocallableOption):
            raise ValueError("AutocallableMCPricer only supports AutocallableOption")

        valuation_date = Settings.evaluation_date
        disc_curve = market_env.get_ir_curve(product.ccy)

        date_grid, time_grid_tensor = self._build_date_grid(product, valuation_date)
        s_t = self._simulate(time_grid_tensor)

        price_pct = self._price_option_leg(
            product, s_t, date_grid, disc_curve, valuation_date
        )
        return tf.constant(price_pct * product.notional, dtype=tf.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_date_grid(self, product: AutocallableOption, valuation_date):
        """Merge product fixing dates with the auxiliary discretization grid.

        Returns:
            date_grid: Sorted list of :class:`datetime.date` objects.
            time_grid_tensor: ``tf.Tensor`` of year fractions (float64).
        """
        schedule_gen = ScheduleGenerator(
            self._calendar, BusinessDayConvention.ModifiedFollowing
        )
        disc_dates = schedule_gen.generate(
            valuation_date,
            product.end_date,
            self._discretization_months,
            TimeUnit.Months,
        )
        date_set = (
            set(product.coupon_fixing_dates)
            | set(product.autocall_fixing_dates)
            | set(disc_dates)
        )
        date_grid = sorted(date_set)
        time_grid = [
            self._daycounter.year_fraction(valuation_date, d) for d in date_grid
        ]
        return date_grid, tf.constant(time_grid, dtype=tf.float64)

    def _simulate(self, time_grid_tensor: tf.Tensor) -> tf.Tensor:
        """Draw Gaussian variates and evolve the model.

        Returns:
            Tensor of shape ``[n_paths, n_steps]``.
        """
        n_steps = time_grid_tensor.shape[0]
        z = tf.random.normal(
            (self._n_paths, n_steps), seed=self._seed, dtype=tf.float64
        )
        return self._model.evolve(time_grid_tensor, z)

    def _price_option_leg(
        self,
        product: AutocallableOption,
        s_t: tf.Tensor,
        date_grid: list,
        disc_curve,
        valuation_date,
    ) -> float:
        """Path-wise Monte Carlo evaluation of the autocallable option leg.

        Loops over all unique observation dates (union of coupon and autocall
        fixing dates) in chronological order.  At each date it checks:

        * **Coupon leg** — conditional coupon (with optional memory) if the
          note is alive and spot ≥ coupon barrier.
        * **Final redemption** — capital-at-risk put payoff on the last coupon
          fixing date if spot ≤ payoff barrier.
        * **Autocall leg** — knocks out paths where spot ≥ autocall barrier.

        Args:
            product: :class:`AutocallableOption` carrying schedules and payoff
                parameters.
            s_t: Simulated spot tensor of shape ``[n_paths, n_dates]``.
            date_grid: List of dates mapped to the columns of *s_t*.
            disc_curve: Discount curve exposing a ``discount(date)`` method.
            valuation_date: Pricing date; past fixing dates are skipped.

        Returns:
            Mean present value as a fraction of notional.
        """
        strike = product.strike
        coupon_fixing_dates = product.coupon_fixing_dates
        coupon_payment_dates = product.coupon_payment_dates
        coupon_rates = product.coupon_rates
        coupon_barriers = product.coupon_barriers
        autocall_fixing_dates = product.autocall_fixing_dates
        autocall_payment_dates = product.autocall_payment_dates
        autocall_barriers = product.autocall_barrier
        payoff_barrier = product.payoff_barrier
        payoff_participation = product.payoff_participation
        memory = product.memory

        n_paths = s_t.shape[0]
        date_to_col = {d: j for j, d in enumerate(date_grid)}

        coupon_map = {
            d: (i, coupon_payment_dates[i])
            for i, d in enumerate(coupon_fixing_dates)
        }
        autocall_map = {
            d: (i, autocall_payment_dates[i])
            for i, d in enumerate(autocall_fixing_dates)
        }

        all_fixing_dates = sorted(
            set(coupon_fixing_dates) | set(autocall_fixing_dates)
        )

        alive = np.ones(n_paths, dtype=bool)
        unpaid_coupons = np.zeros(n_paths, dtype=np.float64)
        coupon_pv = np.zeros(n_paths, dtype=np.float64)
        redemption_pv = np.zeros(n_paths, dtype=np.float64)

        for fix_date in all_fixing_dates:
            if fix_date <= valuation_date:
                continue
            if not alive.any():
                break

            col = date_to_col[fix_date]
            s_i = s_t[:, col].numpy()

            # --- coupon leg ---
            if fix_date in coupon_map:
                c_idx, c_pay_date = coupon_map[fix_date]
                coupon_thr = coupon_barriers[c_idx] / 100.0 * strike
                above_coupon = s_i >= coupon_thr
                paid = alive & above_coupon

                if memory:
                    pay_amount = coupon_rates[c_idx] / 100.0 * (1.0 + unpaid_coupons)
                else:
                    pay_amount = coupon_rates[c_idx] / 100.0

                df = float(disc_curve.discount(c_pay_date))
                coupon_pv += np.where(paid, pay_amount, 0.0) * df

                # --- capital-at-risk put at maturity ---
                if (
                    fix_date == coupon_fixing_dates[-1]
                    and product.redemption_payoff is not None
                ):
                    below_barrier = s_i <= payoff_barrier / 100.0 * strike
                    redemption_pv += np.where(
                        alive & below_barrier,
                        product.redemption_payoff(s_i, strike, payoff_participation),
                        0.0,
                    ) * df

                # --- memory counter ---
                unpaid_coupons = np.where(paid, 0.0, unpaid_coupons)
                unpaid_coupons = np.where(
                    alive & ~above_coupon, unpaid_coupons + 1.0, unpaid_coupons
                )

            # --- autocall leg ---
            if fix_date in autocall_map:
                a_idx, _ = autocall_map[fix_date]
                autocall_thr = autocall_barriers[a_idx] / 100.0 * strike
                above_autocall = s_i >= autocall_thr
                alive = alive & ~above_autocall

        total_pv = coupon_pv + redemption_pv
        self._coupon_pv = float(coupon_pv.mean())
        self._redemption_pv = float(redemption_pv.mean())
        return float(total_pv.mean())
