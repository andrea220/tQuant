from .pricer import Pricer
from ..instruments.option import VanillaOption
from ..markethandles.utils import ExerciseType, OptionType
from ..markethandles.marketenvironment import MarketEnvironment
from ..timehandles.utils import Settings
from ..numericalhandles.dtypes import DTYPE

import math

import tensorflow as tf


def _norm_cdf(x: tf.Tensor) -> tf.Tensor:
    """Standard normal CDF computed via tf.math.erf (no tensorflow_probability)."""
    # sqrt(2) as a Python float: tf.sqrt(2.0) would be computed in float32.
    return 0.5 * (1.0 + tf.math.erf(x / tf.constant(math.sqrt(2.0), dtype=x.dtype)))


def blackscholes_calc(
    spot_price: tf.Tensor,
    strike: tf.Tensor,
    riskfree_rate: tf.Tensor,
    volatility: tf.Tensor,
    time_to_maturity: tf.Tensor,
    dividend_yield: tf.Tensor,
    option_type: OptionType,
) -> tf.Tensor:
    """European Black-Scholes price for a call or put.

    Uses the unified ``φ`` formulation so that a single expression handles both
    option types without branching:

        price = φ · [S·exp(-q·T)·N(φ·d1) - K·exp(-r·T)·N(φ·d2)]

    where φ = +1 for calls and φ = -1 for puts.

    Args:
        spot_price: Spot (or dividend-adjusted spot for discrete model).
        strike: Strike price.
        riskfree_rate: Continuously compounded risk-free rate ``r``.
        volatility: Implied volatility ``σ``.
        time_to_maturity: Time to expiry in year fractions ``T``.
        dividend_yield: Continuous dividend yield ``q`` (repo included).
        option_type: :class:`OptionType` — ``Call`` (+1) or ``Put`` (-1).

    Returns:
        tf.Tensor: Option price.
    """
    phi = tf.constant(float(option_type.value), dtype=spot_price.dtype)
    sqrt_t = tf.sqrt(time_to_maturity)
    d1 = (
        tf.math.log(spot_price / strike)
        + (riskfree_rate - dividend_yield + (volatility**2) / 2) * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return phi * (
        spot_price * tf.exp(-dividend_yield * time_to_maturity) * _norm_cdf(phi * d1)
        - strike * tf.exp(-riskfree_rate * time_to_maturity) * _norm_cdf(phi * d2)
    )


def _phi(
    S: tf.Tensor,
    T: tf.Tensor,
    gamma: tf.Tensor,
    H: tf.Tensor,
    I: tf.Tensor,
    b: tf.Tensor,
    r: tf.Tensor,
    sigma: tf.Tensor,
) -> tf.Tensor:
    """Auxiliary function for the Bjerksund-Stensland (1993) approximation.

    Computes:
        exp(λT) · Sᵞ · [N(d) - (I/S)^κ · N(d - 2·ln(I/S)/(σ√T))]

    where:
        d = -(ln(S/H) + (b + (γ - ½)σ²)T) / (σ√T)
        λ = -r + γb + ½γ(γ-1)σ²
        κ = 2b/σ² + (2γ - 1)
    """
    sqrt_t = tf.sqrt(T)
    d1 = -(tf.math.log(S / H) + (b + (gamma - 0.5) * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - 2.0 * tf.math.log(I / S) / (sigma * sqrt_t)
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma**2
    kappa = 2.0 * b / sigma**2 + (2.0 * gamma - 1.0)
    return tf.exp(lam * T) * tf.pow(S, gamma) * (
        _norm_cdf(d1) - tf.pow(I / S, kappa) * _norm_cdf(d2)
    )


def _bs1993_call(
    S: tf.Tensor,
    K: tf.Tensor,
    r: tf.Tensor,
    b: tf.Tensor,
    sigma: tf.Tensor,
    T: tf.Tensor,
) -> tf.Tensor:
    """Bjerksund-Stensland (1993) price for an American call.

    When b >= r (no dividend pressure), early exercise is never optimal and the
    result collapses to the European Black-Scholes price.
    """
    # European fallback (b >= r: no early exercise for calls)
    european = blackscholes_calc(S, K, r, sigma, T, r - b, OptionType.Call)

    # --- exercise-boundary parameters ---
    beta = (0.5 - b / sigma**2) + tf.sqrt(
        tf.maximum((b / sigma**2 - 0.5)**2 + 2.0 * r / sigma**2, 1e-10)
    )
    B_inf = beta / (beta - 1.0) * K
    # Guard against b ~ r to avoid division by zero
    r_minus_b = tf.maximum(r - b, tf.constant(1e-7, dtype=r.dtype))
    B_0 = tf.maximum(K, r * K / r_minus_b)

    # Trigger price of the 1993 approximation: h(T) = -(b T + 2 sigma sqrt(T)) B_0 / (B_inf - B_0)
    # (was B_0 B_inf / ((B_inf - B_0) K): wrong trigger, early-exercise premium 3-4x too small)
    ht = -(b * T + 2.0 * sigma * tf.sqrt(T)) * B_0 / (B_inf - B_0)
    I = B_0 + (B_inf - B_0) * (1.0 - tf.exp(ht))

    alpha = (I - K) * tf.pow(I, -beta)

    bs1993_price = (
        alpha * tf.pow(S, beta)
        - alpha * _phi(S, T, beta,               I, I, b, r, sigma)
        +         _phi(S, T, tf.ones_like(beta),  I, I, b, r, sigma)
        -         _phi(S, T, tf.ones_like(beta),  K, I, b, r, sigma)
        - K *     _phi(S, T, tf.zeros_like(beta), I, I, b, r, sigma)
        + K *     _phi(S, T, tf.zeros_like(beta), K, I, b, r, sigma)
    )

    # Immediate exercise when S >= I; European price when b >= r
    american = tf.where(S >= I, S - K, bs1993_price)
    return tf.where(b >= r, european, american)


def bjerksund_stensland_calc(
    spot_price: tf.Tensor,
    strike: tf.Tensor,
    riskfree_rate: tf.Tensor,
    volatility: tf.Tensor,
    time_to_maturity: tf.Tensor,
    dividend_yield: tf.Tensor,
    option_type: OptionType,
) -> tf.Tensor:
    """American option price via the Bjerksund-Stensland (1993) approximation.

    Handles calls directly and puts via put-call symmetry:
        AmPut(S, K, r, b, σ, T) = AmCall(K, S, r-b, -b, σ, T)

    Args:
        spot_price: Spot (or spot-net for discrete-dividend model).
        strike: Strike price.
        riskfree_rate: Continuously compounded risk-free rate.
        volatility: Implied volatility.
        time_to_maturity: Time to expiry in year fractions.
        dividend_yield: Continuous dividend yield ``q`` (0 for discrete model).
        option_type: :class:`OptionType` (``Call`` or ``Put``).

    Returns:
        tf.Tensor: American option price.
    """
    S, K, r, sigma, T = spot_price, strike, riskfree_rate, volatility, time_to_maturity
    b = r - dividend_yield  # cost of carry

    if option_type == OptionType.Call:
        return _bs1993_call(S, K, r, b, sigma, T)
    else:
        # Put-call symmetry: AmPut(S, K, r, b) = AmCall(K, S, r-b, -b)
        return _bs1993_call(K, S, r - b, -b, sigma, T)


class BlackScholesPricer(Pricer):
    """European Black-Scholes and American Bjerksund-Stensland (1993) pricer for vanilla options.

    All market inputs come from the underlying's :class:`Equity` view
    (``market_env.get_equity``), i.e. from the market risk factors: the price
    (float64) is differentiable with respect to spot, repo, dividend yield, the
    vol surface nodes and the discount-curve pillars. Get greeks with one
    ``pricer.tape.gradient(product.price, market_env.risk_factors())``.

    Only the time to maturity is a pricer-local variable
    (``product.time_to_maturity``), for theta.

    Dividends:

    - Underlying with discrete dividends (``DIV``): ``dividend_model`` selects
      ``"discrete"`` (``S* = S - PV(div)``, carry = repo) or ``"continuous"``
      (``q_eff = -ln(1 - PV(div)/S) / T`` + repo). In the continuous model
      ``q_eff`` is held fixed when ``S`` moves (sticky yield): the AAD delta is
      ``e^{-qT}N(d1)``, while a bump-and-reprice recomputes ``q_eff`` and gives
      the discrete-model delta.
    - Underlying with only ``DIVYIELD``: carry = div yield + repo, whatever
      ``dividend_model``.
    """

    def __init__(
        self,
        dividend_model: str = "continuous",
        use_implied_repo: bool = True,
    ):
        """Initialize the pricer.

        Args:
            dividend_model (str): ``"discrete"`` or ``"continuous"`` (used when the
                underlying has a discrete dividend schedule, see class docstring).
            use_implied_repo (bool): When ``True``, the market repo is part of the
                carry; when ``False`` the repo is ignored.

        Raises:
            ValueError: If ``dividend_model`` is not supported.
        """
        super().__init__()
        allowed_dividend_models = {"continuous", "discrete"}
        if dividend_model not in allowed_dividend_models:
            raise ValueError(
                f"Invalid dividend_model '{dividend_model}'. "
                f"Allowed values are: {sorted(allowed_dividend_models)}"
            )
        self._dividend_model = dividend_model
        self._use_implied_repo = use_implied_repo

    def _build_inputs(
        self, product: VanillaOption, market_env: MarketEnvironment
    ) -> tuple:
        """Black-Scholes inputs from the underlying's market data.

        Args:
            product (VanillaOption): The option to be priced.
            market_env (MarketEnvironment): Market data.

        Returns:
            tuple: ``(s_net, k, r, q, sigma, t)`` float64 tensors, where ``s_net``
                is the spot (net of PV of dividends in the discrete model) and
                ``q`` the continuous carry deduction.

        Raises:
            ValueError: If the product is not a VanillaOption, if Bermudan
                exercise is requested, or if the underlying has no vol surface.
        """
        if not isinstance(product, VanillaOption):
            raise ValueError("product must be a VanillaOption")
        if product.exercise_type == ExerciseType.Bermudan:
            raise ValueError("Bermudan exercise is not supported")

        eq = market_env.get_equity(product.underlying, ccy=product.ccy)
        if eq.vol_surface is None:
            raise ValueError(f"No volatility surface for underlying '{product.underlying}'")

        # Time to maturity: the only pricer-local variable (theta).
        # r and q_eff use its float value, so theta keeps r and q fixed.
        tenor = eq.vol_surface.daycounter.year_fraction(Settings.evaluation_date, product.end_date)
        t = tf.Variable(tenor, dtype=DTYPE)

        k = tf.cast(product.strike, DTYPE)
        sigma = eq.vol_surface.volatility(strike=float(product.strike), tenor=tenor)
        discount_factor = tf.cast(eq.discount_curve.discount(product.end_date), DTYPE)
        r = -tf.math.log(discount_factor) / tenor

        repo = eq.repo if self._use_implied_repo else tf.constant(0.0, dtype=DTYPE)
        s = eq.spot
        pv_div = tf.constant(0.0, dtype=DTYPE)

        if eq.dividend_model == "discrete" and self._dividend_model == "discrete":
            pv_div = eq.pv_dividends(product.end_date)
            s_net = s - pv_div
            q = repo
        elif eq.dividend_model == "discrete":
            # Equivalent continuous yield; S held fixed in q_eff (sticky yield).
            q_eff = -tf.math.log(1.0 - eq.pv_dividends(product.end_date) / tf.stop_gradient(s)) / tenor
            s_net = s
            q = q_eff + repo
        elif eq.dividend_model == "continuous":
            s_net = s
            q = eq.div_yield + repo
        else:
            s_net = s
            q = repo

        # --- diagnostics on the product (tensors; spot is the market variable) ---
        product.spot = s
        product.spot_net = s_net
        product.pv_discrete_dividends = pv_div
        product.repo_margin = repo
        product.risk_free_rate = r
        product.discount_factor = discount_factor
        product.volatility = sigma
        product.time_to_maturity = t
        product.forward = s_net * tf.exp((r - q) * t)

        return s_net, k, r, q, sigma, t

    def calculate_price(self, product: VanillaOption, market_env: MarketEnvironment) -> tf.Tensor:
        """Price a VanillaOption: Black-Scholes (European) or Bjerksund-Stensland 1993 (American).

        Args:
            product (VanillaOption): The option to be priced.
            market_env (MarketEnvironment): Market data.

        Returns:
            tf.Tensor: Scalar float64 NPV.
        """
        s_net, k, r, q, sigma, t = self._build_inputs(product, market_env)
        if product.exercise_type == ExerciseType.European:
            return blackscholes_calc(s_net, k, r, sigma, t, q, product.option_type)
        return bjerksund_stensland_calc(s_net, k, r, sigma, t, q, product.option_type)
