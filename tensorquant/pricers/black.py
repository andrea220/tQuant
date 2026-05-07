from .pricer import Pricer
from ..instruments.option import VanillaOption
from ..markethandles.utils import ExerciseType, OptionType
from ..markethandles.marketenvironment import MarketEnvironment
from ..timehandles.utils import Settings

import tensorflow as tf


def _norm_cdf(x: tf.Tensor) -> tf.Tensor:
    """Standard normal CDF computed via tf.math.erf (no tensorflow_probability)."""
    return 0.5 * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))


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
    phi = tf.constant(float(option_type.value), dtype=tf.float32)
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
    r_minus_b = tf.maximum(r - b, tf.constant(1e-7, dtype=tf.float32))
    B_0 = tf.maximum(K, r * K / r_minus_b)

    ht = -(b * T + 2.0 * sigma * tf.sqrt(T)) * B_0 * B_inf / ((B_inf - B_0) * K)
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
    def __init__(
        self,
        dividend_model: str = "continuous",
        use_implied_repo: bool = True,
    ):
        """Initialize the Black-Scholes / Bjerksund-Stensland pricer.

        Prices European options with the Black-Scholes formula and American
        options with the Bjerksund-Stensland (1993) approximation.

        Args:
            dividend_model (str): ``'discrete'`` uses ``S* = S - PV(div)``
                (QuantLib / BBG standard); ``'continuous'`` derives an
                equivalent yield ``q_eff = -ln(1 - PV_div/S) / T``.  Both
                models read from the :class:`DividendCurve` in the market.
            use_implied_repo (bool): When ``True``, reads the repo margin from
                the market environment and includes it in the carry.
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
        self._s = None
        self._k = None
        self._t = None
        self._r = None
        self._q = None
        self._sigma = None
        self._f = None

    def _build_inputs(
        self, product: VanillaOption, market_env: MarketEnvironment
    ) -> tuple:
        """Extract and validate all market inputs needed for pricing.

        Performs product / exercise-type validation, market-data lookup and
        converts all inputs to ``tf.Variable`` so the gradient tape can watch
        them.

        Args:
            product (VanillaOption): The option to be priced.
            market_env (MarketEnvironment): Typed accessor for market data.

        Returns:
            tuple: ``(s_net, k, r, q, sigma, t, f)`` as TensorFlow
                variables/tensors.

                Both dividend models read from the ``DividendCurve``:

                - ``'discrete'``: ``s_net = S - PV_div``, ``q = repo`` only.
                - ``'continuous'``: ``s_net = S``,
                  ``q_eff = -ln(1 - PV_div/S) / T`` (+ repo).

        Raises:
            ValueError: If the product is not a VanillaOption, if Bermudan
                exercise is requested, or if any required market datum is
                missing.
        """
        if not isinstance(product, VanillaOption):
            raise ValueError("product must be a VanillaOption")
        if product.exercise_type == ExerciseType.Bermudan:
            raise ValueError("Bermudan exercise is not supported")

        vol_surface = market_env.get_eq_vol_surface(product.underlying, ccy=product.ccy)
        disc_curve = market_env.get_ir_curve(product.ccy)
        spot_value = market_env.get_eq_spot(product.underlying, ccy=product.ccy)

        # Compute tenor once and reuse for both sigma lookup and t Variable
        tenor = vol_surface.daycounter.year_fraction(
            Settings.evaluation_date, product.end_date
        )

        sigma = tf.Variable(
            vol_surface.volatility(strike=product.strike.numpy(), tenor=tenor),
            dtype=tf.float32,
        )
        s = tf.Variable(spot_value, dtype=tf.float32)
        k = product.strike
        t = tf.Variable(tenor, dtype=tf.float32)

        discount_factor = tf.cast(disc_curve.discount(product.end_date), tf.float32)
        r = tf.Variable(-tf.math.log(discount_factor).numpy() / t.numpy(), dtype=tf.float32)

        # --- repo margin (independent Variable → rho w.r.t. repo) -----------
        if self._use_implied_repo:
            repo_margin = tf.Variable(
                market_env.get_eq_repo(product.underlying, ccy=product.ccy),
                dtype=tf.float32,
            )
        else:
            repo_margin = tf.Variable(0.0, dtype=tf.float32)

        # --- dividend model -------------------------------------------------
        # PV of discrete dividends always comes from the DividendCurve.
        div_curve = market_env.get_eq_dividends(product.underlying, ccy=product.ccy)
        pv_div = tf.cast(
            div_curve.pv_dividends(product.end_date, disc_curve), tf.float32
        )
        if self._dividend_model == "discrete":
            # QuantLib / BBG standard: S* = S - PV(div), q = repo only.
            # s_net is a plain tensor (not a new Variable) so the GradientTape
            # correctly propagates ∂P/∂S through s_net = S - PV_div.
            pv_discrete_dividends = pv_div
            s_net = s - pv_div
            q = repo_margin
        else:
            # Equivalent continuous yield derived from the discrete schedule:
            #   q_eff = -ln(1 - PV_div / S) / T
            # Detached via .numpy() → independent Variable, same convention
            # as r and sigma.
            pv_discrete_dividends = tf.constant(0.0, dtype=tf.float32)
            q_eff = tf.Variable(
                (-tf.math.log(1.0 - pv_div / s) / t).numpy(), dtype=tf.float32
            )
            s_net = s
            q = q_eff + repo_margin

        f = s_net * tf.exp((r - q) * t)

        # --- store diagnostics on product -----------------------------------
        product.spot = s
        product.spot_net = s_net
        product.pv_discrete_dividends = pv_discrete_dividends
        product.repo_margin = repo_margin
        product.risk_free_rate = r
        product.discount_factor = discount_factor
        product.volatility = sigma
        product.time_to_maturity = t

        return s_net, k, r, q, sigma, t, f

    def calculate_price(self, product: VanillaOption, market_env: MarketEnvironment) -> tf.Tensor:
        """Price a VanillaOption using Black-Scholes (European) or
        Bjerksund-Stensland 1993 (American).

        Delegates market-data extraction to :meth:`_build_inputs`.
        Intermediate ``tf.Variable`` inputs are stored on ``self`` so that an
        external ``GradientTape`` (managed by :meth:`Pricer.price`) can
        compute greeks.

        Args:
            product (VanillaOption): The option to be priced.
            market_env (MarketEnvironment): The market environment providing
                access to market data (curves, spots, volatilities).

        Returns:
            tf.Tensor: The NPV of the option.
        """
        self._s, self._k, self._r, self._q, self._sigma, self._t, self._f = (
            self._build_inputs(product, market_env)
        )
        product.forward = self._f

        if product.exercise_type == ExerciseType.European:
            return blackscholes_calc(
                self._s, self._k, self._r, self._sigma, self._t, self._q,
                product.option_type,
            )
        else:
            return bjerksund_stensland_calc(
                self._s, self._k, self._r, self._sigma, self._t, self._q,
                product.option_type,
            )
