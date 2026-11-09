from .pricer import Pricer
from ..instruments.barrier import BarrierOption
from ..instruments.digital import DigitalOption
from ..instruments.option import VanillaOption
from ..markethandles.utils import BarrierType, ExerciseType, OptionType, PayoffType
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


def digital_calc(
    spot_price: tf.Tensor,
    strike: tf.Tensor,
    riskfree_rate: tf.Tensor,
    volatility: tf.Tensor,
    time_to_maturity: tf.Tensor,
    dividend_yield: tf.Tensor,
    option_type: OptionType,
    payoff_type: PayoffType,
    cash_amount: float = 1.0,
) -> tf.Tensor:
    """European digital price under Black-Scholes, ``phi = +1`` (call) or ``-1`` (put).

    - cash-or-nothing: ``A e^{-rT} N(phi d2)`` (``N(phi d2)``: risk-neutral probability of finishing in the money);
    - asset-or-nothing: ``S e^{-qT} N(phi d1)``.

    Asset-or-nothing minus ``K`` times the unit cash-or-nothing is the vanilla. The vol is the one at the strike:
    the smile slope is not in the formula (a call spread on the smile includes it).

    Args:
        spot_price, strike, riskfree_rate, volatility, time_to_maturity, dividend_yield: As
            :func:`blackscholes_calc`.
        option_type: :class:`OptionType`.
        payoff_type: ``CashOrNothing`` or ``AssetOrNothing``.
        cash_amount: Amount of a cash-or-nothing digital.

    Returns:
        tf.Tensor: Digital price.
    """
    phi = tf.constant(float(option_type.value), dtype=spot_price.dtype)
    sqrt_t = tf.sqrt(time_to_maturity)
    d1 = (
        tf.math.log(spot_price / strike)
        + (riskfree_rate - dividend_yield + (volatility**2) / 2) * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    if payoff_type == PayoffType.AssetOrNothing:
        return spot_price * tf.exp(-dividend_yield * time_to_maturity) * _norm_cdf(phi * d1)
    return cash_amount * tf.exp(-riskfree_rate * time_to_maturity) * _norm_cdf(phi * d2)


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


# Broadie-Glasserman-Kou constant: -zeta(1/2) / sqrt(2 pi)
_BGK_BETA = 0.5826


def _knock_in_at_expiry(S, K, H, r, sigma, T, q, option_type: OptionType, down: bool) -> tf.Tensor:
    """Knock-in with the barrier observed at expiry only: ``vanilla(S_T) 1{S_T <= H}`` (down) or ``1{S_T >= H}`` (up),
    exact by vanillas and cash digitals struck at the barrier."""
    def van(strike, ot):
        return blackscholes_calc(S, strike, r, sigma, T, q, ot)

    def dig(ot):
        return digital_calc(S, H, r, sigma, T, q, ot, PayoffType.CashOrNothing)

    k, h = float(K), float(H)
    zero = tf.zeros_like(S)
    if option_type == OptionType.Put:
        if down:     # (K - S)+ 1{S <= H}
            return van(K, OptionType.Put) if k <= h else van(H, OptionType.Put) + (K - H) * dig(OptionType.Put)
        # up: (K - S)+ 1{S >= H}
        return zero if k <= h else van(K, OptionType.Put) - van(H, OptionType.Put) - (K - H) * dig(OptionType.Put)
    if down:         # (S - K)+ 1{S <= H}
        return zero if k >= h else van(K, OptionType.Call) - van(H, OptionType.Call) - (H - K) * dig(OptionType.Call)
    # up: (S - K)+ 1{S >= H}
    return van(K, OptionType.Call) if k >= h else van(H, OptionType.Call) + (H - K) * dig(OptionType.Call)


def _knock_in_continuous(S, K, H, r, sigma, T, q, option_type: OptionType, down: bool) -> tf.Tensor:
    """Reiner-Rubinstein (1991) knock-in price, continuous monitoring, no rebate (Haug's A, B, C, D terms)."""
    b = r - q
    phi = 1.0 if option_type == OptionType.Call else -1.0
    eta = 1.0 if down else -1.0
    vol_t = sigma * tf.sqrt(T)
    mu = (b - 0.5 * sigma ** 2) / sigma ** 2
    x1 = tf.math.log(S / K) / vol_t + (1.0 + mu) * vol_t
    x2 = tf.math.log(S / H) / vol_t + (1.0 + mu) * vol_t
    y1 = tf.math.log(H * H / (S * K)) / vol_t + (1.0 + mu) * vol_t
    y2 = tf.math.log(H / S) / vol_t + (1.0 + mu) * vol_t
    fwd_s = S * tf.exp((b - r) * T)
    df_k = K * tf.exp(-r * T)
    hs = H / S
    A = phi * fwd_s * _norm_cdf(phi * x1) - phi * df_k * _norm_cdf(phi * (x1 - vol_t))
    B = phi * fwd_s * _norm_cdf(phi * x2) - phi * df_k * _norm_cdf(phi * (x2 - vol_t))
    C = (phi * fwd_s * tf.pow(hs, 2.0 * (mu + 1.0)) * _norm_cdf(eta * y1)
         - phi * df_k * tf.pow(hs, 2.0 * mu) * _norm_cdf(eta * (y1 - vol_t)))
    D = (phi * fwd_s * tf.pow(hs, 2.0 * (mu + 1.0)) * _norm_cdf(eta * y2)
         - phi * df_k * tf.pow(hs, 2.0 * mu) * _norm_cdf(eta * (y2 - vol_t)))
    strike_above = float(K) > float(H)
    if option_type == OptionType.Call:
        if down:
            return C if strike_above else A - B + D
        return A if strike_above else B - C + D
    if down:
        return B - C + D if strike_above else A
    return A - B + D if strike_above else C


def barrier_calc(
    spot_price: tf.Tensor,
    strike: tf.Tensor,
    barrier: tf.Tensor,
    riskfree_rate: tf.Tensor,
    volatility: tf.Tensor,
    time_to_maturity: tf.Tensor,
    dividend_yield: tf.Tensor,
    option_type: OptionType,
    barrier_type: BarrierType,
    n_monitoring: int | None = None,
) -> tf.Tensor:
    """European single-barrier option under Black-Scholes, no rebate.

    - continuous monitoring (``n_monitoring=None``): Reiner-Rubinstein closed form;
    - one monitoring date at expiry (``n_monitoring=1``): exact, vanilla and cash digitals struck at the barrier;
    - ``n`` dates (taken as equally spaced, the last at expiry): continuous formula with the Broadie-Glasserman-Kou shifted barrier
      ``H exp(+-0.5826 sigma sqrt(T / n))`` (away from the spot).

    Knock-out = vanilla - knock-in. A barrier already crossed by the spot gives knock-in = vanilla.

    Args:
        spot_price, strike, riskfree_rate, volatility, time_to_maturity, dividend_yield: As
            :func:`blackscholes_calc`.
        barrier: Barrier level.
        option_type: :class:`OptionType`.
        barrier_type: :class:`BarrierType`.
        n_monitoring: Number of discrete monitoring dates, ``None`` for continuous monitoring.

    Returns:
        tf.Tensor: Barrier option price.
    """
    S, K, r, sigma, T, q = spot_price, strike, riskfree_rate, volatility, time_to_maturity, dividend_yield
    H = tf.cast(barrier, S.dtype)
    down = barrier_type.is_down
    vanilla = blackscholes_calc(S, K, r, sigma, T, q, option_type)
    crossed = float(S) <= float(H) if down else float(S) >= float(H)
    if crossed:
        knock_in = vanilla
    elif n_monitoring == 0:           # no monitoring date left
        knock_in = tf.zeros_like(vanilla)
    elif n_monitoring == 1:
        knock_in = _knock_in_at_expiry(S, K, H, r, sigma, T, q, option_type, down)
    else:
        if n_monitoring is not None:
            sign = -1.0 if down else 1.0
            H = H * tf.exp(sign * _BGK_BETA * sigma * tf.sqrt(T / n_monitoring))
        knock_in = _knock_in_continuous(S, K, H, r, sigma, T, q, option_type, down)
    return knock_in if barrier_type.is_in else vanilla - knock_in


class BlackScholesPricer(Pricer):
    """Black-Scholes pricer: vanilla options (European, American by Bjerksund-Stensland 1993), European digitals and
    single-barrier options (Reiner-Rubinstein, Broadie-Glasserman-Kou shift for discrete monitoring; the barrier is
    compared with the spot net of dividends in the discrete dividend model).

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

    Digitals: by default the closed form at the vol of the strike (no smile slope). With ``digital_spread = h > 0``
    a European digital is priced as a centred spread of vanillas on the smile, each at its own vol: unit cash
    digital ``u = phi (V(K - hK) - V(K + hK)) / (2 hK)`` (``-dV/dK`` for a call, ``dV/dK`` for a put, so it includes
    ``-phi vega dsigma/dK``); cash-or-nothing ``A u``, asset-or-nothing ``K u + phi V(K)``.
    """

    def __init__(
        self,
        dividend_model: str = "continuous",
        use_implied_repo: bool = True,
        digital_spread: float = None,
    ):
        """Initialize the pricer.

        Args:
            dividend_model (str): ``"discrete"`` or ``"continuous"`` (used when the
                underlying has a discrete dividend schedule, see class docstring).
            use_implied_repo (bool): When ``True``, the market repo is part of the
                carry; when ``False`` the repo is ignored.
            digital_spread (float, optional): Half-width of the vanilla spread as a fraction of the strike
                (e.g. ``0.005``); ``None``: digitals by the closed form at the vol of the strike.

        Raises:
            ValueError: If ``dividend_model`` is not supported or ``digital_spread`` is not positive.
        """
        super().__init__()
        allowed_dividend_models = {"continuous", "discrete"}
        if dividend_model not in allowed_dividend_models:
            raise ValueError(
                f"Invalid dividend_model '{dividend_model}'. "
                f"Allowed values are: {sorted(allowed_dividend_models)}"
            )
        if digital_spread is not None and not digital_spread > 0.0:
            raise ValueError(f"digital_spread must be positive, got {digital_spread}")
        self._dividend_model = dividend_model
        self._use_implied_repo = use_implied_repo
        self._digital_spread = digital_spread

    def _build_inputs(
        self, product, market_env: MarketEnvironment
    ) -> tuple:
        """Black-Scholes inputs from the underlying's market data.

        Args:
            product (VanillaOption | DigitalOption): The option to be priced.
            market_env (MarketEnvironment): Market data.

        Returns:
            tuple: ``(s_net, k, r, q, sigma, t)`` float64 tensors, where ``s_net``
                is the spot (net of PV of dividends in the discrete model) and
                ``q`` the continuous carry deduction.

        Raises:
            ValueError: If the product is not a VanillaOption or DigitalOption, if Bermudan
                exercise is requested, or if the underlying has no vol surface.
        """
        if not isinstance(product, (VanillaOption, DigitalOption, BarrierOption)):
            raise ValueError("product must be a VanillaOption, a DigitalOption or a BarrierOption")
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

    def calculate_price(self, product, market_env: MarketEnvironment) -> tf.Tensor:
        """Price a VanillaOption (Black-Scholes European, Bjerksund-Stensland 1993 American), a DigitalOption or a
        BarrierOption.

        Args:
            product (VanillaOption | DigitalOption): The option to be priced.
            market_env (MarketEnvironment): Market data.

        Returns:
            tf.Tensor: Scalar float64 NPV.
        """
        s_net, k, r, q, sigma, t = self._build_inputs(product, market_env)
        if isinstance(product, BarrierOption):
            n_monitoring = None
            if not product.continuous:
                future = [d for d in product.monitoring_dates if d > Settings.evaluation_date]
                n_monitoring = len(future)
                if n_monitoring == 1 and future[0] != product.end_date:
                    raise ValueError("a single monitoring date must be the expiry")
            return barrier_calc(s_net, k, product.barrier, r, sigma, t, q, product.option_type,
                                product.barrier_type, n_monitoring)
        if isinstance(product, DigitalOption):
            if self._digital_spread is None:
                return digital_calc(s_net, k, r, sigma, t, q, product.option_type, product.payoff_type,
                                    product.cash_amount)
            return self._digital_by_spread(product, market_env, s_net, k, r, sigma, t, q)
        if product.exercise_type == ExerciseType.European:
            return blackscholes_calc(s_net, k, r, sigma, t, q, product.option_type)
        return bjerksund_stensland_calc(s_net, k, r, sigma, t, q, product.option_type)

    def _digital_by_spread(self, product, market_env, s_net, k, r, sigma, t, q) -> tf.Tensor:
        """European digital as a centred spread of vanillas on the smile (see the class docstring)."""
        vol_surface = market_env.get_equity(product.underlying, ccy=product.ccy).vol_surface
        tenor = float(t.numpy())
        strike = float(product.strike)
        h = self._digital_spread * strike
        phi = float(product.option_type.value)

        def vanilla(x: float) -> tf.Tensor:
            vol = vol_surface.volatility(strike=x, tenor=tenor)
            return blackscholes_calc(s_net, tf.constant(x, DTYPE), r, vol, t, q, product.option_type)

        unit = phi * (vanilla(strike - h) - vanilla(strike + h)) / (2.0 * h)
        if product.payoff_type == PayoffType.AssetOrNothing:
            return k * unit + phi * blackscholes_calc(s_net, k, r, sigma, t, q, product.option_type)
        return product.cash_amount * unit
