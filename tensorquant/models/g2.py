import tensorflow
from tensorflow import Variable
from tensorflow.python.framework import dtypes

from .stochasticprocess import StochasticProcess
from .ornsteinuhlenbeck import OrnsteinUhlenbeckProcess
from ..markethandles.ircurve import RateCurve


class G2PlusPlusProcess(StochasticProcess):
    """
    G2++ two-factor Gaussian interest rate model.

    The short rate is decomposed as:
        r(t) = x(t) + y(t) + φ(t)

    where x(t) and y(t) are two correlated zero-mean Ornstein-Uhlenbeck processes:
        dx(t) = -a·x(t)·dt + σ·dW₁(t)
        dy(t) = -b·y(t)·dt + η·dW₂(t)
        dW₁·dW₂ = ρ·dt

    Both factors start at zero: x(0) = y(0) = 0.

    The deterministic shift φ(t) is chosen so that the model fits the observed
    initial term structure exactly:
        φ(t) = f^M(0,t)
               + σ²/(2a²)·(1 - e^{-at})²
               + η²/(2b²)·(1 - e^{-bt})²
               + ρ·σ·η/(a·b)·(1 - e^{-at})·(1 - e^{-bt})

    Zero-coupon bond prices have the closed-form affine expression:
        P(S, T) = A(S, T)·exp(-B_a(τ)·x(S) - B_b(τ)·y(S))

    where τ = T - S and A(S,T), B_a(τ), B_b(τ) are derived analytically.

    Attributes:
        _process_x (OrnsteinUhlenbeckProcess): OU process driving the first factor x.
        _process_y (OrnsteinUhlenbeckProcess): OU process driving the second factor y.
        _a  (tf.Variable): Mean-reversion speed of x.
        _b  (tf.Variable): Mean-reversion speed of y.
        _sigma (tf.Variable): Volatility of x.
        _eta   (tf.Variable): Volatility of y.
        _rho   (tf.Variable): Instantaneous correlation between W₁ and W₂.
        _term_structure (RateCurve): Initial yield curve used to calibrate φ(t).
    """

    def __init__(
        self,
        term_structure: RateCurve,
        a: float,
        b: float,
        sigma: float,
        eta: float,
        rho: float,
    ):
        """
        Initialises the G2++ process.

        Args:
            term_structure (RateCurve): The initial term structure (yield curve).
            a     (float): Mean-reversion speed of the first factor x.
            b     (float): Mean-reversion speed of the second factor y.
            sigma (float): Volatility of the first factor x.
            eta   (float): Volatility of the second factor y.
            rho   (float): Correlation between the two Brownian motions (|rho| < 1).
        """
        self._process_x = OrnsteinUhlenbeckProcess(mr_speed=a, volatility=sigma, x0=0.0)
        self._process_y = OrnsteinUhlenbeckProcess(mr_speed=b, volatility=eta, x0=0.0)
        self._a = Variable(a, dtype=dtypes.float64)
        self._b = Variable(b, dtype=dtypes.float64)
        self._sigma = Variable(sigma, dtype=dtypes.float64)
        self._eta = Variable(eta, dtype=dtypes.float64)
        self._rho = Variable(rho, dtype=dtypes.float64)
        self._term_structure = term_structure

    # ------------------------------------------------------------------
    # StochasticProcess interface
    # ------------------------------------------------------------------

    def size(self) -> int:
        """
        Returns the dimensionality of the process.

        Returns:
            int: 2, since G2++ is a two-factor model.
        """
        return 2

    def initial_values(self) -> list:
        """
        Returns the initial values of the two factors.

        Returns:
            list[float]: [x(0), y(0)] = [0.0, 0.0].
        """
        return [self._process_x.x0, self._process_y.x0]

    def drift(self, t: float, x: float, y: float) -> tuple:
        """
        Computes the drift of the two-factor state vector (x, y).

        Args:
            t (float): Current time (unused; included for interface compatibility).
            x (float): Current value of the first factor.
            y (float): Current value of the second factor.

        Returns:
            tuple[float, float]: Drift terms (drift_x, drift_y) = (-a·x, -b·y).
        """
        drift_x = self._process_x.drift(x)
        drift_y = self._process_y.drift(y)
        return drift_x, drift_y

    def diffusion(self, t: float, x: float, y: float) -> tuple:
        """
        Returns the diffusion parameters of the two-factor state vector.

        The instantaneous covariance matrix is:
            [[σ²,     ρ·σ·η],
             [ρ·σ·η,  η²   ]]

        Args:
            t (float): Current time (unused; included for interface compatibility).
            x (float): Current value of the first factor (unused).
            y (float): Current value of the second factor (unused).

        Returns:
            tuple: (sigma, eta, rho) — the volatilities and their correlation.
        """
        return self._sigma, self._eta, self._rho

    def expectation(self, t0: float, x0: float, y0: float, dt: float) -> tuple:
        """
        Computes the conditional expectation of (x, y) over the interval [t0, t0+dt].

        Uses the exact OU transition:
            E[x(t+dt) | x(t)] = x(t)·e^{-a·dt}
            E[y(t+dt) | y(t)] = y(t)·e^{-b·dt}

        Args:
            t0 (float): Start of the interval (unused; OU is time-homogeneous).
            x0 (float): Value of the first factor at t0.
            y0 (float): Value of the second factor at t0.
            dt (float): Length of the time interval.

        Returns:
            tuple[float, float]: (E[x(t0+dt)], E[y(t0+dt)]).
        """
        ex = self._process_x.expectation(x0=x0, dt=dt)
        ey = self._process_y.expectation(x0=y0, dt=dt)
        return ex, ey

    def std_deviation(self, dt: float, t0=None, x0=None) -> tuple:
        """
        Returns the marginal standard deviations of x and y over the interval dt.

        Args:
            dt  (float): Length of the time interval.
            t0  (float, optional): Start time (unused).
            x0  (float, optional): Initial value (unused).

        Returns:
            tuple[float, float]: (std_x, std_y).
        """
        return (
            self._process_x.std_deviation(dt=dt),
            self._process_y.std_deviation(dt=dt),
        )

    # ------------------------------------------------------------------
    # G2++-specific methods
    # ------------------------------------------------------------------

    @property
    def a(self) -> float:
        """Mean-reversion speed of the first factor x."""
        return self._a.numpy()

    @property
    def b(self) -> float:
        """Mean-reversion speed of the second factor y."""
        return self._b.numpy()

    @property
    def sigma(self) -> float:
        """Volatility of the first factor x."""
        return self._sigma.numpy()

    @property
    def eta(self) -> float:
        """Volatility of the second factor y."""
        return self._eta.numpy()

    @property
    def rho(self) -> float:
        """Instantaneous correlation between the two Brownian motions."""
        return self._rho.numpy()

    def variance_x(self, dt: float) -> float:
        """
        Conditional variance of x(t+dt) given x(t).

        Var[x(t+dt) | x(t)] = σ²·(1 - e^{-2a·dt}) / (2a)

        Args:
            dt (float): Time increment.

        Returns:
            float: Conditional variance of x.
        """
        return self._process_x.variance(dt)

    def variance_y(self, dt: float) -> float:
        """
        Conditional variance of y(t+dt) given y(t).

        Var[y(t+dt) | y(t)] = η²·(1 - e^{-2b·dt}) / (2b)

        Args:
            dt (float): Time increment.

        Returns:
            float: Conditional variance of y.
        """
        return self._process_y.variance(dt)

    def covariance(self, dt: float) -> float:
        """
        Conditional covariance between x(t+dt) and y(t+dt).

        Cov[x(t+dt), y(t+dt) | x(t), y(t)] = ρ·σ·η·(1 - e^{-(a+b)·dt}) / (a+b)

        Args:
            dt (float): Time increment.

        Returns:
            float: Conditional covariance between x and y.
        """
        return (
            self._rho
            * self._sigma
            * self._eta
            / (self._a + self._b)
            * (1 - tensorflow.math.exp(-(self._a + self._b) * dt))
        )

    def phi(self, t: float) -> float:
        """
        Computes the deterministic shift φ(t) that fits the initial term structure.

        φ(t) = f^M(0,t)
               + σ²/(2a²)·(1 - e^{-at})²
               + η²/(2b²)·(1 - e^{-bt})²
               + ρ·σ·η/(a·b)·(1 - e^{-at})·(1 - e^{-bt})

        Args:
            t (float): Time in years.

        Returns:
            float: The value of the deterministic shift at time t.
        """
        f0t = self._term_structure.inst_fwd(t)
        term_x = (
            self._sigma ** 2
            / (2 * self._a ** 2)
            * (1 - tensorflow.math.exp(-self._a * t)) ** 2
        )
        term_y = (
            self._eta ** 2
            / (2 * self._b ** 2)
            * (1 - tensorflow.math.exp(-self._b * t)) ** 2
        )
        term_xy = (
            self._rho
            * self._sigma
            * self._eta
            / (self._a * self._b)
            * (1 - tensorflow.math.exp(-self._a * t))
            * (1 - tensorflow.math.exp(-self._b * t))
        )
        return f0t + term_x + term_y + term_xy

    def short_rate(self, t: float, x: float, y: float) -> float:
        """
        Computes the instantaneous short rate at time t.

        r(t) = φ(t) + x(t) + y(t)

        Args:
            t (float): Current time in years.
            x (float): Current value of the first factor.
            y (float): Current value of the second factor.

        Returns:
            float: The short rate r(t).
        """
        return self.phi(t) + x + y

    # ------------------------------------------------------------------
    # Bond pricing helpers
    # ------------------------------------------------------------------

    def _F(self, h, u):
        """
        Helper integral:

            F(h, u) = ∫₀ᵘ (1 - e^{-hs})² ds
                    = u - 2(1 - e^{-hu})/h + (1 - e^{-2hu})/(2h)

        Used in the computation of the A(S, T) term for zero-coupon bond pricing.

        Args:
            h: Mean-reversion parameter (tf.Variable or float).
            u: Upper limit of integration (float or tensor).

        Returns:
            Scalar: Value of the integral F(h, u).
        """
        return (
            u
            - 2 * (1 - tensorflow.math.exp(-h * u)) / h
            + (1 - tensorflow.math.exp(-2 * h * u)) / (2 * h)
        )

    def _G(self, u):
        """
        Helper integral:

            G(u) = ∫₀ᵘ (1 - e^{-as})(1 - e^{-bs}) ds
                 = u - (1-e^{-au})/a - (1-e^{-bu})/b + (1-e^{-(a+b)u})/(a+b)

        Used in the computation of the cross term in A(S, T).

        Args:
            u: Upper limit of integration (float or tensor).

        Returns:
            Scalar: Value of the integral G(u).
        """
        a, b = self._a, self._b
        return (
            u
            - (1 - tensorflow.math.exp(-a * u)) / a
            - (1 - tensorflow.math.exp(-b * u)) / b
            + (1 - tensorflow.math.exp(-(a + b) * u)) / (a + b)
        )

    def A_Bx_By(self, S: float, T: float) -> tuple:
        """
        Computes the affine bond-pricing coefficients A(S, T), B_x(τ) and B_y(τ).

        The zero-coupon bond price satisfies:
            P(S, T) = A(S, T) · exp(-B_x(τ)·x(S) - B_y(τ)·y(S))

        where τ = T - S and:
            B_x(τ) = (1 - e^{-aτ}) / a
            B_y(τ) = (1 - e^{-bτ}) / b

        The log of A(S, T) equals:
            ln A = ln(P^M(0,T)/P^M(0,S))
                   + σ²/(2a²)·[F(a,τ) - F(a,T) + F(a,S)]
                   + η²/(2b²)·[F(b,τ) - F(b,T) + F(b,S)]
                   + ρ·σ·η/(a·b)·[G(τ) - G(T) + G(S)]

        This expression correctly recovers P(0,T) = P^M(0,T) when S=0 (since
        x(0)=y(0)=0 and all F/G terms vanish at t=0).

        Args:
            S (float): Reference (start) time in years (S ≤ T).
            T (float): Maturity time in years.

        Returns:
            tuple: (A, B_x, B_y) where A is a scalar and B_x, B_y are scalars.
        """
        tau = T - S
        P0T = self._term_structure.discount(T)
        P0S = self._term_structure.discount(S)

        Bx = (1 - tensorflow.math.exp(-self._a * tau)) / self._a
        By = (1 - tensorflow.math.exp(-self._b * tau)) / self._b

        exponent = (
            self._sigma ** 2
            / (2 * self._a ** 2)
            * (self._F(self._a, tau) - self._F(self._a, T) + self._F(self._a, S))
        )
        exponent += (
            self._eta ** 2
            / (2 * self._b ** 2)
            * (self._F(self._b, tau) - self._F(self._b, T) + self._F(self._b, S))
        )
        exponent += (
            self._rho
            * self._sigma
            * self._eta
            / (self._a * self._b)
            * (self._G(tau) - self._G(T) + self._G(S))
        )

        A = (P0T / P0S) * tensorflow.math.exp(exponent)
        return A, Bx, By

    def zero_bond(self, S: float, T: float, xs: float, ys: float) -> float:
        """
        Computes the price of a zero-coupon bond at future time S with maturity T,
        given the factor realisations x(S) = xs and y(S) = ys.

        P(S, T) = A(S, T) · exp(-B_x(τ)·xs - B_y(τ)·ys)

        Args:
            S  (float): Future reference time in years.
            T  (float): Maturity time in years.
            xs (float): Realisation of the first factor at time S.
            ys (float): Realisation of the second factor at time S.

        Returns:
            float: Price of the zero-coupon bond.
        """
        A, Bx, By = self.A_Bx_By(S, T)
        return A * tensorflow.math.exp(-Bx * xs - By * ys)

