from tensorflow import Variable
import tensorflow
from tensorflow.python.framework import dtypes

from .stochasticprocess import StochasticProcess
from .ornsteinuhlenbeck import OrnsteinUhlenbeckProcess
from ..markethandles.ircurve import RateCurve


class HullWhiteProcess(StochasticProcess):
    """
    Hull-White interest rate model class, which models the evolution of short rates under a mean-reverting stochastic process.

    Attributes:
        _process (OrnsteinUhlenbeckProcess): Ornstein-Uhlenbeck process to simulate the short rate evolution.
        _a (tf.Variable): Mean reversion speed (alpha).
        _sigma (tf.Variable): Volatility of the process (sigma).
        _term_structure (RateCurve): The term structure (yield curve) used for forward rate calculations.
    """

    def __init__(self, term_structure: RateCurve, a: float, sigma: float):
        """
        Initializes the Hull-White process.

        Args:
            term_structure (RateCurve): The term structure (yield curve) used for forward rate calculations.
            a (float): Mean reversion speed (alpha) of the process.
            sigma (float): Volatility of the process (sigma).
        """
        self._process = OrnsteinUhlenbeckProcess(
            mr_speed=a, volatility=sigma, x0=term_structure.inst_fwd(0)
        )
        self._a = Variable(a, dtype=dtypes.float64)
        self._sigma = Variable(sigma, dtype=dtypes.float64)
        self._term_structure = term_structure

    def size(self) -> int:
        """
        Returns the dimensionality of the process.

        Returns:
            int: The dimensionality (size) of the underlying process.
        """
        return self._process.size()

    def initial_values(self) -> float:
        """
        Returns the initial value of the short rate.

        Returns:
            float: The initial value of the short rate (x0).
        """
        return self._process.x0

    @property
    def a(self) -> float:
        """
        Mean reversion speed (alpha) of the process.

        Returns:
            float: The mean reversion speed.
        """
        return self._a.numpy()

    @property
    def sigma(self) -> float:
        """
        Volatility (sigma) of the process.

        Returns:
            float: The volatility of the process.
        """
        return self._sigma.numpy()

    @property
    def x0(self) -> float:
        """
        Initial value of the process.

        Returns:
            float: The initial short rate (x0).
        """
        return self._process.x0.numpy()

    def drift(self, t: float, x: float) -> float:
        """
        Calculates the drift term of the process.

        Args:
            t (float): Current time.
            x (float): Current value of the process.

        Returns:
            float: The drift value.
        """
        alpha_drift = (
            self._sigma**2 / (2 * self._a) * (1 - tensorflow.math.exp(-2 * self._a * t))
        )
        shift = 0.0001
        f = self._term_structure.forward_rate(t, t)
        fup = self._term_structure.forward_rate(t + shift, t + shift)
        f_prime = (fup - f) / shift
        alpha_drift += self._a * f + f_prime
        return self._process.drift(t, x) + alpha_drift

    def diffusion(self, t: float, x: float) -> float:
        """
        Calculates the diffusion term of the process.

        Args:
            t (float): Current time.
            x (float): Current value of the process.

        Returns:
            float: The diffusion value.
        """
        return self._process.diffusion(t, x)

    def expectation(self, t0: float, x0: float, dt: float) -> float:
        """
        Computes the expectation of the process over time interval dt.

        Args:
            t0 (float): Start time of the interval.
            x0 (float): Initial value of the process at time t0.
            dt (float): Time increment.

        Returns:
            float: The expected value of the process at time t0 + dt.
        """
        return (
            self._process.expectation(x0=x0, dt=dt)
            + self.alpha(t0 + dt)
            - self.alpha(t0) * tensorflow.math.exp(-self._a * dt)
        )

    def std_deviation(self, dt: float, t0=None, x0=None) -> float:
        """
        Returns the standard deviation of the process over the time interval dt.

        Args:
            dt (float): Time increment.
            t0 (float, optional): Interface placeholder for starting time (default is None).
            x0 (float, optional): Interface placeholder for starting value (default is None).

        Returns:
            float: The standard deviation of the process.
        """
        return self._process.std_deviation(dt=dt)

    def variance(self, dt: float) -> float:
        """
        Returns the variance of the process over the time interval dt.

        Args:
            dt (float): Time increment.

        Returns:
            float: The variance of the process.
        """
        return self._process.variance(dt)

    def alpha(self, t: float) -> float:
        """
        Computes the alpha (mean reversion level) term at time t.

        Args:
            t (float): The time at which alpha is calculated.

        Returns:
            float: The alpha value.
        """
        if self.a > 1e-10:
            alfa = (self._sigma / self._a) * (1 - tensorflow.math.exp(-self._a * t))
        else:
            alfa = self._sigma * t
        alfa = 0.5 * alfa**2
        alfa += self._term_structure.inst_fwd(t)
        return alfa

    def A_B(self, S: float, T: float) -> tuple:
        """
        Computes the time-dependent parameters A(S, T) and B(S, T) of a zero-coupon bond.

        Args:
            S (float): Start time in years (S <= T).
            T (float): Maturity time in years.

        Returns:
            tuple: A(S, T) and B(S, T) parameters used in bond pricing.
        """
        f0S = self._term_structure.inst_fwd(S)
        P0T = self._term_structure.discount(T)
        P0S = self._term_structure.discount(S)

        B = 1 - tensorflow.math.exp(-self._a * (T - S))
        B /= self._a

        exponent = self.sigma * (
            tensorflow.math.exp(-self._a * T) - tensorflow.math.exp(-self._a * S)
        )
        exponent *= exponent
        exponent *= tensorflow.math.exp(2 * self._a * S) - 1
        exponent /= -4 * (self._a**3)
        exponent += B * f0S
        A = tensorflow.math.exp(exponent) * P0T / P0S
        return A, B

    def zero_bond(self, S: float, T: float, rs: float) -> float:
        """
        Computes the price of a zero-coupon bond at future time S with maturity T.

        Args:
            S (float): Future reference time in years.
            T (float): Maturity time in years.
            rs (float): Short rate at time S.

        Returns:
            float: Price of the zero-coupon bond.
        """
        A, B = self.A_B(S, T)
        return A * tensorflow.math.exp(-B * rs)
