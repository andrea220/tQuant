from .stochasticprocess import StochasticProcess
from tensorflow.python.framework import dtypes
from tensorflow import Variable, Tensor
from math import exp, sqrt


class OrnsteinUhlenbeckProcess(StochasticProcess):
    """Ornstein-Uhlenbeck process class for modeling mean-reverting stochastic processes.

    This class represents an Ornstein-Uhlenbeck process, which is a type of mean-reverting stochastic process.
    It is defined by its mean reversion speed, volatility, initial value, and long-term mean level.

    Attributes:
        mr_speed (float): The speed of mean reversion.
        volatility (float): The volatility of the process.
        x0 (float, optional): The initial value of the process. Defaults to 0.0.
        level (float, optional): The long-term mean level of the process. Defaults to 0.0.
    """

    def __init__(self, mr_speed, volatility, x0=0.0, level=0.0):
        """Initializes the Ornstein-Uhlenbeck process.

        Args:
            mr_speed (float): The speed of mean reversion.
            volatility (float): The volatility of the process.
            x0 (float, optional): The initial value of the process. Defaults to 0.0.
            level (float, optional): The long-term mean level of the process. Defaults to 0.0.
        """
        self._x0 = Variable(x0, dtype=dtypes.float64)
        self._mr_speed = Variable(mr_speed, dtype=dtypes.float64)
        self._volatility = Variable(volatility, dtype=dtypes.float64)
        self._level = level

    def size(self):
        """Returns the size of the state space.

        Returns:
            int: The size of the state space, which is 1 for this process.
        """
        return 1

    def initial_values(self):
        """Returns the initial values of the process.

        Returns:
            float: The initial value of the process.
        """
        return self.x0

    def drift(self, x: Tensor) -> Tensor:
        """Calculates the drift term of the process.

        Args:
            x (tensorflow.Tensor): The current value of the process.

        Returns:
            tensorflow.Tensor: The drift term.
        """
        return self._mr_speed * (self._level - x)

    def diffusion(self) -> Tensor:
        """Returns the diffusion term of the process.

        Returns:
            tensorflow.Tensor: The diffusion term (volatility) of the process.
        """
        return self._volatility

    def expectation(self, x0: float, dt: float, t0=None) -> Tensor:
        """Calculates the expected value of the process after a time step.

        Args:
            x0 (float): The initial value of the process.
            dt (float): The time step.
            t0 (float, optional): The initial time. Defaults to None.

        Returns:
            tensorflow.Tensor: The expected value of the process at time t0 + dt.
        """
        return self._level + (x0 - self._level) * exp(-self._mr_speed * dt)

    def std_deviation(self, dt: float, t0=None, x0=None) -> Tensor:
        """Calculates the standard deviation of the process after a time step.

        Args:
            dt (float): The time step.
            t0 (float, optional): The initial time. Defaults to None.
            x0 (float, optional): The initial value. Defaults to None.

        Returns:
            tensorflow.Tensor: The standard deviation of the process at time t0 + dt.
        """
        return sqrt(self.variance(dt))

    def variance(self, dt: float) -> Tensor:
        """Calculates the variance of the process after a time step.

        Args:
            dt (float): The time step.

        Returns:
            tensorflow.Tensor: The variance of the process at time t0 + dt.
        """
        return (
            0.5
            * self._volatility**2
            / self._mr_speed
            * (1 - exp(-2 * self._mr_speed * dt))
        )

    @property
    def x0(self):
        """Returns the initial value of the process.

        Returns:
            float: The initial value of the process.
        """
        return self._x0.numpy()

    @property
    def mr_speed(self):
        """Returns the speed of mean reversion.

        Returns:
            float: The speed of mean reversion.
        """
        return self._mr_speed.numpy()

    @property
    def volatility(self):
        """Returns the volatility of the process.

        Returns:
            float: The volatility of the process.
        """
        return self._volatility.numpy()

    @property
    def level(self):
        """Returns the long-term mean level of the process.

        Returns:
            float: The long-term mean level of the process.
        """
        return self._level
