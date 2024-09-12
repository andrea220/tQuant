from abc import ABC, abstractmethod


class StochasticProcess(ABC):
    """
    Abstract base class for stochastic processes. This defines the common interface for different types
    of stochastic processes used in mathematical finance and other fields.

    Methods:
        size(): Returns the number of dimensions of the process.
        initial_values(): Returns the initial value(s) of the process.
        drift(t0, x0, dt): Computes the drift term of the process.
        diffusion(t0, x0, dt): Computes the diffusion (volatility) term of the process.
        expectation(t0, x0, dt): Returns the expected value of the process after a time increment dt.
        std_deviation(t0, x0, dt): Returns the standard deviation of the process after a time increment dt.
        evolve(t0, x0, dt, dw): Simulates the evolution of the process over a time increment dt using a random term dw.
    """

    def __init__(self):
        """Initializes the StochasticProcess class."""
        pass

    @abstractmethod
    def size(self):
        """
        Returns the number of dimensions of the process.

        Returns:
            int: The number of dimensions (or factors) of the stochastic process.
        """
        pass

    @property
    def factors(self):
        """
        Returns the number of factors (dimensions) in the process.

        Returns:
            int: The number of factors (same as size()).
        """
        return self.size()

    @abstractmethod
    def initial_values(self):
        """
        Returns the initial values of the process.

        Returns:
            float or numpy array: The initial value(s) of the process.
        """
        pass

    @abstractmethod
    def drift(self, t0, x0, dt):
        """
        Computes the drift term of the process, which represents the deterministic trend.

        Args:
            t0 (float): The current time.
            x0 (float or numpy array): The current value(s) of the process.
            dt (float): The time increment.

        Returns:
            float or numpy array: The drift term at time t0 for the process.
        """
        pass

    @abstractmethod
    def diffusion(self, t0, x0, dt):
        """
        Computes the diffusion term of the process, which represents the stochastic component (volatility).

        Args:
            t0 (float): The current time.
            x0 (float or numpy array): The current value(s) of the process.
            dt (float): The time increment.

        Returns:
            float or numpy array: The diffusion (volatility) term at time t0.
        """
        pass

    def expectation(self, t0, x0, dt):  # TODO add discretization schemes
        """
        Computes the expectation (mean) of the process at time t0 + dt, using Euler discretization.

        Args:
            t0 (float): The current time.
            x0 (float or numpy array): The current value(s) of the process.
            dt (float): The time increment.

        Returns:
            float or numpy array: The expected value(s) of the process at time t0 + dt.
        """
        return x0 + self.drift(t0, x0, dt)

    def std_deviation(self, t0, x0, dt):
        """
        Computes the standard deviation of the process at time t0 + dt.

        Args:
            t0 (float): The current time.
            x0 (float or numpy array): The current value(s) of the process.
            dt (float): The time increment.

        Returns:
            float or numpy array: The standard deviation of the process at time t0 + dt.
        """
        return self.diffusion(t0, x0, dt)

    def evolve(self, t0, x0, dt, dw):
        """
        Simulates the evolution of the process over a time increment dt using the Euler scheme.

        Args:
            t0 (float): The current time.
            x0 (float or numpy array): The current value(s) of the process.
            dt (float): The time increment.
            dw (float or numpy array): The random term (typically drawn from a normal distribution).

        Returns:
            float or numpy array: The new value(s) of the process at time t0 + dt.
        """
        return (
            self.expectation(t0=t0, x0=x0, dt=dt)
            + self.std_deviation(t0=t0, x0=x0, dt=dt) * dw
        )
