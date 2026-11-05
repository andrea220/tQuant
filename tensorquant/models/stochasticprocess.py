from abc import ABC, abstractmethod


class StochasticProcess(ABC):
    """Common interface of the simulation models.

    Every model simulates on a time grid from pre-drawn standard normals::

        paths = model.evolve(t_grid, dw)                          # [n_paths, n_steps(, ...)]
        paths = model.evolve(t_grid, dw, observation_times=obs)   # only the observed columns

    ``t_grid`` holds increasing year fractions (> 0, without 0); ``dw`` the standard normals
    (``[n_paths, n_steps]``, or ``[n_paths, n_steps, k]`` for models with ``k`` shocks per step).
    The draws are made by the caller (pricers use ``tf.random.stateless_normal``), so the same
    seed gives the same paths and bumps use common random numbers.
    """

    @abstractmethod
    def size(self) -> int:
        """Number of state dimensions (factors) of the process."""

    @abstractmethod
    def initial_values(self):
        """Initial value(s) of the state."""

    @abstractmethod
    def evolve(self, t_grid, dw, observation_times=None):
        """Simulate the state on *t_grid* from the standard normals *dw*.

        Args:
            t_grid: ``[n_steps]`` increasing times (year fractions, > 0).
            dw: Standard normals, ``[n_paths, n_steps]`` (or with a trailing shock axis).
            observation_times: Optional subset of ``t_grid``: only those columns are returned.

        Returns:
            tf.Tensor: Paths ``[n_paths, n_steps(, n_state)]`` (or the observed columns).
        """

    @property
    def factors(self) -> int:
        """Alias of :meth:`size`."""
        return self.size()
