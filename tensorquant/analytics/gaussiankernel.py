from tensorflow import Tensor, float64, random, stack, fill

from dateutil.relativedelta import relativedelta
from ..models.stochasticprocess import StochasticProcess
from ..models.hullwhite import HullWhiteProcess
from ..timehandles.grid import DateGrid
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..markethandles.ircurve import RateCurve


class GaussianPathGenerator:
    """
    A generator for simulating paths of a Gaussian process.

    Attributes:
        _process (StochasticProcess): The stochastic process being simulated.
        _date_grid (DateGrid): Grid of time points for the simulation.
        _generator: Pseudo-random number generator (currently not implemented).
        _state_variable: The generated paths from the simulation (initialized to None).
    """

    def __init__(self, process: StochasticProcess, date_grid: DateGrid) -> None:
        """
        Initializes the GaussianPathGenerator with a stochastic process and a date grid.

        Args:
            process (StochasticProcess): The stochastic process to be simulated.
            date_grid (DateGrid): The grid of dates over which the simulation will be performed.
        """
        self._process = process
        self._date_grid = date_grid
        self._generator = None  # TODO implementation of pseudo-random number generator
        self._state_variable = None

    def simulate(self, n_paths: int) -> Tensor:
        """
        Simulates paths for the given number of paths using the underlying stochastic process.

        Args:
            n_paths (int): Number of paths to simulate.

        Returns:
            Tensor: Tensor containing the simulated paths.
        """
        path = []
        path.append(fill((n_paths,), value=self._process.initial_values()))
        time_grid = self._date_grid.times
        for i in range(1, len(time_grid)):
            w = random.normal(shape=(n_paths,), mean=0, stddev=1, dtype=float64)
            s = time_grid[i - 1]
            t = time_grid[i]
            dt = t - s
            x_t = self._process.evolve(s, path[i - 1], dt, w)
            path.append(x_t)
        self._state_variable = stack(path, axis=1)

    @property
    def process(self) -> StochasticProcess:
        """
        Returns the stochastic process used in the path generation.

        Returns:
            StochasticProcess: The process being simulated.
        """
        return self._process

    @property
    def date_grid(self) -> DateGrid:
        """
        Returns the date grid used for the simulation.

        Returns:
            DateGrid: The grid of dates for simulation.
        """
        return self._date_grid

    @property
    def generator(self):
        """
        Placeholder for a pseudo-random number generator.

        Returns:
            None: The generator is not implemented yet.
        """
        return None

    @property
    def state_variable(self) -> Tensor:
        """
        Returns the simulated paths after running the `simulate` method.

        Raises:
            ValueError: If `simulate` method has not been called before accessing this property.

        Returns:
            tensorflow.Tensor: The simulated state variables.
        """
        if self._state_variable is None:
            raise ValueError("Must run simulate()")
        return self._state_variable


class HullWhiteShortRateGenerator(GaussianPathGenerator):
    """
    A generator for simulating short-rate curves using the Hull-White process.

    Inherits from GaussianPathGenerator.

    Attributes:
        _process (HullWhiteProcess): The Hull-White process used for short-rate simulation.
    """

    def __init__(self, process: HullWhiteProcess, date_grid: DateGrid) -> None:
        """
        Initializes the GaussianShortRateGenerator with a Hull-White process and a date grid.

        Args:
            process (HullWhiteProcess): The Hull-White process to be used for short-rate simulation.
            date_grid (DateGrid): The grid of dates over which the short-rate curves will be simulated.
        """
        super().__init__(process, date_grid)

    def simulate_curves(self, n_paths: int) -> Tensor:
        """
        Simulates yield curves for the given number of paths using the Hull-White process.

        Args:
            n_paths (int): Number of paths to simulate.

        Returns:
            tensorflow.Tensor: Tensor containing the simulated yield curves.
        """
        self.simulate(n_paths)
        curve_tensor = []
        for i in range(len(self._date_grid.dates)):
            curve_date = self._date_grid.dates[i]

            curve_dates = [curve_date]
            curve_dates += [curve_date + relativedelta(days=i) for i in range(1, 5)]
            curve_dates += [curve_date + relativedelta(weeks=i) for i in range(1, 3)]
            curve_dates += [curve_date + relativedelta(months=i) for i in range(1, 12)]
            curve_dates += [curve_date + relativedelta(years=i) for i in range(1, 15)]
            curve_dates += [
                curve_date + relativedelta(years=i) for i in [20, 25, 30, 40]
            ]
            curve_taus = [
                DayCounter(DayCounterConvention.ActualActual).year_fraction(
                    curve_date, d
                )
                for d in curve_dates
            ]
            bonds_tmp = stack(
                [
                    self._process.zero_bond(
                        self._date_grid.times[i],
                        self._date_grid.times[i] + tau,
                        self.state_variable[:, i],
                    )
                    for tau in curve_taus
                ]
            )  # TODO too slow, should be optimized
            curve_tensor_tmp = RateCurve.from_zcb(
                curve_date,
                curve_taus,
                bonds_tmp,
                "LINEAR",
                DayCounterConvention.ActualActual,
            )
            curve_tensor.append(curve_tensor_tmp)
        return curve_tensor
