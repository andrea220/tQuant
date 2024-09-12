import numpy
import tensorflow

from ..models.hullwhite import HullWhiteProcess
from ..timehandles.grid import DateGrid
from ..instruments.swap import Swap
from ..pricers.swapdiscounting import SwapPricer
from .gaussiankernel import HullWhiteShortRateGenerator
from ..timehandles.utils import Settings
from ..markethandles.utils import curve_map


class SwapExposureGenerator:
    """
    A class for simulating exposure and expected exposure of a swap using the Hull-White short rate model.

    Attributes:
        _model (HullWhiteProcess): The Hull-White process model for simulating interest rates.
        _date_grid (DateGrid): A grid of dates over which the exposure is evaluated.
        _kernel (HullWhiteShortRateGenerator): A kernel for simulating short rates using the Hull-White model.
        _exposure (tf.Tensor): Simulated exposure of the swap (initialized to None).
        _expected_exposure (tf.Tensor): Expected exposure of the swap (initialized to None).
    """

    def __init__(self, model: HullWhiteProcess, date_grid: DateGrid) -> None:
        """
        Initializes the SwapExposureGenerator with a Hull-White model and a date grid.

        Args:
            model (HullWhiteProcess): The Hull-White model to be used for interest rate simulation.
            date_grid (DateGrid): The grid of dates over which the exposure will be simulated.
        """
        self._model = model
        self._date_grid = date_grid
        self._kernel = HullWhiteShortRateGenerator(model, date_grid)
        self._exposure = None
        self._expected_exposure = None

    @property
    def date_grid(self) -> DateGrid:
        """
        Returns the date grid used for exposure simulation.

        Returns:
            DateGrid: The grid of dates for the simulation.
        """
        return self._date_grid

    @property
    def model(self) -> HullWhiteProcess:
        """
        Returns the Hull-White model used for rate simulation.

        Returns:
            HullWhiteProcess: The Hull-White interest rate model.
        """
        return self._model

    @property
    def kernel(self) -> HullWhiteShortRateGenerator:
        """
        Returns the short rate kernel for simulating interest rates.

        Returns:
            HullWhiteShortRateGenerator: The kernel for generating short rates.
        """
        return self._kernel

    @property
    def expected_exposure(self) -> tensorflow.Tensor:
        """
        Returns the expected exposure of the swap.

        Raises:
            ValueError: If the simulate method has not been called before accessing the expected exposure.

        Returns:
            tensorflow.Tensor: The expected exposure of the swap.
        """
        if self._expected_exposure is None:
            raise ValueError("Must call simulate()")
        return self._expected_exposure

    @property
    def exposure(self) -> tensorflow.Tensor:
        """
        Returns the exposure of the swap.

        Raises:
            ValueError: If the simulate method has not been called before accessing the exposure.

        Returns:
            tf.Tensor: The simulated exposure of the swap.
        """
        if self._exposure is None:
            raise ValueError("Must call simulate()")
        return self._exposure

    def simulate(self, n_path: float, product: Swap, last_fixing: float) -> None:
        """
        Simulates swap exposure and expected exposure for the given number of paths and swap product.

        Args:
            n_path (float): Number of paths to simulate.
            product (Swap): The swap instrument for which exposure is being calculated.
            last_fixing (float): The last fixing rate for the floating leg of the swap.

        Raises:
            ValueError: If invalid inputs are provided or simulation encounters an issue.

        Simulates the exposure and stores the results in `_exposure` and `_expected_exposure`.
        """

        simulated_curves = self._kernel.simulate_curves(n_path)

        transaction_fixing_dates = product.floating_leg.display_flows()[
            "fixing_date"
        ].values
        transaction_fixing_rates = numpy.zeros(shape=transaction_fixing_dates.shape[0])
        simulated_fixing_dates = [
            product.index.fixing_date(d) for d in self._date_grid.dates
        ]
        simulated_fixing_rates = tensorflow.reduce_mean(
            self._kernel.state_variable, axis=0
        ).numpy()
        simulated_fixing_rates[0] = last_fixing
        # create fixings lookup table
        fixings_lookup_table = {}
        for i in range(len(simulated_fixing_dates)):
            fixings_lookup_table[simulated_fixing_dates[i]] = simulated_fixing_rates[i]
        # # add transaction fixing rates for a given date from fixings lookup table
        for i in range(len(transaction_fixing_dates)):
            if transaction_fixing_dates[i] in fixings_lookup_table:
                transaction_fixing_rates[i] = fixings_lookup_table[
                    transaction_fixing_dates[i]
                ]
            else:
                # find the nearest fixing from lookup table
                transaction_fixing_rates[i] = fixings_lookup_table.get(
                    transaction_fixing_dates[i],
                    fixings_lookup_table[
                        min(
                            fixings_lookup_table.keys(),
                            key=lambda d: abs(d - transaction_fixing_dates[i]),
                        )
                    ],
                )
        for i, d in enumerate(transaction_fixing_dates):
            product.index.add_fixing(d, transaction_fixing_rates[i])

        exposure = []
        swap_engine = SwapPricer(curve_map)
        for i in range(len(self._date_grid.dates)):
            market_simulated = {}
            market_simulated["EUR:ESTR"] = simulated_curves[i]  # TODO mettere dinamiche
            market_simulated["EUR:6M"] = simulated_curves[i]
            val_date = self._date_grid.dates[i]
            Settings.evaluation_date = val_date
            if self._date_grid.dates[i] > product.end_date:
                pv = tensorflow.zeros(shape=(n_path,), dtype=tensorflow.float64)
            else:
                pv = swap_engine.price(product, val_date, market_simulated)
            exposure.append(pv)
        self._exposure = tensorflow.stack(exposure, axis=0)
        self._expected_exposure = tensorflow.reduce_mean(exposure, axis=1)
