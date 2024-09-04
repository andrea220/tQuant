from ..models.hullwhite import HullWhiteProcess
from ..timehandles.grid import DateGrid
from ..instruments.swap import Swap
from ..pricers.swapdiscounting import SwapPricer
from .gaussiankernel import GaussianShortRateGenerator
from ..timehandles.utils import Settings
from ..markethandles.utils import curve_map
import numpy as np
import tensorflow as tf

class SwapExposureGenerator:
    def __init__(self,
                 model: HullWhiteProcess,
                 date_grid: DateGrid) -> None:
        self._model = model
        self._date_grid = date_grid
        self._kernel = GaussianShortRateGenerator(model, date_grid)
        self._exposure = None
        self._ee = None


    @property
    def expected_exposure(self):
        if self._ee is None:
            raise ValueError("simulate")
        else:
            return self._ee 
        
    @property
    def exposure(self):
        if self._exposure is None:
            raise ValueError("simulate")
        else:
            return self._exposure 
        
    def simulate(self,
                 n_path: float,
                 product: Swap,
                 last_fixing: float):
        
        simulated_curves = self._kernel.simulate_curves(n_path)

        transaction_fixing_dates = product.floating_leg.display_flows()['fixing_date'].values
        transaction_fixing_rates = np.zeros(shape = (transaction_fixing_dates.shape[0]))
        simulated_fixing_dates = [product._index.fixing_date(d) for d in self._date_grid.dates]
        simulated_fixing_rates = tf.reduce_mean(self._kernel._short_rate, axis=0).numpy()
        simulated_fixing_rates[0] = last_fixing
        # create fixings lookup table
        fixings_lookup_table = {}
        for i in range(len(simulated_fixing_dates)):
            fixings_lookup_table[simulated_fixing_dates[i]] = simulated_fixing_rates[i]
        # # add transaction fixing rates for a given date from fixings lookup table
        for i in range(len(transaction_fixing_dates)):
            if transaction_fixing_dates[i] in fixings_lookup_table:
                transaction_fixing_rates[i] = fixings_lookup_table[transaction_fixing_dates[i]]
            else:
                # find the nearest fixing from lookup table
                transaction_fixing_rates[i] = \
                fixings_lookup_table.get(transaction_fixing_dates[i], \
                fixings_lookup_table[min(fixings_lookup_table.keys(), \
                key = lambda d: abs(d - transaction_fixing_dates[i]))])
        for i, d in enumerate(transaction_fixing_dates):
            product._index.add_fixing(d, transaction_fixing_rates[i])

        exposure = []
        swap_engine = SwapPricer(curve_map)
        for i in range(len(self._date_grid.dates)):
            market_simulated = {}
            market_simulated['EUR:ESTR'] = simulated_curves[i] #TODO mettere dinamiche
            market_simulated['EUR:6M'] = simulated_curves[i]
            val_date = self._date_grid.dates[i]
            Settings.evaluation_date = val_date
            if self._date_grid.dates[i] > product.maturity:
                pv = tf.zeros(shape=(n_path,),  dtype=tf.float64)
            else:
                pv = swap_engine.price(product, val_date, market_simulated)
            exposure.append(pv)
        self._exposure = tf.stack(exposure, axis=0)
        self._ee = tf.reduce_mean(exposure, axis=1)
