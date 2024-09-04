import tensorflow as tf
from ..models.stochasticprocess import StochasticProcess
from ..timehandles.grid import DateGrid
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..markethandles.ircurve import RateCurve
from dateutil.relativedelta import relativedelta

class GaussianPathGenerator:
    def __init__(self,
                 process: StochasticProcess,
                 date_grid: DateGrid) -> None:
        self._process = process
        self._date_grid = date_grid
        self._generator = None #TODO implementazione random numbers
        self._short_rate = None

    def simulate(self,
                 n_paths: int) -> tf.Tensor:
        path = []
        path.append(tf.fill((n_paths,),
                                value= self._process.initial_values()))
        time_grid = self._date_grid.times
        for i in range(1, len(time_grid)):
            w = tf.random.normal(shape=(n_paths,), mean=0, stddev=1, dtype= tf.float64)
            s = time_grid[i-1]
            t = time_grid[i]
            dt = t-s
            x_t = self._process.evolve(s, path[i-1], dt, w)
            path.append(x_t)
        path_tensor = tf.stack(path, axis=1)
        return path_tensor


class GaussianShortRateGenerator(GaussianPathGenerator):
    def __init__(self,
                 process: StochasticProcess, 
                 date_grid: DateGrid) -> None:
        super().__init__(process, date_grid)

    # def simulate_curves(self,
    #                     n_paths: int) -> tf.Tensor:
    #     self._short_rate = self.simulate(n_paths)
    #     curve_tensor = []
    #     self._bonds = []
    #     for i, d in enumerate(self._date_grid.dates):
    #         t = self._date_grid.times[i]
    #         bonds_tmp = tf.stack([self._process.zero_bond(t, time, self._short_rate[:,i]) for time in self._process._term_structure._pillars]) #TODO un po' lenta questa parte
    #         curve_tensor_tmp = RateCurve.from_zcb(d, self._process._term_structure._pillars, bonds_tmp , "LINEAR", DayCounterConvention.ActualActual)
    #         self._bonds.append(bonds_tmp)
    #         curve_tensor.append(curve_tensor_tmp)
    #     return curve_tensor

    def simulate_curves(self,
                        n_paths: int) -> tf.Tensor:
        self._short_rate = self.simulate(n_paths)
        curve_tensor = []
        self._bonds = []
        for i in range(len(self._date_grid.dates)):
            curve_date = self._date_grid.dates[i]

            curve_dates = [curve_date]
            curve_dates += [curve_date + relativedelta(days=i) for i in range(1,5)]
            curve_dates += [curve_date + relativedelta(weeks=i) for i in range(1,3)]
            curve_dates += [curve_date + relativedelta(months=i) for i in range(1,12)]
            curve_dates += [curve_date + relativedelta(years=i) for i in range(1,15)]
            curve_dates += [curve_date + relativedelta(years=i) for i in [20, 25, 30, 40]]
            curve_taus = [DayCounter(DayCounterConvention.ActualActual).year_fraction(curve_date, d) for d in curve_dates]
            bonds_tmp = tf.stack([self._process.zero_bond(self._date_grid.times[i], self._date_grid.times[i]+ tau, self._short_rate[:,i]) for tau in curve_taus]) #TODO LENTA
            self._bonds.append(bonds_tmp)
            curve_tensor_tmp = RateCurve.from_zcb(curve_date, curve_taus, bonds_tmp , "LINEAR", DayCounterConvention.ActualActual)
            curve_tensor.append(curve_tensor_tmp)
        return curve_tensor