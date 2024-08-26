import tensorflow as tf
from ..models.stochasticprocess import StochasticProcess

class GaussianPathGenerator:
    def __init__(self,
                 process: StochasticProcess,
                 time_grid: list[float]) -> None:
        self._process = process
        self._time_grid = time_grid
        self._generator = None #TODO implementazione random numbers

    def simulate(self,
                 n_paths: int) -> tf.Tensor:
        path = []
        path.append(tf.fill((n_paths,),
                                value= self._process.initial_values()))
        for i in range(1, len(self._time_grid)):
            w = tf.random.normal(shape=(n_paths,), mean=0, stddev=1, dtype= tf.float64)
            s = self._time_grid[i-1]
            t = self._time_grid[i]
            dt = t-s
            x_t = self._process.evolve(s, path[i-1], dt, w)
            path.append(x_t)
        path_tensor = tf.stack(path, axis=1)
        return path_tensor
