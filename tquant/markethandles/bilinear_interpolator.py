import numpy as np


class BilinearInterpolator:
    def __init__(self, x: list[float], y: list[float], values: np.ndarray):
        self.x = x
        self.y = y
        self.values = values

    def interpolate(self, x: float, y: float):
        if x <= self.x[0]:
            return self._get_value(0, y)
        if x >= self.x[-1]:
            n = len(self.x) - 1
            return self._get_value(n, y)
        i1 = np.searchsorted(self.x, x) - 1
        i2 = i1 + 1
        delta = self.x[i2] - self.x[i1]
        w1 = (self.x[i2] - x) / delta
        w2 = 1.0 - w1
        return w1 * self._get_value(i1, y) + w2 * self._get_value(i2, y)

    def _get_value(self, n: int, y: float):
        if y <= self.y[0]:
            return self.values[n][0]
        if y >= self.y[-1]:
            m = len(self.y) - 1
            return self.values[n][m]
        i1 = np.searchsorted(self.y, y) - 1
        i2 = i1 + 1
        delta = self.y[i2] - self.y[i1]
        w1 = (self.y[i2] - y) / delta
        w2 = 1.0 - w1
        return w1 * self.values[n][i1] + w2 * self.values[n][i2]

