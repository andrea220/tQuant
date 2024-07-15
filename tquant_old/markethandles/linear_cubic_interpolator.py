import numpy as np
from scipy.interpolate import CubicSpline

from tquant import NaturalCubicSpline


class LinearCubicInterpolator:
    def __init__(self, x: np.ndarray[float], y: np.ndarray[float], values: np.ndarray):
        self.x = x
        self.y = y
        self.values = values
        self.cubic_splines = []
        for i in range(len(x)):
            self.cubic_splines.append(NaturalCubicSpline(self.y, values[i]))

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
        return self.cubic_splines[n].interpolate(y)


if __name__ == "__main__":
    x = np.array([1, 2])
    y = np.array([100, 200, 300])
    v = np.array([[0.2, 0.3, 0.4],
                  [0.5, 0.6, 0.7]])

    interpolator = LinearCubicInterpolator(x, y, v)

    for xi in x:
        for yi in y:
            print("Point=(" + str(xi) + "," + str(yi) + ")")
            print(interpolator.interpolate(xi, yi))

    cs = CubicSpline(y, v[0])
    for i in range(100, 300, 10):
        print("Python spline= " + str(cs(i)) + " | My spline= " + str(interpolator.interpolate(1, i)))
