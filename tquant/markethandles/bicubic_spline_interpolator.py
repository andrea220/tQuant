import numpy as np
from scipy.interpolate import CubicSpline

from tquant import NaturalCubicSpline


class BiCubicSplineInterpolator:
    def __init__(self, x: np.ndarray[float], y: np.ndarray[float], values: np.ndarray):
        self.x = x
        self.y = y
        self.values = values
        self.cubic_splines = []
        for j in range(len(y)):
            self.cubic_splines.append(NaturalCubicSpline(self.x, values[:, j]))

    def interpolate(self, x: float, y: float):
        interpolated = np.zeros(len(self.y))
        for j in range(len(self.y)):
            interpolated[j] = self.cubic_splines[j].interpolate(x)
        cs = NaturalCubicSpline(self.y, interpolated)
        return cs.interpolate(y)


if __name__ == "__main__":
    x = np.array([1, 2])
    y = np.array([100, 200, 300])
    v = np.array([[0.2, 0.3, 0.4],
                  [0.5, 0.6, 0.7]])

    interpolator = BiCubicSplineInterpolator(x, y, v)

    for xi in x:
        for yi in y:
            print("Point=(" + str(xi) + "," + str(yi) + ")")
            print(interpolator.interpolate(xi, yi))

    cs = CubicSpline(y, v[0])
    for i in range(100, 300, 10):
        print("Python spline= " + str(cs(i)) + " | My spline= " + str(interpolator.interpolate(1, i)))
