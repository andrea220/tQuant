import numpy as np
from scipy.interpolate import CubicSpline


class LinearInterp:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def interpolate(self, x_new):
        if x_new <= self.x[0]:
            return self.y[0]
        elif x_new >= self.x[-1]:
            return self.y[-1]
        else:
            i = np.searchsorted(self.x, x_new) - 1
            w1 = (self.x[i + 1] - x_new) / (self.x[i + 1] - self.x[i])
            w2 = 1.0 - w1
            r1 = self.y[i]
            r2 = self.y[i + 1]
            return w1 * r1 + w2 * r2



def compute_coefficients(x: np.ndarray, y: np.ndarray):
    n = len(x)
    h = np.diff(x)
    alpha = [3 * (y[i + 1] - y[i]) / h[i] - 3 * (y[i] - y[i - 1]) / h[i - 1] for i in range(1, n - 1)]

    A = np.zeros((n, n))
    b = np.zeros(n)

    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = alpha[i - 1]

    c = np.linalg.solve(A, b)

    a = y[:-1]
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c[:-1], d


class NaturalCubicSpline:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.a, self.b, self.c, self.d = compute_coefficients(x, y)

    def interpolate(self, x_new: float):
        i = np.searchsorted(self.x, x_new) - 1
        if i < 0:
            i = 0
        if i >= len(self.x) - 1:
            i = len(self.x) - 2

        dx = x_new - self.x[i]
        yi = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2 + self.d[i] * dx ** 3

        return yi



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
    x = [0, 1, 2, 3, 4]
    y = [0, 1, 0, 1, 0]

    cs = CubicSpline(x, y, bc_type='natural')
    test_points = np.array([0.0, 0.5, 1.5, 2.5, 3.5, 4.5])
    test_values = cs(test_points)
    print(test_values.tolist())

    ncs = NaturalCubicSpline(x, y)
    comparison = []
    for x in test_points:
        comparison.append(ncs.interpolate(x))

    print(comparison)


    # natural cubic spline example
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
