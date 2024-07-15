import numpy as np

from scipy.interpolate import CubicSpline


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

