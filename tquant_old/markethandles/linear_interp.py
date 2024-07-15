import numpy as np


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
