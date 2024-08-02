import numpy as np
from scipy.interpolate import CubicSpline
import tensorflow as tf
from typing import Optional, List, Union
from enum import Enum



class LinearInterp:
    ''' 
    interpolazione lineare per tensorflow
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def interpolate(self, term):
        for i in range(0, len(self.x) - 1):
            if term < self.x[i + 1]:
                dtr = 1 / (self.x[i + 1] - self.x[i])
                w1 = (self.x[i + 1] - term) * dtr
                w2 = (term - self.x[i]) * dtr
                r1 = w1 * self.y[i]
                r2 = w2 * self.y[i + 1]
                return r1 + r2




###############################################
#TODO da rivedere interpolazioni
# class LinearInterp:
#     def __init__(self, x: np.ndarray, y: np.ndarray):
#         self.x = x
#         self.y = y

#     def interpolate(self, x_new):
#         if x_new <= self.x[0]:
#             return self.y[0]
#         elif x_new >= self.x[-1]:
#             return self.y[-1]
#         else:
#             i = np.searchsorted(self.x, x_new) - 1
#             w1 = (self.x[i + 1] - x_new) / (self.x[i + 1] - self.x[i])
#             w2 = 1.0 - w1
#             r1 = self.y[i]
#             r2 = self.y[i + 1]
#             return w1 * r1 + w2 * r2



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
    

class InterpolationType(Enum):
    Linear = 'linear'
    Quadratic = "quadratic"
    Cubic = "cubic"

    def __str__(self):
        return self.value
    
def polint(xa: List[float], 
           ya: Union[tf.Tensor,tf.Variable], 
           x: float):
    n = len(xa)
    ns = 0
    dif = np.abs(x - xa[0])
    for i in range(n):
        dift = np.abs(x - xa[i])
        if dift < dif:
            ns = i
            dif = dift
    c = tf.Variable(ya)
    d = tf.Variable(ya)
    y = ya[ns]
    ns -= 1
    for m in range(1,n):
        for i in range(1,n - m + 1):
            ho = xa[i - 1] - x
            hp = xa[i + m -1] - x
            den = ho - hp
            if den == 0:
                den = 1e-8
            den = (c[i] - d[i-1])/ den
            c[i-1].assign(ho*den)
            d[i-1].assign(hp*den)
        if 2 * ns < n - m -1:
            y += c[ns+1]
        else:
            y += d[ns]
            ns -= 1
    return y

def interpolate(ax: List[float], 
                ay: Union[tf.Tensor,tf.Variable],
                x: float, 
                no: Optional[InterpolationType] = InterpolationType.Linear,
                logaritm: Optional[bool] = False):
    # N è l'ordine di interpolazione
    if no == InterpolationType.Quadratic and len(ax) >= 2:
        n = 2
    elif no == InterpolationType.Cubic and len(ax) >= 3:
        n = 3
    else:
        n = 1

    if x > ax[-1]:
        j = len(ax)-1
    else:
        j = 0
        while x > ax[j]:
            j = j + 1
    j = j - 1 #Trovo j così che ax[j] < x < ax[j+1]
    k = min(max(j - n // 2, 0), len(ax) - n)
    xx = ax[k:k+n+1]
    yy = ay[k:k+n+1]
    if logaritm:
        yy = tf.math.log(yy)
        #xx = np.array(xx)
        #xx = xx + 1
        #xx = np.log(xx)
        #x = np.log(x+1)
        result = tf.exp(tf.cast(polint(xx,yy,x), dtype=tf.float64))
        #xx = np.exp(xx) - 1
        #x = np.exp(x) - 1
        return result
    else:
        return tf.cast(polint(xx,yy,x), dtype=tf.float64)
    
