#TODO Check velocità interpolazioni ripetute
#import time
from enum import Enum
from typing import Optional, List, Union
import tensorflow as tf
import numpy as np

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