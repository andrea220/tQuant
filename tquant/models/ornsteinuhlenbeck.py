from .stochasticprocess import StochasticProcess
from tensorflow.python.framework import dtypes
import tensorflow as tf

class OrnsteinUhlenbeckProcess(StochasticProcess):
    def __init__(self, mr_speed, volatility, x0=0.0, level=0.0):
        self._x0 =  tf.Variable(x0, dtype=dtypes.float64)
        self._mr_speed =  tf.Variable(mr_speed, dtype=dtypes.float64)
        self._volatility =  tf.Variable(volatility, dtype=dtypes.float64)
        self._level =  level

    def size(self):
        return 1
    
    def initial_values(self):
        return self.x0

    def drift(self, x):
        return self._mr_speed * (self._level - x)

    def diffusion(self):
        return self._volatility

    def expectation(self, x0, dt, t0=None):
        """ overload di euler """
        return self._level + (x0 - self._level) * tf.exp(-self._mr_speed * dt)

    def std_deviation(self, dt, t0=None, x0=None):
        """ parametri none per rispettare interfaccia"""
        return tf.sqrt(self.variance(dt))

    def variance(self, dt):
        return 0.5*self._volatility**2 / self._mr_speed * (1 - tf.exp(-2 * self._mr_speed * dt))
    
    @property
    def x0(self):
        return self._x0.numpy()

    @property
    def mr_speed(self):
        return self._mr_speed.numpy()

    @property
    def volatility(self):
        return self._volatility.numpy()

    @property
    def level(self):
        return self._level