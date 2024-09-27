from .stochasticprocess import StochasticProcess
import tensorflow as tf


class GeometricBrownianMotion(StochasticProcess):

    def __init__(self, mu, sigma, x0):
        self._x0 = tf.Variable(x0, dtype=tf.float64)
        self._mu = tf.Variable(mu, dtype=tf.float64)
        self._sigma = tf.Variable(sigma, dtype=tf.float64)

    def drift(self, dt):
        return (self._mu - (self._sigma**2) / 2) * dt

    def diffusion(self, dt):
        return self._sigma * tf.math.sqrt(dt)

    def initial_values(self):
        return self._x0

    def size(self):
        return 1

    def evolve(self, t, dw):
        dt = t / dw.shape[1]
        exp_factor = tf.math.exp(self.drift(dt) + self.diffusion(dt) * dw)
        return self._x0 * tf.math.cumprod(exp_factor, axis=1)


class ArithmeticBrownianMotion(StochasticProcess):

    def __init__(self, mu, sigma, x0):
        self._x0 = tf.Variable(x0, dtype=tf.float64)
        self._mu = tf.Variable(mu, dtype=tf.float64)
        self._sigma = tf.Variable(sigma, dtype=tf.float64)

    def drift(self, dt):

        return self._mu * dt

    def diffusion(self, dt):

        return self._sigma * tf.math.sqrt(dt)

    def initial_values(self):
        return self._x0

    def size(self):
        return 1

    def evolve(self, t, dw):
        dt = t / dw.shape[1]
        diffusion = self.diffusion(dt) * dw
        drift = self.drift(dt)

        s_t = self._x0 + tf.math.cumsum(drift + diffusion, axis=1)
        return s_t
