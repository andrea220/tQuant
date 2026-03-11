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

    def evolve(self, t_grid, dw):
        """
        Args:
            t_grid: 1D tensor of shape [n_steps] with observation times, e.g. [0.25, 0.5, 0.75, 1.0]
            dw:     tensor of shape [n_paths, n_steps], standard normals
        Returns:
            tensor of shape [n_paths, n_steps] with simulated process values
        """
        t_full = tf.concat([tf.zeros([1], dtype=tf.float64), t_grid], axis=0)  # prepend 0
        dt = tf.reshape(t_full[1:] - t_full[:-1], [1, -1])  # [1, n_steps], one dt per step
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

    def evolve(self, t_grid, dw):
        """
        Args:
            t_grid: 1D tensor of shape [n_steps] with observation times, e.g. [0.25, 0.5, 0.75, 1.0]
            dw:     tensor of shape [n_paths, n_steps], standard normals
        Returns:
            tensor of shape [n_paths, n_steps] with simulated process values
        """
        t_full = tf.concat([tf.zeros([1], dtype=tf.float64), t_grid], axis=0)  # prepend 0
        dt = tf.reshape(t_full[1:] - t_full[:-1], [1, -1])  # [1, n_steps], one dt per step
        return self._x0 + tf.math.cumsum(self.drift(dt) + self.diffusion(dt) * dw, axis=1)
