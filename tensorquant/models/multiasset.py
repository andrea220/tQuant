"""Several equity underlyings simulated jointly: one single-asset process each, correlated Brownian motions."""

import tensorflow as tf

from .stochasticprocess import StochasticProcess
from ..numericalhandles.dtypes import DTYPE


class MultiAssetProcess(StochasticProcess):
    """``n`` single-asset processes (local vol, GBM: one normal per step each) driven by correlated Brownian motions.

    The independent normals ``dw [n_paths, n_steps, n]`` are correlated by the Cholesky factor ``L`` of the
    correlation matrix (``z = dw L^T``: ``corr(z_i, z_j) = rho_ij``) and the column ``i`` drives the process ``i``.
    The matrix is a tensor (e.g. ``market_env.get_correlation(names)``): the paths are differentiable w.r.t. it.
    """

    def __init__(self, processes: list, correlation):
        """
        Args:
            processes (list[StochasticProcess]): One process per underlying, ``noise_size() == 1``.
            correlation: ``[n, n]`` correlation matrix (positive definite), in the order of *processes*.

        Raises:
            ValueError: If a process draws more than one normal per step, or the matrix does not match.
        """
        if any(p.noise_size() != 1 for p in processes):
            raise ValueError("multi-asset: processes with one normal per step only (local vol, GBM)")
        self._processes = list(processes)
        self._correlation = tf.cast(correlation, DTYPE)
        if tuple(self._correlation.shape) != (len(self._processes), len(self._processes)):
            raise ValueError(f"correlation {tuple(self._correlation.shape)} for {len(self._processes)} processes")
        self._cholesky = tf.linalg.cholesky(self._correlation)

    @property
    def processes(self) -> list:
        return self._processes

    def size(self) -> int:
        """Number of underlyings."""
        return len(self._processes)

    def noise_size(self) -> int:
        """One normal per underlying and step."""
        return len(self._processes)

    def initial_values(self) -> tf.Tensor:
        """Spots ``[n]``."""
        return tf.stack([tf.reshape(tf.cast(p.initial_values(), DTYPE), []) for p in self._processes])

    def evolve(self, t_grid, dw, observation_times=None) -> tf.Tensor:
        """Paths of every underlying.

        Args:
            t_grid: ``[n_steps]`` increasing times, without 0.
            dw: ``[n_paths, n_steps, n]`` independent standard normals.
            observation_times: Optional subset of ``t_grid`` (the columns returned).

        Returns:
            tf.Tensor: ``[n_paths, n_steps (or n_observations), n]``.
        """
        dw = tf.cast(tf.convert_to_tensor(dw), DTYPE)
        z = tf.einsum("psj,ij->psi", dw, self._cholesky)                 # correlated normals
        return tf.stack([p.evolve(t_grid, z[:, :, i], observation_times) for i, p in enumerate(self._processes)],
                        axis=-1)
