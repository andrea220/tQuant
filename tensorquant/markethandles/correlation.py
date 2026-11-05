from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf

from ..numericalhandles.dtypes import DTYPE, as_variable


class CorrelationMatrix:
    """Correlation matrix between named risk factors.

    Factor names follow the market-map key prefix, e.g. ``EQ:EUR:TICKER1`` or
    ``FX:USD:EURUSD``, so equity-equity and equity-FX (quanto) correlations live
    in the same object.

    The full ``[n, n]`` matrix is a single float64 ``tf.Variable`` (a risk factor):
    the gradient with respect to the pair (a, b) lands on both (i, j) and (j, i),
    so the sensitivity to rho_ab is their sum. Bumping one entry can make the
    matrix non positive semi-definite; it is validated only at construction.

    Attributes:
        names (list[str]): Factor names, in matrix order.
        matrix (tf.Variable): float64 ``[n, n]`` correlation matrix.
    """

    def __init__(self, names: Sequence[str], matrix, tol: float = 1e-10):
        """Build and validate the matrix.

        Args:
            names (Sequence[str]): Factor names, one per row/column, unique.
            matrix: ``[n, n]`` array-like (or float64 ``tf.Variable``, reused).
            tol (float): Tolerance for symmetry, unit diagonal and PSD checks.

        Raises:
            ValueError: If names are not unique, the shape does not match, or the
                matrix is not symmetric, unit-diagonal, in [-1, 1] and PSD.
        """
        self.names = list(names)
        if len(set(self.names)) != len(self.names):
            raise ValueError("Correlation factor names must be unique")
        self._index = {name: i for i, name in enumerate(self.names)}

        m = np.asarray(matrix.numpy() if isinstance(matrix, tf.Variable) else matrix, dtype=float)
        n = len(self.names)
        if m.shape != (n, n):
            raise ValueError(f"Correlation matrix shape {m.shape} does not match {n} names")
        if not np.allclose(m, m.T, atol=tol):
            raise ValueError("Correlation matrix is not symmetric")
        if not np.allclose(np.diag(m), 1.0, atol=tol):
            raise ValueError("Correlation matrix diagonal must be 1")
        if np.abs(m).max() > 1.0 + tol:
            raise ValueError("Correlations must be in [-1, 1]")
        min_eig = np.linalg.eigvalsh(m).min()
        if min_eig < -tol:
            raise ValueError(f"Correlation matrix is not positive semi-definite (min eigenvalue {min_eig:.3e})")

        self.matrix = as_variable(matrix, name="CORR")

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str, float]]) -> "CorrelationMatrix":
        """Build from ``(factor_1, factor_2, corr)`` triples (one per pair, any order).

        Missing pairs are zero; names are ordered by first appearance.

        Args:
            pairs (Iterable[tuple[str, str, float]]): Pairwise correlations.

        Returns:
            CorrelationMatrix: The validated matrix.

        Raises:
            ValueError: If a pair is repeated with different values, a pair is a
                factor with itself, or the resulting matrix is invalid.
        """
        pairs = list(pairs)
        names: list[str] = []
        for a, b, _ in pairs:
            for name in (a, b):
                if name not in names:
                    names.append(name)
        index = {name: i for i, name in enumerate(names)}
        m = np.eye(len(names))
        seen: dict[tuple[int, int], float] = {}
        for a, b, rho in pairs:
            i, j = sorted((index[a], index[b]))
            if i == j:
                raise ValueError(f"Correlation of '{a}' with itself is not an input")
            if (i, j) in seen and seen[(i, j)] != float(rho):
                raise ValueError(f"Conflicting correlations for ('{a}', '{b}')")
            seen[(i, j)] = float(rho)
            m[i, j] = m[j, i] = float(rho)
        return cls(names, m)

    @property
    def variables(self) -> list[tf.Variable]:
        """Risk-factor variables: ``[matrix]``."""
        return [self.matrix]

    def _indices(self, names: Sequence[str]) -> list[int]:
        missing = [name for name in names if name not in self._index]
        if missing:
            raise KeyError(f"Unknown correlation factors: {missing}")
        return [self._index[name] for name in names]

    def correlation(self, a: str, b: str) -> tf.Tensor:
        """Correlation between two factors (float64 scalar tensor)."""
        i, j = self._indices([a, b])
        return self.matrix[i, j]

    def submatrix(self, names: Sequence[str]) -> tf.Tensor:
        """Correlation matrix of a subset of factors, in the given order (differentiable).

        Args:
            names (Sequence[str]): Factor names.

        Returns:
            tf.Tensor: float64 ``[len(names), len(names)]``.

        Raises:
            KeyError: If a name is unknown.
        """
        # selection matrix E [k, n]: E M E^T has a dense gradient (tf.gather on the variable gives IndexedSlices)
        E = tf.one_hot(self._indices(names), len(self.names), dtype=DTYPE)
        return tf.matmul(tf.matmul(E, self.matrix), E, transpose_b=True)

    def cholesky(self, names: Sequence[str]) -> tf.Tensor:
        """Lower Cholesky factor ``L`` of ``submatrix(names)`` (``L @ L.T`` = submatrix)."""
        return tf.linalg.cholesky(self.submatrix(names))

    def __repr__(self) -> str:
        return f"CorrelationMatrix({len(self.names)} factors, dtype={DTYPE.name})"
