"""Single source of truth for the dtype of market data and the risk-factor variable helper."""

import tensorflow as tf

__all__ = ["DTYPE", "as_variable"]

DTYPE = tf.float64
"""dtype of market data (risk factors) and of the tensors market objects return."""


def as_variable(value, name: str = None) -> tf.Variable:
    """Return *value* as a :data:`DTYPE` ``tf.Variable``.

    A ``tf.Variable`` that already has dtype :data:`DTYPE` is returned as is, so
    its identity (and every reference to it) is preserved. Tensors and other
    variables are cast; Python/numpy data are converted directly to :data:`DTYPE`
    (no float32 round trip).

    Args:
        value: Scalar, nested list, numpy array, ``tf.Tensor`` or ``tf.Variable``.
        name (str, optional): Name of the new variable.

    Returns:
        tf.Variable: A :data:`DTYPE` variable holding *value*.
    """
    if isinstance(value, tf.Variable) and value.dtype == DTYPE:
        return value
    if isinstance(value, (tf.Tensor, tf.Variable)):
        return tf.Variable(tf.cast(value, DTYPE), name=name)
    return tf.Variable(tf.convert_to_tensor(value, dtype=DTYPE), name=name)
