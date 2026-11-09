"""Derivatives of a calibrated optimum w.r.t. its inputs by the implicit function theorem.

A calibration ``p* = argmin_p L(p, z)`` stops where ``g(p*, z) = grad_p L = 0``; when the inputs ``z`` move the
optimum moves with ``dp/dz = -H^{-1} dg/dz`` (``H = dg/dp``). The derivatives are computed **once**, as numbers, at
the optimum (eager, one backward pass per parameter), and :func:`implicit_solution` returns the parameters as a
tensor whose gradient w.r.t. ``z`` is that matrix (custom gradient): the caller's tape never differentiates through
the second-order graph of ``g``. First order only (second derivatives through the calibration are not provided).
"""

from typing import Callable, Sequence

import numpy as np
import tensorflow as tf

from .dtypes import DTYPE

__all__ = ["optimality_derivatives", "newton_step", "implicit_sensitivities", "implicit_solution"]


def optimality_derivatives(gradient: Callable, x: np.ndarray, inputs: Sequence = ()) -> tuple:
    """Gradient ``g``, Hessian ``H = dg/dp`` and ``dg/dz`` of the optimality condition at ``x`` (numbers).

    Args:
        gradient: ``gradient(x, *inputs) -> tf.Tensor`` of the shape of ``x``: ``grad_p L``.
        x (np.ndarray): Parameters (any shape, ``P`` entries).
        inputs: Tensors ``z`` the objective depends on (held at their values: no tape of the caller is touched).

    Returns:
        tuple: ``(g [P], H [P, P], [dg/dz [P, *z.shape] per input])`` as NumPy arrays.
    """
    xs = tf.constant(np.asarray(x, dtype=float), DTYPE)
    zs = [tf.constant(tf.convert_to_tensor(z, DTYPE).numpy()) for z in inputs]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xs)
        tape.watch(zs)
        g = tf.reshape(gradient(xs, *zs), [-1])
    jac = tape.jacobian(g, [xs, *zs], experimental_use_pfor=False,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    del tape
    p = int(g.shape[0])
    return g.numpy(), jac[0].numpy().reshape(p, p), [j.numpy() for j in jac[1:]]


def _masked_hessian(hess: np.ndarray, free: np.ndarray) -> tuple:
    f = np.asarray(free, dtype=float).ravel()
    return hess * f[:, None] * f[None, :] + np.diag(1.0 - f), f


def newton_step(g: np.ndarray, hess: np.ndarray, free: np.ndarray) -> np.ndarray:
    """``H^{-1} g`` on the free parameters (those on a bound get a zero step), flat ``[P]``."""
    h, f = _masked_hessian(hess, free)
    return np.linalg.solve(h, g * f)


def implicit_sensitivities(hess: np.ndarray, dg_dz: Sequence[np.ndarray], free: np.ndarray,
                           shape: tuple) -> list:
    """``dp/dz = -H^{-1} dg/dz`` per input, ``[*shape, *z.shape]``; parameters on a bound do not move."""
    h, f = _masked_hessian(hess, free)
    p = h.shape[0]
    out = []
    for d in dg_dz:
        rhs = d.reshape(p, -1) * f[:, None]
        out.append((-np.linalg.solve(h, rhs)).reshape(tuple(shape) + d.shape[1:]))
    return out


def implicit_solution(value: np.ndarray, inputs: Sequence, sensitivities: Sequence[np.ndarray]) -> tf.Tensor:
    """The optimum as a tensor of the *inputs*: value *value*, gradient ``dp/dz`` (*sensitivities*).

    Args:
        value (np.ndarray): Parameters at the optimum.
        inputs: The tensors ``z`` (connected to the caller's tape: the gradient flows on to the market).
        sensitivities: ``dp/dz`` per input, ``[*value.shape, *z.shape]`` (:func:`implicit_sensitivities`).

    Returns:
        tf.Tensor: float64 tensor of *value*'s shape.
    """
    value = np.asarray(value, dtype=float)
    axes = value.ndim
    sens = [tf.constant(s, DTYPE) for s in sensitivities]

    @tf.custom_gradient
    def solution(*zs):
        def grad(upstream):
            return [tf.tensordot(upstream, s, axes=axes) for s in sens]
        return tf.constant(value, DTYPE), grad

    return solution(*[tf.convert_to_tensor(z, DTYPE) for z in inputs])
