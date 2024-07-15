import time

import numpy as np


def numerical_jacobian(func, x, epsilon=1e-5):
    n = len(x)
    jac = np.zeros((n, n))
    f0 = func(x)

    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += epsilon
        f_plus = func(x_plus)
        jac[:, i] = (f_plus - f0) / epsilon

    return jac


def newton_multidimensional(func, x0, tol=1e-8, max_iter=100):
    x = x0.copy()
    iter_count = 0
    while iter_count < max_iter:
        f = func(x)
        jac = numerical_jacobian(func, x)
        delta_x = np.linalg.solve(jac, -f)
        x += delta_x
        if np.linalg.norm(delta_x) < tol:
            break
        iter_count += 1
    return x
