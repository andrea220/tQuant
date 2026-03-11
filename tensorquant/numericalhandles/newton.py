import numpy


def newton_1d(func, i: int, r0: float, tol: float = 1e-8, max_iter: int = 100) -> float:
    """Scalar Newton's method for a single pillar in curve bootstrapping.

    Finds ``r`` such that ``func(i, r) = 0`` using the analytic derivative
    returned by ``func`` (typically computed via autodiff).

    Args:
        func (callable): Callable with signature ``func(i, r) -> (f, df_dr)``,
            where ``f`` is the NPV of instrument *i* and ``df_dr`` is its
            derivative w.r.t. the rate at pillar *i*.
        i (int): Pillar index being solved.
        r0 (float): Initial guess for the rate at pillar *i*.
        tol (float, optional): Convergence tolerance on both ``|f|`` and the
            Newton step ``|Δr|``. Defaults to 1e-8.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        float: The bootstrapped rate at pillar *i*.

    Raises:
        ValueError: If the derivative is numerically zero or if the method
            fails to converge within *max_iter* iterations.
    """
    r = float(r0)
    for iteration in range(max_iter):
        print(iteration)
        f, df = func(i, r)
        if abs(df) < 1e-14:
            raise ValueError(
                f"Derivative is numerically zero at pillar {i} "
                f"(iteration {iteration}). Newton step is undefined."
            )
        step = -f / df
        r += step
        if abs(f) < tol and abs(step) < tol:
            return r
    raise ValueError(
        f"Newton 1D failed to converge at pillar {i} "
        f"after {max_iter} iterations (last residual: {f:.3e})."
    )


def newton(func, x0, tol=1e-8, max_iter=100):
    """Solves a system of nonlinear equations using Newton's method.

    Args:
        func (callable): A function that returns ``(f(x), jacobian)``, where
            ``f(x)`` is the vector of residuals and ``jacobian`` is the NxN
            Jacobian matrix evaluated at ``x``.
        x0 (numpy.ndarray): Initial guess for the root.
        tol (float, optional): Convergence tolerance. The method stops when
            both ``‖f(x)‖`` and ``‖Δx‖`` are below *tol*. Defaults to 1e-8.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        tuple:
            - numpy.ndarray: Solution vector ``x`` satisfying ``func(x) ≈ 0``.
            - numpy.ndarray: Jacobian matrix at the solution.

    Raises:
        ValueError: If the method fails to converge after *max_iter* iterations.
    """
    x = x0.copy()
    f, jac = None, None
    for iteration in range(max_iter):
        print(iteration)
        f, jac = func(x)
        delta_x = numpy.linalg.solve(jac, -f)
        x += delta_x
        if numpy.linalg.norm(f) < tol and numpy.linalg.norm(delta_x) < tol:
            return x, jac
    raise ValueError(
        f"Newton's method failed to converge after {max_iter} iterations "
        f"(last residual norm: {numpy.linalg.norm(f):.3e})."
    )
