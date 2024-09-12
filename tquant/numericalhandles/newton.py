import numpy


def numerical_jacobian(func, x, epsilon=1e-5):
    """
    Computes the numerical Jacobian matrix for a given function using finite differences.

    Args:
        func (callable): The function for which the Jacobian is computed. It should take a single argument x (array).
        x (numpy array): The point at which to evaluate the Jacobian.
        epsilon (float, optional): A small perturbation used to compute finite differences. Defaults to 1e-5.

    Returns:
        numpy array: The computed Jacobian matrix with shape (n, n), where n is the length of x.
    """
    n = len(x)
    jac = numpy.zeros((n, n))
    f0 = func(x)

    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += epsilon
        f_plus = func(x_plus)
        jac[:, i] = (f_plus - f0) / epsilon

    return jac


def newton(func, x0, tol=1e-8, max_iter=100):
    """
    Solves a system of nonlinear equations using Newton's method.

    Args:
        func (callable): A function that returns a tuple of (f(x), jacobian), where f(x) is the vector of function values and
                         jacobian is the Jacobian matrix evaluated at x.
        x0 (numpy array): Initial guess for the root.
        tol (float, optional): Tolerance for stopping criteria. The method stops when the solution update is smaller than tol. Defaults to 1e-8.
        max_iter (int, optional): Maximum number of iterations allowed. Defaults to 100.

    Returns:
        tuple: A tuple containing:
            - numpy array: The solution vector x that approximately satisfies func(x) = 0.
            - numpy array: The Jacobian matrix at the solution.

    Raises:
        ValueError: If the method fails to converge after max_iter iterations.
    """
    x = x0.copy()
    iter_count = 0
    while iter_count < max_iter:
        f, jac = func(x)
        delta_x = numpy.linalg.solve(jac, -f)
        x += delta_x
        if numpy.linalg.norm(delta_x) < tol:
            break
        iter_count += 1

    if iter_count == max_iter:
        raise ValueError(
            "Newton's method failed to converge after the maximum number of iterations."
        )

    print("total iteration: ", iter_count)
    return x, jac
