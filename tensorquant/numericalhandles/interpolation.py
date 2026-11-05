class LinearInterp:
    """
    Linear interpolation.

    This class provides a simple linear interpolation method for a given set of x and y values. It computes
    interpolated values for a given term by linearly interpolating between the known data points.

    Args:
        x (list or numpy array): Known x-values (independent variable).
        y (list or numpy array): Known y-values (dependent variable).
    """

    def __init__(self, x, y):
        """
        Initializes the LinearInterp class with given x and y data points.

        Args:
            x (list or numpy array): Known x-values.
            y (list or numpy array): Known y-values.
        """
        self.x = x
        self.y = y

    def interpolate(self, term):
        """
        Interpolates a value at the specified term using linear interpolation.

        For a given term (input value), this method finds the two adjacent x-values that bound the term
        and computes the corresponding interpolated y-value.

        Args:
            term (float): The x-value at which interpolation is desired.

        Returns:
            float: The interpolated y-value.

        Raises:
            ValueError: If the term is outside the range of x-values.
        """
        for i in range(0, len(self.x) - 1):
            if term < self.x[i + 1]:
                dtr = 1 / (self.x[i + 1] - self.x[i])
                w1 = (self.x[i + 1] - term) * dtr
                w2 = (term - self.x[i]) * dtr
                r1 = w1 * self.y[i]
                r2 = w2 * self.y[i + 1]
                return r1 + r2

        raise ValueError(f"Term {term} is outside the range of x-values.")


def natural_cubic_spline(x, f, q, max_slope=None):
    """Natural cubic spline through ``(x, f)`` evaluated at ``q``: value, first and second derivative.

    Inside ``[x_0, x_{n-1}]`` the natural cubic spline (zero second derivative at the
    ends); outside, the C1 linear continuation with the spline's end slopes
    (second derivative 0). Pure TensorFlow (tridiagonal solve), float64,
    differentiable w.r.t. both the knots ``x`` and the values ``f``.

    Args:
        x: ``[n]`` increasing knots (``n >= 3``).
        f: ``[n]`` values at the knots.
        q: ``[m]`` query points.
        max_slope (float, optional): Clip the end slopes used outside the knots to
            ``[-max_slope, max_slope]``.

    Returns:
        tuple: ``(value, d/dq, d2/dq2)``, each ``[m]`` float64 tensors.
    """
    import tensorflow as tf
    from .dtypes import DTYPE

    x, f, q = tf.cast(x, DTYPE), tf.cast(f, DTYPE), tf.cast(q, DTYPE)
    zero = tf.zeros([1], DTYPE)
    h = x[1:] - x[:-1]
    df = (f[1:] - f[:-1]) / h
    # second derivatives M_1..M_{n-2} (M_0 = M_{n-1} = 0)
    sup = tf.concat([h[1:-1], zero], 0)
    sub = tf.concat([zero, h[1:-1]], 0)
    diag = 2.0 * (h[:-1] + h[1:])
    rhs = 6.0 * (df[1:] - df[:-1])
    m_in = tf.linalg.tridiagonal_solve([sup, diag, sub], rhs[:, None], diagonals_format="sequence")[:, 0]
    M = tf.concat([zero, m_in, zero], 0)

    n = tf.shape(x)[0]
    qc = tf.clip_by_value(q, x[0], x[-1])
    i = tf.clip_by_value(tf.searchsorted(tf.stop_gradient(x), tf.stop_gradient(qc), side="right") - 1, 0, n - 2)
    xi, xj, fi, fj = tf.gather(x[:-1], i), tf.gather(x[1:], i), tf.gather(f[:-1], i), tf.gather(f[1:], i)
    mi, mj, hi = tf.gather(M[:-1], i), tf.gather(M[1:], i), tf.gather(h, i)
    a, b = xj - qc, qc - xi
    ci, cj = fi / hi - mi * hi / 6.0, fj / hi - mj * hi / 6.0
    value = mi * a**3 / (6.0 * hi) + mj * b**3 / (6.0 * hi) + ci * a + cj * b
    d1 = -mi * a**2 / (2.0 * hi) + mj * b**2 / (2.0 * hi) - ci + cj
    d2 = mi * a / hi + mj * b / hi

    s0 = df[0] - h[0] * (2.0 * M[0] + M[1]) / 6.0       # S'(x_0)
    s1 = df[-1] + h[-1] * (M[-2] + 2.0 * M[-1]) / 6.0   # S'(x_{n-1})
    if max_slope is not None:
        s0 = tf.clip_by_value(s0, -max_slope, max_slope)
        s1 = tf.clip_by_value(s1, -max_slope, max_slope)
    lo, hi_side = q < x[0], q > x[-1]
    ones = tf.ones_like(q)
    value = tf.where(lo, f[0] + s0 * (q - x[0]), tf.where(hi_side, f[-1] + s1 * (q - x[-1]), value))
    d1 = tf.where(lo, s0 * ones, tf.where(hi_side, s1 * ones, d1))
    d2 = tf.where(lo | hi_side, tf.zeros_like(q), d2)
    return value, d1, d2
