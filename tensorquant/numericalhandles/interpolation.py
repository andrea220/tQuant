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

    def derivative(self, term):
        """
        Slope of the linear interpolant at *term*: ``(y[i+1] - y[i]) / (x[i+1] - x[i])`` on the segment that
        :meth:`interpolate` uses (``x[i] <= term < x[i+1]``: at a knot, the segment on its right).

        Args:
            term (float): The x-value.

        Returns:
            The slope (same type as the y-values: tensors keep their gradients).

        Raises:
            ValueError: If the term is outside the range of x-values.
        """
        for i in range(0, len(self.x) - 1):
            if term < self.x[i + 1]:
                return (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i])

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


class MonotoneConvex:
    """Monotone convex interpolation of a yield curve (Hagan and West, "Interpolation methods for curve
    construction", Applied Mathematical Finance 2006), on zero rates ``y`` at times ``x`` (continuously compounded).

    The discrete forwards ``fd_i = (y_i x_i - y_{i-1} x_{i-1}) / (x_i - x_{i-1})`` (``x_0 = 0``) are preserved exactly
    (so are the zero rates at the pillars); the instantaneous forward is continuous, equal at the nodes to
    ``f_i = (dx_i fd_{i+1} + dx_{i+1} fd_i) / (dx_i + dx_{i+1})`` (``f_0``, ``f_n`` extrapolated from the
    neighbours), and on each interval ``fd_i + g(u)`` with the quadratic / piecewise-quadratic ``g`` of the four
    regions of the paper, ``int_0^1 g = 0``: no spurious oscillation (monotone discrete forwards give a monotone
    forward). Flat forward ``f_n`` after the last pillar. No positivity collar (negative rates allowed).

    The four regions share one form, ``int_0^u g = c1 u + c2 u^2 + c3 u^3 + cL (1 - ((eta - u)^+ / eta)^3)
    + cR ((u - eta)^+)^3``, with per-interval coefficients computed once (vectorised) and reused: within a traced
    graph, and in eager mode while the rates are unchanged and no gradient tape records (under a tape they are
    recomputed, so that the gradients reach the rates). Pure TensorFlow, differentiable w.r.t. ``y``.

    Args:
        x: Pillar times (increasing, > 0), numbers.
        y: Zero rates at the pillars (numbers, tensors or variables).
    """

    def __init__(self, x, y):
        self.x = [float(v) for v in x]
        self.y = y
        self._cache_key = None
        self._cache = None
        self._numeric = None
        self._pinned = None
        self._pinned_numeric = None

    def pinned(self):
        """Context manager: the coefficients computed once on entry and reused until exit, also under a gradient
        tape (for many evaluations at fixed rates within one recording, e.g. all the instruments of a bootstrap)."""
        from contextlib import contextmanager

        @contextmanager
        def block():
            self._pinned, self._pinned_numeric = self._compute(), None
            try:
                yield
            finally:
                self._pinned = self._pinned_numeric = None

        return block()

    def _coefficients(self):
        import tensorflow as tf
        from tensorflow.python.eager import record

        if self._pinned is not None:
            return self._pinned

        if not tf.executing_eagerly():
            key = tf.compat.v1.get_default_graph()
        elif not record.could_possibly_record():
            key = tf.stack([tf.cast(v, tf.float64) for v in self.y]).numpy().tobytes()   # one read of the rates
        else:
            key = None
        if key is not None:
            same = key == self._cache_key if isinstance(key, bytes) else key is self._cache_key
            if same:
                return self._cache
        coefficients = self._compute()
        if key is not None:
            self._cache_key, self._cache, self._numeric = key, coefficients, None
        return coefficients

    def _numeric_coefficients(self):
        """The coefficients as numpy arrays when nothing can need a gradient (eager, no tape recording), ``None``
        otherwise: values for display or reporting without one TensorFlow op per evaluation (and, when pinned,
        without reading the rates at each call)."""
        import numpy as np
        import tensorflow as tf
        from tensorflow.python.eager import record

        if not tf.executing_eagerly() or record.could_possibly_record():
            return None
        if self._pinned is not None:
            if self._pinned_numeric is None:
                t, rt, fd, f, coef = self._pinned
                self._pinned_numeric = (np.asarray(t), rt.numpy(), fd.numpy(), f.numpy(), coef.numpy())
            return self._pinned_numeric
        t, rt, fd, f, coef = self._coefficients()           # refreshes the cache (and drops _numeric) if y changed
        if self._numeric is None:
            self._numeric = (np.asarray(t), rt.numpy(), fd.numpy(), f.numpy(), coef.numpy())
        return self._numeric

    def _compute(self):
        """``t`` (numbers), ``rt [n+1]``, ``fd [n]``, ``f [n+1]`` and ``coef [n, 6]`` (``c1, c2, c3, cL, cR, eta``)."""
        import numpy as np
        import tensorflow as tf
        from .dtypes import DTYPE

        t = [0.0] + self.x
        x = tf.constant(self.x, DTYPE)
        y = tf.stack([tf.cast(v, DTYPE) for v in self.y])
        rt = tf.concat([tf.zeros([1], DTYPE), y * x], 0)
        dt = tf.constant(np.diff(t), DTYPE)
        fd = (rt[1:] - rt[:-1]) / dt
        if len(self.x) == 1:
            f = tf.concat([fd, fd], 0)
        else:
            dl, dr = dt[:-1], dt[1:]
            inner = (dl * fd[1:] + dr * fd[:-1]) / (dl + dr)
            f = tf.concat([(fd[0] - 0.5 * (inner[0] - fd[0]))[None], inner,
                           (fd[-1] - 0.5 * (inner[-1] - fd[-1]))[None]], 0)
        g0, g1 = f[:-1] - fd, f[1:] - fd
        zeros, ones = tf.zeros_like(g0), tf.ones_like(g0)

        def safe(d):
            return tf.where(tf.abs(d) < 1e-300, ones, d)

        r1 = ((g0 < 0) & (-g0 / 2 <= g1) & (g1 <= -2 * g0)) | ((g0 > 0) & (-g0 / 2 >= g1) & (g1 >= -2 * g0))
        r2 = ((g0 < 0) & (g1 > -2 * g0)) | ((g0 > 0) & (g1 < -2 * g0))
        r3 = ((g0 > 0) & (g1 < 0) & (g1 > -g0 / 2)) | ((g0 < 0) & (g1 > 0) & (g1 < -g0 / 2))
        zero = (g0 == 0) & (g1 == 0)
        eta2 = tf.clip_by_value((g1 + 2 * g0) / safe(g1 - g0), 0.0, 1.0 - 1e-12)
        eta3 = tf.clip_by_value(3 * g1 / safe(g1 - g0), 1e-12, 1.0)
        eta4 = tf.clip_by_value(g1 / safe(g0 + g1), 1e-12, 1.0 - 1e-12)
        a4 = -g0 * g1 / safe(g0 + g1)
        c1 = tf.where(r1 | r2, g0, tf.where(r3, g1, a4))
        c2 = tf.where(r1, -2 * g0 - g1, zeros)
        c3 = tf.where(r1, g0 + g1, zeros)
        eta = tf.where(r1, 0.5 * ones, tf.where(r2, eta2, tf.where(r3, eta3, eta4)))
        c_left = tf.where(r1 | r2, zeros, tf.where(r3, (g0 - g1) * eta3 / 3, (g0 - a4) * eta4 / 3))
        c_right = tf.where(r1 | r3, zeros, tf.where(r2, (g1 - g0) / (3 * (1 - eta2) ** 2),
                                                    (g1 - a4) / (3 * (1 - eta4) ** 2)))
        coef = tf.stack([c1, c2, c3, c_left, c_right, eta], axis=1)
        coef = tf.where(zero[:, None], tf.concat([tf.zeros_like(coef[:, :5]), 0.5 * tf.ones_like(coef[:, 5:])], 1),
                        coef)
        return t, rt, fd, f, coef

    def _locate(self, term: float):
        for i in range(1, len(self.x) + 1):
            if term <= self.x[i - 1]:
                return i
        return None                                      # after the last pillar

    @staticmethod
    def _shape(row, u):
        """``(g(u), int_0^u g)`` from the coefficients ``row`` of one interval."""
        import tensorflow as tf

        c1, c2, c3, c_left, c_right, eta = tf.unstack(row)
        left, right = tf.maximum(eta - u, 0.0), tf.maximum(u - eta, 0.0)
        integral = c1 * u + c2 * u ** 2 + c3 * u ** 3 + c_left * (1 - (left / eta) ** 3) + c_right * right ** 3
        g = c1 + 2 * c2 * u + 3 * c3 * u ** 2 + 3 * c_left * left ** 2 / eta ** 3 + 3 * c_right * right ** 2
        return g, integral

    @staticmethod
    def _shape_np(row, u):
        """:meth:`_shape` in numpy."""
        c1, c2, c3, c_left, c_right, eta = row
        left, right = max(eta - u, 0.0), max(u - eta, 0.0)
        integral = c1 * u + c2 * u ** 2 + c3 * u ** 3 + c_left * (1 - (left / eta) ** 3) + c_right * right ** 3
        g = c1 + 2 * c2 * u + 3 * c3 * u ** 2 + 3 * c_left * left ** 2 / eta ** 3 + 3 * c_right * right ** 2
        return g, integral

    def _integral_np(self, numeric, term: float) -> float:
        t, rt, fd, f, coef = numeric
        i = self._locate(term)
        if i is None:
            return float(rt[-1] + f[-1] * (term - t[-1]))
        h = t[i] - t[i - 1]
        _, ig = self._shape_np(coef[i - 1], (term - t[i - 1]) / h)
        return float(rt[i - 1] + fd[i - 1] * (term - t[i - 1]) + h * ig)

    def _forward_np(self, numeric, term: float) -> float:
        t, rt, fd, f, coef = numeric
        i = self._locate(term) if term > 0.0 else 1
        if i is None:
            return float(f[-1])
        h = t[i] - t[i - 1]
        g, _ = self._shape_np(coef[i - 1], (term - t[i - 1]) / h)
        return float(fd[i - 1] + g)

    def integrals(self, terms):
        """``int_0^t f`` at many times at once (``[k]`` tensor; 0 for ``t <= 0``): one evaluation for all the dates
        of a leg."""
        import numpy as np
        import tensorflow as tf
        from .dtypes import DTYPE

        terms = np.asarray(terms, dtype=float)
        t, rt, fd, f, coef = self._coefficients()
        tt = np.asarray(t)
        n = len(self.x)
        i = np.minimum(np.searchsorted(np.asarray(self.x), terms, side="left"), n - 1)   # interval i + 1
        t_lo, h = tt[i], tt[i + 1] - tt[i]
        u = tf.constant(np.clip((terms - t_lo) / h, 0.0, 1.0), DTYPE)
        c1, c2, c3, c_left, c_right, eta = tf.unstack(tf.gather(coef, i), axis=1)
        left, right = tf.maximum(eta - u, 0.0), tf.maximum(u - eta, 0.0)
        shape = c1 * u + c2 * u ** 2 + c3 * u ** 3 + c_left * (1 - (left / eta) ** 3) + c_right * right ** 3
        inside = tf.gather(rt, i) + tf.gather(fd, i) * tf.constant(terms - t_lo, DTYPE) + tf.constant(h, DTYPE) * shape
        beyond = rt[-1] + f[-1] * tf.constant(terms - tt[-1], DTYPE)
        out = tf.where(tf.constant(terms > tt[-1]), beyond, inside)
        return tf.where(tf.constant(terms <= 0.0), tf.zeros_like(out), out)

    def integral(self, term: float):
        """``int_0^term f(s) ds = r(term) term`` (``-ln P(0, term)``)."""
        import tensorflow as tf
        from .dtypes import DTYPE

        term = float(term)
        if term <= 0.0:
            return tf.constant(0.0, DTYPE)
        numeric = self._numeric_coefficients()
        if numeric is not None:
            return tf.constant(self._integral_np(numeric, term), DTYPE)
        t, rt, fd, f, coef = self._coefficients()
        i = self._locate(term)
        if i is None:
            return rt[-1] + f[-1] * (term - t[-1])
        h = t[i] - t[i - 1]
        _, ig = self._shape(coef[i - 1], tf.constant((term - t[i - 1]) / h, DTYPE))
        return rt[i - 1] + fd[i - 1] * (term - t[i - 1]) + h * ig

    def forward(self, term: float):
        """Instantaneous forward ``f(term)``."""
        import tensorflow as tf
        from .dtypes import DTYPE

        term = max(float(term), 0.0)
        numeric = self._numeric_coefficients()
        if numeric is not None:
            return tf.constant(self._forward_np(numeric, term), DTYPE)
        t, rt, fd, f, coef = self._coefficients()
        i = self._locate(term) if term > 0.0 else 1
        if i is None:
            return f[-1]
        h = t[i] - t[i - 1]
        g, _ = self._shape(coef[i - 1], tf.constant((term - t[i - 1]) / h, DTYPE))
        return fd[i - 1] + g
