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
