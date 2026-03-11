from datetime import date
from ..timehandles.tqcalendar import Calendar
from ..timehandles.daycounter import DayCounter, DayCounterConvention
from ..timehandles.utils import Settings
import tensorflow as tf


class VolatilitySurface:

    def __init__(
        self,
        reference_date: date,
        calendar: Calendar,
        daycounter: DayCounter,
        strike: list[float],
        maturity: list[float],
        volatility_matrix,
    ) -> None:
        self._reference_date = reference_date
        self._calendar = calendar
        self._daycounter = daycounter
        self._strike = strike
        self._maturity = maturity
        self._volatility_matrix = volatility_matrix

    @property
    def reference_date(self):
        return self._reference_date

    @property
    def calendar(self):
        return self._calendar

    @property
    def daycounter(self):
        return self._daycounter

    @property
    def strike(self):
        return self._strike

    @property
    def maturity(self):
        return self._maturity

    @property
    def volatility_matrix(self):
        return self._volatility_matrix

    def _find_bounds(self, array, value):
        """Find the two closest indices in `array` that bound `value`."""
        lower = 0
        upper = len(array) - 1

        # Handle boundary cases
        if value <= array[lower]:
            return lower, lower
        if value >= array[upper]:
            return upper, upper

        # Search for the correct bounds
        for i in range(len(array) - 1):
            if array[i] <= value <= array[i + 1]:
                return i, i + 1

        return lower, upper  # should not reach here if value is within bounds

    def volatility(self, strike: float, tenor: float):
        """bi-linear interpolation"""
        # Find the indices of surrounding strike and maturity values
        x1_idx, x2_idx = self._find_bounds(self._strike, strike)
        y1_idx, y2_idx = self._find_bounds(self._maturity, tenor)

        # Get the actual strike and maturity values from the indices
        x1, x2 = self._strike[x1_idx], self._strike[x2_idx]
        y1, y2 = self._maturity[y1_idx], self._maturity[y2_idx]

        # Get the volatilities at the four corner points
        V11 = self._volatility_matrix[y1_idx][x1_idx]
        V21 = self._volatility_matrix[y1_idx][x2_idx]
        V12 = self._volatility_matrix[y2_idx][x1_idx]
        V22 = self._volatility_matrix[y2_idx][x2_idx]

        # If the point matches an existing grid point exactly
        if x1 == x2 and y1 == y2:
            return V11
        elif x1 == x2:
            # Linear interpolation in the y direction (maturity)
            return V11 + (V12 - V11) * (tenor - y1) / (y2 - y1)
        elif y1 == y2:
            # Linear interpolation in the x direction (strike)
            return V11 + (V21 - V11) * (strike - x1) / (x2 - x1)

        # Bilinear interpolation formula
        denominator = (x2 - x1) * (y2 - y1)
        value = (
            V11 * (x2 - strike) * (y2 - tenor)
            + V21 * (strike - x1) * (y2 - tenor)
            + V12 * (x2 - strike) * (tenor - y1)
            + V22 * (strike - x1) * (tenor - y1)
        ) / denominator

        return value

    def variance(self, strike, maturity: date):
        implied_vol = self.volatility(strike, maturity)
        t = self.daycounter.year_fraction(Settings.evaluation_date, maturity)
        return t * implied_vol * implied_vol


class BlackConstantVolatility(VolatilitySurface):
    """Constant volatility surface that returns the same volatility value regardless of strike and tenor.

    This class provides the same interface as VolatilitySurface but always returns
    a constant volatility value, ignoring the strike and tenor parameters.
    """

    def __init__(
        self,
        reference_date: date,
        volatility: float,
        calendar: Calendar = None,
        daycounter: DayCounter = None,
    ) -> None:
        """Initialize a constant volatility surface.

        Args:
            reference_date (date): The reference date for the volatility surface.
            calendar (Calendar, optional): Calendar for date calculations. Defaults to None.
            daycounter (DayCounter, optional): Day counter for time calculations.
                Defaults to Actual365 if not provided.
            volatility (float, optional): The constant volatility value. Can be a scalar
                or a tf.Variable/tf.Tensor. If None, must be provided via volatility_matrix.
        """
        if daycounter is None:
            daycounter = DayCounter(DayCounterConvention.Actual365)
        # For constant volatility, strike and maturity lists are not used
        # but we need to provide them for the parent class
        strike = [0.0]  # Dummy value, not used
        maturity = [0.0]  # Dummy value, not used
        self.flat_vol = volatility
        super().__init__(reference_date, calendar, daycounter, strike, maturity, volatility)

    def volatility(self, strike: float = None, tenor: float = None):
        """Return the constant volatility value, ignoring strike and tenor.

        This method has the same signature as VolatilitySurface.volatility() for
        interface compatibility, but always returns the constant volatility value
        regardless of the input parameters.

        Args:
            strike (float, optional): Strike price (ignored for constant volatility).
            tenor (float, optional): Time to maturity in years (ignored for constant volatility).

        Returns:
            tf.Variable: The constant volatility value as a TensorFlow variable.
        """
        
        # Otherwise, convert to tf.Variable
        return tf.Variable(self.flat_vol, dtype=tf.float32)
