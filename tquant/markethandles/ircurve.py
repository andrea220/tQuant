import tensorflow 
from tensorflow.python.framework import dtypes
import numpy 
from datetime import date, timedelta
from typing import Union

from ..numericalhandles.interpolation import LinearInterp
from ..timehandles.daycounter import DayCounter, DayCounterConvention




class RateCurve:
    """Represents a financial rate curve used for discounting, zero rates, or forward rates.

    Attributes:
        _reference_date (date): The starting date of the curve.
        _daycounter_convention (DayCounterConvention): Convention for day counting.
        _daycounter (DayCounter): Day counter instance based on the convention.
        _dates (list[date]): Dates that represent the curve's pillars.
        _pillars (list[float]): Year fractions corresponding to the pillar dates.
        _pillar_days (list[int]): Day counts between the reference date and the pillars.
        __rates (list[float]): Interest rates associated with the pillars.
        _rates (list[Tensor]): Interest rates stored as TensorFlow variables.
        interpolation_type (str): Type of interpolation used (e.g., 'LINEAR').
        interp (LinearInterp): Interpolation object used for rate calculations.
        _jacobian (numpy.ndarray): Jacobian matrix of the curve, if applicable.
    """
    def __init__(
        self,
        reference_date: date,
        pillars: Union[list[date], list[float]],
        rates: list[float],
        interp: str,
        daycounter_convention: DayCounterConvention,
    ):
        """Initializes the RateCurve with a reference date, pillars, rates, and an interpolation method.

        Args:
            reference_date (date): The reference date for the curve.
            pillars (Union[list[date], list[float]]): A list of dates or year fractions.
            rates (list[float]): A list of interest rates corresponding to the pillars.
            interp (str): The interpolation method to use (e.g., 'LINEAR').
            daycounter_convention (DayCounterConvention): The day count convention to use for calculating time differences.

        Raises:
            TypeError: If the pillars are not all of type date or float.
            ValueError: If the interpolation type is unsupported.
        """
        self._reference_date = reference_date
        self._daycounter_convention = daycounter_convention
        self._daycounter = DayCounter(daycounter_convention)
        if all(isinstance(d, date) for d in pillars):
            self._dates = pillars
            self._pillars = [
                self._daycounter.year_fraction(reference_date, d) for d in pillars
            ]
            self._pillar_days = [
                self._daycounter.day_count(reference_date, d) for d in self._dates
            ]
        elif all(isinstance(d, float) for d in pillars):
            self._dates = [reference_date + timedelta(days=d * 365) for d in pillars]
            self._pillars = pillars
            self._pillar_days = [
                self._daycounter.day_count(reference_date, d) for d in self._dates
            ]
        else:
            raise TypeError("Pillars must be a list of either date or float types")
        self.__rates = rates  # float
        self._rates = [tensorflow.Variable(r, dtype=dtypes.float64) for r in rates]  # tensor
        self.interpolation_type = interp
        if self.interpolation_type == "LINEAR":
            self.interp = LinearInterp(self._pillars, self._rates)
        else:
            raise ValueError("Unsupported interpolation type")

        self._jacobian = None

    @classmethod
    def from_zcb(
        cls,
        reference_date: date,
        pillars: Union[list[date], list[int]],
        discount_factors: list[float],
        interp: str,
        daycounter_convention: DayCounterConvention,
    ):
        """Creates a RateCurve instance from zero-coupon bond discount factors.

        Args:
            reference_date (date): The reference date for the curve.
            pillars (Union[list[date], list[int]]): A list of dates or year fractions for the curve.
            discount_factors (list[float]): A list of discount factors (e.g., from zero-coupon bonds).
            interp (str): Interpolation method to use (e.g., 'LINEAR').
            daycounter_convention (DayCounterConvention): Day count convention for time calculations.

        Returns:
            RateCurve: A new RateCurve instance.
        """
        daycounter = DayCounter(daycounter_convention)
        if all(isinstance(d, date) for d in pillars):
            dates = pillars
            pillars = [daycounter.year_fraction(reference_date, d) for d in pillars]
        elif all(isinstance(d, float) for d in pillars):
            dates = [reference_date + timedelta(days=d * 365) for d in pillars]
            pillars = [daycounter.year_fraction(reference_date, d) for d in dates]
        else:
            raise TypeError("dates must be a list of either date or float types")
        rates = [
            -numpy.log(df) / tau if tau > 0 else 0
            for df, tau in zip(discount_factors, pillars)
        ]
        return cls(reference_date, dates, rates, interp, daycounter_convention)

    @property
    def nodes(self):
        """Returns the curve's nodes (pillar dates and rates).

        Returns:
            list[tuple]: A list of tuples, where each tuple contains a date and the corresponding rate.
        """
        return [(t, r) for t, r in zip(self._dates, self.__rates)]

    @property
    def reference_date(self):
        """Returns the reference date of the curve.

        Returns:
            date: The reference date.
        """
        return self._reference_date
    
    @property
    def daycounter_convention(self):
        """Returns the day count convention used by the curve.

        Returns:
            DayCounterConvention: The day count convention.
        """
        return self._daycounter_convention
    
    @property
    def daycounter(self):
        """Returns the day counter used by the curve.

        Returns:
            DayCounter: The day counter object.
        """
        return self._daycounter
    
    @property
    def dates(self):
        """Returns the pillar dates of the curve.

        Returns:
            list[date]: A list of dates representing the curve's pillars.
        """
        return self._dates
    
    @property
    def pillars(self):
        """Returns the year fractions (pillars) of the curve.

        Returns:
            list[float]: A list of year fractions corresponding to the pillars.
        """
        return self._pillars
    
    @property
    def pillar_days(self):
        """Returns the day counts for the pillars from the reference date.

        Returns:
            list[int]: A list of day counts between the reference date and each pillar.
        """
        return self._pillar_days
    
    @property
    def rates(self):
        """Returns the list of interest rates for the curve.

        Returns:
            list[float]: The rates corresponding to the pillars.
        """
        return self.__rates

    @property
    def jacobian(self) -> numpy.ndarray:
        """Returns the Jacobian matrix of the curve.

        Returns:
            numpy.ndarray: The Jacobian matrix if set, otherwise None.
        """
        return self._jacobian
    
    @jacobian.setter
    def jacobian(self, value: numpy.ndarray):
        """Sets the Jacobian matrix for the curve.

        Args:
            value (numpy.ndarray): The Jacobian matrix to set.
        """
        self._jacobian = value
    
    def discount(self, term: Union[date, float]) -> float:
        """Calculates the discount factor for a given term.

        Args:
            term (Union[date, float]): The term for which to calculate the discount factor. Can be a date or year fraction.

        Returns:
            float: The discount factor.

        Raises:
            TypeError: If the term is not a date or float.
        """
        if isinstance(term, date):
            term = self._daycounter.year_fraction(self._reference_date, term)
        elif isinstance(term, float):
            pass
        else:
            raise TypeError("term must be of type date or float")

        if term == 0.0:
            return 1

        if term <= self._pillars[0]:
            nrt = -term * self._rates[0]
            df = tensorflow.exp(nrt)
            return df
        if term >= self._pillars[-1]:
            nrt = -term * self._rates[-1]
            df = tensorflow.exp(nrt)
            return df
        else:
            nrt = -term * self.interp.interpolate(term)
            df = tensorflow.exp(nrt)
            return df

    def zero_rate(self, term: Union[date, float]) -> float:
        """Calculates the zero rate for a given term.

        Args:
            term (Union[date, float]): The term for which to calculate the zero rate. Can be a date or year fraction.

        Returns:
            float: The zero rate.

        Raises:
            TypeError: If the term is not a date or float.
        """
        if isinstance(term, date):
            term = self._daycounter.year_fraction(self._reference_date, term)
        elif isinstance(term, float):
            pass
        else:
            raise TypeError("term must be of type date or float")
        P_tT = self.discount(term)
        return (1 - P_tT) / (term * P_tT)

    def forward_rate(self, d1: Union[date, float], d2: Union[date, float]) -> float:
        """Calculates the forward rate between two dates or year fractions.

        Args:
            d1 (Union[date, float]): The start of the period.
            d2 (Union[date, float]): The end of the period.

        Returns:
            float: The forward rate.

        Raises:
            TypeError: If d1 and d2 are not both dates or both floats.
        """
        if isinstance(d1, date) and isinstance(
            d2, date
        ):  
            tau = self._daycounter.year_fraction(d1, d2)
            df1 = self.discount(
                self._daycounter.year_fraction(self._reference_date, d1)
            )
            df2 = self.discount(
                self._daycounter.year_fraction(self._reference_date, d2)
            )
        elif isinstance(d1, float) and isinstance(d2, float):
            tau = d2 - d1
            df1 = self.discount(d1)
            df2 = self.discount(d2)
        else:
            raise TypeError("d1 e d2 devono essere entrambi date oppure entrambi float")

        return (df1 / df2 - 1) / tau

    def inst_fwd(self, t: float):
        """Calculates the instantaneous forward rate at a specific time.

        Args:
            t (float): The time (in year fractions) to calculate the instantaneous forward rate.

        Returns:
            float: The instantaneous forward rate.
        """
        # time-step needed for differentiation
        dt = 0.01
        expr = -(
            tensorflow.math.log(self.discount(t + dt)) - tensorflow.math.log(self.discount(t - dt))
        ) / (2 * dt)
        return expr

    def _set_rates(self, rates: list[float]) -> None:
        """Sets the rates for the curve and updates the interpolation.

        Args:
            rates (list[float]): The new rates to set.
        """
        self.__rates = rates
        self._rates = [tensorflow.Variable(r, dtype=dtypes.float64) for r in rates]
        self.interp.y = self._rates
