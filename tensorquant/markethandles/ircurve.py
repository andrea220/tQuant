from tensorflow import Variable, exp
from math import log
from tensorflow.python.framework import dtypes
import numpy
from datetime import date, timedelta
from typing import Union, Optional

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
        self._rates = [Variable(r, dtype=dtypes.float64) for r in rates]  # tensor
        self.interpolation_type = interp
        if self.interpolation_type == "LINEAR":
            self.interp = LinearInterp(self._pillars, self._rates)
        else:
            raise ValueError("Unsupported interpolation type")

        self._jacobian = None
        self._name = None

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

    @property
    def name(self) -> Optional[str]:
        """Returns the name of the curve (market key identifier).

        Returns:
            Optional[str]: The name/identifier of the curve, typically set by
                MarketEnvironment when accessing the curve. None if not set.
        """
        return self._name

    @name.setter
    def name(self, value: Optional[str]):
        """Sets the name of the curve.

        Args:
            value (Optional[str]): The name/identifier to set (e.g., "IR:EUR:ESTR:SPOT").
        """
        self._name = value

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
            df = exp(nrt)
            return df
        if term >= self._pillars[-1]:
            nrt = -term * self._rates[-1]
            df = exp(nrt)
            return df
        else:
            nrt = -term * self.interp.interpolate(term)
            df = exp(nrt)
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
        if isinstance(d1, date) and isinstance(d2, date):
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
        expr = -(log(self.discount(t + dt)) - log(self.discount(t - dt))) / (2 * dt)
        return expr

    def _set_rates(self, rates: list[float]) -> None:
        """Sets the rates for the curve and updates the interpolation.

        Args:
            rates (list[float]): The new rates to set.
        """
        self.__rates = rates
        self._rates = [Variable(r, dtype=dtypes.float64) for r in rates]
        self.interp.y = self._rates


class DefaultCurve:
    """Survival-probability / default-probability curve with piecewise-constant hazard rates.

    Given a set of pillars and corresponding survival probabilities Q(t_i), the hazard
    rate over each interval [t_{i-1}, t_i] is kept constant:

        h_i = -ln(Q(t_i) / Q(t_{i-1})) / (t_i - t_{i-1})

    This allows cheap computation of Q(t) for any t via

        Q(t) = Q(t_{i-1}) * exp(-h_i * (t - t_{i-1}))

    A convenience constructor :meth:`from_flat_hazard_rate` builds a flat (single-segment)
    curve from a constant hazard rate λ, i.e. Q(t) = exp(-λ·t).
    """

    def __init__(
        self,
        reference_date: date,
        pillars: Union[list[date], list[float]],
        survival_probs: list[float],
        daycounter_convention: DayCounterConvention,
    ):
        """Initialises the DefaultCurve from a vector of survival probabilities.

        Args:
            reference_date (date): Curve reference date (t = 0 anchor).
            pillars (Union[list[date], list[float]]): Pillar dates or year fractions.
                The first pillar should correspond to t = 0 with Q = 1, or the list
                may start directly from t > 0 – in either case Q(0) = 1 is prepended
                internally.
            survival_probs (list[float]): Survival probabilities Q(t_i) ∈ (0, 1].
                Must be monotonically non-increasing.
            daycounter_convention (DayCounterConvention): Day count convention for
                converting dates to year fractions.
        """
        self._reference_date = reference_date
        self._daycounter = DayCounter(daycounter_convention)
        self._daycounter_convention = daycounter_convention

        # Convert pillars to year fractions
        if all(isinstance(p, date) for p in pillars):
            self._dates = list(pillars)
            self._pillars = [
                self._daycounter.year_fraction(reference_date, d) for d in pillars
            ]
        elif all(isinstance(p, float) for p in pillars):
            self._pillars = list(pillars)
            self._dates = [
                reference_date + timedelta(days=p * 365) for p in pillars
            ]
        else:
            raise TypeError("pillars must be a list of either date or float types")

        # Prepend t=0 / Q=1 anchor if not already present
        if self._pillars[0] > 1e-10:
            self._pillars = [0.0] + self._pillars
            self._dates = [reference_date] + self._dates
            self._survival_probs = [1.0] + list(survival_probs)
        else:
            self._survival_probs = list(survival_probs)

        # Derive piecewise-constant hazard rates for each interval
        self._hazard_rates: list[float] = []
        for i in range(1, len(self._pillars)):
            dt = self._pillars[i] - self._pillars[i - 1]
            if dt <= 0:
                raise ValueError(f"Pillars must be strictly increasing (failed at index {i})")
            q_prev = self._survival_probs[i - 1]
            q_curr = self._survival_probs[i]
            if q_curr <= 0 or q_prev <= 0:
                raise ValueError("Survival probabilities must be strictly positive")
            h = -numpy.log(q_curr / q_prev) / dt
            self._hazard_rates.append(h)

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_flat_hazard_rate(
        cls,
        reference_date: date,
        hazard_rate: float,
        daycounter_convention: DayCounterConvention,
        horizon: float = 100.0,
    ):
        """Creates a flat DefaultCurve from a constant hazard rate λ.

        Q(t) = exp(-λ·t).

        Args:
            reference_date (date): Curve reference date.
            hazard_rate (float): Constant hazard rate λ (e.g. 0.03 for 3 %).
            daycounter_convention (DayCounterConvention): Day count convention.
            horizon (float): Far pillar in years (default 100). Determines the
                range over which the flat rate is valid.

        Returns:
            DefaultCurve: A flat default curve.
        """
        far_date = reference_date + timedelta(days=int(horizon * 365))
        pillars = [reference_date, far_date]
        survival_probs = [1.0, numpy.exp(-hazard_rate * horizon)]
        return cls(reference_date, pillars, survival_probs, daycounter_convention)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def reference_date(self) -> date:
        """Reference date of the curve."""
        return self._reference_date

    @property
    def pillars(self) -> list[float]:
        """Year-fraction pillars (including the t = 0 anchor)."""
        return self._pillars

    @property
    def survival_probs(self) -> list[float]:
        """Survival probabilities at each pillar."""
        return self._survival_probs

    @property
    def hazard_rates(self) -> list[float]:
        """Piecewise-constant hazard rates for each interval between pillars."""
        return self._hazard_rates

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def _time(self, term: Union[date, float]) -> float:
        """Converts a date or float to a year fraction."""
        if isinstance(term, date):
            return self._daycounter.year_fraction(self._reference_date, term)
        return float(term)

    def survival_prob(self, t: Union[date, float]) -> float:
        """Returns the survival probability Q(0, t).

        Args:
            t (Union[date, float]): Evaluation time (date or year fraction).

        Returns:
            float: Survival probability Q(0, t).
        """
        t = self._time(t)
        if t <= 0.0:
            return 1.0

        # Find the interval [t_{i-1}, t_i] that contains t
        for i in range(1, len(self._pillars)):
            if t <= self._pillars[i]:
                dt = t - self._pillars[i - 1]
                return self._survival_probs[i - 1] * numpy.exp(-self._hazard_rates[i - 1] * dt)

        # Beyond last pillar: extrapolate with last hazard rate
        dt = t - self._pillars[-1]
        return self._survival_probs[-1] * numpy.exp(-self._hazard_rates[-1] * dt)

    def marginal_pd(self, t1: Union[date, float], t2: Union[date, float]) -> float:
        """Returns the marginal (conditional) probability of default in (t1, t2].

        PD(t1, t2) = Q(t1) - Q(t2).

        Args:
            t1 (Union[date, float]): Start of the period.
            t2 (Union[date, float]): End of the period.

        Returns:
            float: Probability of default in the interval (t1, t2].
        """
        return self.survival_prob(t1) - self.survival_prob(t2)


class FlatCurve(RateCurve):
    """A flat (constant-rate) curve implementation.

    This curve assumes a single constant continuously-compounded rate for all maturities.
    It is implemented as a `RateCurve` with two very distant pillars carrying the same rate,
    so interpolation naturally keeps the curve flat.
    """

    def __init__(
        self,
        reference_date: date,
        rate: float,
        daycounter_convention: DayCounterConvention,
    ):
        """Initializes a flat curve with a constant rate.

        Args:
            reference_date (date): The reference date for the curve.
            rate (float): Constant continuously-compounded rate for all maturities.
            daycounter_convention (DayCounterConvention): Day count convention.
        """
        # Use two far-apart pillars with identical rates to obtain a flat curve
        far_date = reference_date + timedelta(days=365 * 100)  # 100Y horizon
        super().__init__(
            reference_date=reference_date,
            pillars=[reference_date, far_date],
            rates=[rate, rate],
            interp="LINEAR",
            daycounter_convention=daycounter_convention,
        )

    @property
    def rate(self) -> float:
        """Returns the flat rate of the curve."""
        # All rates are identical by construction; return the first one.
        return self._RateCurve__rates[0]

    @rate.setter
    def rate(self, value: float) -> None:
        """Sets the flat rate and updates the underlying curve representation."""
        # Keep both internal representations (float list and Tensor list) consistent.
        self._RateCurve__rates = [value, value]
        self._rates = [Variable(value, dtype=dtypes.float64) for _ in range(2)]
        self.interp.y = self._rates