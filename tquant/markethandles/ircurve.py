import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np

from ..numericalhandles.interpolation import LinearInterp
from ..timehandles.daycounter import DayCounter, DayCounterConvention

from datetime import date, timedelta
from typing import Union



class RateCurve:
    """
    Represents an interest rate curve, which is used in financial markets to determine 
    the present value of future cash flows.

    The curve can be defined by a set of dates (pillars) or times to maturity, and the 
    corresponding rates. This class also supports interpolation between these pillars 
    and allows calculation of discount factors, zero rates, forward rates, and instantaneous 
    forward rates.

    Attributes:
    -----------
    _reference_date: date
        The base date from which all other dates are measured.
    _dates: list[date]
        The list of dates corresponding to the pillars of the curve.
    _pillars: list[float]
        The times to maturity in years corresponding to the pillars of the curve.
    _pillar_days: list[int]
        The number of days between the reference date and each pillar.
    __rates: list[float]
        The interest rates corresponding to the pillars of the curve.
    _rates: list[tf.Variable]
        TensorFlow variables representing the rates for computational purposes.
    interpolation_type: str
        The type of interpolation used to estimate rates between the pillars.
    _jacobian: Any
        The Jacobian matrix for the curve, used in sensitivity analysis.
    """
    def __init__(self,
                 reference_date: date,
                 pillars: Union[list[date], list[float]], 
                 rates: list[float],
                 interp: str):
        """
        Initializes a RateCurve instance with the specified attributes.

        Parameters:
        -----------
        reference_date: date
            The base date from which all other dates are measured.
        pillars: Union[list[date], list[float]]
            The list of dates or times to maturity (in years) that define the curve.
        rates: list[float]
            The interest rates corresponding to the pillars.
        interp: str
            The interpolation method used to estimate rates between the pillars.
        """
        self._reference_date = reference_date
        self._daycounter = DayCounter(DayCounterConvention.ActualActual)
        if all(isinstance(d, date) for d in pillars):
            self._dates = pillars
            self._pillars = [self._daycounter.year_fraction(reference_date, d) for d in pillars]
            self._pillar_days = [self._daycounter.day_count(reference_date, d) for d in self._dates]
        elif all(isinstance(d, float) for d in pillars):
            self._dates = [reference_date + timedelta(days=d*365) for d in pillars]
            self._pillars = pillars #[self._daycounter.year_fraction(reference_date, d) for d in self._dates]
            self._pillar_days = [self._daycounter.day_count(reference_date, d) for d in self._dates]
        else:
            raise TypeError("Pillars must be a list of either date or float types")
        self.__rates = rates                                                    # float 
        self._rates = [tf.Variable(r, dtype=dtypes.float64) for r in rates]     # tensor 
        self.interpolation_type = interp
        if self.interpolation_type == "LINEAR":
            self.interp = LinearInterp(self._pillars, self._rates)
        else:
            raise ValueError("Unsupported interpolation type")
        
        self._jacobian = None

    @classmethod
    def from_zcb(cls,
                 reference_date: date,
                 pillars: Union[list[date], list[int]],
                 discount_factors: list[float],
                 interp: str):
        """
        Creates a RateCurve instance from zero-coupon bond (ZCB) discount factors.

        Parameters:
        -----------
        reference_date: date
            The base date from which all other dates are measured.
        pillars: Union[list[date], list[float]]
            The list of dates or times to maturity (in years) that define the curve.
        discount_factors: list[float]
            The discount factors corresponding to the pillars.
        interp: str
            The interpolation method used to estimate rates between the pillars.

        Returns:
        -----------
        RateCurve: A new RateCurve instance.
        """
        daycounter = DayCounter(DayCounterConvention.ActualActual)
        if all(isinstance(d, date) for d in pillars):
            dates = pillars
            pillars = [daycounter.year_fraction(reference_date, d) for d in pillars]
        elif all(isinstance(d, float) for d in pillars):
            dates = [reference_date + timedelta(days=d*365) for d in pillars]
            pillars = [daycounter.year_fraction(reference_date, d) for d in dates]
        else:
            raise TypeError("dates must be a list of either date or float types")
        # pillars = [daycounter.year_fraction(reference_date, d) for d in dates]
        rates = [-np.log(df) / tau if tau > 0 else 0 for df, tau in zip(discount_factors, pillars)]
        return cls(reference_date, dates, rates, interp)
    
    @property
    def nodes(self):
        """
        Returns the nodes of the curve, which consist of the dates and corresponding rates.

        Returns:
        -----------
        list[tuple]: A list of tuples where each tuple contains a date and a rate.
        """
        return [(t,r) for t, r in zip(self._dates, self.__rates)]

    @property
    def reference_date(self):
        """
        Returns the reference date of the curve.

        Returns:
        -----------
        date: The reference date.
        """
        return self._reference_date
            
    def discount(self,
                 term: Union[date, float]) -> float:
        """
        Calculates the discount factor for a given term.

        Parameters:
        -----------
        term: Union[date, float]
            The term for which the discount factor is calculated. This can be either a 
            date or a time to maturity in years.

        Returns:
        -----------
        float: The discount factor for the given term.
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
            df = tf.exp(nrt)
            return df
        if term >= self._pillars[-1]:
            nrt = -term * self._rates[-1]
            df = tf.exp(nrt)
            return df
        else:
            nrt = -term * self.interp.interpolate(term)
            df = tf.exp(nrt)
            return df
    
    def zero_rate(self,
                  term: Union[date, float]) -> float:
        """
        Calculates the zero-coupon rate for a given term.

        Parameters:
        -----------
        term: Union[date, float]
            The term for which the zero-coupon rate is calculated. This can be either a 
            date or a time to maturity in years.

        Returns:
        -----------
        float: The zero-coupon rate for the given term.
        """
        if isinstance(term, date):
            term = self._daycounter.year_fraction(self._reference_date, term)
        elif isinstance(term, float):
            pass 
        else:
            raise TypeError("term must be of type date or float")
        P_tT = self.discount(term)
        return (1 - P_tT) / (term*P_tT)

    def forward_rate(self,
                     d1: Union[date, float],
                     d2: Union[date, float]) -> float:
        """
        Calculates the forward rate between two dates or times to maturity.

        Parameters:
        -----------
        d1: Union[date, float]
            The start date or time to maturity.
        d2: Union[date, float]
            The end date or time to maturity.

        Returns:
        -----------
        float: The forward rate between d1 and d2.
        """
        if isinstance(d1, date) and isinstance(d2, date): #TODO aggiungere anche daycounter
            tau = self._daycounter.year_fraction(d1, d2)
            df1 = self.discount(self._daycounter.year_fraction(self._reference_date, d1))
            df2 = self.discount(self._daycounter.year_fraction(self._reference_date, d2))
        elif isinstance(d1, float) and isinstance(d2, float):
            tau = d2 - d1
            df1 = self.discount(d1)
            df2 = self.discount(d2)
        else:
            raise TypeError("d1 e d2 devono essere entrambi date oppure entrambi float")

        return (df1 / df2 - 1) / tau
    
    def inst_fwd(self,
                 t: float):
        """
        Calculates the instantaneous forward rate at a given time.

        Parameters:
        -----------
        t: float
            The time to maturity at which the instantaneous forward rate is calculated.

        Returns:
        -----------
        float: The instantaneous forward rate at time t.
        """
        # time-step needed for differentiation
        dt = 0.01
        expr = - (tf.math.log(self.discount(t + dt)) - tf.math.log(self.discount(t - dt))) / (2 * dt)
        return expr
    
    def _set_rates(self,
                   rates: list[float]) -> None: 
        """
        Sets new rates for the curve and updates the interpolation function.

        Parameters:
        -----------
        rates: list[float]
            The new rates to be set on the curve.
        """
        self.__rates = rates 
        self._rates = [tf.Variable(r, dtype=dtypes.float64) for r in rates]
        self.interp.y = self._rates

    @property
    def jacobian(self) -> np.ndarray:
        """
        Returns the Jacobian matrix of the curve.

        The Jacobian matrix is used in sensitivity analysis and represents the 
        derivatives of the curve's outputs with respect to its inputs.

        Returns:
        -----------
        np.ndarray: The Jacobian matrix of the curve.
        """
        return self._jacobian

    @jacobian.setter
    def jacobian(self, value: np.ndarray):
        """
        Sets the Jacobian matrix of the curve.

        Parameters:
        -----------
        value: np.ndarray
            The Jacobian matrix to be set.
        """
        self._jacobian = value

