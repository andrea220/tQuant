import math
from ..timehandles.utils import CompoundingType, Frequency
from ..timehandles.daycounter import DayCounter


class InterestRate:
    """
    Represents an interest rate with associated day count convention, compounding type, and frequency.

    Attributes:
        _r (float): The interest rate.
        _daycounter (DayCounter): Day count convention for year fraction calculations.
        _compounding (CompoundingType): The compounding type (e.g., Simple, Compounded, Continuous).
        _frequency (int): Frequency of compounding per year.
    """

    def __init__(
        self,
        r: float,
        daycounter: DayCounter,
        compounding: CompoundingType,
        frequency: Frequency,
    ) -> None:
        """
        Initializes the InterestRate with the given rate, day count convention, compounding type, and frequency.

        Args:
            r (float): The interest rate.
            daycounter (DayCounter): The day count convention to use for time calculations.
            compounding (CompoundingType): The type of compounding (e.g., Simple, Compounded, Continuous).
            frequency (Frequency): The frequency of compounding (e.g., annually, semi-annually).

        """
        self._r = r
        self._daycounter = daycounter
        self._compounding = compounding
        self._frequency = frequency.value

    @property
    def rate(self) -> float:
        """
        Returns the interest rate.

        Returns:
            float: The interest rate.
        """
        return self._r

    @property
    def daycounter(self) -> DayCounter:
        """
        Returns the day counter used.

        Returns:
            DayCounter: The day counter.
        """
        return self._daycounter

    @property
    def compounding(self) -> CompoundingType:
        """
        Returns the compounding type.

        Returns:
            CompoundingType: The compounding type (e.g., Simple, Compounded, Continuous).
        """
        return self._compounding

    @property
    def frequency(self) -> int:
        """
        Returns the frequency of compounding.

        Returns:
            int: The frequency of compounding per year.
        """
        return self._frequency

    def discount_factor(self, t: float) -> float:
        """
        Calculates the discount factor for a given time period.

        Args:
            t (float): The time period in years.

        Returns:
            float: The discount factor.
        """
        return 1 / self.compound_factor(t)

    def compound_factor(self, *args):
        """
        Calculates the compound factor over a given time period or between two dates.

        Args:
            *args: A single argument (t, float) representing the time in years, or four arguments (d1, d2, refStart, refEnd)
                   where d1 and d2 are the start and end dates, and refStart and refEnd are optional reference dates.

        Returns:
            float: The compound factor.

        Raises:
            ValueError: If the number of arguments is invalid or if negative time is provided.
        """
        if len(args) == 1:
            t = args[0]
        elif len(args) == 4:
            d1, d2, refStart, refEnd = args
            t = self._daycounter.year_fraction(d1, d2)
            return self.compound_factor(t)
        else:
            raise ValueError("Invalid number of arguments")

        if t < 0.0:
            raise ValueError(f"Negative time ({t}) not allowed")
        if self._r is None:
            raise ValueError("Null interest rate")

        if self.compounding == CompoundingType.Simple:
            return 1.0 + self._r * t
        elif self.compounding == CompoundingType.Compounded:
            return math.pow(1.0 + self._r / self.frequency, self.frequency * t)
        elif self.compounding == CompoundingType.Continuous:
            return math.exp(self._r * t)
        else:
            raise ValueError("Unknown compounding convention")

    @staticmethod
    def implied_rate(compound, daycounter, comp, freq, t):
        """
        Calculates the implied interest rate from a given compound factor.

        Args:
            compound (float): The compound factor.
            daycounter (DayCounter): The day count convention for time calculations.
            comp (CompoundingType): The compounding type (e.g., Simple, Compounded, Continuous).
            freq (Frequency): The compounding frequency per year.
            t (float): The time period in years.

        Returns:
            InterestRate: The implied InterestRate object.

        Raises:
            ValueError: If the compound factor is non-positive, or if an invalid time period is provided.
        """
        if compound <= 0.0:
            raise ValueError("Positive compound factor required")

        r = None
        if compound == 1.0:
            if t < 0.0:
                raise ValueError(f"Non-negative time ({t}) required")
            r = 0.0
        else:
            if t <= 0.0:
                raise ValueError(f"Positive time ({t}) required")
            if comp == CompoundingType.Simple:
                r = (compound - 1.0) / t
            elif comp == CompoundingType.Compounded:
                r = (math.pow(compound, 1.0 / (freq * t)) - 1.0) * freq
            elif comp == CompoundingType.Continuous:
                r = math.log(compound) / t
            else:
                raise ValueError(f"Unknown compounding convention ({int(comp)})")
        return InterestRate(r, daycounter, comp, freq)

    def __str__(self):
        """
        Returns a string representation of the InterestRate object, including the rate and compounding type.

        Returns:
            str: A string representing the interest rate, its compounding type, and frequency (if applicable).

        Raises:
            ValueError: If the compounding type is unknown.
        """
        if self._r is None:
            return "null interest rate"
        result = f"{self.rate:.6f}"
        if self.compounding == CompoundingType.Simple:
            result += " simple compounding"
        elif self.compounding == CompoundingType.Compounded:
            result += f" {self.frequency} compounding"
        elif self.compounding == CompoundingType.Continuous:
            result += " continuous compounding"
        else:
            raise ValueError(
                f"Unknown compounding convention ({int(self.compounding)})"
            )
        return result
