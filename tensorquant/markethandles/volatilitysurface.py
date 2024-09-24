from datetime import date 
from ..timehandles.tqcalendar import Calendar
from ..timehandles.daycounter import DayCounter
from ..timehandles.utils import Settings

class VolatilitySurface:

    def __init__(self,
                 reference_date: date,
                 calendar: Calendar, 
                 daycounter: DayCounter,
                 strike: list[float],
                 maturity: list[float],
                 volatility_matrix) -> None:
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
    
    def volatility(self, strike: float, maturity: float):
        """ bi-linear interpolation"""
        # Find the indices of surrounding strike and maturity values
        x1_idx, x2_idx = self._find_bounds(self._strike, strike)
        y1_idx, y2_idx = self._find_bounds(self._maturity, maturity)

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
            return V11 + (V12 - V11) * (maturity - y1) / (y2 - y1)
        elif y1 == y2:
            # Linear interpolation in the x direction (strike)
            return V11 + (V21 - V11) * (strike - x1) / (x2 - x1)

        # Bilinear interpolation formula
        denominator = (x2 - x1) * (y2 - y1)
        value = (V11 * (x2 - strike) * (y2 - maturity) +
                 V21 * (strike - x1) * (y2 - maturity) +
                 V12 * (x2 - strike) * (maturity - y1) +
                 V22 * (strike - x1) * (maturity - y1)) / denominator

        return value
    
    def variance(self, strike, maturity):
        implied_vol = self.volatility(strike, maturity)
        t = self.daycounter.year_fraction(Settings.evaluation_date, maturity)
        return t * implied_vol*implied_vol   

class BlackConstantVolatility(VolatilitySurface):

    def __init__(self, reference_date, calendar, daycounter, volatility) -> None:
        super().__init__(reference_date, calendar, daycounter, 0, 0, volatility)

    def volatility(self, strike, maturity):
        return self._volatility_matrix
    
    # def variance(self, strike, maturity):
    #     implied_vol = self.volatility(strike, maturity)
    #     t = self.daycounter.year_fraction(tq.Settings.evaluation_date, maturity)
    #     return t * implied_vol*implied_vol