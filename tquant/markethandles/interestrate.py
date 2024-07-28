import math 
from ..timehandles.utils import CompoundingType, Frequency
from ..timehandles.daycounter import DayCounter

class InterestRate:
    
    def __init__(self,
                 r: float,
                 daycounter: DayCounter,
                 compounding: CompoundingType, 
                 frequency: Frequency) -> None:
        self._r = r
        self._daycounter = daycounter
        self._compounding = compounding
        self._frequency = frequency.value

    @property
    def rate(self):
        return self._r

    @property
    def daycounter(self):
        return self._daycounter
    
    @property
    def compounding(self):
        return self._compounding

    @property
    def frequency(self):
        return self._frequency
    
    def discount_factor(self, t:float) -> float:
        return 1/self.compound_factor(t)
    
    def compound_factor(self, *args):
        ''' 
        
        Da implementare il daycounter.yearfraction riga 35
        '''
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
        ''' 
        Returns the InterestRate object given a compound factor
        '''
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
            raise ValueError(f"Unknown compounding convention ({int(self.compounding)})")
        return result