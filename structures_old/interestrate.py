import math 
from structures_old.compounding import *

class InterestRate:
    
    def __init__(self,
                 r: float,
                 compounding: Compounding,
                 frequency: Frequency) -> None:
        self._r = r
        self._compounding = compounding
        self._frequency = frequency.value

    @property
    def rate(self):
        return self._r

    @property
    def compounding(self):
        return self._compounding

    @property
    def frequency(self):
        return self._frequency
    
    def compoundFactor(self, *args):
        ''' 
        
        ##WARNING## Da implementare il daycounter.yearfraction riga 35
        '''
        if len(args) == 1:
            t = args[0]
        elif len(args) == 4:
            d1, d2, refStart, refEnd = args
            tau_i = (d2 - d1)
            t = tau_i.days/365 #dc_.yearFraction(d1, d2, refStart, refEnd)
            return self.compoundFactor(t)
        else:
            raise ValueError("Invalid number of arguments")
        
        if t < 0.0:
            raise ValueError(f"Negative time ({t}) not allowed")
        if self._r is None:
            raise ValueError("Null interest rate")

        if self.compounding == Compounding.Simple:
            return 1.0 + self._r * t
        elif self.compounding == Compounding.Compounded:
            return math.pow(1.0 + self._r / self.frequency, self.frequency * t)
        elif self.compounding == Compounding.Continuous:
            return math.exp(self._r * t)
        else:
            raise ValueError("Unknown compounding convention")
        
    
    @staticmethod
    def impliedRate(compound, comp, freq, t):
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
            if comp == Compounding.Simple:
                r = (compound - 1.0) / t
            elif comp == Compounding.Compounded:
                r = (math.pow(compound, 1.0 / (freq * t)) - 1.0) * freq
            elif comp == Compounding.Continuous:
                r = math.log(compound) / t
            else:
                raise ValueError(f"Unknown compounding convention ({int(comp)})")
        return InterestRate(r, comp, freq)
    

    def __str__(self):
        if self._r is None:
            return "null interest rate"
        result = f"{self.rate:.6f}"
        if self.compounding == Compounding.Simple:
            result += " simple compounding"
        elif self.compounding == Compounding.Compounded:
            result += f" {self.frequency} compounding"
        elif self.compounding == Compounding.Continuous:
            result += " continuous compounding"
        else:
            raise ValueError(f"Unknown compounding convention ({int(self.compounding)})")
        return result