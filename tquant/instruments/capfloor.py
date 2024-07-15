from datetime import date
from ..interface.trade import Trade
from ..flows.floatingleg import FloatingRateLeg
from ..interface.index import Index
from ..interface.tqcalendar import Calendar
from ..utilities.utils import DayCounterConvention, Position


class IrOption(Trade):
    def __init__(self, 
                 leg: FloatingRateLeg,
                 strike: float
                ) -> None:
        super().__init__()
        self.leg = leg
        self.strike = strike

    @property
    def start_date(self):
        pass

    @property
    def maturity_date(self):
        pass 

    @property
    def implied_volatility(self):
        ''' Black implied vol '''
        pass 

    @property
    def atm_rate(self):
        pass 

    @property
    def is_expired(self):
        pass


class Cap(IrOption):
    def __init__(self,
                 leg: FloatingRateLeg,
                 strike: float) -> None:
        super().__init__(leg, strike)


class Floor(IrOption):
    def __init__(self,
                 leg: FloatingRateLeg,
                 strike: float) -> None:
        super().__init__(leg, strike)


class Collar(IrOption):
    def __init__(self,
                 leg: FloatingRateLeg,
                 strike: float) -> None:
        super().__init__(leg, strike)