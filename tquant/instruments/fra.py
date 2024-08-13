from datetime import date
from .product import Product
from ..index.index import Index
from ..timehandles.tqcalendar import Calendar
from ..timehandles.utils import DayCounterConvention
from ..markethandles.utils import Position


class ForwardRateAgreement(Product): #TODO da eliminare, deprecata
    def __init__(self, 
                start_date: date,
                maturity_date: date,
                position: Position,
                rate: float,
                notional: float,
                index: Index,
                day_counter: DayCounterConvention,
                calendar: Calendar) -> None:
        super().__init__("currency", start_date, maturity_date)
        self._start_date = start_date
        self._maturity_date = maturity_date
        self._position = position
        self._rate = rate
        self._notional = notional
        self._index = index 
        self._day_counter = day_counter
        self._calendar = calendar

    @property
    def accrual(self):
        return self._day_counter.year_fraction(self._start_date, self._maturity_date)
    
    @property
    def fixing_date(self):
        return self._index.fixing_date(self.start_date)
