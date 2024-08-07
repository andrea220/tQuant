from datetime import date
from .product import Product
from ..index.index import Index
from ..timehandles.utils import DayCounterConvention
from ..timehandles.tqcalendar import Calendar
from ..markethandles.utils import Currency

class Fra(Product):
    def __init__(self,
                 ccy: Currency,
                 start_date: date,
                 end_date: date,
                 notional: float,
                 quote: float,
                 day_counter: DayCounterConvention,
                 index: Index):
        super().__init__(ccy, start_date, end_date, quote)
        self.start_date = start_date
        self.end_date = end_date
        self.notional = notional
        self.day_counter = day_counter
        self.index = index
