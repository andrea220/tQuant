from datetime import date
from ..interface.trade import Trade
from ..interface.index import Index
from ..interface.tqcalendar import Calendar
from ..utilities.utils import DayCounterConvention, Position


class ForwardRateAgreement(Trade):
    def __init__(self, 
                start_date: date,
                maturity_date: date,
                position: Position,
                rate: float,
                notional: float,
                index: Index,
                day_counter: DayCounterConvention,
                calendar: Calendar) -> None:
        super().__init__()
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

    # @property 
    # def pricing_engine(self):
    #     if self._pricing_engine is not None:
    #         return self._pricing_engine
    #     else:
    #         raise ValueError("You must set a valid Pricing-Engine")
        
    # @pricing_engine.setter
    # def pricing_engine(self, value: FraDiscountingEngine):
    #     self._pricing_engine = value

    # def price(self):
    #     return self._pricing_engine.price()

    # def price_aad(self):
    #     return self._pricing_engine.price_aad()

    # def implied_forward(self):
    #     return self._pricing_engine.implied_rate()