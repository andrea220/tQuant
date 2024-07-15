import datetime

from abc import ABC


class Product(ABC):
    def __init__(self,
                 ccy: str,
                 start_date: datetime,
                 maturity: datetime,
                 quote: float):
        self.ccy = ccy
        self.start_date = start_date
        self.maturity = maturity
        self.quote = quote
