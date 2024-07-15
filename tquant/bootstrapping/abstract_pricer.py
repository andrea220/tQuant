import datetime
from abc import abstractmethod
from typing import Dict

from tquant import Product, RateCurve


class AbstractPricer:
    @abstractmethod
    def price(self,
              product: Product,
              trade_date: datetime,
              curves: Dict[str, RateCurve]):
        pass
