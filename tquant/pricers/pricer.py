from abc import ABC, abstractmethod
import tensorflow as tf
from datetime import date
from ..instruments.product import Product

class Pricer(ABC):
    @abstractmethod
    def calculate_price(self,
                        product,
                        trade_date,
                        curves):
        return 

    def price(self,
              product: Product,
              trade_date: date,
              market: dict,
              autodiff: bool=False):
        if autodiff:
            with tf.GradientTape() as tape:
                npv = self.calculate_price(product, trade_date, market)
            return npv, tape
        else:
            return self.calculate_price(product, trade_date, market)
