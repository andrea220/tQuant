from abc import ABC, abstractmethod
from ..markethandles.utils import Currency
from datetime import date 

# class ProductAP(ABC):
#     def __init__(self,
#                  ccy: str,
#                  start_date: date,
#                  maturity: date,
#                  quote: float):
#         self.ccy = ccy
#         self.start_date = start_date
#         self.maturity = maturity
#         self.quote = quote

class Product(ABC):
    """
    An abstract base class representing a generic financial product.

    This class serves as a foundation for various financial products, providing 
    essential attributes such as currency, start date, maturity date, and quote. 
    Specific financial products, such as deposits, bonds, or derivatives, can inherit 
    from this class and implement additional functionality.
    """
    def __init__(self,
                 ccy: Currency,
                 start_date: date,
                 maturity: date,
                 quote: float):
        """
        Initializes a Product instance with the specified attributes.

        Parameters:
        -----------
        ccy: Currency
            The currency in which the product is denominated.
        start_date: date
            The date on which the product starts or the initial date for the product's 
            lifecycle.
        maturity: date
            The date on which the product matures or the final date for the product's 
            lifecycle.
        quote: float
            The quoted value for the product, which could represent an interest rate, 
            price, or other relevant metric.
        """
        self.ccy = ccy
        self.start_date = start_date
        self.maturity = maturity
        self.quote = quote

