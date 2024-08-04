from datetime import date 
from .ircurve import RateCurve
from ..instruments.product import Product
from ..pricers.pricer import Pricer
import numpy as np



class ObjectiveFunction:
    def __init__(self,
                 trade_date: date,
                 rate_curve: RateCurve,
                 products: list[Product],
                 pricers: list[Pricer],
                 curves: dict[str, RateCurve]):
        self.trade_date = trade_date
        self.rate_curve = rate_curve
        self.products = products
        self.pricers = pricers
        self.curves = curves

    def __call__(self, x):
        self.rate_curve.set_rates(x)
        res = np.zeros(len(self.pricers))
        jac = np.zeros((len(x), len(x)))
        for i, (pricer, product) in enumerate(zip(self.pricers, self.products)):
            pv, tape = pricer.price_aad(product, self.trade_date, self.curves)
            gradients = tape.gradient(pv, [self.rate_curve.rates])
            res[i] = pv
            jac[i,:] = np.array(gradients[0])
        return res, np.nan_to_num(jac, nan=0.0)