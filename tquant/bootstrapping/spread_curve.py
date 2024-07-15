import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

from tquant import RateCurve


class SpreadCurve(RateCurve):
    def __init__(self, pillars: list[float], rates: list[float], base_curve: RateCurve):
        super().__init__(pillars, rates)
        self.base_curve = base_curve

    def discount(self, term):
        base_df = self.base_curve.discount(term)
        df = super().discount(term)
        return base_df * df


