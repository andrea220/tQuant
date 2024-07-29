from .pricer import Pricer
from ..instruments.swap import InterestRateSwap
from ..markethandles.utils import SwapType
from .floatingflow import FloatingLegDiscounting, FloatingCouponDiscounting
from .fixedflow import FixedLegDiscounting, FixedCouponDiscounting
from datetime import date 


import tensorflow as tf


class SwapAnalyticEngine(Pricer):

    def __init__(self, swap: InterestRateSwap) -> None:
        self.swap = swap
        self.floating_leg_pricer = FloatingLegDiscounting(swap.floating_leg)
        self.fixed_leg_pricer = FixedLegDiscounting(swap.fixed_leg)

    def price(self, discount_curve, estimation_curve, evaluation_date: date):
        self.npv_float = self.floating_leg_pricer.price(discount_curve, estimation_curve, evaluation_date, FloatingCouponDiscounting)
        self.npv_fixed = self.fixed_leg_pricer.price(discount_curve, evaluation_date, FixedCouponDiscounting)
        if self.swap.swap_type == SwapType.Payer:
            self.swap.price =  self.npv_float - self.npv_fixed  
        else:
            self.swap.price =   self.npv_fixed - self.npv_float
        return self.swap.price
    
    def price_aad(self, discount_curve, estimation_curve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, estimation_curve, evaluation_date)
        return npv, tape

    def implied_rate(self):
        pass

