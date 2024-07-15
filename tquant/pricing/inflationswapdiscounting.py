from ..interface.pricer import Pricer
from ..instruments.inflationswap import InflationSwap
from .inflationflow import InflationLegDiscounting, InflationCouponDiscounting
from .fixedflow import FixedLegDiscounting, FixedCouponDiscounting
from datetime import date 
import tensorflow as tf


class InflationSwapAnalyticEngine(Pricer):

    def __init__(self, inflation_swap: InflationSwap) -> None:
        self.inflation_swap = inflation_swap
        self.inflation_leg_pricer = InflationLegDiscounting(inflation_swap.inflation_leg)
        self.fixed_leg_pricer = FixedLegDiscounting(inflation_swap.fixed_leg)

    def price(self, discount_curve, estimation_curve, evaluation_date: date):
        npv_inflation = self.inflation_leg_pricer.price(discount_curve, estimation_curve, evaluation_date, InflationCouponDiscounting)
        npv_fixed = self.fixed_leg_pricer.price(discount_curve, evaluation_date, FixedCouponDiscounting)
        return self.inflation_swap.sign*(npv_fixed - npv_inflation)
    
    def price_aad(self, discount_curve, estimation_curve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, estimation_curve, evaluation_date)
        return npv, tape

    def implied_rate(self):
        pass
