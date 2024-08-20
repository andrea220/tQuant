# DEPRECATED

from .pricer import Pricer
from ..instruments.cds import CreditDefaultSwap
from .defaultflow import DefaultLegDiscount, DefaultCouponDiscounting
from .premiumflow import PremiumLegDiscount, PremiumCouponDiscounting
from datetime import date 
import tensorflow as tf


class CDSAnalyticEngine(Pricer):

    def __init__(self, cds: CreditDefaultSwap) -> None:
        self.cds = cds
        self.default_leg_pricer = DefaultLegDiscount(cds.default_leg)
        self.premium_leg_pricer = PremiumLegDiscount(cds.premium_leg)

    def price(self, discount_curve, estimation_curve, evaluation_date: date):
        npv_default = self.default_leg_pricer.price(discount_curve, estimation_curve, evaluation_date, DefaultCouponDiscounting)
        npv_premium = self.premium_leg_pricer.price(discount_curve, estimation_curve, evaluation_date, PremiumCouponDiscounting)
        return self.cds.sign*(-npv_premium + npv_default)
    
    def price_aad(self, discount_curve, estimation_curve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, estimation_curve, evaluation_date)
        return npv, tape

    def implied_rate(self):
        pass
