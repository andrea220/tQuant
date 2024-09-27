from .pricer import Pricer

from ..instruments.ois import Ois
from ..instruments.swap import Swap
from ..markethandles.utils import SwapType
from .floatingflow import (
    FloatingLegDiscounting,
    OisLegDiscounting,
)
from .fixedflow import FixedLegDiscounting  # , FixedCouponDiscounting
from datetime import date

# from ..timehandles.utils import DayCounterConvention
# from ..timehandles.daycounter import DayCounter


class OisPricer(Pricer):
    def __init__(self, market_map):
        super().__init__()
        self._market_map = market_map

    def calculate_price(self, product: Ois, market: dict):
        if isinstance(product, Ois):
            try:
                # get single curve for ois valuation and discounting
                curve_usage = product._index.name
                curve_ccy, curve_tenor = curve_usage.split(":")
                product.discount_curve = self._market_map[f"IR:{curve_ccy}"][
                    curve_tenor
                ]
                product.estimation_curve = product.discount_curve
                dc = market[product.discount_curve]
            except:
                raise ValueError("Unknown Curve")

            floating_leg_pricer = OisLegDiscounting(product.floating_leg)
            fixed_leg_pricer = FixedLegDiscounting(product.fixed_leg)

            if product.swap_type == SwapType.Payer:
                product.floating_leg.price = floating_leg_pricer.calculate_price(dc)
                product.fixed_leg.price = -fixed_leg_pricer.calculate_price(dc)
            else:
                product.floating_leg.price = -floating_leg_pricer.calculate_price(dc)
                product.fixed_leg.price = fixed_leg_pricer.calculate_price(dc)
            return product.floating_leg.price + product.fixed_leg.price
        else:
            raise TypeError("Wrong product type")


class SwapPricer(Pricer):
    def __init__(self, market_map):
        super().__init__()
        self._market_map = market_map

    def calculate_price(self, product: Swap, market: dict):
        if isinstance(product, Swap):
            try:
                curve_usage = product.ccy.value + ":ON"
                curve_ccy, curve_tenor = curve_usage.split(":")
                product.discount_curve = self._market_map[f"IR:{curve_ccy}"][
                    curve_tenor
                ]
                dc = market[product.discount_curve]

                curve_usage = product._index.name
                curve_ccy, curve_tenor = curve_usage.split(":")
                product.estimation_curve = self._market_map[f"IR:{curve_ccy}"][
                    curve_tenor
                ]
                fc = market[product.estimation_curve]

            except:
                raise ValueError("Unknown Curve")

            floating_leg_pricer = FloatingLegDiscounting(product.floating_leg)
            fixed_leg_pricer = FixedLegDiscounting(product.fixed_leg)
            if product.swap_type == SwapType.Payer:
                product.floating_leg.price = floating_leg_pricer.calculate_price(dc, fc)
                product.fixed_leg.price = -fixed_leg_pricer.calculate_price(dc)
            else:
                product.floating_leg.price = -floating_leg_pricer.calculate_price(
                    dc, fc
                )
                product.fixed_leg.price = fixed_leg_pricer.calculate_price(dc)

            return product.floating_leg.price + product.fixed_leg.price
        else:
            raise TypeError("Wrong product type")
