from .pricer import Pricer
from ..instruments.swap import InterestRateSwap
from ..instruments.ois import Ois
from ..instruments.swap import Swap
from ..markethandles.utils import SwapType
from .floatingflow import FloatingLegDiscounting, FloatingCouponDiscounting, OisLegDiscounting
from .fixedflow import FixedLegDiscounting, FixedCouponDiscounting
from datetime import date 
from ..timehandles.utils import DayCounterConvention
from ..timehandles.daycounter import DayCounter
import tensorflow as tf
         
class OisPricer(Pricer):
    def __init__(self, curve_map):
        super().__init__()
        self._curve_map = curve_map

    def calculate_price(self,
              product,
              as_of_date: date,
              curves):
        if isinstance(product, Ois):
            ois = product
            try:
                curve_usage = product._index.name
                curve_ccy, curve_tenor = curve_usage.split(":")
                dc = curves[self._curve_map[curve_ccy][curve_tenor]]
            except:
                raise ValueError("Unknown Curve")

            floating_leg_pricer = OisLegDiscounting(ois.floating_leg)
            fixed_leg_pricer = FixedLegDiscounting(ois.fixed_leg)

            pv_flt = floating_leg_pricer.calculate_price(dc, as_of_date)
            pv_fix = fixed_leg_pricer.calculate_price(dc, as_of_date)
            self.pv_flt = pv_flt
            self.pv_fix = pv_fix
            return pv_flt - pv_fix
        else:
            raise TypeError("Wrong product type")

           
class SwapPricer(Pricer):
    def __init__(self, curve_map):
        super().__init__()
        self._curve_map = curve_map

    def calculate_price(self,
                        product,
                        as_of_date: date,
                        curves):
        if isinstance(product, Swap):
            try:
                curve_usage = product.ccy.value + ":ON"
                curve_ccy, curve_tenor = curve_usage.split(":")
                dc = curves[self._curve_map[curve_ccy][curve_tenor]]

                curve_usage = product._index.name
                curve_ccy, curve_tenor = curve_usage.split(":")
                fc = curves[self._curve_map[curve_ccy][curve_tenor]]
            except:
                raise ValueError("Unknown Curve")

            floating_leg_pricer = FloatingLegDiscounting(product.floating_leg)
            fixed_leg_pricer = FixedLegDiscounting(product.fixed_leg)

            pv_flt = floating_leg_pricer.calculate_price(dc, fc, as_of_date)
            pv_fix = fixed_leg_pricer.calculate_price(dc, as_of_date)
            self.pv_flt = pv_flt
            self.pv_fix = pv_fix
            return pv_flt - pv_fix
        else:
            raise TypeError("Wrong product type")

           