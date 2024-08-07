from .pricer import Pricer, AbstractPricerAP
from ..instruments.swap import InterestRateSwap
from ..instruments.ois import Ois, OisAP, OisTest
from ..markethandles.utils import SwapType
from .floatingflow import FloatingLegDiscounting, FloatingCouponDiscounting, OisLegDiscounting
from .fixedflow import FixedLegDiscounting, FixedCouponDiscounting
# from ..flows.fixedcoupon import FixedRateLeg
# from ..flows.floatingcoupon import FloatingRateLeg
from datetime import date 
from ..timehandles.utils import DayCounterConvention
from ..timehandles.daycounter import DayCounter

from datetime import timedelta
import tensorflow as tf
import numpy as np


class SwapAnalyticEngine(Pricer):

    def __init__(self, swap: InterestRateSwap):
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


class OisAnalyticEngine(Pricer):

    def __init__(self, swap: Ois):
        self.swap = swap
        self.floating_leg_pricer = FloatingLegDiscounting(swap.floating_leg)
        self.fixed_leg_pricer = FixedLegDiscounting(swap.fixed_leg)

    def price(self, discount_curve, evaluation_date: date):
        self.npv_float = self.floating_leg_pricer.price(discount_curve, discount_curve, evaluation_date, FloatingCouponDiscounting)
        self.npv_fixed = self.fixed_leg_pricer.price(discount_curve, evaluation_date, FixedCouponDiscounting)
        if self.swap.swap_type == SwapType.Payer:
            self.swap.price =  self.npv_float - self.npv_fixed  
        else:
            self.swap.price =   self.npv_fixed - self.npv_float
        return self.swap.price
    
    def price_aad(self, discount_curve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, evaluation_date)
        return npv, tape

    def implied_rate(self):
        pass



def compute_growing_factor(ois, 
                           fixing_dates: list[date],
                           fixing_rates: list[float],
                           start_date: date,
                           as_of_date: date):
    if not fixing_dates:
        return 0.0
    first = 0
    for i, fixing_date in enumerate(fixing_dates):
        if start_date <= fixing_date < start_date + timedelta(days=1):
            first = i
            break
    gf = 1.0
    for i in range(first, len(fixing_dates) - 1):
        t1 = fixing_dates[i]
        t2 = fixing_dates[i + 1]
        yf = ois.day_counter.year_fraction(t1, t2)
        gf *= 1.0 + fixing_rates[i] * yf
    gf *= 1.0 + fixing_rates[-1] * ois.day_counter.year_fraction(fixing_dates[-1], as_of_date)
    return gf


def calculate_forward(as_of_date: date,
                      fc,
                      index_start: date,
                      index_end: date,
                      yf: float):
    act365 = DayCounterConvention.Actual365
    day_counter = DayCounter(act365)
    t1 = day_counter.year_fraction(as_of_date, index_start)
    t2 = day_counter.year_fraction(as_of_date, index_end)
    df1 = fc.discount(t1)
    df2 = fc.discount(t2)
    forward = (df1 / df2 - 1.0) / yf
    return np.float64(forward)


class OisPricerAP(AbstractPricerAP):
    def __init__(self, curve_assignment):
        super().__init__()
        self.curve_assignment = curve_assignment

    def price(self,
              product,
              as_of_date: date,
              curves):
        if isinstance(product, OisAP):
            ois = product
            discount = {"CCY": ois.ccy, "USAGE": "DISCOUNT"}
            dc_name = self.curve_assignment.get_curve_name(discount)
            dc = curves[dc_name]
            act365 = DayCounterConvention.Actual365
            day_counter = DayCounter(act365)
            pv_fix = 0.0
            for i in range(len(ois.pay_dates_fix)):
                pay_date = ois.pay_dates_fix[i]
                if pay_date > as_of_date:
                    yf = ois.day_counter_fix.year_fraction(ois.start_dates_fix[i], ois.end_dates_fix[i])
                    cashflow = ois.notional * ois.quote * yf
                    pv_fix += cashflow * dc.discount(day_counter.year_fraction(as_of_date, pay_date))

            pv_flt = 0.0
            for i in range(len(ois.pay_dates_flt)):
                pay_date = ois.pay_dates_flt[i]
                if pay_date > as_of_date:
                    yf = ois.day_counter_flt.year_fraction(ois.start_dates_flt[i],
                                                           ois.end_dates_flt[i])
                    if ois.start_dates_flt[i] < as_of_date:
                        growing_factor1 = compute_growing_factor(ois,
                                                                ois.fixing_dates,
                                                                 ois.fixing_rates,
                                                                 ois.start_dates_flt[i],
                                                                 as_of_date)
                        growing_factor2 = 1.0 / dc.discount(
                            day_counter.year_fraction(as_of_date, ois.end_dates_flt[i]))
                        rate = (growing_factor1 * growing_factor2 - 1.0) / yf
                    else:
                        rate = calculate_forward(as_of_date,
                                                 dc,
                                                 ois.start_dates_flt[i],
                                                 ois.end_dates_flt[i],
                                                 yf)
                    cashflow = ois.notional * rate * yf
                    pv_flt += cashflow * dc.discount(day_counter.year_fraction(as_of_date, pay_date))

            return pv_flt - pv_fix
        else:
            raise TypeError("Wrong product type")


class OisPricerTest(AbstractPricerAP):
    def __init__(self, curve_name):
        super().__init__()
        self.curve_name = curve_name

    def price(self,
              product,
              as_of_date: date,
              curves):
        if isinstance(product, OisAP):
            ois = product
            dc = curves[self.curve_name]
            act365 = DayCounterConvention.Actual365
            day_counter = DayCounter(act365)
            pv_fix = 0.0
            for i in range(len(ois.pay_dates_fix)):
                pay_date = ois.pay_dates_fix[i]
                if pay_date > as_of_date:
                    yf = ois.day_counter_fix.year_fraction(ois.start_dates_fix[i], ois.end_dates_fix[i])
                    cashflow = ois.notional * ois.quote * yf
                    pv_fix += cashflow * dc.discount(day_counter.year_fraction(as_of_date, pay_date))

            pv_flt = 0.0
            for i in range(len(ois.pay_dates_flt)):
                pay_date = ois.pay_dates_flt[i]
                if pay_date > as_of_date:
                    yf = ois.day_counter_flt.year_fraction(ois.start_dates_flt[i],
                                                           ois.end_dates_flt[i])
                    if ois.start_dates_flt[i] < as_of_date:
                        growing_factor1 = compute_growing_factor(ois,
                                                                ois.fixing_dates,
                                                                 ois.fixing_rates,
                                                                 ois.start_dates_flt[i],
                                                                 as_of_date)
                        growing_factor2 = 1.0 / dc.discount(
                            day_counter.year_fraction(as_of_date, ois.end_dates_flt[i]))
                        rate = (growing_factor1 * growing_factor2 - 1.0) / yf
                    else:
                        rate = dc.forward_rate(ois.start_dates_flt[i],
                                               ois.end_dates_flt[i],
                                               ois.day_counter_flt,
                                               as_of_date)
                        
                    cashflow = ois.notional * rate * yf
                    pv_flt += cashflow * dc.discount(day_counter.year_fraction(as_of_date, pay_date))

            return pv_flt - pv_fix
        else:
            raise TypeError("Wrong product type")


class OisPricerTestLegs(AbstractPricerAP):
    def __init__(self, curve_name):
        super().__init__()
        self.curve_name = curve_name

    def price(self,
              product,
              as_of_date: date,
              curves):
        if isinstance(product, OisTest):
            ois = product
            dc = curves[self.curve_name]

            floating_leg_pricer = OisLegDiscounting(ois.floating_leg)
            fixed_leg_pricer = FixedLegDiscounting(ois.fixed_leg)

            pv_flt = floating_leg_pricer.price(dc, as_of_date)
            pv_fix = fixed_leg_pricer.price(dc, as_of_date)#, tq.FixedCouponDiscounting)


            # pv_fix = 0.0
            # for i in range(len(ois.pay_dates_fix)):
            #     pay_date = ois.pay_dates_fix[i]
            #     if pay_date > as_of_date:
            #         yf = ois.day_counter_fix.year_fraction(ois.start_dates_fix[i], ois.end_dates_fix[i])
            #         cashflow = ois.notional * ois.quote * yf
            #         pv_fix += cashflow * dc.discount(ois.day_counter_fix.year_fraction(as_of_date, pay_date))

            # pv_flt = 0.0
            # for i in range(len(ois.pay_dates_flt)):
            #     pay_date = ois.pay_dates_flt[i]
            #     if pay_date > as_of_date:
            #         yf = ois.day_counter_flt.year_fraction(ois.start_dates_flt[i],
            #                                                ois.end_dates_flt[i])
            #         if ois.start_dates_flt[i] < as_of_date:
            #             growing_factor1 = compute_growing_factor(ois,
            #                                                     ois.fixing_dates,
            #                                                      ois.fixing_rates,
            #                                                      ois.start_dates_flt[i],
            #                                                      as_of_date)
            #             growing_factor2 = 1.0 / dc.discount(
            #                 ois.day_counter_flt.year_fraction(as_of_date, ois.end_dates_flt[i]))
            #             rate = (growing_factor1 * growing_factor2 - 1.0) / yf
            #         else:
            #             rate = dc.forward_rate(ois.start_dates_flt[i],
            #                                    ois.end_dates_flt[i],
            #                                    ois.day_counter_flt,
            #                                    as_of_date)
                        
            #         cashflow = ois.notional * rate * yf
            #         pv_flt += cashflow * dc.discount(ois.day_counter_flt.year_fraction(as_of_date, pay_date))

            self.pv_flt = pv_flt
            self.pv_fix = pv_fix
            return pv_flt - pv_fix
        else:
            raise TypeError("Wrong product type")

    def price_aad(self, product,
              as_of_date: date,
              curves):
        with tf.GradientTape() as tape:
            npv = self.price(product, as_of_date, curves)
        return npv, tape
           
class OisPricer(Pricer):
    def __init__(self, curve_map):
        super().__init__()
        self._curve_map = curve_map

    def price(self,
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

            pv_flt = floating_leg_pricer.price(dc, as_of_date)
            pv_fix = fixed_leg_pricer.price(dc, as_of_date)
            self.pv_flt = pv_flt
            self.pv_fix = pv_fix
            return pv_flt - pv_fix
        else:
            raise TypeError("Wrong product type")

    def price_aad(self, product,
              as_of_date: date,
              curves):
        with tf.GradientTape() as tape:
            npv = self.price(product, as_of_date, curves)
        return npv, tape
           