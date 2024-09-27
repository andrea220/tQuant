from .pricer import Pricer
from ..instruments.forward import Fra
from ..timehandles.utils import TimeUnit, BusinessDayConvention, Settings


class FraPricer(Pricer):
    def __init__(self, market_map):
        super().__init__()
        self._market_map = market_map

    def calculate_price(self, product: Fra, market: dict):
        if isinstance(product, Fra):
            try:
                # get discount curve for product ccy
                curve_usage = product.ccy.value + ":ON"
                curve_ccy, curve_tenor = curve_usage.split(":")
                dc = market[self._market_map[f"IR:{curve_ccy}"][curve_tenor]]
                # get forward curve for the index
                curve_usage = product._index.name
                curve_ccy, curve_tenor = curve_usage.split(":")
                fc = market[self._market_map[f"IR:{curve_ccy}"][curve_tenor]]
            except:
                raise ValueError("Unknown Curve")

            pv = 0.0
            if product.start_date > Settings.evaluation_date:
                accrual = product.day_counter.year_fraction(
                    product.start_date, product.end_date
                )
                fixing_d = product.fixing_date
                d1 = product._index.fixing_calendar.advance(
                    fixing_d, 2, TimeUnit.Days, BusinessDayConvention.ModifiedFollowing
                )  # valuedate-start date
                d2 = product._index.fixing_maturity(d1)
                t = product._index.daycounter.year_fraction(d1, d2)
                disc1 = fc.discount(d1)
                disc2 = fc.discount(d2)
                fwd = (disc1 / disc2 - 1) / t
                self.fwd = fwd
                pv += (
                    product.notional
                    * accrual
                    * (fwd - product.fixed_rate)
                    * product.side.value
                    * dc.discount(
                        product.day_counter.year_fraction(
                            Settings.evaluation_date, product.end_date
                        )
                    )
                )
            return pv / (1 + fwd * accrual)
        else:
            raise TypeError("Wrong product type")
