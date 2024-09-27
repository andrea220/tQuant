import datetime

from .product import Product
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency, SwapType
from ..index.curverateindex import OvernightIndex
from ..flows.fixedcoupon import FixedRateLeg
from ..flows.floatingcoupon import FloatingRateLeg


class Ois(Product):

    def __init__(
        self,
        ccy: Currency,
        start_date: datetime.date,
        end_date: datetime.date,
        start_dates_fix: list[datetime.date],
        end_dates_fix: list[datetime.date],
        pay_dates_fix: list[datetime.date],
        start_dates_flt: list[datetime.date],
        end_dates_flt: list[datetime.date],
        pay_dates_flt: list[datetime.date],
        fixed_rate: float,
        notional: float,
        day_counter_fix: DayCounter,
        day_counter_flt: DayCounter,
        index: OvernightIndex,
        swap_type: SwapType = SwapType.Payer,
    ):

        super().__init__(ccy, start_date, end_date)

        self._notional = notional
        self._day_counter_fix = day_counter_fix
        self._day_counter_flt = day_counter_flt

        self._fixed_rate = fixed_rate
        self._notionals = [notional] * len(pay_dates_fix)
        self._rates = [fixed_rate] * len(pay_dates_fix)
        self._gearings = [1] * len(pay_dates_flt)
        self._margins = [0] * len(pay_dates_flt)
        self._index = index
        self._swap_type = swap_type

        self._fixed_leg = FixedRateLeg(
            pay_dates_fix,
            start_dates_fix,
            end_dates_fix,
            self._notionals,
            self._rates,
            day_counter_fix,
        )
        self._floating_leg = FloatingRateLeg(
            pay_dates_flt,
            start_dates_flt,
            end_dates_flt,
            self._notionals,
            self._gearings,
            self._margins,
            index,
            day_counter_flt,
        )

        self._discount_curve = None
        self._estimation_curve = None

    @property
    def notional(self) -> float:
        """
        Get the notional amount for the swap.

        Returns:
            float: The notional amount.
        """
        return self._notional

    @property
    def swap_type(self) -> SwapType:
        """
        Get the swap type, either Payer or Receiver.

        Returns:
            SwapType: The type of the swap.
        """
        return self._swap_type

    @property
    def fixed_leg(self) -> FixedRateLeg:
        """
        Get the fixed leg of the OIS.

        Returns:
            FixedRateLeg: The fixed leg of the swap.
        """
        return self._fixed_leg

    @property
    def floating_leg(self) -> FloatingRateLeg:
        """
        Get the floating leg of the OIS.

        Returns:
            FloatingRateLeg: The floating leg of the swap.
        """
        return self._floating_leg

    @property
    def fixed_rate(self):
        return self._fixed_rate

    @property
    def discount_curve(self) -> str:
        if self._discount_curve is None:
            raise ValueError("you must define a pricer")
        return self._discount_curve

    @discount_curve.setter
    def discount_curve(self, value: str):
        self._discount_curve = value

    @property
    def estimation_curve(self) -> str:
        if self._estimation_curve is None:
            raise ValueError("you must define a pricer")
        return self._estimation_curve

    @estimation_curve.setter
    def estimation_curve(self, value: str):
        self._estimation_curve = value
