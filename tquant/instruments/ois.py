import datetime

from .product import Product
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency, SwapType
from ..index.curverateindex import OvernightIndex
from ..flows.fixedcoupon import FixedRateLeg
from ..flows.floatingcoupon import FloatingRateLeg


class Ois(Product):
    """
    Represents an Overnight Indexed Swap (OIS) product.

    Attributes:
        ccy (Currency): The currency of the swap.
        start_date (datetime.date): The start date of the swap.
        end_date (datetime.date): The end date of the swap.
        start_dates_fix (list[datetime.date]): Start dates for fixed leg payments.
        end_dates_fix (list[datetime.date]): End dates for fixed leg payments.
        pay_dates_fix (list[datetime.date]): Payment dates for fixed leg.
        start_dates_flt (list[datetime.date]): Start dates for floating leg payments.
        end_dates_flt (list[datetime.date]): End dates for floating leg payments.
        pay_dates_flt (list[datetime.date]): Payment dates for floating leg.
        fixing_dates (list[datetime.date]): Dates on which floating leg rates are fixed.
        fixing_rates (list[float]): Fixing rates for the floating leg.
        quote (float): The quoted rate for the fixed leg.
        notional (float): The notional amount of the swap.
        day_counter_fix (DayCounter): Day count convention for the fixed leg.
        day_counter_flt (DayCounter): Day count convention for the floating leg.
        index (OvernightIndex): The overnight index used for the floating leg.
        swap_type (SwapType): Type of the swap, either Payer or Receiver.
        fixed_leg (FixedRateLeg): Fixed leg of the OIS.
        floating_leg (FloatingRateLeg): Floating leg of the OIS.
    """

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
        fixing_dates: list[datetime.date],
        fixing_rates: list[float],
        quote: float,
        notional: float,
        day_counter_fix: DayCounter,
        day_counter_flt: DayCounter,
        index: OvernightIndex,
        swap_type: SwapType = SwapType.Payer,
    ):
        """
        Initializes the Ois instance.

        Args:
            ccy (Currency): The currency in which the OIS is denominated.
            start_date (datetime.date): The start date of the swap.
            end_date (datetime.date): The end date of the swap.
            start_dates_fix (list[datetime.date]): Start dates for fixed leg cash flows.
            end_dates_fix (list[datetime.date]): End dates for fixed leg cash flows.
            pay_dates_fix (list[datetime.date]): Payment dates for the fixed leg.
            start_dates_flt (list[datetime.date]): Start dates for floating leg cash flows.
            end_dates_flt (list[datetime.date]): End dates for floating leg cash flows.
            pay_dates_flt (list[datetime.date]): Payment dates for the floating leg.
            fixing_dates (list[datetime.date]): Fixing dates for the floating leg rates.
            fixing_rates (list[float]): The fixing rates corresponding to the floating leg.
            quote (float): The interest rate for the fixed leg.
            notional (float): The notional amount for the swap.
            day_counter_fix (DayCounter): Day count convention for the fixed leg.
            day_counter_flt (DayCounter): Day count convention for the floating leg.
            index (OvernightIndex): The overnight index used for the floating leg.
            swap_type (SwapType, optional): The type of swap (Payer or Receiver). Defaults to SwapType.Payer.
        """
        super().__init__(ccy, start_date, end_date, quote)
        self._start_dates_fix = start_dates_fix
        self._end_dates_fix = end_dates_fix
        self._pay_dates_fix = pay_dates_fix
        self._start_dates_flt = start_dates_flt
        self._end_dates_flt = end_dates_flt
        self._pay_dates_flt = pay_dates_flt
        self._fixing_dates = fixing_dates  # TODO a che servono?
        self._fixing_rates = fixing_rates  # TODO a che servono?
        self._notional = notional
        self._day_counter_fix = day_counter_fix
        self._day_counter_flt = day_counter_flt

        self._notionals = [notional] * len(pay_dates_fix)
        self._rates = [quote] * len(pay_dates_fix)
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

    @property
    def start_dates_fix(self) -> list[datetime.date]:
        """
        Get the start dates for the fixed leg cash flows.

        Returns:
            list[datetime.date]: The start dates for the fixed leg.
        """
        return self._start_dates_fix

    @property
    def end_dates_fix(self) -> list[datetime.date]:
        """
        Get the end dates for the fixed leg cash flows.

        Returns:
            list[datetime.date]: The end dates for the fixed leg.
        """
        return self._end_dates_fix

    @property
    def pay_dates_fix(self) -> list[datetime.date]:
        """
        Get the payment dates for the fixed leg cash flows.

        Returns:
            list[datetime.date]: The payment dates for the fixed leg.
        """
        return self._pay_dates_fix

    @property
    def start_dates_flt(self) -> list[datetime.date]:
        """
        Get the start dates for the floating leg cash flows.

        Returns:
            list[datetime.date]: The start dates for the floating leg.
        """
        return self._start_dates_flt

    @property
    def end_dates_flt(self) -> list[datetime.date]:
        """
        Get the end dates for the floating leg cash flows.

        Returns:
            list[datetime.date]: The end dates for the floating leg.
        """
        return self._end_dates_flt

    @property
    def pay_dates_flt(self) -> list[datetime.date]:
        """
        Get the payment dates for the floating leg cash flows.

        Returns:
            list[datetime.date]: The payment dates for the floating leg.
        """
        return self._pay_dates_flt

    @property
    def fixing_dates(self) -> list[datetime.date]:
        """
        Get the fixing dates for the floating leg rates.

        Returns:
            list[datetime.date]: The fixing dates for the floating leg.
        """
        return self._fixing_dates

    @property
    def fixing_rates(self) -> list[float]:
        """
        Get the fixing rates for the floating leg.

        Returns:
            list[float]: The fixing rates for the floating leg.
        """
        return self._fixing_rates

    @property
    def notional(self) -> float:
        """
        Get the notional amount for the swap.

        Returns:
            float: The notional amount.
        """
        return self._notional

    @property
    def day_counter_fix(self) -> DayCounter:
        """
        Get the day count convention for the fixed leg of the OIS.

        Returns:
            DayCounter: The day count convention used for the fixed leg.
        """
        return self._day_counter_fix

    @property
    def day_counter_flt(self) -> DayCounter:
        """
        Get the day count convention for the floating leg of the OIS.

        Returns:
            DayCounter: The day count convention used for the floating leg.
        """
        return self._day_counter_flt

    @property
    def notionals(self) -> list[float]:
        """
        Get the notionals for each period of the fixed and floating legs.

        Returns:
            list[float]: The notional amounts for each payment period.
        """
        return self._notionals

    @property
    def rates(self) -> list[float]:
        """
        Get the fixed rates for the fixed leg.

        Returns:
            list[float]: The fixed rates for the fixed leg.
        """
        return self._rates

    @property
    def gearings(self) -> list[float]:
        """
        Get the gearings for the floating leg, which adjust the floating rate.

        Returns:
            list[float]: The gearing multipliers for the floating leg.
        """
        return self._gearings

    @property
    def margins(self) -> list[float]:
        """
        Get the margins for the floating leg, which are added to the floating rate.

        Returns:
            list[float]: The margins for the floating leg.
        """
        return self._margins

    @property
    def index(self) -> OvernightIndex:
        """
        Get the overnight index used for the floating leg of the OIS.

        Returns:
            OvernightIndex: The index used for the floating leg.
        """
        return self._index

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
