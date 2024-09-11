from datetime import date
from .product import Product
from ..index.index import Index
from ..flows.fixedcoupon import FixedRateLeg
from ..flows.floatingcoupon import FloatingRateLeg
from ..timehandles.daycounter import DayCounter
from ..markethandles.utils import Currency, SwapType


class Swap(Product):
    """
    Represents an interest rate swap contract, with both fixed and floating legs.

    Attributes:
        ccy (Currency): The currency of the swap.
        start_date (date): The start date of the swap.
        end_date (date): The end date of the swap.
        start_dates_fix (list[date]): The start dates for the fixed leg payments.
        end_dates_fix (list[date]): The end dates for the fixed leg.
        pay_dates_fix (list[date]): The payment dates for the fixed leg.
        start_dates_flt (list[date]): The start dates for the floating leg payments.
        end_dates_flt (list[date]): The end dates for the floating leg.
        pay_dates_flt (list[date]): The payment dates for the floating leg.
        quote (float): The interest rate applied to the fixed leg.
        notional (float): The notional principal amount of the swap.
        day_counter_fix (DayCounter): The day count convention for the fixed leg.
        day_counter_flt (DayCounter): The day count convention for the floating leg.
        index (Index): The index used for the floating leg.
        swap_type (SwapType): The type of swap (Payer or Receiver).
        fixed_leg (FixedRateLeg): The fixed rate leg of the swap.
        floating_leg (FloatingRateLeg): The floating rate leg of the swap.
    """

    def __init__(
        self,
        ccy: Currency,
        start_date: date,
        end_date: date,
        start_dates_fix: list[date],
        end_dates_fix: list[date],
        pay_dates_fix: list[date],
        start_dates_flt: list[date],
        end_dates_flt: list[date],
        pay_dates_flt: list[date],
        quote: float,
        notional: float,
        day_counter_fix: DayCounter,
        day_counter_flt: DayCounter,
        index: Index,
        swap_type: SwapType = SwapType.Payer,
    ) -> None:
        """
        Initializes a new instance of the Swap class.

        Args:
            ccy (Currency): The currency of the swap.
            start_date (date): The start date of the swap.
            end_date (date): The end date of the swap.
            start_dates_fix (list[date]): Start dates for the fixed leg.
            end_dates_fix (list[date]): End dates for the fixed leg.
            pay_dates_fix (list[date]): Payment dates for the fixed leg.
            start_dates_flt (list[date]): Start dates for the floating leg.
            end_dates_flt (list[date]): End dates for the floating leg.
            pay_dates_flt (list[date]): Payment dates for the floating leg.
            quote (float): The fixed rate applied to the fixed leg.
            notional (float): The notional principal for the swap.
            day_counter_fix (DayCounter): The day count convention for the fixed leg.
            day_counter_flt (DayCounter): The day count convention for the floating leg.
            index (Index): The floating rate index.
            swap_type (SwapType): Type of swap, Payer or Receiver.
        """
        super().__init__(ccy, start_date, end_date, quote)
        self._start_dates_fix = start_dates_fix
        self._end_dates_fix = end_dates_fix
        self._pay_dates_fix = pay_dates_fix
        self._start_dates_flt = start_dates_flt
        self._end_dates_flt = end_dates_flt
        self._pay_dates_flt = pay_dates_flt
        self._notional = notional
        self._day_counter_fix = day_counter_fix
        self._day_counter_flt = day_counter_flt

        self._fix_notionals = [notional] * len(pay_dates_fix)
        self._rates = [quote] * len(pay_dates_fix)
        self._float_notionals = [notional] * len(pay_dates_flt)
        self._gearings = [1] * len(pay_dates_flt)
        self._margins = [0] * len(pay_dates_flt)
        self._index = index
        self._swap_type = swap_type

        self._fixed_leg = FixedRateLeg(
            pay_dates_fix,
            start_dates_fix,
            end_dates_fix,
            self._fix_notionals,
            self._rates,
            day_counter_fix,
        )
        self._floating_leg = FloatingRateLeg(
            pay_dates_flt,
            start_dates_flt,
            end_dates_flt,
            self._float_notionals,
            self._gearings,
            self._margins,
            index,
            day_counter_flt,
        )

    @property
    def start_dates_fix(self) -> list[date]:
        """
        Get the start dates for the fixed leg.

        Returns:
            list[date]: The start dates for the fixed leg.
        """
        return self._start_dates_fix

    @property
    def end_dates_fix(self) -> list[date]:
        """
        Get the end dates for the fixed leg.

        Returns:
            list[date]: The end dates for the fixed leg.
        """
        return self._end_dates_fix

    @property
    def pay_dates_fix(self) -> list[date]:
        """
        Get the payment dates for the fixed leg.

        Returns:
            list[date]: The payment dates for the fixed leg.
        """
        return self._pay_dates_fix

    @property
    def start_dates_flt(self) -> list[date]:
        """
        Get the start dates for the floating leg.

        Returns:
            list[date]: The start dates for the floating leg.
        """
        return self._start_dates_flt

    @property
    def end_dates_flt(self) -> list[date]:
        """
        Get the end dates for the floating leg.

        Returns:
            list[date]: The end dates for the floating leg.
        """
        return self._end_dates_flt

    @property
    def pay_dates_flt(self) -> list[date]:
        """
        Get the payment dates for the floating leg.

        Returns:
            list[date]: The payment dates for the floating leg.
        """
        return self._pay_dates_flt

    @property
    def notional(self) -> float:
        """
        Get the notional principal amount of the swap.

        Returns:
            float: The notional principal amount.
        """
        return self._notional

    @property
    def day_counter_fix(self) -> DayCounter:
        """
        Get the day count convention for the fixed leg.

        Returns:
            DayCounter: The day count convention used for the fixed leg.
        """
        return self._day_counter_fix

    @property
    def day_counter_flt(self) -> DayCounter:
        """
        Get the day count convention for the floating leg.

        Returns:
            DayCounter: The day count convention used for the floating leg.
        """
        return self._day_counter_flt

    @property
    def fix_notionals(self) -> list[float]:
        """
        Get the notionals for the fixed leg.

        Returns:
            list[float]: The notionals for the fixed leg.
        """
        return self._fix_notionals

    @property
    def rates(self) -> list[float]:
        """
        Get the interest rates for the fixed leg.

        Returns:
            list[float]: The interest rates for the fixed leg.
        """
        return self._rates

    @property
    def float_notionals(self) -> list[float]:
        """
        Get the notionals for the floating leg.

        Returns:
            list[float]: The notionals for the floating leg.
        """
        return self._float_notionals

    @property
    def gearings(self) -> list[float]:
        """
        Get the gearings (multipliers) for the floating leg.

        Returns:
            list[float]: The gearings for the floating leg.
        """
        return self._gearings

    @property
    def margins(self) -> list[float]:
        """
        Get the margins for the floating leg.

        Returns:
            list[float]: The margins for the floating leg.
        """
        return self._margins

    @property
    def index(self) -> Index:
        """
        Get the index used for the floating leg.

        Returns:
            Index: The index for the floating leg.
        """
        return self._index

    @property
    def swap_type(self) -> SwapType:
        """
        Get the type of swap (Payer or Receiver).

        Returns:
            SwapType: The type of the swap.
        """
        return self._swap_type

    @property
    def fixed_leg(self) -> FixedRateLeg:
        """
        Get the fixed leg of the swap.

        Returns:
            FixedRateLeg: The fixed leg of the swap.
        """
        return self._fixed_leg

    @property
    def floating_leg(self) -> FloatingRateLeg:
        """
        Get the floating leg of the swap.

        Returns:
            FloatingRateLeg: The floating leg of the swap.
        """
        return self._floating_leg
