import datetime
import pandas

from .coupon import Coupon
from ..markethandles.interestrate import InterestRate
from ..timehandles.utils import CompoundingType, Frequency
from ..timehandles.daycounter import DayCounter


class FixedCoupon(Coupon):
    """
    Represents a fixed-rate coupon.

    This class is a concrete implementation of the `Coupon` class for fixed-rate coupons,
    with methods to calculate the coupon amount, rate, and accrued amount.
    """

    def __init__(
        self,
        payment_date: datetime.date,
        nominal: float,
        accrual_start_date: datetime.date,
        accrual_end_date: datetime.date,
        ref_period_start: datetime.date,
        ref_period_end: datetime.date,
        r: float,
        daycounter: DayCounter,
    ):
        """
        Initializes a FixedCoupon instance with the given attributes.

        Args:
            payment_date (datetime.date): The payment date of the coupon.
            nominal (float): The nominal (face value) amount of the coupon.
            accrual_start_date (datetime.date): The start date of the accrual period.
            accrual_end_date (datetime.date): The end date of the accrual period.
            ref_period_start (datetime.date): The reference period start date.
            ref_period_end (datetime.date): The reference period end date.
            r (float): The fixed interest rate applied to the coupon.
            daycounter (DayCounter): The day count convention used to calculate accrued interest.
        """
        super().__init__(
            payment_date,
            nominal,
            daycounter,
            accrual_start_date,
            accrual_end_date,
            ref_period_start,
            ref_period_end,
        )
        self._rate = InterestRate(
            r, daycounter, CompoundingType.Simple, Frequency.Annual
        )
        self._daycounter = daycounter
        self._amount = self.nominal * (
            self._rate.compound_factor(
                self.accrual_start_date,
                self.accrual_end_date,
                self.ref_period_start,
                self.ref_period_end,
            )
            - 1
        )

    @property
    def rate(self) -> InterestRate:
        """
        Returns the fixed interest rate of the coupon.

        Returns:
            InterestRate: The fixed interest rate of the coupon.
        """
        return self._rate

    @property
    def day_counter(self) -> DayCounter:
        """
        Returns the day count convention used for the coupon.

        Returns:
            DayCounter: The day count convention for accrued interest calculation.
        """
        return self._daycounter

    def display(self) -> pandas.DataFrame:
        """
        Displays a summary of the coupon details in a pandas DataFrame format.

        Returns:
            pandas.DataFrame: A DataFrame containing the coupon's reference period, payment date,
            nominal value, accrual period, day count convention, interest rate, and the coupon amount.
        """
        coupon_display = pandas.DataFrame(
            [
                self.ref_period_start,
                self.ref_period_end,
                self.date,
                self._nominal,
                self.accrual_period,
                self._daycounter.day_counter_convention.name,
                self._rate.rate,
                self.amount,
            ]
        ).T

        coupon_display.columns = [
            "start_period",
            "end_period",
            "payment_date",
            "notional",
            "accrual",
            "day_counter",
            "rate",
            "amount",
        ]
        return coupon_display

    @property
    def amount(self) -> float:
        """
        Returns the total coupon amount.

        The amount is calculated as the nominal value multiplied by the compounded factor
        of the interest rate over the accrual period.

        Returns:
            float: The total (not discounted) coupon payment amount.
        """
        return self._amount

    @property
    def accrual_period(self):
        """
        Returns the fraction of the year representing the accrual period.

        Uses the `DayCounter` to calculate the year fraction between the accrual start
        and end dates.

        Returns:
            float: The fraction of the year that represents the accrual period.
        """
        return self._daycounter.year_fraction(
            self.accrual_start_date, self.accrual_end_date
        )

    def accrued_amount(self, d: datetime.date):
        """
        Calculates the accrued coupon amount up to the given date.

        Args:
            d (datetime.date): The date up to which the accrued amount is calculated.

        Returns:
            float: The accrued amount up to the specified date. If the date is before
            the accrual start date or after the payment date, returns 0.
        """
        if d <= self.accrual_start_date or d > self._payment_date:
            return 0
        return self.nominal * (
            self._rate.compound_factor(
                self.accrual_start_date,
                min(d, self.accrual_end_date),
                self.ref_period_start,
                self.ref_period_end,
            )
            - 1
        )


class FixedRateLeg:
    """
    Represents a leg of fixed-rate coupon payments.

    This class models a series of fixed-rate coupon payments across multiple periods.
    It constructs a sequence of `FixedCoupon` objects with specified attributes.
    """

    def __init__(
        self,
        payment_dates: list[datetime.date],
        period_start_dates: list[datetime.date],
        period_end_dates: list[datetime.date],
        notionals: list[float],
        coupon_rates: list[float],
        daycounter: DayCounter,
        compounding: CompoundingType = CompoundingType.Simple,
        frequency: Frequency = Frequency.Annual,
    ) -> None:
        """
        Initializes a FixedRateLeg with a sequence of payment dates, notionals, rates, and other attributes.

        Args:
            payment_dates (list[datetime.date]): A list of coupon payment dates.
            period_start_dates (list[datetime.date]): A list of accrual period start dates.
            period_end_dates (list[datetime.date]): A list of accrual period end dates.
            notionals (list[float]): A list of notional amounts for each coupon.
            coupon_rates (list[float]): A list of fixed interest rates for each coupon.
            daycounter (DayCounter): The day count convention used to calculate accrued interest.
            compounding (CompoundingType, optional): The compounding method (default is Simple).
            frequency (Frequency, optional): The frequency of coupon payments (default is Annual).
        """

        self._notionals = notionals
        self._rates = coupon_rates
        self._daycounter = daycounter
        self._compounding = compounding
        self._frequency = frequency
        self._payment_dates = payment_dates
        self._period_start_dates = period_start_dates
        self._period_end_dates = period_end_dates

        self.leg_flows = []
        for i in range(len(payment_dates)):
            self.leg_flows.append(
                FixedCoupon(
                    payment_dates[i],
                    notionals[i],
                    period_start_dates[i],
                    period_end_dates[i],
                    period_start_dates[i],
                    period_end_dates[i],
                    coupon_rates[i],
                    daycounter,
                )
            )

    @property
    def coupon_rates(self) -> list[InterestRate]:
        """
        Returns a list of `InterestRate` objects corresponding to each coupon rate.

        Each rate is associated with the day count convention, compounding, and frequency.

        Returns:
            list[InterestRate]: A list of `InterestRate` objects for the coupon rates.
        """
        return [
            InterestRate(r, self._daycounter, self._compounding, self._frequency)
            for r in self._rates
        ]

    def display_flows(self) -> pandas.DataFrame:
        """
        Displays a summary of the cash flows for the fixed-rate leg.

        This method concatenates the display data for each `FixedCoupon` in the leg into a
        pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame summarizing the cash flows for the entire leg.
        """
        flows = self.leg_flows
        leg_display = pandas.DataFrame()
        for i in range(len(flows)):
            coupon_flow = flows[i].display()
            leg_display = pandas.concat([leg_display, coupon_flow], axis=0)
        return leg_display
    
    @property
    def price(self) -> float:
        """
        Get the price associated with the Leg.

        Returns:
            float: The price of the Leg.
        """
        if self._price is None:
            raise ValueError("you must define a pricer")
        return self._price
    
    @price.setter
    def price(self, value):
        self._price = value