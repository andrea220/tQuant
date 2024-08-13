from datetime import date 
from .coupon import Coupon
from ..markethandles.interestrate import InterestRate
from ..timehandles.utils import CompoundingType, Frequency
from ..timehandles.daycounter import DayCounter
import pandas as pd
from pandas import DataFrame

class FixedCoupon(Coupon):
    """
    A concrete implementation of the `Coupon` class representing a fixed-rate coupon.

    This class models a coupon with a fixed interest rate, providing methods to calculate 
    the amount, rate, and accrued amount based on the specified day count convention and 
    other parameters.
    """

    def __init__(self,
                 payment_date: date,
                 nominal: float,
                 accrual_start_date: date,
                 accrual_end_date: date,
                 ref_period_start: date,
                 ref_period_end: date,
                 r: float,
                 daycounter: DayCounter
                 ):
        """
        Initializes a FixedCoupon instance with the specified attributes.

        Parameters:
        -------
        payment_date: date
            The date on which the coupon payment is made.
        nominal: float
            The nominal (face value) amount of the coupon.
        accrual_start_date: date
            The start date of the accrual period.
        accrual_end_date: date
            The end date of the accrual period.
        ref_period_start: date
            The start date of the reference period.
        ref_period_end: date
            The end date of the reference period.
        r: float
            The fixed interest rate applied to the coupon.
        daycounter: DayCounter
            The day count convention used to calculate accrued interest.
        """
        super().__init__(payment_date, nominal, daycounter, accrual_start_date, accrual_end_date, ref_period_start, ref_period_end)
        self._rate = InterestRate(r, daycounter, CompoundingType.Simple, Frequency.Annual)
        self._daycounter = daycounter

        
    @property
    def rate(self) -> InterestRate:
        """
        Returns the fixed interest rate associated with the coupon.

        Returns:
        -------
        InterestRate
            The fixed interest rate of the coupon.
        """
        return self._rate
    
    @property
    def day_counter(self) -> DayCounter:
        """
        Returns the day count convention used by the coupon.

        Returns:
        -------
        DayCounter
            The day count convention used for calculating accrued interest.
        """
        return self._daycounter
    
    def display(self) -> DataFrame:
        """
        Displays a summary of the coupon's details in a DataFrame format.

        The DataFrame includes the start and end of the reference period, payment date, 
        nominal amount, accrual period, day count convention, interest rate, and the coupon 
        amount.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame summarizing the coupon's details.
        """
        coupon_display = pd.DataFrame([self.ref_period_start,
                                        self.ref_period_end,
                                        self.date,
                                        self._nominal,
                                        self.accrual_period,
                                        self._daycounter.day_counter_convention.name,
                                        self._rate.rate,
                                        self.amount
                                        ]).T

        coupon_display.columns = ['start_period',
                                'end_period',
                                'payment_date',
                                'notional',
                                'accrual',
                                'day_counter',
                                'rate',
                                'amount'
                                ]
        return coupon_display
    
    @property
    def amount(self)-> float:
        """
        Returns the total amount of the coupon, calculated based on the nominal value, 
        interest rate, and accrual period.

        Returns:
        -------
        float
            The total amount of the coupon payment.
        """
        self._amount = self.nominal * (self._rate.compound_factor(self.accrual_start_date,
                                                                    self.accrual_end_date,
                                                                    self.ref_period_start,
                                                                    self.ref_period_end
                                                                    ) - 1)
        return self._amount
    
    @property
    def accrual_period(self):
        """
        Returns the fraction of the year that represents the accrual period.

        Uses the `DayCounter` to calculate the year fraction between the accrual start and 
        end dates.

        Returns:
        -------
        float
            The fraction of the year that represents the accrual period.
        """
        return self._daycounter.year_fraction(self.accrual_start_date, self.accrual_end_date)
        
    
    def accrued_amount(self, d: date):
        """
        Returns the accrued amount up to the specified date.

        The accrued amount is calculated as the nominal value times the compounded factor 
        of the rate over the period from the accrual start date to the minimum of the given 
        date and the accrual end date.

        Parameters:
        -------
        d: date
            The date up to which the accrued amount is calculated.

        Returns:
        -------
        float
            The accrued amount up to the specified date. Returns 0 if the date is outside 
            the accrual period.
        """
        if d <= self.accrual_start_date or d > self._payment_date:
            return 0
        else:
            return self.nominal * (self._rate.compound_factor(self.accrual_start_date,
                                                            min(d, self.accrual_end_date),
                                                            self.ref_period_start,
                                                            self.ref_period_end
                                                            ) - 1)


class FixedRateLeg:
    """
    A class representing a leg of fixed-rate coupon payments.

    This class models a series of fixed-rate coupon payments over multiple periods, 
    constructing a sequence of `FixedCoupon` objects based on the specified payment dates, 
    notionals, rates, and other parameters.
    """
    def __init__(self,
                 payment_dates: list[date],
                 period_start_dates: list[date],
                 period_end_dates: list[date],
                 notionals: list[float],
                 coupon_rates: list[float], 
                 daycounter: DayCounter,
                 compounding: CompoundingType = CompoundingType.Simple,
                 frequency: Frequency = Frequency.Annual) -> None:
        """
        Initializes a FixedRateLeg instance with the specified attributes.

        Parameters:
        -------
        payment_dates: list[date]
            A list of dates on which the coupon payments are made.
        period_start_dates: list[date]
            A list of start dates for each accrual period.
        period_end_dates: list[date]
            A list of end dates for each accrual period.
        notionals: list[float]
            A list of nominal (face value) amounts for each coupon.
        coupon_rates: list[float]
            A list of fixed interest rates for each coupon.
        daycounter: DayCounter
            The day count convention used to calculate accrued interest.
        compounding: CompoundingType, optional
            The compounding convention used (default is `CompoundingType.Simple`).
        frequency: Frequency, optional
            The frequency of the coupon payments (default is `Frequency.Annual`).
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
            self.leg_flows.append(FixedCoupon(payment_dates[i],
                                    notionals[i],
                                    period_start_dates[i],
                                    period_end_dates[i],
                                    period_start_dates[i],
                                    period_end_dates[i],
                                    coupon_rates[i],
                                    daycounter)
                                        )
    @property
    def coupon_rates(self) -> list[InterestRate]:
        """
        Returns a list of `InterestRate` objects corresponding to the coupon rates.

        This method constructs an `InterestRate` object for each rate in the list, applying
        the specified day count convention, compounding, and frequency.

        Returns:
        -------
        list[InterestRate]
            A list of `InterestRate` objects for each coupon rate.
        """
        return [InterestRate(r, self._daycounter, self._compounding, self._frequency) for r in self._rates]

    def display_flows(self) -> DataFrame:
        """
        Displays a summary of the cash flows for the entire leg in a DataFrame format.

        This method concatenates the display data of each `FixedCoupon` in the leg into a 
        single DataFrame.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame summarizing the cash flows of the fixed-rate leg.
        """
        flows = self.leg_flows
        leg_display = pd.DataFrame()
        for i in range(len(flows)):
            coupon_flow = flows[i].display()
            leg_display = pd.concat([leg_display, coupon_flow], axis = 0)
        return leg_display

