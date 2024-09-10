from datetime import date
from .coupon import Coupon
from ..timehandles.daycounter import DayCounter
from ..timehandles.utils import TimeUnit, BusinessDayConvention
from ..index.curverateindex import IborIndex
import pandas as pd
from pandas import DataFrame
from tensorflow import Tensor

        
class FloatingCoupon(Coupon):
    """
    A concrete implementation of the `Coupon` class representing a floating-rate coupon.

    This class models a coupon with a floating interest rate, which is typically linked 
    to an index (e.g., LIBOR). The floating rate is adjusted by a spread and may include 
    gearing, and it can account for whether the rate is set in advance or in arrears.
    """
    def __init__(self,
                 payment_date: date,
                 nominal: float,
                 accrual_start_date: date,
                 accrual_end_date: date,
                 index: IborIndex, 
                 gearing: float, 
                 spread: float, 
                 ref_period_start: date,
                 ref_period_end: date,
                 daycounter: DayCounter,
                 is_in_arrears: bool = False, 
                 fixing_days: int = None,
                 ) -> None:
        """
        Initializes a FloatingCoupon instance with the specified attributes.

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
        index: IborIndex
            The index (e.g., LIBOR) to which the floating rate is tied.
        gearing: float
            The multiplicative coefficient applied to the index rate.
        spread: float
            The fixed spread added to the index rate.
        ref_period_start: date
            The start date of the reference period.
        ref_period_end: date
            The end date of the reference period.
        daycounter: DayCounter
            The day count convention used to calculate accrued interest.
        is_in_arrears: bool, optional
            Indicates whether the rate is set in arrears (default is False).
        fixing_days: int, optional
            The number of days before the fixing date; if None, uses the index's fixing days.
        """
        super().__init__(payment_date, nominal, daycounter, accrual_start_date, accrual_end_date, ref_period_start, ref_period_end)
        self._day_counter = daycounter
        self._fixing_days = fixing_days 
        self._index = index
        self._gearing = gearing
        self._spread = spread
        self._is_in_arrears = is_in_arrears
        self._rate = None
        self._amount = None
        self._convexity_adj = None


    @property
    def day_counter(self) -> DayCounter:
        """
        Returns the day count convention used by the coupon.

        Returns:
        -------
        DayCounter
            The day count convention used for calculating accrued interest.
        """
        return self._day_counter
    
    @property
    def fixing_days(self) -> int:
        """
        Returns the number of days before the fixing date.

        If fixing days were not provided during initialization, this property will return 
        the default fixing days of the index. If the index is not provided, it defaults to 0.

        Returns:
        -------
        int
            The number of days before the fixing date.
        """
        if self._fixing_days is None:
            if self.index is not None:
                return self.index.fixing_days
            else:
                return 0
    
    @property
    def index(self) -> IborIndex:
        """
        Returns the index to which the floating rate is tied.

        Returns:
        -------
        IborIndex
            The index associated with the floating rate.
        """
        return self._index
            
    @property
    def is_in_arrears(self) -> bool:
        """
        Indicates whether the rate is set in arrears.

        Returns:
        -------
        bool
            True if the rate is set in arrears, False otherwise.
        """
        return self._is_in_arrears
    
    @property
    def fixing_date(self) -> date:
        """
        Returns the fixing date for the floating rate.

        If the rate is in arrears, the fixing date is based on the accrual end date;
        otherwise, it is based on the accrual start date.

        Returns:
        -------
        date
            The date on which the rate is fixed.
        """
        if self.is_in_arrears:
            ref_date = self.accrual_end_date 
        else:
            ref_date = self.accrual_start_date
        return self._index.fixing_calendar.advance(ref_date,
                                                    -self._index.fixing_days,
                                                    TimeUnit.Days,
                                                    BusinessDayConvention.Preceding)
        
    @property
    def accrual_period(self) -> float: 
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

    
    def display(self) -> DataFrame:
        """
        Displays a summary of the floating coupon's details in a DataFrame format.

        The DataFrame includes the start and end of the reference period, payment date, 
        nominal amount, fixing date, fixing days, index, accrual period, arrears status, 
        gearing, spread, and day count convention.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame summarizing the floating coupon's details.
        """
        if isinstance(self._rate, Tensor):
            r = self._rate.numpy()
        else:
            r = None

        if isinstance(self._amount, Tensor):
            a = self._amount.numpy()
        else:
            a = None

        coupon_display = pd.DataFrame([self.accrual_start_date,
                                       self.accrual_end_date,
                                        self.ref_period_start,
                                        self.ref_period_end,
                                        self.day_counter, 
                                        self.accrual_period,
                                        self.fixing_date,
                                        self.date,
                                        self._nominal,
                                        self._index.name,
                                        r,
                                        self._spread,
                                        self._gearing,
                                        a, 
                                        self._convexity_adj
                                        ]).T

        coupon_display.columns = ['accr_start',
                                  'accr_end',
                                'start_period',
                                'end_period',
                                'dc',
                                'accrual',
                                'fixing_date',
                                'pay_date',
                                'notional',
                                'index',
                                'rate',
                                'spread',
                                'gearing',
                                'amount',
                                'convexity'
                                ]
        return coupon_display
        
    @property
    def amount(self)-> float: 
        """
        Returns the total amount of the floating coupon.

        The amount is pre-calculated and stored in the `_amount` attribute.

        Returns:
        -------
        float
            The total amount of the floating coupon payment.
        """
        return self._amount
    
    @property
    def rate(self) -> float:
        """
        Returns the interest rate applied to the floating coupon.

        The rate is pre-calculated and stored in the `_rate` attribute.

        Returns:
        -------
        float
            The interest rate of the floating coupon.
        """
        return self._rate
       
    def accrued_amount(self, d: date) -> float:
        """
        Calculates and returns the accrued amount of the floating coupon up to the specified date.

        The accrued amount is calculated as the nominal value multiplied by the compound factor 
        of the rate over the period from the accrual start date to the minimum of the given date 
        and the accrual end date.

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
        

class FloatingRateLeg:
    """
    A class representing a leg of floating-rate coupon payments.

    This class models a series of floating-rate coupon payments over multiple periods, 
    constructing a sequence of `FloatingCoupon` objects based on the specified payment dates, 
    notionals, gearings, spreads, and other parameters.
    """
    def __init__(self,
                 payment_dates: list[date],
                 period_start_dates: list[date],
                 period_end_dates: list[date],
                 notionals: list[float],
                 gearings: list[float],
                 spreads: list[float],
                 index: IborIndex,
                 daycounter: DayCounter,
                 is_in_arrears: bool = False
                 ) -> None:
        """
        Initializes a FloatingRateLeg instance with the specified attributes.

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
        gearings: list[float]
            A list of gearing coefficients (multiplicative factors) for each coupon.
        spreads: list[float]
            A list of fixed spreads added to the index rate for each coupon.
        index: IborIndex
            The index (e.g., LIBOR) to which the floating rate is tied.
        daycounter: DayCounter
            The day count convention used to calculate accrued interest.
        """
        self._notionals = notionals
        self._gearings = gearings
        self._spreads = spreads 
        self._index = index
        self._daycounter = daycounter
        self._is_in_arrears = is_in_arrears

        self.leg_flows = []
        for i in range(len(payment_dates)):
            self.leg_flows.append(FloatingCoupon(payment_dates[i],
                                                self._notionals[i],
                                                period_start_dates[i],
                                                period_end_dates[i],
                                                self._index,
                                                self._gearings[i],
                                                self._spreads[i],
                                                period_start_dates[i],
                                                period_end_dates[i],
                                                self._daycounter
                                                )
                                        )
    
   
    def display_flows(self) -> DataFrame: 
        """
        Displays a summary of the cash flows for the entire floating-rate leg in a DataFrame format.

        This method concatenates the display data of each `FloatingCoupon` in the leg into a 
        single DataFrame.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame summarizing the cash flows of the floating-rate leg.
        """
        flows = self.leg_flows
        leg_display = pd.DataFrame()
        for i in range(len(flows)):
            coupon_flow = flows[i].display()
            leg_display = pd.concat([leg_display, coupon_flow], axis = 0)
        return leg_display

        
