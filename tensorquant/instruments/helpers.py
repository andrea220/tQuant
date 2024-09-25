import datetime
from abc import ABC, abstractmethod

from ..markethandles.utils import Currency
from ..timehandles.utils import (
    BusinessDayConvention,
    DayCounterConvention,
    TimeUnit,
    decode_term,
)
from ..timehandles.daycounter import DayCounter
from ..timehandles.schedule import ScheduleGenerator
from ..timehandles.targetcalendar import TARGET, Calendar
from ..instruments.deposit import Deposit
from ..instruments.ois import Ois
from ..instruments.forward import Fra
from ..instruments.swap import Swap
from ..index.curverateindex import IborIndex, OvernightIndex, Index


class ProductGenerator(ABC):
    """
    Abstract base class for generating financial products.

    Attributes:
        name (str): The name or type of the product generator.
        ccy (str): The currency of the product.
        notional (float): The notional amount for the product.
    """

    def __init__(self, name: str, ccy: str, notional: float):
        """
        Initializes the product generator with basic attributes.

        Args:
            name (str): The name or type of the product generator.
            ccy (str): The currency of the product.
            notional (float): The notional amount for the product.
        """
        self.name = name
        self.ccy = ccy
        self.notional = notional

    @abstractmethod
    def build(self, trade_date, quote: float, term: str):
        """
        Abstract method to build a financial product.

        Args:
            trade_date (datetime.date): The trade date for the product.
            quote (float): The quote or rate for the product.
            term (str): The term or duration of the product.

        Returns:
            Product: The constructed financial product.
        """
        return


class DepositGenerator(ProductGenerator):
    """
    Generator for creating deposit products.

    Attributes:
        start_delay (int): Number of days to delay the start of the deposit.
        roll_convention (BusinessDayConvention): Business day convention for rolling.
        day_count_convention (DayCounterConvention): Day count convention for the deposit.
        calendar (Calendar): Calendar used for date adjustments.
    """

    def __init__(
        self,
        ccy: Currency,
        start_delay: int,
        roll_convention: BusinessDayConvention,
        day_count_convention: DayCounterConvention,
        notional: float,
        calendar: Calendar,
    ):
        """
        Initializes the deposit generator with specific attributes.

        Args:
            ccy (Currency): The currency of the deposit.
            start_delay (int): Number of days to delay the start of the deposit.
            roll_convention (BusinessDayConvention): Business day convention for rolling.
            day_count_convention (DayCounterConvention): Day count convention for the deposit.
            notional (float): The notional amount of the deposit.
            calendar (Calendar): Calendar used for date adjustments.
        """
        super().__init__("depo", ccy, notional)
        self.start_delay = start_delay
        self.roll_convention = roll_convention
        self.day_count_convention = day_count_convention
        self.calendar = calendar

    def build(self, trade_date: datetime.date, quote: float, term: str):
        """
        Builds a deposit product based on the provided trade date, quote, and term.

        Args:
            trade_date (datetime.date): The trade date for the deposit.
            quote (float): The quote or rate for the deposit.
            term (str): The term or duration of the deposit.

        Returns:
            Deposit: The constructed deposit product.
        """
        if term == "O/N" or term == "ON":
            start_date = trade_date
            maturity = self.calendar.advance(
                start_date, 1, TimeUnit.Days, self.roll_convention
            )
        elif term == "T/N" or term == "TN":
            start_date = self.calendar.advance(
                trade_date, 1, TimeUnit.Days, self.roll_convention
            )
            maturity = self.calendar.advance(
                start_date, 1, TimeUnit.Days, self.roll_convention
            )
        elif term == "S/N" or term == "SN":
            start_date = self.calendar.advance(
                trade_date, 2, TimeUnit.Days, self.roll_convention
            )
            maturity = self.calendar.advance(
                start_date, 1, TimeUnit.Days, self.roll_convention
            )
        else:
            tenor, time_unit = decode_term(term)
            start_date = self.calendar.advance(
                trade_date, 2, TimeUnit.Days, self.roll_convention
            )
            maturity = self.calendar.advance(
                trade_date, tenor, time_unit, self.roll_convention
            )

        return Deposit(
            self.ccy,
            quote,
            trade_date,
            start_date,
            maturity,
            self.notional,
            self.day_count_convention,
        )


class OisGenerator(ProductGenerator):
    """
    Generator for creating Overnight Index Swap (OIS) products.

    Attributes:
        start_delay (int): Number of days to delay the start of the OIS.
        fixing_days (int): Number of days for fixing.
        period_fix (str): Fixing period.
        period_flt (str): Floating period.
        roll_convention (BusinessDayConvention): Business day convention for rolling.
        day_count_convention_fix (DayCounterConvention): Day count convention for the fixed leg.
        day_count_convention_flt (DayCounterConvention): Day count convention for the floating leg.
        calendar (Calendar): Calendar used for date adjustments.
        index (Index): Index used for the OIS.
    """

    def __init__(
        self,
        ccy: str,
        start_delay: int,
        fixing_days: int,
        period_fix: str,
        period_flt: str,
        roll_convention: BusinessDayConvention,
        notional: float,
        day_count_convention_fix: DayCounterConvention,
        day_count_convention_flt: DayCounterConvention,
        calendar: Calendar,
        index: Index,
    ):
        """
        Initializes the OIS generator with specific attributes.

        Args:
            ccy (str): The currency of the OIS.
            start_delay (int): Number of days to delay the start of the OIS.
            fixing_days (int): Number of days for fixing.
            period_fix (str): Fixing period.
            period_flt (str): Floating period.
            roll_convention (BusinessDayConvention): Business day convention for rolling.
            notional (float): The notional amount of the OIS.
            day_count_convention_fix (DayCounterConvention): Day count convention for the fixed leg.
            day_count_convention_flt (DayCounterConvention): Day count convention for the floating leg.
            calendar (Calendar): Calendar used for date adjustments.
            index (Index): Index used for the OIS.
        """
        super().__init__("ois", ccy, notional)
        self.start_delay = start_delay
        self.fixing_days = fixing_days
        self.period_fix = period_fix
        self.period_flt = period_flt
        self.roll_convention = roll_convention
        self.notional = notional
        self.day_count_convention_fix = day_count_convention_fix
        self.day_count_convention_flt = day_count_convention_flt
        self.calendar = calendar
        self.index = index

    def build(self, trade_date: datetime.date, quote: float, term: str):
        """
        Builds an OIS product based on the provided trade date, quote, and term.

        Args:
            trade_date (datetime.date): The trade date for the OIS.
            quote (float): The quote or rate for the OIS.
            term (str): The term or duration of the OIS.

        Returns:
            Ois: The constructed OIS product.
        """
        start_date = self.calendar.advance(
            trade_date, self.start_delay, TimeUnit.Days, self.roll_convention
        )

        period_maturity, time_unit = decode_term(term)
        maturity = self.calendar.advance(
            start_date, period_maturity, time_unit, self.roll_convention
        )

        period_fix, time_unit_fix = decode_term(self.period_fix)
        period_float, time_unit_float = decode_term(self.period_flt)
        schedule_generator = ScheduleGenerator(self.calendar, self.roll_convention)
        schedule_fix = schedule_generator.generate(
            start_date, maturity, period_fix, time_unit_fix
        )
        schedule_float = schedule_generator.generate(
            start_date, maturity, period_float, time_unit_float
        )

        start_fix = []
        end_fix = []
        pay_fix = []
        for i in range(len(schedule_fix) - 1):
            start_fix.append(schedule_fix[i])
            end_fix.append(schedule_fix[i + 1])
            pay_fix.append(
                self.calendar.adjust(schedule_fix[i + 1], self.roll_convention)
            )

        day_count_fix = DayCounter(self.day_count_convention_fix)
        day_count_flt = DayCounter(self.day_count_convention_flt)

        start_flt = []
        end_flt = []
        pay_flt = []
        for i in range(len(schedule_float) - 1):
            start_flt.append(schedule_float[i])
            end_flt.append(schedule_float[i + 1])
            pay_flt.append(
                self.calendar.adjust(schedule_float[i + 1], self.roll_convention)
            )


        return Ois(
            self.ccy,
            start_date,
            maturity,
            start_fix,
            end_fix,
            pay_fix,
            start_flt,
            end_flt,
            pay_flt,
            quote,
            self.notional,
            day_count_fix,
            day_count_flt,
            self.index,
        )


class FraGenerator(ProductGenerator):
    """
    Generator for creating Forward Rate Agreement (FRA) products.

    Attributes:
        start_delay (int): Number of days to delay the start of the FRA.
        fixing_days (int): Number of days for fixing.
        index_term (str): Term for the index.
        roll_convention (BusinessDayConvention): Business day convention for rolling.
        day_count_convention (DayCounterConvention): Day count convention for the FRA.
        calendar (Calendar): Calendar used for date adjustments.
        index (Index): Index used for the FRA.
    """

    def __init__(
        self,
        ccy: str,
        start_delay: int,
        fixing_days: int,
        index_term: str,
        roll_convention: BusinessDayConvention,
        notional: float,
        day_count_convention: DayCounterConvention,
        calendar: Calendar,
        index: Index,
    ):
        """
        Initializes the FRA generator with specific attributes.

        Args:
            ccy (str): The currency of the FRA.
            start_delay (int): Number of days to delay the start of the FRA.
            fixing_days (int): Number of days for fixing.
            index_term (str): Term for the index.
            roll_convention (BusinessDayConvention): Business day convention for rolling.
            notional (float): The notional amount of the FRA.
            day_count_convention (DayCounterConvention): Day count convention for the FRA.
            calendar (Calendar): Calendar used for date adjustments.
            index (Index): Index used for the FRA.
        """
        super().__init__("fra", ccy, notional)
        self.ccy = ccy
        self.start_delay = start_delay
        self.fixing_days = fixing_days
        self.index_term = index_term
        self.roll_convention = roll_convention
        self.notional = notional
        self.day_count_convention = day_count_convention
        self.calendar = calendar
        self.index = index

    def build(self, trade_date: datetime.date, quote: float, term: str):
        """
        Builds a FRA product based on the provided trade date, quote, and term.

        Args:
            trade_date (datetime.date): The trade date for the FRA.
            quote (float): The quote or rate for the FRA.
            term (str): The term or duration of the FRA.

        Returns:
            Fra: The constructed FRA product.
        """
        shifters = term.split("-")
        if len(shifters) != 2:
            raise ValueError("Wrong term specified: " + term)

        settlement_date = self.calendar.advance(
            trade_date, self.start_delay, TimeUnit.Days, self.roll_convention
        )

        start_period, start_time_unit = decode_term(shifters[0])
        start_date = self.calendar.advance(
            settlement_date, start_period, start_time_unit, self.roll_convention
        )

        end_period, end_time_unit = decode_term(shifters[1])
        end_date = self.calendar.advance(
            settlement_date, end_period, end_time_unit, self.roll_convention
        )

        return Fra(
            self.ccy,
            start_date,
            end_date,
            self.notional,
            quote,
            self.day_count_convention,
            self.index,
        )


class SwapGenerator(ProductGenerator):
    """
    Generator for creating Swap products.

    Attributes:
        start_delay (int): Number of days to delay the start of the swap.
        period_fix (str): Fixing period.
        period_flt (str): Floating period.
        roll_convention (BusinessDayConvention): Business day convention for rolling.
        day_count_convention_fix (DayCounterConvention): Day count convention for the fixed leg.
        day_count_convention_flt (DayCounterConvention): Day count convention for the floating leg.
        calendar (Calendar): Calendar used for date adjustments.
        index (Index): Index used for the swap.
    """

    def __init__(
        self,
        ccy: Currency,
        start_delay: int,
        period_fix: str,
        period_flt: str,
        roll_convention: BusinessDayConvention,
        notional: float,
        day_count_convention_fix: DayCounterConvention,
        day_count_convention_flt: DayCounterConvention,
        calendar: Calendar,
        index: Index,
    ):
        """
        Initializes the swap generator with specific attributes.

        Args:
            ccy (Currency): The currency of the swap.
            start_delay (int): Number of days to delay the start of the swap.
            period_fix (str): Fixing period.
            period_flt (str): Floating period.
            roll_convention (BusinessDayConvention): Business day convention for rolling.
            notional (float): The notional amount of the swap.
            day_count_convention_fix (DayCounterConvention): Day count convention for the fixed leg.
            day_count_convention_flt (DayCounterConvention): Day count convention for the floating leg.
            calendar (Calendar): Calendar used for date adjustments.
            index (Index): Index used for the swap.
        """
        super().__init__("swap", ccy, notional)
        self.start_delay = start_delay
        self.period_fix = period_fix
        self.period_flt = period_flt
        self.roll_convention = roll_convention
        self.notional = notional
        self.day_count_convention_fix = day_count_convention_fix
        self.day_count_convention_flt = day_count_convention_flt
        self.calendar = calendar
        self.index = index

    def build(self, trade_date: datetime.date, quote: float, term: str):
        """
        Builds a swap product based on the provided trade date, quote, and term.

        Args:
            trade_date (datetime.date): The trade date for the swap.
            quote (float): The quote or rate for the swap.
            term (str): The term or duration of the swap.

        Returns:
            Swap: The constructed swap product.
        """
        start_date = self.calendar.advance(
            trade_date, self.start_delay, TimeUnit.Days, self.roll_convention
        )

        period_maturity, time_unit = decode_term(term)
        maturity = self.calendar.advance(
            start_date, period_maturity, time_unit, self.roll_convention
        )

        period_fix, time_unit_fix = decode_term(self.period_fix)
        period_float, time_unit_float = decode_term(self.period_flt)

        schedule_generator = ScheduleGenerator(self.calendar, self.roll_convention)
        schedule_fix = schedule_generator.generate(
            start_date, maturity, period_fix, time_unit_fix
        )
        schedule_float = schedule_generator.generate(
            start_date, maturity, period_float, time_unit_float
        )

        start_fix = []
        end_fix = []
        pay_fix = []
        for i in range(len(schedule_fix) - 1):
            start_fix.append(schedule_fix[i])
            end_fix.append(schedule_fix[i + 1])
            pay_fix.append(
                self.calendar.adjust(schedule_fix[i + 1], self.roll_convention)
            )

        day_count_fix = DayCounter(self.day_count_convention_fix)
        day_count_flt = DayCounter(self.day_count_convention_flt)

        start_flt = []
        end_flt = []
        pay_flt = []
        for i in range(len(schedule_float) - 1):
            start_flt.append(schedule_float[i])
            end_flt.append(schedule_float[i + 1])
            pay_flt.append(
                self.calendar.adjust(schedule_float[i + 1], self.roll_convention)
            )

        return Swap(
            self.ccy,
            start_date,
            maturity,
            start_fix,
            end_fix,
            pay_fix,
            start_flt,
            end_flt,
            pay_flt,
            quote,
            self.notional,
            day_count_fix,
            day_count_flt,
            self.index,
        )
