from datetime import datetime, timedelta
from typing import Dict

from tquant import RateCurve, BusinessDayConvention, DayCounterConvention, DayCounter, Product
from tquant.bootstrapping.abstract_pricer import AbstractPricer
from tquant.bootstrapping.curve_assignment import CurveAssignment
from tquant.bootstrapping.ois import Ois
from tquant.bootstrapping.ois_builder import OisBuilder
from tquant.bootstrapping.util import calculate_forward


def compute_growing_factor(fixing_dates: list[datetime],
                           fixing_rates: list[float],
                           start_date: datetime,
                           as_of_date: datetime):
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


class OisPricer(AbstractPricer):
    def __init__(self, curve_assignment: CurveAssignment):
        super().__init__()
        self.curve_assignment = curve_assignment

    def price(self,
              product: Product,
              as_of_date: datetime,
              curves: Dict[str, RateCurve]):
        if isinstance(product, Ois):
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
                        growing_factor1 = compute_growing_factor(ois.fixing_dates,
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


if __name__ == "__main__":
    today = datetime.now()
    builder = OisBuilder("OIS",
                         "EUR",
                         2,
                         2,
                         "1Y",
                         "6M", BusinessDayConvention.ModifiedFollowing,
                         1.0,
                         DayCounterConvention.Actual360,
                         DayCounterConvention.Actual360)
    market_data = {"EONIA": RateCurve([1.0 / 365.0], [0.01])}
    ois = builder.build(datetime.now(), 0.01, "2Y")
    data = [["EUR", "DISCOUNT", "", "EONIA"]]
    attributes = ["CCY", "USAGE", "INDEX_TENOR"]
    curve_assignment = CurveAssignment(data, attributes)
    pricer = OisPricer(curve_assignment)
    pv = pricer.price(ois, today, market_data)
    print(pv)
