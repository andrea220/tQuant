import datetime
from typing import Dict

from dateutil.relativedelta import relativedelta

from tquant import RateCurve, BusinessDayConvention, DayCounterConvention, DayCounter, Product
from tquant.bootstrapping.abstract_pricer import AbstractPricer
from tquant.bootstrapping.curve_assignment import CurveAssignment
from tquant.bootstrapping.fra import Fra
from tquant.bootstrapping.fra_builder import FraBuilder
from tquant.bootstrapping.util import calculate_forward


class FraPricer(AbstractPricer):
    def __init__(self, curve_assignment: CurveAssignment):
        super().__init__()
        self.curve_assignment = curve_assignment

    def price(self,
              product: Product,
              as_of_date: datetime,
              curves: Dict[str, RateCurve]):
        if isinstance(product, Fra):
            fra = product
            discount = {"CCY": fra.ccy, "USAGE": "DISCOUNT"}
            forecast = {"CCY": fra.ccy, "USAGE": "FORECAST", "INDEX_TENOR": fra.period}
            fc_name = self.curve_assignment.get_curve_name(forecast)
            dc_name = self.curve_assignment.get_curve_name(discount)
            fc = curves[fc_name]
            dc = curves[dc_name]
            start_date = fra.start_date
            pv = 0.0
            if start_date > as_of_date:
                index_yf = fra.index_day_counter.year_fraction(fra.index_start_date, fra.index_end_date)
                forward = calculate_forward(as_of_date, fc, fra.index_start_date, fra.index_end_date, index_yf)
                yf = fra.day_counter.year_fraction(fra.start_date, fra.end_date)
                act365 = DayCounterConvention.Actual365
                day_counter = DayCounter(act365)
                df = dc.discount(day_counter.year_fraction(as_of_date, fra.start_date))
                pv = fra.notional * yf * (forward - fra.quote) * df / (1.0 + forward * yf)
            return pv

        else:
            raise TypeError("Wrong product type")


if __name__ == "__main__":
    today = datetime.datetime.now()
    builder = FraBuilder("FRA",
                         "EUR",
                         2,
                         2,
                         "6M",
                         BusinessDayConvention.ModifiedFollowing,
                         1.0,
                         DayCounterConvention.Thirty360,
                         DayCounterConvention.Actual360)
    market_data = {"EURIBOR 3M": RateCurve([1.0 / 365.0], [0.01]),
                   "EONIA": RateCurve([1.0 / 365.0], [0.01])}
    fra = builder.build(datetime.datetime.now(), 0.01, "3M-6M")
    data = [["EUR", "DISCOUNT", "", "EONIA"],
            ["EUR", "FORECAST", "3M", "EURIBOR 3M"]]
    attributes = ["CCY", "USAGE", "INDEX_TENOR"]
    curve_assignment = CurveAssignment(data, attributes)
    pricer = FraPricer(curve_assignment)
    pv = pricer.price(fra, today, market_data)
    print(pv)
