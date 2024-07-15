import datetime
from typing import Dict

from tquant import Product, RateCurve, DayCounterConvention
from tquant.bootstrapping.abstract_pricer import AbstractPricer
from tquant.bootstrapping.curve_assignment import CurveAssignment
from tquant.bootstrapping.future import Future
from tquant.bootstrapping.future_builder import FutureBuilder
from tquant.bootstrapping.util import calculate_forward


class FuturePricer(AbstractPricer):
    def __init__(self,
                 curve_assignment: CurveAssignment):
        super().__init__()
        self.curve_assignment = curve_assignment

    def price(self,
              product: Product,
              as_of_date: datetime,
              curves: Dict[str, RateCurve]):
        if isinstance(product, Future):
            future = product
            forecast = {"CCY": future.ccy, "USAGE": "FORECAST", "INDEX_TENOR": future.period}
            fc_name = self.curve_assignment.get_curve_name(forecast)
            fc = curves[fc_name]
            start_date = future.start_date
            end_date = future.maturity
            yf = future.day_counter.year_fraction(start_date, end_date)
            forward = calculate_forward(as_of_date, fc, start_date, end_date, yf)
            return 1.0 - forward - future.quote / 100.0


if __name__ == "__main__":
    today = datetime.datetime.today()
    future = FutureBuilder("fut", "EUR", 1.0, "6M", DayCounterConvention.Actual360)\
        .build(today, 98, "JUN 24")
    market_data = {"EUR EURIBOR 6M": RateCurve([1.0 / 365.0], [0.01])}
    data = [["EUR", "DISCOUNT", "6M", "EUR EURIBOR 6M"]]
    attributes = ["CCY", "USAGE", "INDEX_TENOR"]
    curve_assignment = CurveAssignment(data, attributes)
    pricer = FuturePricer(curve_assignment)
    pv = pricer.price(future, today, market_data)
    print(pv)
