import datetime
from datetime import timedelta
from typing import Dict

from tquant import Deposit, RateCurve, DayCounterConvention, DayCounter, Product
from dateutil.relativedelta import relativedelta

from tquant.bootstrapping.abstract_pricer import AbstractPricer
from tquant.bootstrapping.curve_assignment import CurveAssignment


class DepositPricer(AbstractPricer):
    def __init__(self,
                 curve_assignment: CurveAssignment):
        super().__init__()
        self.curve_assignment = curve_assignment

    def price(self,
              product: Product,
              as_of_date: datetime,
              curves: Dict[str, RateCurve]):
        if isinstance(product, Deposit):
            deposit = product
            instance = {"CCY": deposit.ccy, "USAGE": "DISCOUNT"}
            curve_name = self.curve_assignment.get_curve_name(instance)
            curve = curves[curve_name]
            act365 = DayCounterConvention.Actual365
            day_counter = DayCounter(act365)
            ts = day_counter.year_fraction(as_of_date, deposit.start_date)
            te = day_counter.year_fraction(as_of_date, deposit.end_date)
            df_s = curve.discount(ts)
            df_e = curve.discount(te)
            start_cashflow = 0.0
            if ts >= 0.0:
                start_cashflow = 1.0
            end_cashflow = 0.0
            if te > 0.0:
                yf = deposit.day_counter.year_fraction(deposit.start_date, deposit.end_date)
                end_cashflow = 1.0 + deposit.quote * yf
            start_cashflow *= deposit.notional
            end_cashflow *= deposit.notional

            return start_cashflow * df_s - end_cashflow * df_e
        else:
            raise TypeError("Wrong product type")


if __name__ == "__main__":
    today = datetime.datetime.now()
    spot_date = today + timedelta(days=2)
    maturity = spot_date + relativedelta(months=6)
    deposit = Deposit("EUR", 0.01, today, spot_date, maturity, 1.0, DayCounterConvention.Actual360)
    market_data = {"EUR EONIA": RateCurve([1.0 / 365.0], [0.01])}
    data = [["EUR", "DISCOUNT", "", "EUR EONIA"]]
    attributes = ["CCY", "USAGE", "INDEX_TENOR"]
    curve_assignment = CurveAssignment(data, attributes)
    pricer = DepositPricer(curve_assignment)
    pv = pricer.price(deposit, today, market_data)
    print(pv)
