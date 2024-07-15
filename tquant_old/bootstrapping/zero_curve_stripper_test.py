import datetime

from tquant import BusinessDayConvention, DayCounterConvention, DepositBuilder
from tquant.bootstrapping.curve_assignment import CurveAssignment
from tquant.bootstrapping.ois_builder import OisBuilder
from tquant.bootstrapping.zero_curve_stripper import ZeroCurveStripper

import pandas as pd


def test_ois_fake():
    trade_date = datetime.date(2024, 6, 14)
    df = pd.read_csv('market_data_test.csv')
    generators = df['Generator'].astype(str).tolist()
    maturities = df['Maturity'].astype(str).tolist()
    quotes = df['Quote'].astype(float).tolist()
    data = [["EUR", "DISCOUNT", "EUR EONIA"]]
    attributes = ["CCY", "USAGE"]
    curve_assignment = CurveAssignment(data, attributes)
    depo_builder = DepositBuilder("depo",
                                  "EUR",
                                  2,
                                  BusinessDayConvention.ModifiedFollowing,
                                  DayCounterConvention.Actual360,
                                  1.0)
    ois_builder = OisBuilder("ois",
                             "EUR",
                             2,
                             2,
                             "1Y",
                             "6M",
                             BusinessDayConvention.ModifiedFollowing,
                             1.0,
                             DayCounterConvention.Actual360,
                             DayCounterConvention.Actual360)
    generator_map = {"depo": depo_builder, "ois": ois_builder}
    stripper = ZeroCurveStripper()
    curve = stripper.strip(
        "EUR EONIA", trade_date, generators, maturities, quotes, curve_assignment, generator_map, {}, "GLOBAL")
    for pillar in curve.pillars:
        print("Pillar=" + str(pillar))
    for pillar in curve.pillars:
        print("Df=" + str(curve.discount(pillar).numpy()))
    for rate in curve.rates:
        print("Rate=" + str(rate))


if __name__ == "__main__":
    test_ois_fake()
