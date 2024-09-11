import unittest
from datetime import date
from tquant.markethandles.ircurve import RateCurve
from tquant.markethandles.utils import Currency, curve_map
from tquant.timehandles.daycounter import DayCounterConvention
from tquant.instruments.deposit import Deposit
from tquant.pricers.deposit import DepositPricer

class TestDepositExample(unittest.TestCase):
    def test_pricing(self):
        market_data = {}
        evaluation_date = date(2024, 4, 30)
        market_data["EUR:ESTR"] = RateCurve(reference_date=evaluation_date,
                                                pillars=[1.0/365, 1.0, 2.0, 4.0],
                                                rates=[0.01, 0.02, 0.022, 0.03],
                                                interp='LINEAR',
                                                daycounter_convention=DayCounterConvention.ActualActual)

        spot_date = date(2024, 5, 2)
        maturity = date(2025, 2, 3)
        deposit = Deposit(ccy= Currency.EUR,
                            quote=0.03,
                            trade_date=evaluation_date,
                            start_date=spot_date,
                            end_date=maturity,
                            notional=100,
                            day_count_convention=DayCounterConvention.Actual360)
        depo_engine = DepositPricer(curve_map)
        npv_depo, tape = depo_engine.price(deposit, evaluation_date, market_data, True)
        sensitivities = tape.gradient(npv_depo, market_data["EUR:ESTR"]._rates)
        self.assertIsNotNone(npv_depo)
        self.assertIsNotNone(sensitivities)

if __name__ == "__main__":
    unittest.main()