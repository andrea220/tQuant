import unittest
from datetime import date
from dateutil.relativedelta import relativedelta
import sys
sys.path.append('C:/dev/tQuant') #local path

from tquant.instruments.deposit import Deposit
from tquant.timehandles.utils import DayCounterConvention
from tquant.markethandles.utils import Currency
from tquant.markethandles.ircurve import RateCurve
from tquant.pricers.deposit import DepositEngine

class TestDepositExample(unittest.TestCase):
    def test_pricing(self):
        today = date(2024, 4, 30)
        spot_date = date(2024, 5, 2)
        maturity = spot_date + relativedelta(months=6)
        deposit = Deposit(Currency.EUR, today, spot_date, maturity, 100, DayCounterConvention.Actual360, 0.01)
        disc_curve = RateCurve([1.0/365, 1.0, 2.0, 4.0], [0.01, 0.02, 0.022, 0.03])
        pricer = DepositEngine(deposit)
        pv, tape = pricer.price_aad(disc_curve, today)
        sensitivities = tape.gradient(pv, [disc_curve.rates])
        print(pv)
        print(sensitivities)
        self.assertIsNotNone(pv)
        self.assertIsNotNone(sensitivities)

if __name__ == "__main__":
    unittest.main()
