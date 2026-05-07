import unittest
from datetime import date

from tensorquant.markethandles.dividendcurve import DividendCurve
from tensorquant.markethandles.ircurve import RateCurve
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.utils import Currency
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention


class TestMarketEnvironmentStatic(unittest.TestCase):
    def setUp(self):
        self.evaluation_date = date(2026, 1, 2)
        dc = DayCounter(DayCounterConvention.Actual365)

        eur_curve = RateCurve(
            reference_date=self.evaluation_date,
            pillars=[0.25, 1.0, 2.0, 5.0],
            rates=[0.02, 0.022, 0.023, 0.025],
            interp="LINEAR",
            daycounter_convention=DayCounterConvention.Actual365,
        )

        vol_surface = VolatilitySurface(
            reference_date=self.evaluation_date,
            calendar=None,
            daycounter=dc,
            strike=[80.0, 100.0, 120.0],
            maturity=[0.5, 1.0, 2.0],
            volatility_matrix=[
                [0.24, 0.22, 0.23],
                [0.23, 0.21, 0.22],
                [0.22, 0.20, 0.21],
            ],
        )

        dividends = DividendCurve(
            reference_date=self.evaluation_date,
            ex_dates=[date(2026, 6, 15), date(2026, 12, 15)],
            amounts=[1.2, 1.3],
            currency=Currency.EUR,
            daycounter_convention=DayCounterConvention.Actual365,
        )

        market = {
            "IR:EUR:ESTR:SPOT": eur_curve,
            "EQ:EUR:SX5E:SPOT": 5000.0,
            "EQ:EUR:SX5E:VOL": vol_surface,
            "EQ:EUR:SX5E:REPO": 0.01,
            "EQ:EUR:SX5E:DIVYIELD": 0.02,
            "EQ:EUR:SX5E:DIV": dividends,
        }

        self.market_env = MarketEnvironment(market=market)

    def test_get_ir_curve_default_ticker(self):
        curve = self.market_env.get_ir_curve(Currency.EUR)
        self.assertIsNotNone(curve)
        self.assertEqual(curve.name, "IR:EUR:ESTR:SPOT")

    def test_get_eq_spot(self):
        spot = self.market_env.get_eq_spot("SX5E", Currency.EUR)
        self.assertEqual(spot, 5000.0)

    def test_get_eq_vol_surface(self):
        vol_surface = self.market_env.get_eq_vol_surface("SX5E", Currency.EUR)
        self.assertIsNotNone(vol_surface)
        self.assertAlmostEqual(vol_surface.volatility(100.0, 1.0), 0.21)

    def test_get_repo_dividend_yield_and_dividends(self):
        repo = self.market_env.get_eq_repo("SX5E", Currency.EUR)
        div_yield = self.market_env.get_eq_div_yield("SX5E", Currency.EUR)
        div_curve = self.market_env.get_eq_dividends("SX5E", Currency.EUR)

        self.assertEqual(repo, 0.01)
        self.assertEqual(div_yield, 0.02)
        self.assertEqual(len(div_curve), 2)

    def test_get_currency(self):
        ccy = self.market_env.get_currency("SX5E")
        self.assertEqual(ccy, Currency.EUR)


if __name__ == "__main__":
    unittest.main()
