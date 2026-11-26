"""Vega by expiry and skew (``tq.vega_by_expiry``, ``tq.skew_weights``) on Black-Scholes prices: the sums of the node
vegas against bumps of the vol matrix with the same shape."""

import unittest
from datetime import date

import numpy as np

import tensorquant as tq

EVAL_DATE = date(2026, 5, 15)


def market():
    tq.Settings.evaluation_date = EVAL_DATE
    return tq.generate_market(EVAL_DATE, seed=7, tickers_per_currency=(("EUR", 1), ("GBP", 1), ("USD", 1)))


class TestVegaByExpiry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mkt = market()
        key = next(k for k in cls.mkt.risk_factors() if k.startswith("EQ:EUR:") and k.endswith(":VOL"))
        cls.ticker = key.split(":")[2]
        cls.eq = cls.mkt.get_equity(cls.ticker)

    def price(self, strike: float, years: int, autodiff: bool = False):
        tq.Settings.evaluation_date = EVAL_DATE
        option = tq.VanillaOption(tq.Currency.EUR, start_date=EVAL_DATE, end_date=date(2026 + years, 5, 15),
                                  option_type=tq.OptionType.Put, strike=strike, underlying=self.ticker,
                                  exercise_type=tq.ExerciseType.European)
        pricer = tq.BlackScholesPricer()
        pricer.price(option, self.mkt, autodiff=autodiff)
        return option, pricer

    def test_weights(self):
        """+1 vol point per 10% of strike below the forward of the expiry, -1 above, held beyond 50%-150%."""
        w = tq.skew_weights(self.eq)
        vs = self.eq.vol_surface
        strikes = np.array([float(k) for k in vs.strike])
        for i, t in enumerate(vs.maturity):
            m = np.clip(strikes / float(self.eq.forward(float(t))), 0.5, 1.5)
            np.testing.assert_allclose(w[i], (1.0 - m) / 0.1, atol=1e-12)
        self.assertLessEqual(w.max(), 5.0 + 1e-12)
        self.assertGreaterEqual(w.min(), -5.0 - 1e-12)

    def test_level_and_skew_vs_bumps(self):
        """Level, cumulative and skew of a put = the central bumps of the matrix by the same shapes (1e-4 relative),
        all on the expiry rows around the option's maturity."""
        spot = float(self.eq.spot)
        option, pricer = self.price(0.85 * spot, 2, autodiff=True)
        vol = self.eq.vol_surface.volatility_matrix
        vega = np.asarray(pricer.tape.gradient(option.price, vol)) / 100.0
        table = tq.vega_by_expiry(vega, self.eq)
        np.testing.assert_allclose(table["level"], vega.sum(axis=1), atol=1e-14)
        np.testing.assert_allclose(table["cumulative"].iloc[-1], vega.sum(), rtol=1e-12)
        base = vol.numpy().copy()

        def bumped(shift):
            vol.assign(base + shift / 100.0)
            try:
                return float(self.price(0.85 * spot, 2)[0].price)
            finally:
                vol.assign(base)

        h = 0.01
        for shape, value in ((np.ones_like(base), table["level"].sum()),
                             (tq.skew_weights(self.eq), table["skew"].sum())):
            fd = (bumped(h * shape) - bumped(-h * shape)) / (2 * h)
            self.assertAlmostEqual(fd / value, 1.0, delta=1e-4)
        self.assertGreater(table["skew"].sum(), 0.0)     # a put below the forward: long the steeper skew


class TestVegaByTenor(unittest.TestCase):
    def test_split(self):
        """On a tenor: all there; between two: split linearly in time; before 3M on 3M, beyond the last on it; total
        kept; tenors up to the first at or beyond the last expiry."""
        out = tq.vega_by_tenor([1.0, 2.0, 4.0, 8.0, 16.0], [0.02, 0.5, 1.5, 2.0, 6.0])
        self.assertEqual(out["label"].tolist(), ["3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y"])
        np.testing.assert_allclose(out["vega"], [1.0, 2.0, 2.0, 2.0 + 8.0, 0.0, 0.0, 0.0, 16.0], atol=1e-12)
        self.assertAlmostEqual(out["vega"].sum(), 31.0)
        tail = tq.vega_by_tenor([1.0], [0.7])
        self.assertEqual(tail["label"].tolist(), ["3M", "6M", "1Y"])
        np.testing.assert_allclose(tail["vega"], [0.0, 0.6, 0.4], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
