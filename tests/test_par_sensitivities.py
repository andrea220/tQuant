"""Par sensitivities of prices to the curve quotes (``par_sensitivities``): stored bootstrap Jacobians, the Ibor curve
carried to its OIS discount curve."""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import tensorquant as tq  # noqa: E402
from markers import slow  # noqa: E402
from tensorquant.markethandles.bootstrapping import (QUOTE_CONVENTIONS, _quote_instrument, bootstrap_from_quotes,  # noqa: E402
                                                     par_sensitivities)
from tensorquant.markethandles.marketenvironment import MarketEnvironment  # noqa: E402
from test_bootstrap import EVAL_DATE, market_quotes  # noqa: E402

EUR = tq.Currency.EUR


def quotes():
    q = market_quotes()
    return q[(q["ccy"] == "EUR")].reset_index(drop=True)          # ESTR and 6M (discounted on ESTR)


def zero_greeks(product, mkt):
    pricer = tq.PricerAssignment.create(product)
    with tf.GradientTape() as tape:
        price = pricer.calculate_price(product, mkt)
    return float(price), tape.gradient(price, mkt.risk_factors())


class TestParSensitivities(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tq.Settings.evaluation_date = EVAL_DATE
        cls.mkt = MarketEnvironment({})
        bootstrap_from_quotes(quotes(), EVAL_DATE, cls.mkt)

    def setUp(self):
        tq.Settings.evaluation_date = EVAL_DATE

    def instrument(self, curve, instrument, tenor):
        q = quotes()
        quote = float(q[(q["curve"] == curve) & (q["instrument"] == instrument) & (q["tenor"] == tenor)]["quote"].iloc[0])
        return _quote_instrument(EUR, QUOTE_CONVENTIONS[("EUR", curve)], instrument, tenor, quote, EVAL_DATE)

    def test_calibration_instrument_on_its_own_quote(self):
        """A calibration instrument (its fixed rate held): minus its quote DV01 on its own quote (the market quote up
        1bp moves the instrument as its own fixed rate down 1bp), zero on every other quote; for an instrument of the
        6M curve also zero on the ESTR quotes (the 6M curve is rebootstrapped on the new discount curve and keeps it
        at par)."""
        for curve, instrument, tenor in [("ESTR", "ois", "5Y"), ("6M", "swap", "10Y"), ("6M", "fra", "12M"),
                                         ("6M", "fixing", "6M")]:
            product = self.instrument(curve, instrument, tenor)
            _, greeks = zero_greeks(product, self.mkt)
            par = par_sensitivities(self.mkt, greeks)
            own = self.mkt.get_ir_curve(EUR, curve)
            label = f"{instrument} {tenor}"
            with self.subTest(label=label, curve=curve):
                self.assertAlmostEqual(par[f"IR:EUR:{curve}:SPOT"][label], -own.calib_quote_dv01[label],
                                       delta=1e-6 * abs(own.calib_quote_dv01[label]))
                for key, series in par.items():
                    rest = series.drop(label) if key == f"IR:EUR:{curve}:SPOT" else series
                    self.assertLess(float(np.max(np.abs(rest))), 1e-6 * abs(own.calib_quote_dv01[label]), key)

    def test_curve_without_jacobian(self):
        """A curve not bootstrapped from quotes has no par sensitivity (``None``)."""
        tq.Settings.evaluation_date = EVAL_DATE
        mkt = MarketEnvironment({"IR:EUR:ESTR:SPOT": tq.RateCurve(EVAL_DATE, [1.0, 5.0], [0.02, 0.025], "LINEAR",
                                                                  tq.DayCounterConvention.Actual365)})
        product = tq.swap_generator("EUR", "ESTR", 1e6).build(EVAL_DATE, 0.02, "3Y")
        _, greeks = zero_greeks(product, mkt)
        self.assertIsNone(par_sensitivities(mkt, greeks)["IR:EUR:ESTR:SPOT"])

    @slow
    def test_parallel_quote_bump(self):
        """A forward-start 6M swap (not a calibration instrument): the sum of its par sensitivities x 1bp = its PV
        change when every quote of both curves moves 1bp and the curves are rebootstrapped (first order: 2%)."""
        product = tq.swap_generator("EUR", "6M", 1e6).build(EVAL_DATE, 0.033, "4Y", forward="1Y")
        price, greeks = zero_greeks(product, self.mkt)
        par = par_sensitivities(self.mkt, greeks)
        predicted = 1e-4 * sum(float(s.sum()) for s in par.values() if s is not None)
        bumped = quotes().assign(quote=lambda q: q["quote"] + 1e-4)
        mkt = MarketEnvironment({})
        bootstrap_from_quotes(bumped, EVAL_DATE, mkt)
        tq.Settings.evaluation_date = EVAL_DATE
        moved, _ = zero_greeks(product, mkt)
        self.assertAlmostEqual(predicted, moved - price, delta=0.02 * abs(moved - price))


if __name__ == "__main__":
    unittest.main()
