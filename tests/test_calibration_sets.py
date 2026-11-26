"""Calibration sets (``calibration/models.py``): models calibrated on a snapshot, stored apart, reloaded; fit and
drift; the model test battery against Black and the local vol."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import tensorquant as tq
from markers import slow
from tensorquant.calibration.models import (_payoff, _quality, _test_products, calibrate_snapshot_models,
                                            check_calibration, list_calibration_sets, load_calibration_set, load_model,
                                            model_fit, model_test)
from tensorquant.markethandles.dividendcurve import DividendCurve
from tensorquant.markethandles.ircurve import RateCurve
from tensorquant.markethandles.marketenvironment import MarketEnvironment
from tensorquant.markethandles.utils import Currency
from tensorquant.markethandles.volatilitysurface import VolatilitySurface
from tensorquant.timehandles.daycounter import DayCounter, DayCounterConvention

EVAL_DATE = date(2026, 5, 15)
A365 = DayCounterConvention.Actual365
S0, VOL = 100.0, 0.22


def flat_market() -> MarketEnvironment:
    """Flat 22% surface (a grid, for the local vol), discrete dividends, a rate curve."""
    surface = VolatilitySurface(reference_date=EVAL_DATE, calendar=None, daycounter=DayCounter(A365),
                                strike=[40.0, 70.0, 100.0, 130.0, 200.0], maturity=[0.25, 1.0, 2.0, 4.0],
                                volatility_matrix=[[VOL] * 5] * 4)
    return MarketEnvironment(market={
        "IR:EUR:ESTR:SPOT": RateCurve(EVAL_DATE, [0.5, 1.0, 2.0, 5.0], [0.02, 0.021, 0.022, 0.024], "LINEAR", A365),
        "EQ:EUR:X:SPOT": S0, "EQ:EUR:X:VOL": surface,
        "EQ:EUR:X:DIV": DividendCurve(reference_date=EVAL_DATE, ex_dates=[date(2026, 9, 15), date(2027, 9, 15)],
                                      amounts=[1.0, 1.1], currency=Currency.EUR, daycounter_convention=A365)})


class TestLocalVolCheck(unittest.TestCase):
    def test_flat_surface(self):
        """Local vol in a set (a check, pricing rebuilds it): on a flat surface no arbitrage, no cell at the floor or
        the cap, and the Monte Carlo repricing of the nodes within 0.5 vol points; the report survives the round
        trip."""
        tq.Settings.evaluation_date = EVAL_DATE
        eq = flat_market().get_equity("X")
        model = tq.LocalVolatilityModel.calibrate(eq)
        report = model.report.iloc[0]
        self.assertEqual((report["arbitrage_fraction"], report["floored_fraction"], report["at_cap_fraction"]),
                         (0.0, 0.0, 0.0))
        self.assertAlmostEqual(report["max_local_vol"], VOL, delta=0.01)
        fit = model_fit(model, eq, mc_paths=4_000)
        inner = fit[(fit["t"] >= 0.25) & fit["moneyness"].between(0.7, 1.3)]
        self.assertLess(float((inner["error_volpts"] ** 2).mean() ** 0.5), 0.5)
        back = type(model).from_frames(model.to_frames())
        self.assertEqual(float(back.report.iloc[0]["max_local_vol"]), float(report["max_local_vol"]))

    def test_unreachable_nodes_and_grid(self):
        """Nodes beyond the reach of the paths (fewer than 30 expected beyond the strike) are left out of the fit
        (NaN, rms finite); the simulation grid is daily over three months, weekly after (uniform without them)."""
        from tensorquant.calibration.models import MIN_PATHS_BEYOND, _fit_errors
        from tensorquant.models.brownian import simulation_grid
        tq.Settings.evaluation_date = EVAL_DATE
        eq = flat_market().get_equity("X")
        fit = model_fit(tq.LocalVolatilityModel.calibrate(eq), eq, mc_paths=2_000)
        far = fit[fit["model_vol"].isna()]
        self.assertGreater(len(far), 0)
        self.assertFalse(((far["moneyness"] > 0.9) & (far["moneyness"] < 1.1)).any())   # near the money: measured
        self.assertTrue(np.isfinite(_fit_errors(fit)["rms_volpts"]))
        self.assertEqual(MIN_PATHS_BEYOND, 30)
        grid = simulation_grid(1.0)
        steps = np.diff(np.concatenate([[0.0], grid]))
        np.testing.assert_allclose(steps[grid < 0.25], 1.0 / 365.0, atol=1e-12)
        self.assertLessEqual(steps[grid > 0.3].max(), 1.0 / 52.0 + 1e-12)
        np.testing.assert_allclose(simulation_grid(1.0, fine_horizon=0.0), np.arange(1.0 / 52.0, 1.0, 1.0 / 52.0))


class TestBattery(unittest.TestCase):
    def test_products(self):
        products = _test_products(S0)
        kinds = pd.Series([p["product"] for p in products]).value_counts().to_dict()
        self.assertEqual(kinds, {"Vanilla": 20, "Down-and-in put (at expiry)": 6, "Digital (cash)": 12})
        vanillas = [p for p in products if p["product"] == "Vanilla"]
        self.assertTrue(all((p["option_type"] == "Call") == (p["strike_pct"] > 100) for p in vanillas))

    def test_payoffs(self):
        s = np.linspace(30.0, 170.0, 141)
        put = {"product": "Vanilla", "option_type": "Put", "strike": 100.0}
        dip = {"product": "Down-and-in put (at expiry)", "option_type": "Put", "strike": 100.0, "barrier": 70.0}
        np.testing.assert_allclose(_payoff(dip, s), np.where(s <= 70.0, _payoff(put, s), 0.0))
        call = {"product": "Digital (cash)", "option_type": "Call", "strike": 100.0}
        dput = {"product": "Digital (cash)", "option_type": "Put", "strike": 100.0}
        np.testing.assert_allclose((_payoff(call, s) + _payoff(dput, s))[s != 100.0], 1.0)

    def test_quality(self):
        self.assertEqual([_quality(x) for x in (0.4, 1.0, 2.0, 3.0)], ["good", "good", "fair", "poor"])


@slow
class TestModelTestFlat(unittest.TestCase):
    """Flat smile: the local vol and a Heston with almost no vol of vol (v0 = theta = 0.22^2) are Black."""

    def test_flat_market_is_black(self):
        tq.Settings.evaluation_date = EVAL_DATE
        mkt = flat_market()
        heston = tq.HestonModel(v0=VOL ** 2, kappa=1.0, theta=VOL ** 2, xi=1e-3, rho=0.0)
        res = model_test({"Heston": heston}, mkt.get_equity("X"), mkt, n_paths=8_000)
        for name in ("LocalVol", "Heston"):
            z = (res[name] - res["Black"]) / res[f"{name}_se"].clip(lower=1e-9)
            self.assertLess(float(z.abs().max()), 4.0, name)
        vols = res.dropna(subset=["LocalVol_vol"])
        self.assertLess(float((vols["LocalVol_vol"] - VOL).abs().max()), 0.006)


@slow
class TestCalibrationSet(unittest.TestCase):
    """A synthetic snapshot calibrated (displaced diffusion, one underlying): files, summary, reload, fit, drift."""

    def test_round_trip(self):
        tq.Settings.evaluation_date = EVAL_DATE
        mkt = tq.generate_market(EVAL_DATE, 42)
        with tempfile.TemporaryDirectory() as tmp:
            snap = mkt.to_folder(Path(tmp) / "snap", "1200")
            root = Path(tmp) / "calibrations"
            folder = calibrate_snapshot_models(snap, root, plan={"TICKER1": ["DisplacedDiffusion", "G2pp"],
                                                                 "TICKER2": []})       # flags per underlying
            sets = list_calibration_sets(root)
            self.assertEqual([(s["date"], s["snapshot"]["set"]) for s in sets], [(EVAL_DATE.isoformat(), "1200")])
            summary = load_calibration_set(folder)["summary"]
            self.assertEqual(list(summary["model"]), ["DisplacedDiffusion", "G2pp"])
            self.assertEqual(summary.iloc[1]["status"], "skipped")              # not a set model (yet)
            row = summary.iloc[0]
            self.assertEqual((row["ticker"], row["model"], row["status"]), ("TICKER1", "DisplacedDiffusion", "ok"))
            self.assertIn(row["quality"], ("good", "fair", "poor"))
            model = load_model(folder, "TICKER1", "DisplacedDiffusion")
            eq = mkt.get_equity("TICKER1")
            fresh = tq.DisplacedDiffusionModel.calibrate(eq)
            for name, value in fresh.parameters.items():
                np.testing.assert_allclose(model.parameters[name].numpy(), value.numpy(), rtol=0, atol=1e-12)
            stored = pd.read_csv(folder / "TICKER1" / "DisplacedDiffusion_fit.csv")
            np.testing.assert_allclose(model_fit(model, eq)["model_vol"], stored["model_vol"], rtol=0, atol=1e-10)
            drift = check_calibration(model, eq)                    # same market: the stored errors
            self.assertAlmostEqual(drift["rms_volpts"], float(row["rms_volpts"]), places=8)
            self.assertIsNotNone(model.jacobian)


@slow
class TestCalibrationSetSLV(unittest.TestCase):
    """Heston SLV through the files (nested tables: Heston, local vol, leverage tables; JSON settings)."""

    def test_slv_round_trip(self):
        tq.Settings.evaluation_date = EVAL_DATE
        mkt = tq.generate_market(EVAL_DATE, 42)
        with tempfile.TemporaryDirectory() as tmp:
            snap = mkt.to_folder(Path(tmp) / "snap", "1200")
            folder = calibrate_snapshot_models(snap, Path(tmp) / "cal", models=(), tickers=["TICKER1"],
                                               slv=["TICKER1"])
            row = load_calibration_set(folder)["summary"].iloc[0]
            self.assertEqual((row["model"], row["status"]), ("HestonSLV", "ok"))
            model = load_model(folder, "TICKER1", "HestonSLV")
            self.assertIsInstance(model, tq.HestonSLVModel)
            stored = pd.read_csv(folder / "TICKER1" / "HestonSLV_leverage_nodes.csv")
            self.assertEqual(model.leverage_table["nodes"].size, len(stored))
            self.assertEqual(tuple(model.parameters["expectation"].shape), model.leverage_table["nodes"].shape)
            self.assertIsInstance(model.heston, tq.HestonModel)


if __name__ == "__main__":
    unittest.main()
