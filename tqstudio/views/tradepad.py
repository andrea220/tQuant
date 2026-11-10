"""Tradepad: define a trade on one underlying and price it with Black-Scholes and the calibrated equity models.

Models are calibrated once per underlying (:class:`ModelCache`) and priced as instances; market greeks come from
one AAD pass mapped through the calibration (:func:`tensorquant.par_greeks`).
"""

from __future__ import annotations

import math
import time
from datetime import date

import numpy as np
import tensorflow as tf

import tensorquant as tq

MODELS = ["Black-Scholes", "Local vol", "Heston", "Displaced diffusion", "Heston SLV"]
PRODUCTS = ["Vanilla", "Digital", "Barrier"]
BARRIER_TYPES = [b.value for b in tq.BarrierType]
STEPS_PER_YEAR, MIN_STEPS = 100, 25


class ModelCache:
    """Calibrated model instances per ``(ticker, model)``, with their calibration time."""

    def __init__(self):
        self._models: dict = {}

    def get(self, eq, name: str) -> tuple:
        """``(model, seconds spent calibrating now)``: 0 when it was already calibrated."""
        key = (eq.ticker, name)
        if key in self._models:
            return self._models[key], 0.0
        t0 = time.perf_counter()
        if name == "Local vol":
            model = tq.LocalVolatilityModel.calibrate(eq)
        elif name == "Heston":
            model = tq.HestonModel.calibrate(eq)
        elif name == "Displaced diffusion":
            model = tq.DisplacedDiffusionModel.calibrate(eq)
        elif name == "Heston SLV":                  # reuses the cached Heston and local vol
            heston, _ = self.get(eq, "Heston")
            local_vol, _ = self.get(eq, "Local vol")
            model = tq.HestonSLVModel(heston, local_vol).calibrate_leverage(eq)
        else:
            raise ValueError(f"unknown model {name}")
        self._models[key] = model
        return model, time.perf_counter() - t0

    def clear(self) -> None:
        self._models.clear()


def build_product(mkt: tq.MarketEnvironment, trade: dict):
    """Product of a trade: ``ticker``, ``product`` (Vanilla / Digital / Barrier), ``option_type`` (Call / Put),
    ``strike_pct`` (strike / spot, %), ``expiry`` (ISO date); Digital: ``payoff`` (Cash / Asset), ``cash_amount``;
    Barrier: ``barrier_pct`` (barrier / spot, %), ``barrier_type`` (DownOut, ...), continuous monitoring."""
    eq = mkt.get_equity(trade["ticker"])
    s0 = float(eq.spot)
    expiry = date.fromisoformat(trade["expiry"])
    if expiry <= mkt.evaluation_date:
        raise ValueError("the expiry must be after the snapshot date")
    option_type = tq.OptionType.Call if trade["option_type"] == "Call" else tq.OptionType.Put
    strike = s0 * float(trade["strike_pct"]) / 100
    common = (eq.currency, mkt.evaluation_date, expiry, option_type, strike)
    kind = trade["product"]
    if kind == "Vanilla":
        return tq.VanillaOption(*common, underlying=eq.ticker)
    if kind == "Digital":
        payoff = tq.PayoffType.AssetOrNothing if trade.get("payoff") == "Asset" else tq.PayoffType.CashOrNothing
        return tq.DigitalOption(*common, payoff_type=payoff, cash_amount=float(trade.get("cash_amount", 1.0)),
                                underlying=eq.ticker)
    if kind == "Barrier":
        return tq.BarrierOption(*common, barrier=s0 * float(trade["barrier_pct"]) / 100,
                                barrier_type=tq.BarrierType(trade["barrier_type"]), underlying=eq.ticker)
    raise ValueError(f"unknown product {kind}")


def _greeks(eq, price, tape, model=None) -> tuple:
    params = model.parameters if model is not None else {}
    g = tape.gradient(price, {"model": params, "market": eq.risk_factors()})
    market = {k: [None if x is None else x.numpy() for x in v] for k, v in g["market"].items()}
    if model is not None:
        return tq.par_greeks(model, {k: v.numpy() for k, v in g["model"].items()}, market, equity=eq)
    return {k: [np.zeros(()) if x is None else x for x in v] for k, v in market.items()}


def price_trade(mkt: tq.MarketEnvironment, cache: ModelCache, trade: dict, models: list[str],
                n_paths: int = 50_000, seed: int = 1) -> dict:
    """Price *trade* with each of *models*: price, standard error, delta, vega, rho (scaled by ``quantity``), vega
    per surface node, times.

    Returns:
        dict: ``trade`` (spot, strike, forward and vol at the strike, expiry), ``rows`` (one per model), ``vega_grid``
        (expiries and strikes / spot of the surface).
    """
    eq = mkt.get_equity(trade["ticker"])
    s0 = float(eq.spot)
    qty = float(trade.get("quantity", 1.0))
    probe = build_product(mkt, trade)
    T = eq.year_fraction(probe.end_date)
    vs = eq.vol_surface
    spot_key = next(k for k in eq.risk_factors() if k.startswith("EQ:") and k.endswith(":SPOT"))
    vol_key = next(k for k in eq.risk_factors() if k.startswith("EQ:") and k.endswith(":VOL"))
    n_steps = max(MIN_STEPS, math.ceil(STEPS_PER_YEAR * T))

    rows = []
    for name in models:
        product = build_product(mkt, trade)
        row = {"model": name, "calibration_s": 0.0}
        try:
            if name == "Black-Scholes":
                pricer = tq.BlackScholesPricer(dividend_model="discrete" if eq.dividend_model == "discrete"
                                               else "continuous")
                model = None
            else:
                model, row["calibration_s"] = cache.get(eq, name)
                pricer = tq.VanillaMCPricer(model=model, n_paths=int(n_paths), n_steps=n_steps, seed=int(seed))
            t0 = time.perf_counter()
            pricer.price(product, mkt, autodiff=True)
            greeks = _greeks(eq, product.price, pricer.tape, model)
            row["pricing_s"] = time.perf_counter() - t0
        except Exception as exc:                                     # shown on the page, the others still priced
            row["error"] = str(exc)
            rows.append(row)
            continue
        se = getattr(product, "standard_error", None)
        rho = sum(float(np.sum(x)) for k, v in greeks.items() if k.startswith("IR:") for x in v)
        vega = np.asarray(greeks[vol_key][0], float) / 100
        row.update({"unit_price": float(product.price),
                    "standard_error": None if se is None or name == "Black-Scholes" else float(se),
                    "price": qty * float(product.price), "delta": qty * float(np.sum(greeks[spot_key][0])),
                    "vega": qty * float(vega.sum()), "rho": qty * rho * 1e-4, "vega_nodes": (qty * vega).tolist()})
        rows.append(row)

    forward = float(eq.forward(T))
    return {"trade": {"spot": s0, "strike": float(probe.strike), "expiry": probe.end_date.isoformat(), "T": T,
                      "forward": forward, "rate": float(eq.discount_curve.zero_rate(T)),
                      "discount_factor": float(eq.discount_curve.discount(T)), "repo": float(eq.repo),
                      "pv_dividends": float(eq.pv_dividends(T)), "implied_div_yield": float(eq.implied_div_yield(T)),
                      "vol_at_strike": float(vs.volatility(strike=float(probe.strike), tenor=T)),
                      "vol_atm_forward": float(vs.volatility(strike=forward, tenor=T)),
                      "barrier": float(probe.barrier) if hasattr(probe, "barrier") else None,
                      "quantity": qty, "n_steps": n_steps, "n_paths": int(n_paths)},
            "rows": rows,
            "vega_grid": {"expiries": [float(t) for t in vs.maturity],
                          "moneyness": [float(k) / s0 for k in vs.strike]}}
