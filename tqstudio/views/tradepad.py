"""Tradepad: pick an asset class and a product type (the library's instruments), fill its deal fields, pick a model,
price it with greeks.

Each product declares its fields (rendered as the form by the web page), its models and how it is built from the
fields. Equity options: Black-Scholes closed forms (digitals as a vanilla spread on the smile) or local vol (Monte
Carlo on the calibrated local vol, cached per underlying); autocallables: local vol Monte Carlo; rates products:
analytic pricers (discounting, Bachelier / Black on caplet vols and on the swaption cube). Greeks by AAD to the market
risk factors: delta, vega per vol node, rho per curve pillar.

Market inputs of a deal (:func:`market_inputs`: spot, vol at the strike, rate and curve at the expiry) can be
overridden at pricing (equity: ``spot``, ``vol``): the spot replaces the market spot, the vol shifts the whole surface
in parallel so that the vol at the strike is the given one; the market is restored after the pricing.
"""

from __future__ import annotations

import math
import re
import time
from contextlib import contextmanager
from datetime import date, timedelta

import numpy as np
import tensorflow as tf
from dateutil.relativedelta import relativedelta

import tensorquant as tq

from .market import equity_tickers

STEPS_PER_YEAR, MIN_STEPS = 100, 25
DIGITAL_SPREAD = 0.005   # Black-Scholes digitals: centred vanilla spread on the smile, strikes K (1 +- 0.5%)
TENORS = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
SHORT_TERMS = ["1M", "3M", "6M", "9M", "1Y"]
CAL = tq.TARGET()
MF = tq.BusinessDayConvention.ModifiedFollowing
A360 = tq.DayCounterConvention.Actual360
IBOR = "6M"              # the Ibor index of the rates products (curve IR:<ccy>:6M)


CALIBRATED = {"Heston": "Heston", "Displaced diffusion": "DisplacedDiffusion", "Heston SLV": "HestonSLV"}
# GBM on an autocallable: for debugging only (the Black-Scholes model of one expiry, wrong at the intermediate dates)
GBM_DEBUG = "GBM (debug)"
SAME_CCY = "Underlying currency"         # payment currency of a product that is not quanto
PAYMENT_CCY = "Payment currency"         # price currency: the payment one (no conversion)
NO_TICKER = "None"                       # an empty extra underlying (ignored)
MAX_UNDERLYINGS = 5                      # underlyings of a worst-of autocallable (the form adds them one by one)
WORST_OF_MODELS = ("Local vol", "GBM (debug)")    # models of a worst-of (one normal per step and underlying)


class ModelCache:
    """Model instances per ``(snapshot, ticker, model)``: the local vol built on the snapshot, the calibrated models
    (Heston, displaced diffusion, Heston SLV) loaded from the latest calibration set with the underlying (their
    parameters held on the current market; ``source_set`` names the set)."""

    def __init__(self, calibrations=None):
        self._models: dict = {}
        self._calibrations = calibrations

    def get(self, eq, name: str, snapshot: str = "") -> tuple:
        """``(model, seconds spent calibrating now)``: 0 when it was already there (or loaded from a set)."""
        key = (snapshot, eq.ticker, name)
        if key in self._models:
            return self._models[key], 0.0
        t0 = time.perf_counter()
        if name == "Local vol":
            model = tq.LocalVolatilityModel.calibrate(eq)
        elif name in CALIBRATED:
            model = self._calibrated(eq.ticker, CALIBRATED[name], name)
        else:
            raise ValueError(f"unknown model {name}")
        self._models[key] = model
        return model, time.perf_counter() - t0 if name == "Local vol" else 0.0

    def _calibrated(self, ticker: str, stored: str, label: str):
        from tensorquant.calibration.models import list_calibration_sets, load_calibration_set, load_model

        for s in list_calibration_sets(self._calibrations) if self._calibrations else []:
            summary = load_calibration_set(s["path"])["summary"]
            ok = summary[(summary["ticker"] == ticker) & (summary["model"] == stored) & (summary["status"] == "ok")]
            if len(ok):
                model = load_model(s["path"], ticker, stored)
                model.source_set = f"{s['date']} {s['set']}"
                return model
        raise ValueError(f"no calibration set with {label} for {ticker}: run one in the Calibration tab")

    def clear(self) -> None:
        self._models.clear()


# ----------------------------------------------------------------------------------------------------------------
# product catalog: fields and models
# ----------------------------------------------------------------------------------------------------------------

def _group(name: str, *fields) -> list:
    """Fields shown together under the heading *name* (a sub-panel of the deal form)."""
    return [{**f, "group": name} for f in fields]


def _select(name, label, options, default=None):
    return {"name": name, "label": label, "type": "select", "options": list(options),
            "default": default if default is not None else (list(options) or [None])[0]}


def _number(name, label, default, step=1.0):
    return {"name": name, "label": label, "type": "number", "default": default, "step": step}


def _date(name, label, default):
    return {"name": name, "label": label, "type": "date", "default": default}


def _rate_currencies(mkt: tq.MarketEnvironment, needs: str) -> list[str]:
    """Currencies with the market data a rates product needs: ``"ois"`` (default overnight curve), ``"ibor"`` (the
    6M curve), ``"caplet"`` (its caplet vols), ``"swaption"`` (its swaption cube)."""
    keys = mkt.risk_factors()
    out = []
    for ccy in ("EUR", "USD", "GBP"):
        ok = {"ois": any(k.startswith(f"IR:{ccy}:") and k.endswith(":SPOT") and k.split(":")[2] != IBOR for k in keys),
              "ibor": f"IR:{ccy}:{IBOR}:SPOT" in keys,
              "caplet": f"IR:{ccy}:{IBOR}:VOL" in keys,
              "swaption": f"IR:{ccy}:{IBOR}:SWVOL" in keys}[needs]
        if ok:
            out.append(ccy)
    return out


FORWARD_STARTS = ["Spot", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y"]


def swap_generators(mkt: tq.MarketEnvironment) -> dict:
    """The swap generators of the curves in *mkt*, from their market conventions (``QUOTE_CONVENTIONS``, the same
    ones the curves were bootstrapped with): ``{label: {ccy, curve, kind, conventions, discount, projection}}``."""
    from tensorquant.markethandles.bootstrapping import QUOTE_CONVENTIONS

    keys = mkt.risk_factors()
    out = {}
    for (ccy, name), conv in QUOTE_CONVENTIONS.items():
        if f"IR:{ccy}:{name}:SPOT" not in keys:
            continue
        cal = type(conv["calendar"]()).__name__
        lag = f"T+{conv['spot']}"
        if conv["kind"] == "ois":
            dc = conv["day_count"].name
            label = f"{ccy} {name} OIS"
            legs = {"fixed": f"1Y, {dc}", "floating": f"{name} compounded, 1Y, {dc}"}
            discount = projection = f"IR:{ccy}:{name}:SPOT"
        else:
            if f"IR:{ccy}:{conv['discount']}:SPOT" not in keys:
                continue
            label = f"{ccy} {name} Euribor IRS" if ccy == "EUR" else f"{ccy} {name} IRS"
            legs = {"fixed": f"{conv['fixed_period']}, {conv['fixed_day_count'].name}",
                    "floating": f"{ccy} {name} index, {conv['index_months']}M, {conv['float_day_count'].name}"}
            discount, projection = f"IR:{ccy}:{conv['discount']}:SPOT", f"IR:{ccy}:{name}:SPOT"
        out[label] = {"ccy": ccy, "curve": name, "kind": conv["kind"], "discount": discount, "projection": projection,
                      "spot": int(conv["spot"]), "calendar": conv["calendar"],
                      "conventions": {"Spot lag": lag, "Calendar": cal, "Roll": "Modified Following",
                                      "Fixed leg": legs["fixed"], "Floating leg": legs["floating"],
                                      "Discount curve": discount, "Projection curve": projection}}
    return out


def catalog(mkt: tq.MarketEnvironment) -> dict:
    """Asset classes -> product types -> ``{fields, models}`` for the market *mkt* (the form of the Tradepad)."""
    tickers = equity_tickers(mkt)
    one_year = date(mkt.evaluation_date.year + 1, mkt.evaluation_date.month, mkt.evaluation_date.day).isoformat()
    # autocallable defaults: strike date 10 days after the pricing date, maturity 3 years after the strike date
    strike_default = mkt.evaluation_date + timedelta(days=10)
    maturity_default = date(strike_default.year + 3, strike_default.month, min(strike_default.day, 28)
                            if strike_default.month == 2 else strike_default.day).isoformat()
    underlying = _select("ticker", "Underlying", tickers)
    option = [underlying, _select("option_type", "Type", ["Put", "Call"]), _number("strike_pct", "Strike (% spot)", 100.0, 0.5),
              _date("expiry", "Expiry", one_year), _number("quantity", "Quantity", 1.0)]
    equity_models = ["Black-Scholes", "Local vol", *CALIBRATED]
    ccys = _rate_currencies(mkt, "ois")
    payment = [_select("payment_ccy", "Payment currency", [SAME_CCY, *ccys], SAME_CCY),
               _select("fx_mode", "FX treatment", ["Quanto", "Composite"], "Quanto"),
               _select("price_ccy", "Price currency", [PAYMENT_CCY, *ccys], PAYMENT_CCY)]

    def rates_common(needs):
        return [_select("ccy", "Currency", _rate_currencies(mkt, needs)), _number("notional", "Notional", 1e6, 1e5),
                _select("position", "Position", ["Long", "Short"])]

    def swap_side():
        return _select("side", "Side", ["Payer", "Receiver"])

    products = {
        "Equity": {
            "Vanilla": {"fields": option[:2] + [_select("exercise", "Exercise", ["European", "American"])] + option[2:]
                        + payment, "models": equity_models},
            "Digital": {"fields": option + [_select("payoff", "Payoff", ["Cash", "Asset"]),
                                            _number("cash_amount", "Cash amount", 100.0)], "models": equity_models},
            "Barrier": {"fields": option + [_select("barrier_type", "Barrier type", [b.value for b in tq.BarrierType]),
                                            _number("barrier_pct", "Barrier (% spot)", 70.0, 0.5)],
                        "models": equity_models},
        },
        "Rates": {
            "Deposit": {"fields": rates_common("ois") + [_select("term", "Term", SHORT_TERMS, "6M"),
                                                         _number("rate", "Rate (%)", 2.0, 0.01)],
                        "models": ["Discounting"]},
            "FRA": {"fields": rates_common("ibor") + [_select("start", "Start", ["1M", "3M", "6M", "9M", "12M", "18M"], "6M"),
                                                      _number("rate", "Rate (%)", 2.5, 0.01)],
                    "models": ["Discounting"]},
            "Swap": {"fields": [_select("generator", "Generator", list(swap_generators(mkt)))]
                     + _group("Trade", _select("side", "Fixed leg", ["Pay fixed", "Receive fixed"]),
                              _number("notional", "Notional", 1e7, 1e6),
                              _select("forward", "Start", FORWARD_STARTS, "Spot"),
                              _select("tenor", "Maturity", TENORS, "5Y"))
                     + _group("Rates", _number("rate", "Fixed rate (%)", 2.5, 0.01),
                              _number("spread_bp", "Floating spread (bp)", 0.0, 1.0)),
                     "models": ["Discounting"], "legs": True},
            "Cap/Floor": {"fields": rates_common("caplet") + [_select("cap_floor", "Type", ["Cap", "Floor"]),
                                                              _select("tenor", "Maturity", TENORS[:-1], "5Y"),
                                                              _number("strike", "Strike (%)", 2.5, 0.05)],
                          "models": ["Analytic (caplet vols)"]},
            "Collar": {"fields": rates_common("caplet") + [_select("tenor", "Maturity", TENORS[:-1], "5Y"),
                                                           _number("cap_strike", "Cap strike (%)", 3.5, 0.05),
                                                           _number("floor_strike", "Floor strike (%)", 1.5, 0.05)],
                       "models": ["Analytic (caplet vols)"]},
            "Swaption": {"fields": rates_common("swaption") + [_select("payer", "Type", ["Payer", "Receiver"]),
                                                               _select("expiry", "Expiry", ["6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"], "1Y"),
                                                               _select("tenor", "Swap tenor", ["1Y", "2Y", "5Y", "10Y", "20Y"], "5Y"),
                                                               _number("strike", "Strike (%)", 2.8, 0.05)],
                         "models": ["Analytic (swaption cube)"]},
        },
        "Exotics": {
            "Autocallable": {"fields": [underlying]
                             + _group("Deal parameters", _number("notional", "Notional", 100.0),
                                      # one currency: the flows are paid and the price is shown in it
                                      _select("payment_ccy", "Currency", ccys, "EUR" if "EUR" in ccys else None),
                                      payment[1],
                                      _date("strike_date", "Strike date", strike_default.isoformat()),
                                      _date("maturity", "Maturity", maturity_default))
                             + _group("Coupon", _select("coupon_type", "Type", ["Standard", "Memory"]),
                                      _number("coupon", "Coupon per period (%)", 2.0, 0.25),
                                      _number("coupon_barrier", "Coupon barrier (%)", 80.0, 1.0),
                                      _select("coupon_frequency", "Frequency", ["3M", "6M", "1Y"], "6M"))
                             + _group("Final redemption", _number("put_barrier", "Put barrier (%)", 80.0, 1.0),
                                      _number("put_strike", "Put strike (%)", 100.0, 1.0),
                                      _number("put_participation", "Put participation (%)", 100.0, 5.0))
                             + _group("Early redemption (autocall)",
                                      _number("autocall_barrier", "Autocall barrier (%)", 100.0, 1.0),
                                      _number("autocall_step", "Step-down per observation (%)", 0.0, 0.5),
                                      _number("autocall_floor", "Step-down floor (%)", 80.0, 1.0),
                                      _select("autocall_frequency", "Frequency", ["3M", "6M", "1Y"], "6M"),
                                      _select("first_autocall", "First observation", ["3M", "6M", "1Y", "2Y"], "1Y")),
                             "models": ["Local vol", *CALIBRATED, GBM_DEBUG], "create_trade": True,
                             "more_underlyings": MAX_UNDERLYINGS - 1,       # worst-of: ticker_2, ticker_3, ...
                             "spots": {t: float(mkt.get_equity(t).spot) for t in tickers}},
        },
    }
    products["Rates"] = {name: spec for name, spec in products["Rates"].items()      # products with data only
                         if spec["fields"][0]["options"]}
    return {"snapshot": mkt.evaluation_date.isoformat(), "asset_classes": products}


# ----------------------------------------------------------------------------------------------------------------
# building the products
# ----------------------------------------------------------------------------------------------------------------

def _payment_ccy(f: dict, eq):
    """The payment currency of the deal fields: ``None`` when it is the underlying's (not quanto)."""
    name = f.get("payment_ccy") or SAME_CCY
    if name == SAME_CCY or name == eq.currency.name:
        return None
    return tq.Currency[name]


def _fx_mode(f: dict) -> str:
    return "composite" if f.get("fx_mode") == "Composite" else "quanto"


def _composite_fx(mkt, f: dict, eq) -> float:
    """FX spot of a composite (1 unit of the underlying's currency in the payment one), 1 otherwise: the levels of
    a composite are in the payment currency (``S X``)."""
    pay = _payment_ccy(f, eq)
    return float(mkt.get_fx_spot(eq.currency, pay)) if pay is not None and _fx_mode(f) == "composite" else 1.0


def _price_ccy(f: dict):
    name = f.get("price_ccy") or PAYMENT_CCY
    return None if name == PAYMENT_CCY else tq.Currency[name]


def _equity_option(mkt, f: dict, kind: str):
    eq = mkt.get_equity(f["ticker"])
    s0 = float(f.get("_ref_spot", eq.spot))   # strikes on the market spot, not on an override
    expiry = date.fromisoformat(f["expiry"])
    if expiry <= mkt.evaluation_date:
        raise ValueError("the expiry must be after the snapshot date")
    option_type = tq.OptionType.Call if f["option_type"] == "Call" else tq.OptionType.Put
    common = (eq.currency, mkt.evaluation_date, expiry, option_type, s0 * float(f["strike_pct"]) / 100)
    if kind == "Vanilla":                          # composite: the strike in % of the spot in the payment currency
        exercise = tq.ExerciseType.American if f.get("exercise") == "American" else tq.ExerciseType.European
        strike = common[4] * _composite_fx(mkt, f, eq)
        return tq.VanillaOption(*common[:4], strike, underlying=eq.ticker, exercise_type=exercise,
                                payment_ccy=_payment_ccy(f, eq), fx_mode=_fx_mode(f))
    if kind == "Digital":
        payoff = tq.PayoffType.AssetOrNothing if f.get("payoff") == "Asset" else tq.PayoffType.CashOrNothing
        return tq.DigitalOption(*common, payoff_type=payoff, cash_amount=float(f.get("cash_amount", 1.0)),
                                underlying=eq.ticker)
    return tq.BarrierOption(*common, barrier=s0 * float(f["barrier_pct"]) / 100,
                            barrier_type=tq.BarrierType(f["barrier_type"]), underlying=eq.ticker)


def _months(term: str) -> int:
    return int(term[:-1]) * (12 if term.endswith("Y") else 1)


def _backward_dates(today: date, maturity: date, months: int) -> list:
    """Observation dates rolled backward from the maturity every *months* (each from the unadjusted maturity, then
    Modified Following): the maturity is always an observation and schedules of different frequencies align."""
    out, k = [], 0
    while True:
        d = maturity - relativedelta(months=k * months)
        if d <= today:
            return sorted(set(out))
        out.append(CAL.adjust(d, MF))
        k += 1


def _tickers(f: dict) -> list:
    """Underlyings of the autocallable: the first, then the worst-of ones (distinct)."""
    extra = sorted((k for k in f if re.fullmatch(r"ticker_\d+", k)), key=lambda k: int(k.split("_")[1]))
    tickers = [f["ticker"]] + [f[k] for k in extra if f.get(k) not in (None, "", NO_TICKER)]
    if len(tickers) > MAX_UNDERLYINGS:
        raise ValueError(f"at most {MAX_UNDERLYINGS} underlyings")
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"the same underlying twice: {tickers}")
    return tickers


def _autocallable(mkt, f: dict):
    """Phoenix autocallable: coupons on their own schedule (standard or memory), autocall observations from the first
    observation date to maturity, capital-at-risk put (barrier, strike, participation). Struck at the market spot
    when the strike date is the pricing date; forward start when it is later (the strike is the spot fixed then,
    the schedules follow the strike date); a live trade when it is earlier (the strike and the past observations
    are the fixings of the snapshot, ``eq_fixings.csv``)."""
    eq = mkt.get_equity(f["ticker"])
    today = mkt.evaluation_date
    strike_date = CAL.adjust(date.fromisoformat(f.get("strike_date") or today.isoformat()), MF)
    forward_start = strike_date > today
    struck_today = strike_date == today                               # earlier: a live trade, struck on its fixing
    maturity = date.fromisoformat(f["maturity"])
    end = CAL.adjust(maturity, MF)
    if end <= strike_date:
        raise ValueError("the maturity must be after the strike date")
    first = CAL.advance(strike_date, _months(f["first_autocall"]), tq.TimeUnit.Months, MF)
    if first >= end:
        raise ValueError("the first autocall observation must be before the maturity")
    coupons = _backward_dates(strike_date, maturity, _months(f["coupon_frequency"]))
    autocalls = [d for d in _backward_dates(strike_date, maturity, _months(f["autocall_frequency"])) if d >= first]
    step = float(f.get("autocall_step", 0.0) or 0.0)                      # step-down, with its floor only then
    tickers = _tickers(f)
    if len(tickers) > 1:                    # worst-of: in the payment currency, the foreign underlyings quanto
        if _fx_mode(f) == "composite":
            raise ValueError("worst-of: composite not supported (quanto only)")
        ccy, payment, underlying = _payment_ccy(f, eq) or eq.currency, None, tickers
        strike = None if not struck_today else [float(f.get("_ref_spot", eq.spot))] + [
            float(mkt.get_equity(t).spot) for t in tickers[1:]]
    else:
        ccy, payment, underlying = eq.currency, _payment_ccy(f, eq), eq.ticker
        strike = None if not struck_today else float(f.get("_ref_spot", eq.spot)) * _composite_fx(mkt, f, eq)
    return tq.AutocallableOption(
        ccy=ccy, notional=float(f["notional"]), start_date=min(today, strike_date), end_date=end,
        strike=strike,
        strike_date=strike_date,
        coupon_fixing_dates=coupons, coupon_payment_dates=coupons,
        coupon_rates=[float(f["coupon"])], coupon_barriers=[float(f["coupon_barrier"])],
        memory=f["coupon_type"] == "Memory", autocall_fixing_dates=autocalls, autocall_payment_dates=autocalls,
        autocall_barrier=[float(f["autocall_barrier"])], payoff_barrier=float(f["put_barrier"]),
        payoff_participation=float(f["put_participation"]) / 100, put_strike=float(f["put_strike"]),
        underlying=underlying, autocall_step=step, payment_ccy=payment, fx_mode=_fx_mode(f),
        autocall_floor=float(f["autocall_floor"]) if step > 0 and f.get("autocall_floor") not in (None, "") else None)


def _ibor(ccy):
    return tq.IborIndex(CAL, int(IBOR[:-1]), tq.TimeUnit.Months, ccy)


def _rates_product(mkt, f: dict, kind: str, rate_override: float = None):
    ccy = tq.Currency[f["ccy"]]
    notional = float(f["notional"])
    rate = float(f.get("rate", 0.0)) / 100 if rate_override is None else rate_override
    if kind == "Deposit":
        return tq.DepositGenerator(ccy, 2, MF, A360, notional, CAL).build(mkt.evaluation_date, rate, f["term"])
    if kind == "FRA":
        start = int(f["start"][:-1])
        return tq.FraGenerator(ccy, 2, 2, IBOR, MF, notional, A360, CAL, _ibor(ccy)).build(
            mkt.evaluation_date, rate, f"{start}M-{start + int(IBOR[:-1])}M")
    if kind == "Swap":                                  # the generator of the curve conventions (as the bootstrap)
        g = f["_generator"]
        forward = None if f.get("forward", "Spot") == "Spot" else f["forward"]
        return tq.swap_generator(g["ccy"], g["curve"], notional).build(
            mkt.evaluation_date, rate, f["tenor"], forward=forward, spread=float(f.get("spread_bp", 0.0)) / 1e4)
    caps = tq.CapFloorGenerator(ccy, 2, IBOR, MF, notional, A360, CAL, _ibor(ccy))
    if kind == "Cap/Floor":
        option_type = tq.OptionType.Call if f["cap_floor"] == "Cap" else tq.OptionType.Put
        return caps.build(mkt.evaluation_date, float(f["strike"]) / 100, f["tenor"], option_type)
    if kind == "Collar":
        return caps.build_collar(mkt.evaluation_date, float(f["cap_strike"]) / 100, float(f["floor_strike"]) / 100,
                                 f["tenor"])
    if kind == "Swaption":
        swaps = tq.SwapGenerator(ccy, 2, "1Y", IBOR, MF, notional, A360, A360, CAL, _ibor(ccy))
        return tq.SwaptionGenerator(swaps).build(mkt.evaluation_date, f["expiry"], f["tenor"], float(f["strike"]) / 100,
                                                 payer=f["payer"] == "Payer")
    raise ValueError(f"unknown product {kind}")


# ----------------------------------------------------------------------------------------------------------------
# greeks
# ----------------------------------------------------------------------------------------------------------------

def _to_numpy(grads: dict) -> dict:
    return {k: [np.zeros(()) if x is None else np.asarray(x.numpy() if hasattr(x, "numpy") else x, float) for x in v]
            for k, v in grads.items()}


def _rho_buckets(mkt, greeks: dict, sign: float) -> list[dict]:
    """Rho per pillar (1 bp) of every curve the price depends on."""
    out = []
    for key, blocks in greeks.items():
        if not (key.startswith("IR:") and key.endswith(":SPOT")):
            continue
        values = np.array([float(np.sum(b)) for b in blocks]) * 1e-4 * sign
        if not np.any(values):
            continue
        _, ccy, name, _ = key.split(":")
        curve = mkt.get_ir_curve(tq.Currency[ccy], name)
        out.append({"curve": f"{ccy} {name}", "dates": [d.isoformat() for d in curve.dates],
                    "rho": values.tolist(), "total": float(values.sum())})
    return out


def _sensitivity_table(mkt, greeks: dict, sign: float) -> dict | None:
    """Rho of a rates product per curve pillar (1 bp) in two blocks: Par (to the quotes the curves were bootstrapped
    from, :func:`par_sensitivities`; ``None`` for a curve without a stored Jacobian) and Zero (to the zero rates of the
    pillars). Rows: the pillars of the curves by maturity, labelled by the tenor of their calibration instrument
    (pillar dates for curves not built from quotes); the first row is the total.

    Returns:
        dict: ``curves`` (names), ``par_available`` (per curve), ``total`` and ``rows`` (``label``, ``par`` and
        ``zero``: one value per curve, ``None`` where the curve has no such pillar), or ``None`` without curves.
    """
    from tensorquant.markethandles.bootstrapping import par_sensitivities

    zero_keys = [k for k, blocks in greeks.items() if k.startswith("IR:") and k.endswith(":SPOT")
                 and any(np.any(np.asarray(b)) for b in blocks)]
    if not zero_keys:
        return None
    par = par_sensitivities(mkt, {k: greeks[k] for k in zero_keys})
    keys = list(dict.fromkeys(zero_keys + [k for k, v in par.items() if v is not None and np.any(v.to_numpy())]))
    rows, names = {}, []
    for j, key in enumerate(keys):
        _, ccy, name, _ = key.split(":")
        names.append(f"{ccy} {name}")
        curve = mkt.get_ir_curve(tq.Currency[ccy], name)
        dv01 = getattr(curve, "calib_quote_dv01", None)
        def tenor_label(label: str) -> str:              # 12M -> 1Y, 24M -> 2Y: one row per maturity across curves
            n = label[:-1]
            return f"{int(n) // 12}Y" if label.endswith("M") and n.isdigit() and int(n) % 12 == 0 else label

        labels = ([tenor_label(lab.split(" ", 1)[1]) for lab in dv01.index]
                  if dv01 is not None and len(dv01) == len(curve.pillars) else [_tenor(float(t)) for t in curve.pillars])
        zero = [float(np.sum(b)) * 1e-4 * sign for b in greeks[key]] if key in greeks else [0.0] * len(labels)
        par_values = par.get(key)
        for i, (label, t) in enumerate(zip(labels, curve.pillars)):
            row = rows.setdefault(label, {"label": label, "T": float(t), "par": [None] * len(keys),
                                          "zero": [None] * len(keys)})
            row["T"] = min(row["T"], float(t))
            row["zero"][j] = zero[i]
            if par_values is not None:
                row["par"][j] = float(par_values.iloc[i]) * 1e-4 * sign
    ordered = sorted(rows.values(), key=lambda r: r["T"])
    available = [par.get(k) is not None for k in keys]
    total = {"par": [sum(r["par"][j] or 0.0 for r in ordered) if available[j] else None for j in range(len(keys))],
             "zero": [sum(r["zero"][j] or 0.0 for r in ordered) for j in range(len(keys))]}
    return {"curves": names, "par_available": available, "total": total, "rows": ordered}


def _equity_vega_map(eq, greeks: dict, scale: float) -> dict:
    vol_key = next(k for k in eq.risk_factors() if k.startswith("EQ:") and k.endswith(":VOL"))
    vega = scale * np.asarray(greeks[vol_key][0], float) / 100
    vs = eq.vol_surface
    s0 = float(eq.spot)
    return {"title": "Vega per surface node (1 vol point)", "z": vega.tolist(),
            "x": [f"{100 * float(k) / s0:.0f}%" for k in vs.strike], "x_title": "strike / spot",
            "y": [f"{float(t):.2f}y" for t in vs.maturity], "y_title": "expiry", "total": float(vega.sum())}


def _tenor(t: float) -> str:
    return f"{round(12 * t)}M" if t < 1 else f"{t:.0f}Y" if abs(t - round(t)) < 0.02 else f"{t:.1f}Y"


# ----------------------------------------------------------------------------------------------------------------
# market inputs and overrides
# ----------------------------------------------------------------------------------------------------------------

def _fields(mkt, request: dict) -> tuple:
    spec = catalog(mkt)["asset_classes"].get(request.get("asset_class"), {}).get(request.get("product"))
    if spec is None:
        raise ValueError(f"unknown product {request.get('asset_class')} / {request.get('product')}")
    fields = {fld["name"]: fld["default"] for fld in spec["fields"]}
    fields.update(request.get("fields", {}))
    if "generator" in fields:                      # swaps: currency and curves from the generator
        generator = swap_generators(mkt).get(fields["generator"])
        if generator is None:
            raise ValueError(f"unknown swap generator {fields['generator']}")
        fields["ccy"], fields["_generator"] = generator["ccy"], generator
    return spec, fields


def _equity_strike_expiry(mkt, kind: str, f: dict) -> tuple:
    """Strike and expiry the market inputs refer to (the autocallable: its initial fixing and final date)."""
    product = _autocallable(mkt, f) if kind == "Autocallable" else _equity_option(mkt, f, kind)
    strike = (product.strike[0] if isinstance(product.strike, list) else product.strike
              if product.strike is not None else float(f.get("_ref_spot", mkt.get_equity(f["ticker"]).spot)))
    return float(strike), product.end_date              # forward start: the spot (the strike is not known yet)


def market_inputs(mkt: tq.MarketEnvironment, request: dict) -> dict:
    """Market data the deal of *request* uses: equity: spot, vol at the strike and expiry, zero rate of the discount
    curve at the expiry (and its name), forward, PV of the dividends, repo; rates: the curves and their zero rates
    at the maturity.

    Returns:
        dict: ``editable`` (``{spot, vol}`` for equity, else ``{}``) and ``info`` (``[label, value, decimals]``).
    """
    _, f = _fields(mkt, request)
    if request["asset_class"] in ("Equity", "Exotics"):
        eq = mkt.get_equity(f["ticker"])
        strike, expiry = _equity_strike_expiry(mkt, request["product"], f)
        T = eq.year_fraction(expiry)
        curve = eq.discount_curve
        _, ccy, name, _ = curve.name.split(":")
        s0 = float(eq.spot)
        dividends = []                                 # discrete dividends with ex-date up to the expiry
        if eq.dividends is not None:
            pays = eq.dividends.payment_dates or eq.dividends.ex_dates
            for ex, pay, amount in zip(eq.dividends.ex_dates, pays, eq.dividends.amounts):
                if ex <= expiry:
                    dividends.append({"ex_date": ex.isoformat(), "pay_date": pay.isoformat(), "amount": float(amount),
                                      "pct_spot": 100 * float(amount) / s0,
                                      "pv": float(amount) * float(curve.discount(pay))})
        div_model = {"model": eq.dividend_model,
                     "yield": None if eq.div_yield is None else float(eq.div_yield)}
        editable = {"spot": s0}
        if eq.vol_surface is not None:
            editable["vol"] = float(eq.vol_surface.volatility(strike=strike, tenor=T))
        return {"editable": editable,
                "info": [["Rate (zero, %)", 100 * float(curve.zero_rate(T)), 3], ["Curve", f"{ccy} {name}", None],
                         ["Forward", float(eq.forward(T)), 2], ["PV dividends", float(eq.pv_dividends(T)), 2],
                         ["Repo (%)", 100 * float(eq.repo_rate(expiry)), 3], ["Strike", strike, 2],
                         ["Expiry", f"{expiry.isoformat()} ({T:.2f}y)", None]],
                "dividends": dividends, "dividend_model": div_model}
    product = _rates_product(mkt, f, request["product"])
    end = getattr(product, "end_date", None) or getattr(product, "expiry", None)
    if end is None and hasattr(product, "swap"):
        end = product.swap.end_date
    ccy = f["ccy"]
    info = []
    for key in sorted(k for k in mkt.risk_factors() if k.startswith(f"IR:{ccy}:") and k.endswith(":SPOT")):
        _, _, name, _ = key.split(":")
        curve = mkt.get_ir_curve(tq.Currency[ccy], name)
        if end is not None:
            info.append([f"{ccy} {name} zero at maturity (%)", 100 * float(curve.zero_rate(end)), 3])
    if end is not None:
        info.append(["Maturity", end.isoformat(), None])
    return {"editable": {}, "info": info}


def swap_legs(mkt: tq.MarketEnvironment, f: dict) -> dict:
    """The swap of the deal fields *f* and its two legs on the market: per leg its direction, conventions, the curves
    it is indexed to and discounted on, and one row per coupon (accrual start / end, payment date, fixing date,
    year fraction, rate (fixed, or the index forward / past fixing), spread, amount, discount factor, PV), signed
    for the deal's side (paid < 0, received > 0).

    Returns:
        dict: ``summary`` (``[label, value, decimals]``: generator conventions, dates), ``legs`` (fixed, floating)
        and ``npv`` (sum of the leg PVs).
    """
    from tensorquant.pricers.floatingflow import FloatingCouponDiscounting, OisCouponDiscounting

    g = f["_generator"]
    product = _rates_product(mkt, f, "Swap")
    ccy = tq.Currency[g["ccy"]]
    discount = mkt.get_ir_curve(ccy, g["discount"].split(":")[2])
    projection = mkt.get_ir_curve(ccy, g["projection"].split(":")[2])
    pay_fixed = f.get("side", "Pay fixed") == "Pay fixed"
    today = mkt.evaluation_date
    fixed_rows, float_rows = [], []
    for c in product.fixed_leg.leg_flows:
        if c.date <= today:
            continue
        df = float(discount.discount(c.date))
        amount = (-1 if pay_fixed else 1) * float(c.amount)
        fixed_rows.append({"start": c.accrual_start_date.isoformat(), "end": c.accrual_end_date.isoformat(),
                           "payment": c.date.isoformat(), "fixing": None, "year_fraction": float(c.accrual_period),
                           "rate": float(product.fixed_rate), "spread": None, "amount": amount, "df": df,
                           "pv": amount * df})
    ois = g["kind"] == "ois"
    cal = g["calendar"]()

    def fixing_date(start):                             # the index fixing: spot lag business days before the start
        d, n = start, g["spot"]
        while n > 0:
            d -= timedelta(days=1)
            if cal.is_business_day(d):
                n -= 1
        return d

    for c in product.floating_leg.leg_flows:
        if c.date <= today:
            continue
        flow = OisCouponDiscounting(c) if ois else FloatingCouponDiscounting(c)
        rate = float(flow.floating_rate(c.ref_period_start, c.ref_period_end, projection))
        spread = float(c._spread)
        df = float(discount.discount(c.date))
        amount = (1 if pay_fixed else -1) * c.nominal * (rate + spread) * float(c.accrual_period)
        fixing = None if ois else fixing_date(c.accrual_start_date)
        float_rows.append({"start": c.accrual_start_date.isoformat(), "end": c.accrual_end_date.isoformat(),
                           "payment": c.date.isoformat(), "fixing": fixing.isoformat() if fixing else None,
                           "year_fraction": float(c.accrual_period), "rate": rate, "spread": spread, "amount": amount,
                           "df": df, "pv": amount * df,
                           "known": bool(fixing is not None and fixing <= today)})
    conv = g["conventions"]
    fixed_name, float_name = ("Pay", "Receive") if pay_fixed else ("Receive", "Pay")
    legs = [{"name": "Fixed leg", "direction": fixed_name, "conventions": conv["Fixed leg"],
             "index": None, "projection": None, "discount": g["discount"], "rows": fixed_rows,
             "pv": sum(r["pv"] for r in fixed_rows)},
            {"name": "Floating leg", "direction": float_name, "conventions": conv["Floating leg"],
             "index": f"{g['ccy']} {g['curve']}" + (" overnight (compounded)" if ois else " Ibor"),
             "projection": g["projection"], "discount": g["discount"], "rows": float_rows,
             "pv": sum(r["pv"] for r in float_rows)}]
    summary = [["Generator", f["generator"], None], ["Start date", product.start_date.isoformat(), None],
               ["Maturity date", product.end_date.isoformat(), None]] + \
              [[k, v, None] for k, v in conv.items() if k not in ("Fixed leg", "Floating leg")]
    return {"summary": summary, "legs": legs, "npv": legs[0]["pv"] + legs[1]["pv"]}


def describe_trade(mkt: tq.MarketEnvironment, request: dict) -> dict:
    """The trade built from the deal fields, before pricing: its main terms and, for the autocallable, the schedule
    (one row per fixing: fixing date, autocall date when it is an autocall observation, payment date, coupon
    barrier and coupon amount in % of the strike / notional).

    Returns:
        dict: ``summary`` (``[label, value, decimals]``) and ``schedule`` (list of rows, or ``None``).
    """
    _, f = _fields(mkt, request)
    if request.get("product") == "Swap":
        return {**swap_legs(mkt, f), "schedule": None}
    if request.get("product") != "Autocallable":
        return {"summary": [], "schedule": None}
    product = _autocallable(mkt, f)
    table = product.schedule_dataframe()
    state = tq.AutocallableMCPricer().trade_state(product, mkt)
    past = {r["date"]: r for r in state["rows"]}
    schedule = []
    for _, row in table.iterrows():
        coupon = row["coupon_rate_pct"] is not None and row["coupon_rate_pct"] == row["coupon_rate_pct"]
        autocall = row["autocall_barrier_pct"] is not None and row["autocall_barrier_pct"] == row["autocall_barrier_pct"]
        schedule.append({"fixing_date": row["fixing_date"].isoformat(),
                         "autocall_date": row["fixing_date"].isoformat() if autocall else None,
                         "payment_date": row["payment_date"].isoformat(),
                         "coupon_barrier": float(row["coupon_barrier_pct"]) if coupon else None,
                         "coupon_amount": float(row["coupon_rate_pct"]) if coupon else None,
                         "autocall_barrier": float(row["autocall_barrier_pct"]) if autocall else None,
                         **_past_outcome(past.get(row["fixing_date"]), state["called_on"], row["fixing_date"])})
    strikes = state["strikes"]
    strike = (f"fixed on {product.strike_date.isoformat()} (forward start)" if strikes is None
              else ", ".join(f"{t} {k:,.2f}" for t, k in zip(product.underlyings, strikes))
              if product.is_worst_of else float(strikes[0]))
    summary = [["Strike (initial fixing)", strike, 2], ["Strike date", product.strike_date.isoformat(), None],
               *([["Worst-of", " / ".join(product.underlyings), None]] if product.is_worst_of else []),
               ["Maturity", product.end_date.isoformat(), None],
               *([["Status", f"called on {state['called_on'].isoformat()}" if state["called_on"] else
                   f"live, {len(state['rows'])} observations fixed"
                   + (f", {state['unpaid_coupons']} coupons in memory" if state["unpaid_coupons"] else ""), None]]
                 if state["rows"] else []),
               ["Notional", float(product.notional), 2],
               ["Payment currency", f"{product.payment_ccy.name} (quanto)" if product.is_quanto
                else f"{product.payment_ccy.name} (composite: levels on S X)" if product.is_composite
                else _worst_of_ccy(mkt, product), None],
               ["Coupon type", "Memory" if product.memory else "Standard", None],
               ["Put", f"barrier {product.payoff_barrier:g}%, strike {product.put_strike:g}%, "
                       f"participation {100 * product.payoff_participation:g}%", None],
               ["Autocall", f"barrier {product.autocall_barrier[0]:g}%"
                + (f", step-down {product.autocall_step:g}% per observation"
                   + (f", floor {product.autocall_floor:g}%" if product.autocall_floor is not None else "")
                   if product.autocall_step else ""), None],
               ["Coupons / autocall dates", f"{len(product.coupon_fixing_dates)} / {len(product.autocall_fixing_dates)}",
                None]]
    return {"summary": summary, "schedule": schedule, "market_data": _autocallable_market_data(mkt, product)}


def _autocallable_market_data(mkt, product) -> dict:
    """Market data an autocallable is priced on (the Market Data tab): per underlying the spot, the level at the
    strike date (forward if the strike date is future, the spot if it is today, the historical fixing if it is past),
    the repo and the dividends up to the maturity, the quanto inputs; the discount curves (payment currency and the
    underlyings' currencies) with their zero rates up to the maturity."""
    today = mkt.evaluation_date
    end = product.end_date
    pay_ccy = product.payment_ccy
    strike_date = product.strike_date
    underlyings = []
    for i, ticker in enumerate(product.underlyings):
        eq = mkt.get_equity(ticker)
        spot = float(eq.spot)
        if strike_date > today:
            level, source = float(eq.forward(strike_date)), f"forward to {strike_date.isoformat()}"
        elif strike_date == today:
            level, source = (product.strikes[i] if product.strikes else spot), "spot (struck today)"
        else:
            given = product.fixings.get(strike_date)
            if given is not None:
                level = float(given[i] if isinstance(given, (list, tuple)) else given)
                source = f"fixing of {strike_date.isoformat()} (on the trade)"
            else:
                level = float(mkt.get_eq_fixing(ticker, strike_date))
                source = f"fixing of {strike_date.isoformat()} (close)"
        dividends = []
        if eq.dividends is not None:
            pays = eq.dividends.payment_dates or eq.dividends.ex_dates
            for ex, pay, amount in zip(eq.dividends.ex_dates, pays, eq.dividends.amounts):
                if today < ex <= end:
                    dividends.append({"ex_date": ex.isoformat(), "pay_date": pay.isoformat(), "amount": float(amount),
                                      "pct_spot": 100 * float(amount) / spot,
                                      "pv": float(amount) * float(eq.discount_curve.discount(pay))})
        row = {"ticker": ticker, "ccy": eq.currency.name, "spot": spot, "strike_level": level, "strike_source": source,
               "repo": 100 * float(eq.repo_rate(end)), "forward": float(eq.forward(end)),
               "pv_dividends": float(eq.pv_dividends(eq.year_fraction(end))), "dividend_model": eq.dividend_model,
               "div_yield": None if eq.div_yield is None else 100 * float(eq.div_yield), "dividends": dividends,
               "quanto": None}
        if eq.currency != pay_ccy:
            rho, sigma_x = mkt.get_quanto(ticker, eq.currency, pay_ccy, eq.year_fraction(end))
            row["quanto"] = {"correlation": float(rho), "fx_vol": 100 * float(sigma_x)}
        underlyings.append(row)
    curves, seen = [], set()
    roles = [(pay_ccy, "discounting of the payments")] + [(mkt.get_equity(t).currency, f"forward of {t}")
                                                            for t in product.underlyings]
    for ccy, role in roles:
        curve = mkt.get_ir_curve(ccy)
        if curve.name in seen:
            next(c for c in curves if c["name"] == curve.name)["role"] += f", {role}"
            continue
        seen.add(curve.name)
        dates = list(curve.dates)
        last = next((i for i, d in enumerate(dates) if d >= end), len(dates) - 1)
        pillars = [{"date": d.isoformat(), "zero": 100 * float(curve.zero_rate(d)), "df": float(curve.discount(d))}
                   for d in dates[:last + 1] if d > today]
        _, c, name, _ = curve.name.split(":")
        curves.append({"name": curve.name, "label": f"{c} {name}", "role": role,
                       "zero_at_maturity": 100 * float(curve.zero_rate(end)), "df_at_maturity": float(curve.discount(end)),
                       "pillars": pillars})
    return {"maturity": end.isoformat(), "underlyings": underlyings, "curves": curves}


def _projections(state: dict) -> dict:
    """Projections of a priced autocallable from the paths of its Monte Carlo (computed on request, after the price):
    expected cash flows (% of the notional, not discounted), autocall probabilities, maturity, mean simulated levels
    against the market forwards (a quanto underlying: the mean carries the quanto drift)."""
    product, pricer = state["product"], state["pricer"]
    p = pricer.projections()
    quanto = set(getattr(product, "quanto", None) or {}) | ({product.underlying} if product.is_quanto else set())
    spots = {eq.ticker: float(eq.spot) for eq in getattr(pricer, "_equities", [])}
    return {
        "n_paths": p["n_paths"],
        "cash_flows": [{"date": r["date"].isoformat(), "coupon": 100 * r["coupon"], "redemption": 100 * r["redemption"],
                        "put": 100 * r["put"], "total": 100 * r["total"]} for r in p["cash_flows"]],
        # the price split as priced (smoothed digitals): coupons and put, % of the notional
        "pv": {"coupons": 100 * float(product.coupon_pv), "put": -100 * float(product.put_pv),
               "total": 100 * (float(product.coupon_pv) - float(product.put_pv))},
        "autocall": [{"date": r["date"].isoformat(), "probability": 100 * r["probability"],
                      "cumulative": 100 * r["cumulative"], "known": r["known"]} for r in p["autocall"]],
        "maturity": {k: v if k == "expected_life" else 100 * v for k, v in p["maturity"].items()},
        "worst": [{"ticker": w["ticker"], "maturity": 100 * w["maturity"], "knock_in": 100 * w["knock_in"]}
                  for w in p["worst"]],
        "forwards": [{"ticker": f["ticker"], "quanto": f["ticker"] in quanto, "spot": spots.get(f["ticker"]),
                      "rows": [{"date": r["date"].isoformat(), "mean": r["mean"], "p10": r["p10"], "p90": r["p90"],
                                "forward": r["forward"]} for r in f["rows"]]} for f in p["forwards"]],
    }


def _worst_of_ccy(mkt, product) -> str:
    """Currency of a worst-of with its quanto underlyings (those in another currency), or just the currency."""
    quanto = [t for t in product.underlyings if mkt.get_equity(t).currency != product.payment_ccy]
    return product.payment_ccy.name + (f" (quanto: {', '.join(quanto)})" if quanto else "")


def _past_outcome(row, called_on, fixing_date) -> dict:
    """Schedule columns of a past observation: the fixing (worst performance, % of the strike) and the outcome."""
    if row is None:
        return {"performance": None, "outcome": "after the call" if called_on and fixing_date > called_on else None}
    parts = []
    if row["coupon"] == "paid":
        parts.append(f"coupon {row['coupon_paid']:g}%")
    elif row["coupon"] == "missed":
        parts.append("coupon missed")
    if row["autocall"]:
        parts.append("called")
    return {"performance": 100 * row["performance"], "outcome": ", ".join(parts) or "-"}


@contextmanager
def _overridden(eq, overrides: dict, strike: float, T: float):
    """Temporarily set the spot and shift the vol surface so that the vol at (strike, T) is ``overrides["vol"]``."""
    spot0 = float(eq.spot)
    matrix = eq.vol_surface.volatility_matrix
    vols0 = matrix.numpy().copy()
    try:
        if overrides.get("spot") is not None:
            eq.spot.assign(float(overrides["spot"]))
        if overrides.get("vol") is not None:
            shift = float(overrides["vol"]) - float(eq.vol_surface.volatility(strike=strike, tenor=T))
            matrix.assign(vols0 + shift)
        yield
    finally:
        eq.spot.assign(spot0)
        matrix.assign(vols0)


# ----------------------------------------------------------------------------------------------------------------
# pricing
# ----------------------------------------------------------------------------------------------------------------

def _vanilla_gamma_theta(mkt, eq, f, kind, make_pricer, analytic: bool) -> tuple:
    """Gamma and theta (per calendar day) of one unit. Closed form: gamma by second-order AAD on the spot, theta
    ``-dV/dT`` by AAD on the pricer's time to maturity. Monte Carlo (the pathwise second derivative of a kinked
    payoff is zero): gamma as the difference of the AAD deltas at the spot +- 1%, theta as the change of the price
    with the expiry one day shorter, on the same random numbers."""
    spot = eq.spot
    if analytic:
        option = _equity_option(mkt, f, kind)
        pricer = make_pricer(option)
        with tf.GradientTape() as outer:
            pricer.price(option, mkt, autodiff=True)
            delta = pricer.tape.gradient(option.price, spot)
        gamma = float(outer.gradient(delta, spot))
        option = _equity_option(mkt, f, kind)
        pricer = make_pricer(option)
        pricer.price(option, mkt, autodiff=True)
        theta = -float(pricer.tape.gradient(option.price, option.time_to_maturity)) / 365
        return gamma, theta
    s0, h = float(spot), 0.01 * float(spot)
    deltas = []
    try:
        for x in (s0 + h, s0 - h):
            spot.assign(x)
            option = _equity_option(mkt, f, kind)
            pricer = make_pricer(option)
            pricer.price(option, mkt, autodiff=True)
            deltas.append(float(pricer.tape.gradient(option.price, spot)))
    finally:
        spot.assign(s0)
    option = _equity_option(mkt, f, kind)
    make_pricer(option).price(option, mkt)
    shorter = _equity_option(mkt, {**f, "expiry": (date.fromisoformat(f["expiry"]) - timedelta(days=1)).isoformat()}, kind)
    make_pricer(option).price(shorter, mkt)
    return (deltas[0] - deltas[1]) / (2 * h), float(shorter.price) - float(option.price)


def _price_equity(mkt, cache, kind, f, model, n_paths, seed) -> dict:
    """Price and greeks of an equity product in one go (:func:`_equity_price` then :func:`_equity_greeks`)."""
    return _equity_greeks(mkt, _equity_price(mkt, cache, kind, f, model, n_paths, seed))


def _equity_price(mkt, cache, kind, f, model, n_paths, seed) -> dict:
    """First phase: the product priced under the tape. Returns the state :func:`_equity_greeks` completes (pricer
    with its tape, product, model, timings)."""
    eq = mkt.get_equity(f["ticker"])
    snap = mkt.evaluation_date.isoformat()
    s0 = float(eq.spot)
    calibration_s = 0.0
    if kind == "Autocallable" and len(_tickers(f)) > 1:      # worst-of: one model per underlying
        product = _autocallable(mkt, f)
        qty, instance = 1.0, None
        if model not in WORST_OF_MODELS:
            raise ValueError(f"worst-of: {' or '.join(WORST_OF_MODELS)} only (one normal per step and underlying)")
        if model == GBM_DEBUG:
            models = tq.GeometricBrownianMotion
        else:
            models = {}
            for t in product.underlyings:
                models[t], seconds = cache.get(mkt.get_equity(t), model, snap)
                calibration_s += seconds
        pricer = tq.AutocallableMCPricer(model=models, n_paths=int(n_paths), seed=int(seed))
    elif kind == "Autocallable":
        product = _autocallable(mkt, f)
        qty = 1.0
        if model == GBM_DEBUG:                                           # class path, nothing calibrated
            instance = None
            pricer = tq.AutocallableMCPricer(model=tq.GeometricBrownianMotion, n_paths=int(n_paths), seed=int(seed))
        else:
            instance, calibration_s = cache.get(eq, model, snap)
            pricer = tq.AutocallableMCPricer(model=instance, n_paths=int(n_paths), seed=int(seed))
    else:
        product = _equity_option(mkt, f, kind)
        qty = float(f.get("quantity", 1.0))
        if model == "Black-Scholes":
            instance = None
            pricer = tq.BlackScholesPricer(dividend_model="discrete" if eq.dividend_model == "discrete" else "continuous",
                                           digital_spread=DIGITAL_SPREAD if kind == "Digital" else None)
        else:
            if product.exercise_type == tq.ExerciseType.American:
                raise ValueError("American exercise is priced with Black-Scholes (Bjerksund-Stensland) only: the local "
                                 "vol Monte Carlo has no early exercise")
            instance, calibration_s = cache.get(eq, model, snap)
            T = eq.year_fraction(product.end_date)
            pricer = tq.VanillaMCPricer(model=instance, n_paths=int(n_paths),
                                        n_steps=max(MIN_STEPS, math.ceil(STEPS_PER_YEAR * T)), seed=int(seed))
    t0 = time.perf_counter()
    price_ccy = _price_ccy(f)
    pricer.price(product, mkt, autodiff=True, price_currency=price_ccy)
    to_price = 1.0 if price_ccy is None else float(mkt.get_fx_spot(product.payment_ccy, price_ccy))
    # reference of the % in the price currency: the spot (in the payment currency for a composite) or the notional
    ref = (s0 * _composite_fx(mkt, f, eq) if kind != "Autocallable" else product.notional) * to_price
    return {"kind": kind, "f": f, "model": model, "n_paths": n_paths, "seed": seed, "eq": eq, "s0": s0, "qty": qty,
            "product": product, "pricer": pricer, "instance": instance, "calibration_s": calibration_s, "t0": t0,
            "price_s": time.perf_counter() - t0, "ref": ref, "to_price": to_price}


def _quick_headline(state: dict) -> list:
    """The price of the first phase, as the headline cards (the greeks follow)."""
    kind, product, qty, s0 = state["kind"], state["product"], state["qty"], state["s0"]
    return [["Price", qty * float(product.price), 4],
            ["Premium (% spot)" if kind != "Autocallable" else "Price (% notional)",
             100 * float(product.price) / state["ref"], 3],
            ["Pricing [s]", state["price_s"], 1]]


def _delta_rows(equities, greeks: dict, scale: float, notional: float) -> list:
    """Delta per underlying: per 1 of spot, and the price change for a +1% spot move (currency, % of the notional)."""
    rows = []
    for eq in equities:
        delta = scale * float(np.sum(greeks[f"EQ:{eq.currency.name}:{eq.ticker}:SPOT"][0]))
        spot = float(eq.spot)
        rows.append({"ticker": eq.ticker, "ccy": eq.currency.name, "spot": spot, "delta": delta,
                     "delta_1pct": 0.01 * spot * delta, "delta_1pct_notional": 100 * 0.01 * spot * delta / notional})
    return rows


def _worst_of_greeks(mkt, state: dict) -> dict:
    """Greeks of a worst-of autocallable: delta and vega per underlying, vega map of all of them (rows by underlying
    and expiry, strikes in % of each spot), rho of the curves, sensitivity per correlation pair (+0.01), quanto rows
    of the foreign underlyings."""
    product, pricer, t0 = state["product"], state["pricer"], state["t0"]
    equities = [mkt.get_equity(t) for t in product.underlyings]
    factors = {}
    for eq in equities:
        factors.update(eq.risk_factors())
    factors.update({k: v for k, v in mkt.risk_factors().items()
                    if k.startswith(f"IR:{product.payment_ccy.name}:") or k.startswith("CORR:") or k.startswith("FX:")})
    greeks = _to_numpy(pricer.tape.gradient(product.price, factors))
    pricing_s = time.perf_counter() - t0
    se = product.standard_error
    vegas = [_equity_vega_map(eq, greeks, 1.0) for eq in equities]
    rho = _rho_buckets(mkt, greeks, 1.0)
    headline = [["Price", float(product.price), 4], ["Price (% notional)", 100 * float(product.price) / state["ref"], 3],
                ["Std error", float(se), 4]]
    deltas = _delta_rows(equities, greeks, 1.0, product.notional)                 # a table in Risk
    headline += [[f"Vega {eq.ticker} (1 vol pt)", v["total"], 4] for eq, v in zip(equities, vegas)]
    headline += [["Rho (1 bp)", sum(r["total"] for r in rho), 4],
                 ["Calibration [s]", state["calibration_s"], 1], ["Pricing [s]", pricing_s, 1]]
    T = equities[0].year_fraction(product.end_date)
    info = [["Underlyings", " / ".join(product.underlyings), None],
            ["Expiry", f"{product.end_date.isoformat()} ({T:.2f}y)", None],
            ["Payment currency", product.payment_ccy.name, None]]
    strikes = pricer.trade_state(product, mkt)["strikes"] or [None] * len(equities)
    for eq, k in zip(equities, strikes):
        info += [[f"{eq.ticker} spot", float(eq.spot), 2],
                 [f"{eq.ticker} strike", f"S({product.strike_date.isoformat()}), forward start" if k is None else k, 2],
                 [f"{eq.ticker} vol ATM fwd (%)",
                  100 * float(eq.vol_surface.volatility(strike=float(eq.forward(T)), tenor=T)), 2]]
    names = mkt.get_correlation_matrix().names
    g = np.asarray(greeks.get("CORR:ALL:GLOBAL:SPOT", [np.zeros(())])[0])
    keys = [f"EQ:{eq.currency.name}:{eq.ticker}" for eq in equities]
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            i, j = names.index(keys[a]), names.index(keys[b])
            rho_ab = float(mkt.get_correlation([keys[a], keys[b]])[0, 1])
            info.append([f"Correlation {equities[a].ticker} / {equities[b].ticker}", rho_ab, 3])
            if g.ndim == 2:
                info.append([f"Correlation sensitivity {equities[a].ticker} / {equities[b].ticker} (+0.01)",
                             float(g[i, j] + g[j, i]) / 100, 4])
    for ticker, (rho_x, sigma_x) in (getattr(product, "quanto", None) or {}).items():
        info += [[f"{ticker} quanto: correlation with the FX rate", float(rho_x), 3],
                 [f"{ticker} quanto: FX vol (%)", 100 * float(sigma_x), 2]]
    fx_vega = sum(float(np.sum(v)) for k, vs in greeks.items() if k.startswith("FX:") and k.endswith(":VOL")
                  for v in vs if v is not None)
    if fx_vega:
        info.append(["FX vega (1 vol pt)", fx_vega / 100, 4])
    info.append(["Monte Carlo", f"{int(state['n_paths']):,} paths", None])
    # one vega map per underlying (a sub-tab each in the page)
    maps = [{**v, "ticker": eq.ticker, "title": f"Vega per surface node of {eq.ticker} (1 vol point)"}
            for eq, v in zip(equities, vegas)]
    return {"headline": headline, "info": info, "vega": maps[0], "vega_maps": maps, "rho": rho, "sensitivities": None,
            "deltas": deltas}


def _equity_greeks(mkt, state: dict) -> dict:
    """Second phase: greeks from the tape of the first one (backward pass), maps, info."""
    if getattr(state["product"], "is_worst_of", False):
        return _worst_of_greeks(mkt, state)
    kind, f, model, n_paths, seed = state["kind"], state["f"], state["model"], state["n_paths"], state["seed"]
    eq, s0, qty, product, pricer = state["eq"], state["s0"], state["qty"], state["product"], state["pricer"]
    instance, calibration_s, t0 = state["instance"], state["calibration_s"], state["t0"]
    params = instance.parameters if instance is not None else {}
    factors = eq.risk_factors()
    fx_product = product.is_quanto or product.is_composite
    converted = getattr(product, "price_currency", product.payment_ccy) != product.payment_ccy
    if fx_product or converted:                     # payment curve, FX vols and spots, correlations too
        factors = {**factors, **{k: v for k, v in mkt.risk_factors().items()
                                 if k.startswith(f"IR:{product.payment_ccy.name}:") or k.startswith("CORR:")
                                 or k.startswith("FX:")}}
    g = pricer.tape.gradient(product.price, {"model": params, "market": factors})
    market = _to_numpy(g["market"])
    greeks = (tq.par_greeks(instance, {k: v.numpy() for k, v in g["model"].items()}, market, equity=eq)
              if instance is not None and params else market)
    pricing_s = time.perf_counter() - t0
    spot_key = next(k for k in eq.risk_factors() if k.startswith("EQ:") and k.endswith(":SPOT"))
    se = getattr(product, "standard_error", None)                 # Monte Carlo only
    vega = _equity_vega_map(eq, greeks, qty)
    rho = _rho_buckets(mkt, greeks, qty)
    price = qty * float(product.price)
    headline = [["Price", price, 4], ["Premium (% spot)" if kind != "Autocallable" else "Price (% notional)",
                                      100 * float(product.price) / state["ref"], 3],
                ["Std error", None if se is None else qty * float(se), 4],
                ["Delta", qty * float(np.sum(greeks[spot_key][0])), 4], ["Vega (1 vol pt)", vega["total"], 4],
                ["Rho (1 bp)", sum(r["total"] for r in rho), 4],
                ["Calibration [s]", calibration_s, 1], ["Pricing [s]", pricing_s, 1]]
    deltas = None
    if kind == "Autocallable":                      # the delta in a table of Risk, not in the bar
        headline = [h for h in headline if h[0] != "Delta"]
        deltas = _delta_rows([eq], greeks, qty, product.notional)
    T = eq.year_fraction(product.end_date)
    info = [["Spot", s0, 2], ["Expiry", f"{product.end_date.isoformat()} ({T:.2f}y)", None],
            ["Forward", float(eq.forward(T)), 2], ["Rate (zero, %)", 100 * float(eq.discount_curve.zero_rate(T)), 3],
            ["Discount factor", float(eq.discount_curve.discount(T)), 5], ["PV dividends", float(eq.pv_dividends(T)), 2],
            ["Vol ATM fwd (%)", 100 * float(eq.vol_surface.volatility(strike=float(eq.forward(T)), tenor=T)), 2]]
    if kind == "Autocallable":
        today = mkt.evaluation_date
        info.insert(1, ["Strike", float(product.strike) if product.strike is not None
                        else f"S({product.strike_date.isoformat()}), forward start" if product.strike_date > today
                        else f"{float(mkt.get_eq_fixing(eq.ticker, product.strike_date)):,.2f} "
                             f"(fixing of {product.strike_date.isoformat()})", 2])
    if kind != "Autocallable":
        info.insert(1, ["Strike", float(product.strike), 2])
        info.append(["Vol at strike (%)", 100 * float(eq.vol_surface.volatility(strike=float(product.strike), tenor=T)), 2])
    if fx_product or converted:                     # quanto / composite / converted: inputs and sensitivities
        info += _quanto_info(mkt, eq, product, greeks, qty)
    if instance is not None:
        info.append(["Monte Carlo", f"{int(n_paths):,} paths", None])
    if getattr(instance, "source_set", None):
        info.append(["Calibration set", instance.source_set, None])
    if kind == "Vanilla":           # the desk's view: delta, gamma, theta, vega, rho as numbers, no maps or timings
        T_steps = max(MIN_STEPS, math.ceil(STEPS_PER_YEAR * T))
        if instance is None:
            make = lambda o: tq.BlackScholesPricer(
                dividend_model="discrete" if eq.dividend_model == "discrete" else "continuous")
        else:
            make = lambda o: tq.VanillaMCPricer(model=instance, n_paths=int(n_paths), n_steps=T_steps, seed=int(seed))
        gamma, theta = (state["to_price"] * x for x in _vanilla_gamma_theta(mkt, eq, f, kind, make,
                                                                             analytic=instance is None))
        by_label = {h[0]: h for h in headline}
        headline = [by_label["Price"], by_label["Premium (% spot)"], by_label["Delta"], ["Gamma", qty * gamma, 6],
                    ["Theta (1 day)", qty * theta, 4], by_label["Vega (1 vol pt)"], by_label["Rho (1 bp)"]]
        vega, rho = None, []
    sensitivities = None
    if kind != "Autocallable":                      # greeks per unit and the equivalent (times the size)
        units = {"Delta": "per 1 of spot", "Gamma": "per 1 of spot", "Theta (1 day)": "per calendar day",
                 "Vega (1 vol pt)": "per vol point", "Rho (1 bp)": "per bp"}
        rows = [{"greek": label, "unit": units[label], "per_unit": None if value is None or not qty else value / qty,
                 "equivalent": value} for label, value, _ in headline if label in units]
        sensitivities = {"quantity": qty, "rows": rows}
    return {"headline": headline, "info": info, "vega": vega, "rho": rho, "sensitivities": sensitivities,
            "deltas": deltas}


def _quanto_info(mkt, eq, product, greeks: dict, qty: float) -> list:
    """Quanto rows of the info table: payment currency, correlation with ``X`` (1 unit of the underlying's currency in
    the payment one), FX vol, and the sensitivities to the FX vols (1 vol pt) and to that correlation (+0.01)."""
    mode = " (quanto)" if product.is_quanto else " (composite: S X)" if product.is_composite else ""
    rows = [["Payment currency", f"{product.payment_ccy.name}{mode}", None]]
    if getattr(product, "price_currency", product.payment_ccy) != product.payment_ccy:
        rows.append(["Price currency", f"{product.price_currency.name} (at the FX spot)", None])
    fx_delta = sum(float(np.sum(v)) for k, vs in greeks.items() if k.startswith("FX:") and k.endswith(":SPOT")
                   for v in vs if v is not None)
    if fx_delta:
        rows.append(["FX delta (per 1 of FX spot)", qty * fx_delta, 4])
    if getattr(product, "quanto_correlation", None) is not None:
        rows += [["Correlation with the FX rate", float(product.quanto_correlation), 3],
                 ["FX vol (%)", 100 * float(product.quanto_fx_vol), 2]]
    fx_vega = sum(float(np.sum(v)) for k, vs in greeks.items() if k.startswith("FX:") and k.endswith(":VOL")
                  for v in vs if v is not None)
    rows.append(["FX vega (1 vol pt)", qty * fx_vega / 100, 4])
    corr_key = "CORR:ALL:GLOBAL:SPOT"
    if corr_key in greeks and greeks[corr_key][0] is not None:
        names = mkt.get_correlation_matrix().names
        und = f"EQ:{eq.currency.name}:{eq.ticker}"
        pay = product.payment_ccy.name
        fx = next((n for n in names if n in (f"FX:{pay}:{eq.currency.name}{pay}", f"FX:{eq.currency.name}:{pay}{eq.currency.name}")),
                  None)
        g = np.asarray(greeks[corr_key][0])
        if und in names and fx is not None and g.ndim == 2:
            i, j = names.index(und), names.index(fx)
            rows.append([f"Correlation sensitivity ({fx.split(':')[2]}, +0.01)", qty * float(g[i, j] + g[j, i]) / 100, 4])
    return rows


def _price_rates(mkt, kind, f) -> dict:
    sign = -1.0 if f.get("position") == "Short" else 1.0
    if kind == "Swap" and f.get("side") == "Receive fixed":
        sign = -sign                                  # the generators build payers (pay fixed)
    product = _rates_product(mkt, f, kind)
    pricer = tq.PricerAssignment.create(product)
    t0 = time.perf_counter()
    pricer.price(product, mkt, autodiff=True)
    ccy = f["ccy"]
    factors = {k: v for k, v in mkt.risk_factors().items() if k.startswith(f"IR:{ccy}:")}
    greeks = _to_numpy(pricer.tape.gradient(product.price, factors))
    pricing_s = time.perf_counter() - t0
    price = sign * float(product.price)
    rho = _rho_buckets(mkt, greeks, sign)
    sens_table = _sensitivity_table(mkt, greeks, sign)
    headline = [["PV", price, 2]]
    info = [["Currency", ccy, None], ["Notional", float(f["notional"]), 0]]

    if kind in ("FRA", "Swap"):                       # fair rate: the PV is linear in the fixed rate
        r0, h = float(f["rate"]) / 100, 1e-4
        bumped = _rates_product(mkt, f, kind, r0 + h)
        tq.PricerAssignment.create(bumped).price(bumped, mkt)
        slope = (float(bumped.price) - float(product.price)) / h
        headline.append(["Fair rate (%)", 100 * (r0 - float(product.price) / slope), 4])
    if kind == "Swaption":
        headline.append(["Forward swap rate (%)", 100 * float(product.forward_swap_rate), 4])
        info += [["Annuity", float(product.annuity), 4], ["Vol at strike", float(product.volatility), 6]]
    headline.append(["DV01 (1 bp)", sum(r["total"] for r in rho), 2])

    vega = None
    vol_key = f"IR:{ccy}:{IBOR}:VOL" if kind in ("Cap/Floor", "Collar") else f"IR:{ccy}:{IBOR}:SWVOL" if kind == "Swaption" else None
    if vol_key:
        unit, scale = ("1 bp vol", 1e-4) if (mkt.get_ir_vol_surface(tq.Currency[ccy], IBOR).vol_type if kind != "Swaption"
                                            else mkt.get_swaption_vol_cube(tq.Currency[ccy], IBOR).vol_type) == "normal" \
            else ("1 vol pt", 1e-2)
        g = sign * scale * np.asarray(greeks[vol_key][0], float)
        if kind == "Swaption":
            cube = mkt.get_swaption_vol_cube(tq.Currency[ccy], IBOR)
            z = g.sum(axis=2)                          # summed over the strike offsets
            vega = {"title": f"Vega per expiry x tenor ({unit}, summed over strikes)", "z": z.tolist(),
                    "x": [_tenor(t) for t in cube.tenors], "x_title": "tenor",
                    "y": [_tenor(t) for t in cube.expiries], "y_title": "expiry", "total": float(g.sum())}
        else:
            vs = mkt.get_ir_vol_surface(tq.Currency[ccy], IBOR)
            vega = {"title": f"Vega per caplet vol node ({unit})", "z": g.tolist(),
                    "x": [f"{100 * float(k):.2f}%" for k in vs.strike], "x_title": "strike",
                    "y": [_tenor(float(t)) for t in vs.maturity], "y_title": "expiry", "total": float(g.sum())}
        headline.append([f"Vega ({unit})", vega["total"], 2])
    headline.append(["Pricing [s]", pricing_s, 2])
    return {"headline": headline, "info": info, "vega": vega, "rho": rho, "sens_table": sens_table}


def price_trade(mkt: tq.MarketEnvironment, cache: ModelCache, request: dict, n_paths: int = 50_000,
                seed: int = 1, phase: str = "all") -> dict:
    """Price ``request = {asset_class, product, fields, model, overrides}`` (``overrides``: equity ``spot`` / ``vol``,
    see :func:`_overridden`).

    ``phase="price"`` (equity products): the price only, at once, with ``pending``: a function that computes the
    rest (greeks from the same tape, under the same overrides) and returns the whole result; the caller keeps it
    and calls it next (the web app: ``/api/price/greeks``). Rates products are priced in one phase anyway.

    Returns:
        dict: ``headline`` and ``info`` (``[label, value, decimals]``), ``vega`` (map: ``z``, axes, ``total``, or
        ``None``), ``rho`` (per curve: pillar dates, rho per pillar, total), ``model``; with ``phase="price"`` the
        headline of the price and ``pending``.

    Raises:
        ValueError: Unknown asset class, product or model, or invalid fields.
    """
    spec, fields = _fields(mkt, request)
    model = request.get("model") or spec["models"][0]
    if model not in spec["models"]:
        raise ValueError(f"model {model} not available for {request['product']}: {spec['models']}")
    overrides = {k: v for k, v in (request.get("overrides") or {}).items() if v is not None}
    if request["asset_class"] in ("Equity", "Exotics"):
        eq = mkt.get_equity(fields["ticker"])
        if eq.vol_surface is None:
            raise ValueError(f"no implied vol surface for {eq.ticker} in this snapshot (vols are not in the real data "
                             "yet)")
        strike, expiry = _equity_strike_expiry(mkt, request["product"], fields)
        fields["_ref_spot"] = float(eq.spot)       # the deal's strikes stay on the market spot under an override
        T = eq.year_fraction(expiry)
        meta = {"model": model, "product": request["product"], "overrides": overrides,
                "snapshot": mkt.evaluation_date.isoformat()}
        if phase == "price":
            with _overridden(eq, overrides, strike, T):
                state = _equity_price(mkt, cache, request["product"], fields, model, n_paths, seed)

            def finish() -> dict:                       # the greeks, under the same overrides
                with _overridden(eq, overrides, strike, T):
                    return {**_equity_greeks(mkt, state), **meta}

            projections = (lambda: _projections(state)) if request["product"] == "Autocallable" else None
            return {"headline": _quick_headline(state), "pending": finish, "projections": projections, **meta}
        with _overridden(eq, overrides, strike, T):
            out = _price_equity(mkt, cache, request["product"], fields, model, n_paths, seed)
        return {**out, **meta}
    out = _price_rates(mkt, request["product"], fields)
    return {**out, "model": model, "product": request["product"], "overrides": overrides,
            "snapshot": mkt.evaluation_date.isoformat()}
