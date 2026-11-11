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


class ModelCache:
    """Calibrated model instances per ``(snapshot, ticker, model)``, with their calibration time."""

    def __init__(self):
        self._models: dict = {}

    def get(self, eq, name: str, snapshot: str = "") -> tuple:
        """``(model, seconds spent calibrating now)``: 0 when it was already calibrated."""
        key = (snapshot, eq.ticker, name)
        if key in self._models:
            return self._models[key], 0.0
        t0 = time.perf_counter()
        if name != "Local vol":
            raise ValueError(f"unknown model {name}")
        model = tq.LocalVolatilityModel.calibrate(eq)
        self._models[key] = model
        return model, time.perf_counter() - t0

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


def catalog(mkt: tq.MarketEnvironment) -> dict:
    """Asset classes -> product types -> ``{fields, models}`` for the market *mkt* (the form of the Tradepad)."""
    tickers = equity_tickers(mkt)
    one_year = date(mkt.evaluation_date.year + 1, mkt.evaluation_date.month, mkt.evaluation_date.day).isoformat()
    three_years = date(mkt.evaluation_date.year + 3, mkt.evaluation_date.month, mkt.evaluation_date.day).isoformat()
    underlying = _select("ticker", "Underlying", tickers)
    option = [underlying, _select("option_type", "Type", ["Put", "Call"]), _number("strike_pct", "Strike (% spot)", 100.0, 0.5),
              _date("expiry", "Expiry", one_year), _number("quantity", "Quantity", 1.0)]
    equity_models = ["Black-Scholes", "Local vol"]

    def rates_common(needs):
        return [_select("ccy", "Currency", _rate_currencies(mkt, needs)), _number("notional", "Notional", 1e6, 1e5),
                _select("position", "Position", ["Long", "Short"])]

    def swap_side():
        return _select("side", "Side", ["Payer", "Receiver"])

    products = {
        "Equity": {
            "Vanilla": {"fields": option[:2] + [_select("exercise", "Exercise", ["European", "American"])] + option[2:],
                        "models": equity_models},
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
            "Swap": {"fields": rates_common("ibor") + [swap_side(), _select("tenor", "Tenor", TENORS, "5Y"),
                                                       _number("rate", "Fixed rate (%)", 2.5, 0.01)],
                     "models": ["Discounting"]},
            "OIS": {"fields": rates_common("ois") + [swap_side(), _select("tenor", "Tenor", TENORS, "5Y"),
                                                     _number("rate", "Fixed rate (%)", 2.3, 0.01)],
                    "models": ["Discounting"]},
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
                                      _date("maturity", "Maturity", three_years))
                             + _group("Coupon", _select("coupon_type", "Type", ["Standard", "Memory"]),
                                      _number("coupon", "Coupon per period (%)", 4.0, 0.25),
                                      _number("coupon_barrier", "Coupon barrier (%)", 80.0, 1.0),
                                      _select("coupon_frequency", "Frequency", ["3M", "6M", "1Y"], "6M"))
                             + _group("Final redemption", _number("put_barrier", "Put barrier (%)", 70.0, 1.0),
                                      _number("put_strike", "Put strike (%)", 100.0, 1.0),
                                      _number("put_participation", "Put participation (%)", 100.0, 5.0))
                             + _group("Early redemption (autocall)",
                                      _number("autocall_barrier", "Autocall barrier (%)", 100.0, 1.0),
                                      _select("autocall_frequency", "Frequency", ["3M", "6M", "1Y"], "6M"),
                                      _select("first_autocall", "First observation", ["3M", "6M", "1Y", "2Y"], "1Y")),
                             "models": ["Local vol"], "create_trade": True},
        },
    }
    return {"snapshot": mkt.evaluation_date.isoformat(), "asset_classes": products}


# ----------------------------------------------------------------------------------------------------------------
# building the products
# ----------------------------------------------------------------------------------------------------------------

def _equity_option(mkt, f: dict, kind: str):
    eq = mkt.get_equity(f["ticker"])
    s0 = float(f.get("_ref_spot", eq.spot))   # strikes on the market spot, not on an override
    expiry = date.fromisoformat(f["expiry"])
    if expiry <= mkt.evaluation_date:
        raise ValueError("the expiry must be after the snapshot date")
    option_type = tq.OptionType.Call if f["option_type"] == "Call" else tq.OptionType.Put
    common = (eq.currency, mkt.evaluation_date, expiry, option_type, s0 * float(f["strike_pct"]) / 100)
    if kind == "Vanilla":
        exercise = tq.ExerciseType.American if f.get("exercise") == "American" else tq.ExerciseType.European
        return tq.VanillaOption(*common, underlying=eq.ticker, exercise_type=exercise)
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


def _autocallable(mkt, f: dict):
    """Phoenix autocallable struck at the market spot: coupons on their own schedule (standard or memory), autocall
    observations from the first observation date to maturity, capital-at-risk put (barrier, strike, participation)."""
    eq = mkt.get_equity(f["ticker"])
    today = mkt.evaluation_date
    maturity = date.fromisoformat(f["maturity"])
    end = CAL.adjust(maturity, MF)
    if end <= today:
        raise ValueError("the maturity must be after the pricing date")
    first = CAL.advance(today, _months(f["first_autocall"]), tq.TimeUnit.Months, MF)
    if first >= end:
        raise ValueError("the first autocall observation must be before the maturity")
    coupons = _backward_dates(today, maturity, _months(f["coupon_frequency"]))
    autocalls = [d for d in _backward_dates(today, maturity, _months(f["autocall_frequency"])) if d >= first]
    return tq.AutocallableOption(
        ccy=eq.currency, notional=float(f["notional"]), start_date=today, end_date=end,
        strike=float(f.get("_ref_spot", eq.spot)), coupon_fixing_dates=coupons, coupon_payment_dates=coupons,
        coupon_rates=[float(f["coupon"])], coupon_barriers=[float(f["coupon_barrier"])],
        memory=f["coupon_type"] == "Memory", autocall_fixing_dates=autocalls, autocall_payment_dates=autocalls,
        autocall_barrier=[float(f["autocall_barrier"])], payoff_barrier=float(f["put_barrier"]),
        payoff_participation=float(f["put_participation"]) / 100, put_strike=float(f["put_strike"]),
        underlying=eq.ticker)


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
    if kind == "Swap":
        return tq.SwapGenerator(ccy, 2, "1Y", IBOR, MF, notional, A360, A360, CAL, _ibor(ccy)).build(
            mkt.evaluation_date, rate, f["tenor"])
    if kind == "OIS":
        return tq.OisGenerator(ccy, 2, "1Y", "1Y", MF, notional, A360, A360, CAL, tq.OvernightIndex(CAL, ccy)).build(
            mkt.evaluation_date, rate, f["tenor"])
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
    return spec, fields


def _equity_strike_expiry(mkt, kind: str, f: dict) -> tuple:
    """Strike and expiry the market inputs refer to (the autocallable: its initial fixing and final date)."""
    product = _autocallable(mkt, f) if kind == "Autocallable" else _equity_option(mkt, f, kind)
    return float(product.strike), product.end_date


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
        return {"editable": {"spot": s0, "vol": float(eq.vol_surface.volatility(strike=strike, tenor=T))},
                "info": [["Rate (zero, %)", 100 * float(curve.zero_rate(T)), 3], ["Curve", f"{ccy} {name}", None],
                         ["Forward", float(eq.forward(T)), 2], ["PV dividends", float(eq.pv_dividends(T)), 2],
                         ["Repo (%)", 100 * float(eq.repo), 3], ["Strike", strike, 2],
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


def describe_trade(mkt: tq.MarketEnvironment, request: dict) -> dict:
    """The trade built from the deal fields, before pricing: its main terms and, for the autocallable, the schedule
    (one row per fixing: fixing date, autocall date when it is an autocall observation, payment date, coupon
    barrier and coupon amount in % of the strike / notional).

    Returns:
        dict: ``summary`` (``[label, value, decimals]``) and ``schedule`` (list of rows, or ``None``).
    """
    _, f = _fields(mkt, request)
    if request.get("product") != "Autocallable":
        return {"summary": [], "schedule": None}
    product = _autocallable(mkt, f)
    table = product.schedule_dataframe()
    schedule = []
    for _, row in table.iterrows():
        coupon = row["coupon_rate_pct"] is not None and row["coupon_rate_pct"] == row["coupon_rate_pct"]
        autocall = row["autocall_barrier_pct"] is not None and row["autocall_barrier_pct"] == row["autocall_barrier_pct"]
        schedule.append({"fixing_date": row["fixing_date"].isoformat(),
                         "autocall_date": row["fixing_date"].isoformat() if autocall else None,
                         "payment_date": row["payment_date"].isoformat(),
                         "coupon_barrier": float(row["coupon_barrier_pct"]) if coupon else None,
                         "coupon_amount": float(row["coupon_rate_pct"]) if coupon else None,
                         "autocall_barrier": float(row["autocall_barrier_pct"]) if autocall else None})
    summary = [["Strike (initial fixing)", float(product.strike), 2], ["Maturity", product.end_date.isoformat(), None],
               ["Notional", float(product.notional), 2], ["Coupon type", "Memory" if product.memory else "Standard", None],
               ["Put", f"barrier {product.payoff_barrier:g}%, strike {product.put_strike:g}%, "
                       f"participation {100 * product.payoff_participation:g}%", None],
               ["Coupons / autocall dates", f"{len(product.coupon_fixing_dates)} / {len(product.autocall_fixing_dates)}",
                None]]
    return {"summary": summary, "schedule": schedule}


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
    eq = mkt.get_equity(f["ticker"])
    snap = mkt.evaluation_date.isoformat()
    s0 = float(eq.spot)
    calibration_s = 0.0
    if kind == "Autocallable":
        product = _autocallable(mkt, f)
        qty = 1.0
        instance, calibration_s = cache.get(eq, "Local vol", snap)
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
    pricer.price(product, mkt, autodiff=True)
    params = instance.parameters if instance is not None else {}
    g = pricer.tape.gradient(product.price, {"model": params, "market": eq.risk_factors()})
    market = _to_numpy(g["market"])
    greeks = (tq.par_greeks(instance, {k: v.numpy() for k, v in g["model"].items()}, market, equity=eq)
              if instance is not None and params else market)
    pricing_s = time.perf_counter() - t0
    spot_key = next(k for k in eq.risk_factors() if k.startswith("EQ:") and k.endswith(":SPOT"))
    se = getattr(product, "standard_error", None) if instance is not None else None
    vega = _equity_vega_map(eq, greeks, qty)
    rho = _rho_buckets(mkt, greeks, qty)
    price = qty * float(product.price)
    headline = [["Price", price, 4], ["Premium (% spot)" if kind != "Autocallable" else "Price (% notional)",
                                      100 * float(product.price) / (s0 if kind != "Autocallable" else product.notional), 3],
                ["Std error", None if se is None else qty * float(se), 4],
                ["Delta", qty * float(np.sum(greeks[spot_key][0])), 4], ["Vega (1 vol pt)", vega["total"], 4],
                ["Rho (1 bp)", sum(r["total"] for r in rho), 4],
                ["Calibration [s]", calibration_s, 1], ["Pricing [s]", pricing_s, 1]]
    T = eq.year_fraction(product.end_date)
    info = [["Spot", s0, 2], ["Expiry", f"{product.end_date.isoformat()} ({T:.2f}y)", None],
            ["Forward", float(eq.forward(T)), 2], ["Rate (zero, %)", 100 * float(eq.discount_curve.zero_rate(T)), 3],
            ["Discount factor", float(eq.discount_curve.discount(T)), 5], ["PV dividends", float(eq.pv_dividends(T)), 2],
            ["Vol ATM fwd (%)", 100 * float(eq.vol_surface.volatility(strike=float(eq.forward(T)), tenor=T)), 2]]
    if kind != "Autocallable":
        info.insert(1, ["Strike", float(product.strike), 2])
        info.append(["Vol at strike (%)", 100 * float(eq.vol_surface.volatility(strike=float(product.strike), tenor=T)), 2])
    if instance is not None:
        info.append(["Monte Carlo", f"{int(n_paths):,} paths", None])
    if kind == "Vanilla":           # the desk's view: delta, gamma, theta, vega, rho as numbers, no maps or timings
        T_steps = max(MIN_STEPS, math.ceil(STEPS_PER_YEAR * T))
        if instance is None:
            make = lambda o: tq.BlackScholesPricer(
                dividend_model="discrete" if eq.dividend_model == "discrete" else "continuous")
        else:
            make = lambda o: tq.VanillaMCPricer(model=instance, n_paths=int(n_paths), n_steps=T_steps, seed=int(seed))
        gamma, theta = _vanilla_gamma_theta(mkt, eq, f, kind, make, analytic=instance is None)
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
    return {"headline": headline, "info": info, "vega": vega, "rho": rho, "sensitivities": sensitivities}


def _price_rates(mkt, kind, f) -> dict:
    sign = -1.0 if f.get("position") == "Short" else 1.0
    if kind in ("Swap", "OIS") and f.get("side") == "Receiver":
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
    headline = [["PV", price, 2]]
    info = [["Currency", ccy, None], ["Notional", float(f["notional"]), 0]]

    if kind in ("FRA", "Swap", "OIS"):                # fair rate: the PV is linear in the fixed rate
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
    return {"headline": headline, "info": info, "vega": vega, "rho": rho}


def price_trade(mkt: tq.MarketEnvironment, cache: ModelCache, request: dict, n_paths: int = 50_000,
                seed: int = 1) -> dict:
    """Price ``request = {asset_class, product, fields, model, overrides}`` (``overrides``: equity ``spot`` / ``vol``,
    see :func:`_overridden`).

    Returns:
        dict: ``headline`` and ``info`` (``[label, value, decimals]``), ``vega`` (map: ``z``, axes, ``total``, or
        ``None``), ``rho`` (per curve: pillar dates, rho per pillar, total), ``model``.

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
        strike, expiry = _equity_strike_expiry(mkt, request["product"], fields)
        fields["_ref_spot"] = float(eq.spot)       # the deal's strikes stay on the market spot under an override
        with _overridden(eq, overrides, strike, eq.year_fraction(expiry)):
            out = _price_equity(mkt, cache, request["product"], fields, model, n_paths, seed)
    else:
        out = _price_rates(mkt, request["product"], fields)
    return {**out, "model": model, "product": request["product"], "overrides": overrides,
            "snapshot": mkt.evaluation_date.isoformat()}
