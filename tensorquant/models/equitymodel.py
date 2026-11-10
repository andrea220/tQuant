"""Equity models as instances, separate from the market.

A model holds **settings** (how it is built) and **calibrated parameters** (float64 ``tf.Variable``s), never the market:
the underlying is given to :meth:`EquityModel.calibrate` and to :meth:`EquityModel.process`, which binds the model to
an :class:`Equity` and returns a :class:`StochasticProcess` (the fast, market-dependent part, e.g. the forwards and the
local vol grid, is rebuilt there, inside the pricing tape). The same instance can be reused across dates, scenarios
and pricers, and serialized with :meth:`EquityModel.to_frames` / :meth:`EquityModel.from_frames`.

Greeks of a price computed with an instance:

- **model greeks**: ``tape.gradient(price, model.parameters)`` (parameters held, AAD on the variables);
- **market greeks**: ``tape.gradient(price, equity.risk_factors())`` (forwards, and the vol nodes for the models that
  read them at pricing, e.g. local vol);
- **par greeks**: :func:`par_greeks` maps the model greeks to the market inputs through the calibration Jacobian
  ``dp / d(market inputs)`` stored by :meth:`EquityModel.calibrate` (``None`` for a model built by hand).

A model **class** passed to a pricer keeps the original behaviour: ``Model.from_equity(equity, maturity, strike)``
calibrates inside the pricing tape and the greeks go through the calibration.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from .stochasticprocess import StochasticProcess
from ..numericalhandles.dtypes import DTYPE

__all__ = ["EquityModel", "par_greeks"]


class EquityModel(ABC):
    """Base class of the equity model instances (settings + calibrated parameters, no market).

    Attributes:
        ticker (str | None): Underlying the model was calibrated on (``None`` if built by hand).
        calibration_date (date | None): Evaluation date of the calibration.
        jacobian (dict | None): ``{parameter: {market key: [array per variable]}}``, ``dp / d(market input)`` with the
            shape ``parameter.shape + variable.shape``, keys as in :meth:`Equity.risk_factors`; ``None`` if built by
            hand.
        report (pd.DataFrame | None): Calibration report (fit errors).
    """

    #: Settings stored and serialized (name -> default); subclasses override.
    SETTINGS: dict = {}

    def __init__(self, **settings):
        unknown = set(settings) - set(self.SETTINGS)
        if unknown:
            raise TypeError(f"unknown settings for {type(self).__name__}: {sorted(unknown)}")
        self.settings = {**self.SETTINGS, **settings}
        self.ticker: Optional[str] = None
        self.calibration_date: Optional[date] = None
        self.jacobian: Optional[dict] = None
        self.report: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> dict:
        """Calibrated parameters ``{name: tf.Variable}`` (empty for a model without parameters)."""
        return {}

    @abstractmethod
    def process(self, equity) -> StochasticProcess:
        """Bind the model to *equity*: a new process (the model is not modified) for the Monte Carlo pricers.

        Called by the pricers inside their ``GradientTape``: the market-dependent part (forwards, local vol grid, ...)
        is rebuilt there, the parameters enter as the model's variables.
        """

    @classmethod
    @abstractmethod
    def calibrate(cls, equity, **settings) -> "EquityModel":
        """Calibrate to the market data of *equity*: an instance with parameters, Jacobian, report and metadata."""

    @classmethod
    @abstractmethod
    def from_equity(cls, equity, maturity=None, strike=None, **settings) -> StochasticProcess:
        """Process calibrated inside the caller's tape (greeks through the calibration): the factory used by the
        pricers when they are given the model class."""

    def map_model_greeks(self, model_greeks: dict, equity=None) -> dict:
        """Model greeks mapped to the market inputs, ``{market key: [sum_p dV/dp . dp/d variable]}``: through the
        stored calibration :attr:`jacobian` (*equity* unused); subclasses calibrated by simulation override it with
        the adjoint of their calibration.

        Raises:
            ValueError: If the model has no Jacobian (built by hand).
        """
        if self.jacobian is None:
            raise ValueError(f"{type(self).__name__} has no calibration Jacobian (built by hand): only model greeks")
        out = {}
        for name, per_key in self.jacobian.items():
            g = model_greeks.get(name)
            if g is None:
                continue
            g = np.asarray(g)
            for key, blocks in per_key.items():
                mapped = [np.tensordot(g, block, axes=g.ndim) for block in blocks]
                out[key] = [a + b for a, b in zip(out[key], mapped)] if key in out else mapped
        return out

    # ------------------------------------------------------------------
    # Helpers for the subclasses
    # ------------------------------------------------------------------

    def _record_calibration(self, equity, params: dict, tape: Optional[tf.GradientTape], report=None):
        """Store metadata, report and the Jacobian of the differentiable *params* (``{name: tensor}`` computed on
        *tape*) with respect to the market risk factors of *equity*."""
        from ..timehandles.utils import Settings

        self.ticker = equity.ticker
        self.calibration_date = Settings.evaluation_date
        self.report = report
        if tape is None or not params:
            return
        factors = equity.risk_factors()
        variables = [v for vs in factors.values() for v in vs]
        jac = {}
        for name, value in params.items():
            # one backward pass per parameter entry, to all the variables at once
            blocks = tape.jacobian(value, variables, experimental_use_pfor=False)
            blocks = [np.zeros(tuple(value.shape) + tuple(v.shape)) if b is None else b.numpy()
                      for b, v in zip(blocks, variables)]
            jac[name], i = {}, 0
            for key, vs in factors.items():
                jac[name][key] = blocks[i:i + len(vs)]
                i += len(vs)
        self.jacobian = jac

    @staticmethod
    def _variable(value, name: str) -> tf.Variable:
        return tf.Variable(np.asarray(tf.stop_gradient(value) if tf.is_tensor(value) else value, dtype=np.float64),
                           dtype=DTYPE, name=name)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_frames(self) -> dict:
        """Tables of the model: ``model`` (class, ticker, date), ``settings``, ``parameters`` and ``jacobian`` (long
        format), ``report``. :meth:`from_frames` rebuilds the same instance."""
        meta = pd.DataFrame([{"model": type(self).__name__, "ticker": self.ticker,
                              "calibration_date": self.calibration_date.isoformat() if self.calibration_date else None}])
        settings = pd.DataFrame([{"name": k, "value": json.dumps(v)} for k, v in self.settings.items()],
                                columns=["name", "value"])
        params = pd.DataFrame([{"name": n, "index": i, "value": float(x)}
                               for n, v in self.parameters.items() for i, x in enumerate(np.ravel(v.numpy()))],
                              columns=["name", "index", "value"])
        rows = []
        for n, per_key in (self.jacobian or {}).items():
            for key, blocks in per_key.items():
                for j, block in enumerate(blocks):
                    shape = json.dumps(list(np.shape(block)))
                    rows += [{"parameter": n, "key": key, "variable": j, "shape": shape, "index": i, "value": float(x)}
                             for i, x in enumerate(np.ravel(block))]
        jacobian = pd.DataFrame(rows, columns=["parameter", "key", "variable", "shape", "index", "value"])
        frames = {"model": meta, "settings": settings, "parameters": params, "jacobian": jacobian}
        if self.report is not None:
            frames["report"] = self.report.reset_index(drop=True)
        return frames

    @classmethod
    def from_frames(cls, frames: dict) -> "EquityModel":
        """Rebuild a model saved by :meth:`to_frames` (the class must match)."""
        meta = frames["model"].iloc[0]
        if meta["model"] != cls.__name__:
            raise ValueError(f"tables of a {meta['model']}, not of a {cls.__name__}")
        settings = {r["name"]: json.loads(r["value"]) for _, r in frames["settings"].iterrows()}
        values = {}
        for name, group in frames["parameters"].groupby("name", sort=False):
            values[name] = group.sort_values("index")["value"].to_numpy()
        model = cls._from_values(values, settings, frames)
        model.ticker = None if pd.isna(meta["ticker"]) else meta["ticker"]
        model.calibration_date = None if pd.isna(meta["calibration_date"]) else date.fromisoformat(meta["calibration_date"])
        model.report = frames.get("report")
        jac = frames.get("jacobian")
        if jac is not None and len(jac):
            model.jacobian = {}
            for (name, key, _), group in jac.groupby(["parameter", "key", "variable"], sort=False):
                block = group.sort_values("index")["value"].to_numpy().reshape(json.loads(group["shape"].iloc[0]))
                model.jacobian.setdefault(name, {}).setdefault(key, []).append(block)
        return model

    @classmethod
    def _from_values(cls, values: dict, settings: dict, frames: dict) -> "EquityModel":
        """Instance from the serialized parameter values (subclasses with parameters override)."""
        return cls(**settings)

    def __repr__(self) -> str:
        params = {n: np.round(v.numpy(), 6).tolist() for n, v in self.parameters.items()}
        return f"{type(self).__name__}(ticker={self.ticker!r}, parameters={params})"


def par_greeks(model: EquityModel, model_greeks: dict, market_greeks: Optional[dict] = None, equity=None) -> dict:
    """Greeks to the market inputs: the direct market greeks plus the model greeks mapped through the calibration,
    ``dV/dm = dV/dm|direct + sum_p dV/dp . dp/dm`` (:meth:`EquityModel.map_model_greeks`: the stored Jacobian, or
    the adjoint of the calibration for the models that need the market, e.g. :class:`HestonSLVModel`).

    Args:
        model (EquityModel): The calibrated model.
        model_greeks (dict): ``{parameter: dV/dp}`` (e.g. ``tape.gradient(price, model.parameters)``); for a portfolio
            on one underlying, the sum over the trades.
        market_greeks (dict, optional): ``{market key: [dV/d variable]}`` direct greeks
            (e.g. ``tape.gradient(price, equity.risk_factors())``).
        equity (Equity, optional): The underlying, for the models mapped by the adjoint of their calibration.

    Returns:
        dict: ``{market key: [array per variable]}``.

    Raises:
        ValueError: If the model has parameters but no calibration to map them through (built by hand), or needs
            *equity* and none is given.
    """
    out = {}
    for key, grads in (market_greeks or {}).items():
        out[key] = [np.zeros(()) if g is None else np.asarray(g) for g in grads]
    if not model.parameters:
        return out
    for key, mapped in model.map_model_greeks(model_greeks, equity).items():
        if key in out:
            out[key] = [a + b for a, b in zip(out[key], mapped)]
        else:
            out[key] = mapped
    return out


def bind(model, equity, maturity=None, strike=None) -> StochasticProcess:
    """Process for a pricer: ``model.process(equity)`` for an instance (ticker checked), ``model.from_equity(equity,
    maturity, strike)`` for a class (calibration inside the caller's tape)."""
    if isinstance(model, EquityModel):
        if model.ticker is not None and model.ticker != equity.ticker:
            raise ValueError(f"{type(model).__name__} calibrated on '{model.ticker}', priced on '{equity.ticker}'")
        return model.process(equity)
    return model.from_equity(equity, maturity, strike)


def check_model(model) -> None:
    """Pricer argument check: an :class:`EquityModel` instance or a class with a ``from_equity`` factory."""
    if not (isinstance(model, EquityModel) or (isinstance(model, type) and hasattr(model, "from_equity"))):
        raise TypeError(f"model must be an EquityModel instance or a model class with a from_equity factory, got {model!r}")
