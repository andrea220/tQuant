from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np
import tensorflow as tf

from ..numericalhandles.dtypes import DTYPE, as_variable
from ..timehandles.daycounter import DayCounter


def _linear_weights(grid: tf.Tensor, x) -> tuple:
    """Bracketing index and weight on a sorted 1D grid, flat outside (differentiable in x)."""
    # numbers straight to float64 (tf.cast of a Python float goes through float32)
    x = x if isinstance(x, (tf.Tensor, tf.Variable)) else tf.constant(np.float64(x), DTYPE)
    x = tf.clip_by_value(tf.cast(x, DTYPE), grid[0], grid[-1])
    n = tf.shape(grid)[0]
    i = tf.clip_by_value(tf.searchsorted(grid, tf.reshape(tf.stop_gradient(x), [1]), side="right")[0] - 1, 0, n - 2)
    w = (x - grid[i]) / (grid[i + 1] - grid[i])
    return i, w


class SwaptionVolCube:
    """Swaption vols on (option expiry x underlying swap tenor x strike offset from the ATM forward).

    The vols are one float64 ``tf.Variable`` ``[n_expiries, n_tenors, n_offsets]`` (a risk factor: vega per
    node). ``volatility(expiry, tenor, offset)`` interpolates linearly in each dimension (flat outside the
    grid); ``offset = K - F`` with ``F`` the forward swap rate.

    Args:
        reference_date (date): Reference date.
        daycounter (DayCounter): Day count of the expiry year fractions.
        expiries (Sequence[float]): Option expiries (years, increasing).
        tenors (Sequence[float]): Underlying swap tenors (years, increasing).
        strike_offsets (Sequence[float]): Strike minus ATM forward (decimal, increasing; include 0).
        volatilities: ``[n_expiries, n_tenors, n_offsets]`` vols (array-like or float64 ``tf.Variable``).
        vol_type (str): ``"normal"`` (Bachelier, default) or ``"lognormal"`` (Black).

    Raises:
        ValueError: If the shape does not match the axes or ``vol_type`` is unknown.
    """

    def __init__(self, reference_date: date, daycounter: DayCounter, expiries: Sequence[float],
                 tenors: Sequence[float], strike_offsets: Sequence[float], volatilities, vol_type: str = "normal"):
        if vol_type not in ("lognormal", "normal"):
            raise ValueError(f"vol_type must be 'lognormal' or 'normal', got '{vol_type}'")
        self.reference_date = reference_date
        self.daycounter = daycounter
        self.expiries = [float(x) for x in expiries]
        self.tenors = [float(x) for x in tenors]
        self.strike_offsets = [float(x) for x in strike_offsets]
        self.vol_type = vol_type
        self.volatilities = as_variable(volatilities, name="SWVOL")
        shape = (len(self.expiries), len(self.tenors), len(self.strike_offsets))
        if tuple(self.volatilities.shape) != shape:
            raise ValueError(f"volatilities shape {tuple(self.volatilities.shape)} does not match the axes {shape}")
        for axis in (self.expiries, self.tenors, self.strike_offsets):
            if len(axis) < 2 or np.any(np.diff(axis) <= 0):
                raise ValueError("each axis needs at least 2 increasing points")
        self._e = tf.constant(self.expiries, DTYPE)
        self._t = tf.constant(self.tenors, DTYPE)
        self._k = tf.constant(self.strike_offsets, DTYPE)

    @property
    def variables(self) -> list[tf.Variable]:
        """Risk-factor variables: ``[volatilities]``."""
        return [self.volatilities]

    def volatility(self, expiry, tenor, strike_offset) -> tf.Tensor:
        """Trilinear vol at (expiry years, tenor years, strike - forward), flat outside the grid.

        Differentiable w.r.t. the vol nodes and the strike offset (hence the forward).
        """
        ie, we = _linear_weights(self._e, expiry)
        it, wt = _linear_weights(self._t, tenor)
        ik, wk = _linear_weights(self._k, strike_offset)
        v = self.volatilities
        out = tf.constant(0.0, DTYPE)
        for de, fe in ((0, 1.0 - we), (1, we)):
            for dt, ft in ((0, 1.0 - wt), (1, wt)):
                for dk, fk in ((0, 1.0 - wk), (1, wk)):
                    out += fe * ft * fk * v[ie + de, it + dt, ik + dk]
        return out
