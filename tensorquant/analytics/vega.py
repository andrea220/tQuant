"""Vega of an equity product read by expiry: level (the whole smile of an expiry up 1 vol point) and skew (the smile
of an expiry rotated around its forward), from the vega per node of the vol surface.

The vol surface enters a price only through its nodes, so the first-order price change of any deformation
``dsigma[i, j]`` of the nodes (vol points) is ``sum_j vega[i, j] dsigma[i, j]``: a sensitivity to a shape of the
smile is a weighted sum of the node vegas, with no repricing. Node vegas of a local vol price alternate in sign
along a row (a single node moves the slope and the curvature of the smile, so the local vol around it); these sums
are the stable reading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..markethandles.equity import Equity

__all__ = ["SKEW_STEP", "SKEW_RANGE", "VEGA_TENORS", "skew_weights", "vega_by_expiry", "vega_by_tenor"]

SKEW_STEP = 0.10              # skew scenario: +1 vol point per 10% of strike below the forward, -1 per 10% above
SKEW_RANGE = (0.5, 1.5)       # strikes / forward beyond which the shift is held (+5 / -5 vol points)
VEGA_TENORS = (0.25, 0.5) + tuple(float(y) for y in range(1, 31))   # 3M, 6M, 1Y, 2Y, ... 30Y


def skew_weights(equity: Equity, step: float = SKEW_STEP, moneyness_range: tuple = SKEW_RANGE) -> np.ndarray:
    """Shift of each node of the surface of *equity* (vol points) in the +1 skew scenario: ``(1 - m) / step`` with
    ``m = K / F(T)`` (forward of the expiry) clipped to *moneyness_range*. Positive puts, negative calls: a steeper
    skew; its negative is a flattening.

    Args:
        equity (Equity): The underlying (its vol surface and forward).
        step (float): Moneyness per vol point.
        moneyness_range (tuple): ``(low, high)`` of ``K / F`` beyond which the shift is constant.

    Returns:
        np.ndarray: ``[n_maturities, n_strikes]``.
    """
    surface = equity.vol_surface
    strikes = np.array([float(k) for k in surface.strike])
    forwards = np.array([float(equity.forward(float(t))) for t in surface.maturity])
    m = np.clip(strikes[None, :] / forwards[:, None], *moneyness_range)
    return (1.0 - m) / step


def vega_by_expiry(vega: np.ndarray, equity: Equity, step: float = SKEW_STEP,
                   moneyness_range: tuple = SKEW_RANGE) -> pd.DataFrame:
    """Vega per expiry of a node-vega matrix.

    Args:
        vega (np.ndarray): ``[n_maturities, n_strikes]`` price change per vol point of each node.
        equity (Equity): The underlying whose surface *vega* refers to.
        step, moneyness_range: The skew scenario (:func:`skew_weights`).

    Returns:
        pd.DataFrame: per maturity ``t``, ``level`` (row sum: all strikes of the expiry +1 vol point),
        ``cumulative`` (running sum of ``level``), ``skew`` (the +1 skew scenario on that expiry).
    """
    vega = np.asarray(vega, dtype=float)
    level = vega.sum(axis=1)
    skew = (vega * skew_weights(equity, step, moneyness_range)).sum(axis=1)
    t = [float(x) for x in equity.vol_surface.maturity]
    return pd.DataFrame({"t": t, "level": level, "cumulative": np.cumsum(level), "skew": skew})


def vega_by_tenor(level, maturities, tenors=VEGA_TENORS) -> pd.DataFrame:
    """Vega per expiry gathered on standard tenors: the vega of an expiry between two tenors is split between them
    linearly in time (all of it on a tenor it falls on), before the first tenor on the first, after the last used one
    on it; the total is kept.

    Args:
        level: Vega of each expiry (``vega_by_expiry(...)["level"]``).
        maturities: The expiries (years), ascending.
        tenors: Candidate tenors (years), ascending; those up to the first at or beyond the last expiry are used.

    Returns:
        pd.DataFrame: ``tenor`` (years), ``label`` (``3M``, ``6M``, ``1Y``, ...), ``vega``.
    """
    level, maturities = np.asarray(level, float), np.asarray(maturities, float)
    tenors = np.asarray(tenors, float)
    last = int(np.searchsorted(tenors, maturities.max() - 1e-9))
    grid = tenors[: min(last, len(tenors) - 1) + 1]
    vega = np.zeros(len(grid))
    for t, v in zip(maturities, level):
        if t <= grid[0]:
            vega[0] += v
        elif t >= grid[-1]:
            vega[-1] += v
        else:
            j = int(np.searchsorted(grid, t)) - 1
            w = (grid[j + 1] - t) / (grid[j + 1] - grid[j])
            vega[j] += w * v
            vega[j + 1] += (1.0 - w) * v
    labels = [f"{round(12 * t)}M" if t < 1 else f"{t:.0f}Y" for t in grid]
    return pd.DataFrame({"tenor": grid, "label": labels, "vega": vega})
