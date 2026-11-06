"""Test markers.

Slow tests (Monte Carlo with tight statistical tolerances, bump-and-reprice through calibrations, long AAD backward
passes) run only with ``TQ_SLOW=1``:

    python -m unittest discover -s tests                 # fast suite (default)
    TQ_SLOW=1 python -m unittest discover -s tests       # full suite (before a commit, or when touching a model)
"""

import os
import unittest

RUN_SLOW = os.environ.get("TQ_SLOW", "") not in ("", "0")


def slow(test):
    """Skip *test* unless ``TQ_SLOW=1``."""
    return unittest.skipUnless(RUN_SLOW, "slow: set TQ_SLOW=1")(test)
