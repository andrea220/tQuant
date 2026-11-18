"""Where the market data are: the shared ``marketdata`` folder (snapshots, calibration sets, configuration, price
history, caches), the first valid one of ``tqstudio/config.yaml`` (``marketdata_roots``, tried in order) unless the
environment variable ``TQSTUDIO_MARKETDATA`` names one (``python -m tqstudio --marketdata <folder>``). A folder is valid
when it exists and holds ``config/settings.yaml``. The code (``marketdata/mdbuild``) stays in the repository."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CODE_MARKETDATA = REPO / "marketdata"                     # the mdbuild package (code), run from here
CONFIG = Path(__file__).resolve().parent / "config.yaml"
ENV = "TQSTUDIO_MARKETDATA"


def is_valid_root(folder) -> bool:
    """A market data folder: it exists and holds ``config/settings.yaml``."""
    return (Path(folder) / "config" / "settings.yaml").is_file()


def candidate_roots(config: Path = CONFIG) -> list[Path]:
    """The ``marketdata_roots`` of *config*, in order (the repository's ``marketdata`` without a config)."""
    data = yaml.safe_load(Path(config).read_text(encoding="utf-8")) if Path(config).is_file() else {}
    roots = (data or {}).get("marketdata_roots") or [str(CODE_MARKETDATA)]
    return [Path(str(r).strip()) for r in roots]


def resolve_root(override: str | None = None, config: Path = CONFIG) -> Path:
    """The market data folder: *override* (or ``TQSTUDIO_MARKETDATA``) when given, else the first valid candidate.

    Raises:
        FileNotFoundError: The override is not a valid folder, or no candidate is (the message lists them).
    """
    override = override or os.environ.get(ENV)
    if override:
        if not is_valid_root(override):
            raise FileNotFoundError(f"{override}: not a market data folder (no config/settings.yaml)")
        return Path(override)
    tried = candidate_roots(config)
    for root in tried:
        if is_valid_root(root):
            return root
    raise FileNotFoundError("no market data folder among " + ", ".join(str(r) for r in tried)
                            + f" (marketdata_roots in {config})")


MARKETDATA = resolve_root()
