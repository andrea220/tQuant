"""mdbuild: market data snapshots from Bloomberg (and, later, the internal service for curves and vols).

Independent of tensorquant: it writes the snapshot tables (CSV, the layout ``MarketEnvironment.from_folder`` reads)
and keeps the price history (Parquet). Run from the ``marketdata`` folder: ``python -m mdbuild``.
"""

from .universe import Instrument, load_universe

__all__ = ["Instrument", "load_universe"]
