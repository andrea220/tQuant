"""Data sources. A source answers three questions: reference fields of tickers, bulk rows of one field (the dividend
projections) and daily history. :class:`BloombergSource` is the Desktop API (the terminal on this machine); tests
use an in-memory source with the same methods."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class Source(Protocol):
    def reference(self, tickers: list[str], fields: list[str]) -> pd.DataFrame:
        """``[ticker x field]`` values (missing as NaN)."""

    def bulk(self, ticker: str, field: str) -> pd.DataFrame:
        """Rows of a bulk field (one column per sub-field)."""

    def history(self, tickers: list[str], field: str, start: date, end: date) -> pd.DataFrame:
        """Long table ``date, ticker, value`` of daily values."""


class BloombergSource:
    """Bloomberg Desktop API (``//blp/refdata``): ReferenceDataRequest and HistoricalDataRequest."""

    def __init__(self, host: str = "localhost", port: int = 8194, timeout_ms: int = 30000,
                 adjust_splits: bool = True, adjust_dividends: bool = False):
        import blpapi                                   # only needed when Bloomberg is used

        self._blpapi = blpapi
        self._timeout = int(timeout_ms)
        self._adjust = (bool(adjust_splits), bool(adjust_dividends))
        options = blpapi.SessionOptions()
        options.setServerHost(host)
        options.setServerPort(int(port))
        self._session = blpapi.Session(options)
        if not self._session.start():
            raise ConnectionError(f"cannot start a Bloomberg session on {host}:{port} (is the terminal running?)")
        if not self._session.openService("//blp/refdata"):
            raise ConnectionError("cannot open //blp/refdata")
        self._ref = self._session.getService("//blp/refdata")

    def close(self) -> None:
        self._session.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    def _messages(self, request):
        blpapi = self._blpapi
        self._session.sendRequest(request)
        while True:
            event = self._session.nextEvent(self._timeout)
            if event.eventType() == blpapi.Event.TIMEOUT:
                raise TimeoutError("Bloomberg request timed out")
            yield from event
            if event.eventType() == blpapi.Event.RESPONSE:
                return

    @staticmethod
    def _value(element):
        if element.isNull():                               # empty field (e.g. a record date not set yet)
            return None
        value = element.getValue()
        return value.isoformat() if hasattr(value, "isoformat") and not isinstance(value, str) else value

    def reference(self, tickers: list[str], fields: list[str]) -> pd.DataFrame:
        request = self._ref.createRequest("ReferenceDataRequest")
        for t in tickers:
            request.getElement("securities").appendValue(t)
        for f in fields:
            request.getElement("fields").appendValue(f)
        rows = {t: {} for t in tickers}
        for msg in self._messages(request):
            if not msg.hasElement("securityData"):
                continue
            for sd in msg.getElement("securityData").values():
                ticker = sd.getElementAsString("security")
                if sd.hasElement("securityError"):
                    raise ValueError(f"Bloomberg: unknown security '{ticker}'")
                data = sd.getElement("fieldData")
                for f in fields:
                    if data.hasElement(f):
                        rows[ticker][f] = self._value(data.getElement(f))
        return pd.DataFrame.from_dict(rows, orient="index", columns=fields)

    def bulk(self, ticker: str, field: str, overrides: dict = None) -> pd.DataFrame:
        """Bulk field of one security, one row per item; *overrides* ``{field id: value}`` (e.g. the ``CHAIN_*``
        filters of ``CHAIN_TICKERS``)."""
        request = self._ref.createRequest("ReferenceDataRequest")
        request.getElement("securities").appendValue(ticker)
        request.getElement("fields").appendValue(field)
        for name, value in (overrides or {}).items():
            override = request.getElement("overrides").appendElement()
            override.setElement("fieldId", name)
            override.setElement("value", str(value))
        rows = []
        for msg in self._messages(request):
            if not msg.hasElement("securityData"):
                continue
            for sd in msg.getElement("securityData").values():
                data = sd.getElement("fieldData")
                if not data.hasElement(field):
                    continue
                for item in data.getElement(field).values():
                    rows.append({str(e.name()): self._value(e) for e in item.elements()})
        return pd.DataFrame(rows)

    def history(self, tickers: list[str], field: str, start: date, end: date) -> pd.DataFrame:
        request = self._ref.createRequest("HistoricalDataRequest")
        for t in tickers:
            request.getElement("securities").appendValue(t)
        request.getElement("fields").appendValue(field)
        request.set("startDate", start.strftime("%Y%m%d"))
        request.set("endDate", end.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")
        request.set("adjustmentFollowDPDF", False)          # explicit adjustments, not the terminal's settings
        request.set("adjustmentSplit", self._adjust[0])
        request.set("adjustmentNormal", self._adjust[1])
        request.set("adjustmentAbnormal", self._adjust[1])
        rows = []
        for msg in self._messages(request):
            if not msg.hasElement("securityData"):
                continue
            sd = msg.getElement("securityData")
            ticker = sd.getElementAsString("security")
            for point in sd.getElement("fieldData").values():
                if point.hasElement(field):
                    rows.append({"date": point.getElementAsDatetime("date"), "ticker": ticker,
                                 "value": point.getElementAsFloat(field)})
        frame = pd.DataFrame(rows, columns=["date", "ticker", "value"])
        frame["date"] = pd.to_datetime(frame["date"])
        return frame
