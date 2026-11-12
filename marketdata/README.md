# marketdata

Market data snapshots for tQuant / tqstudio, self-contained (own requirements, no tensorquant import).

## Layout

| Path | Content |
|---|---|
| `config/universe.yaml` | the universe: Bloomberg tickers only (`Equity`, `Curncy`, `Index` yellow keys) |
| `config/settings.yaml` | Bloomberg connection, fields, history length, correlation window, output folders |
| `mdbuild/` | the builder: `universe.py`, `sources.py` (Bloomberg Desktop API), `history.py` (Parquet history, correlations), `snapshot.py` (tables), `internal.py` (internal service: curves and vols, to be written) |
| `live/<yyyymmdd>/<hhmm>/` | real snapshots (git-ignored: licensed data) |
| `history/prices.parquet` | daily closes of the universe, 10 years, split-adjusted (git-ignored) |
| `<yyyymmdd>/<hhmm>/` | synthetic snapshots (`synthetic/generate_synthetic.py`, needs tensorquant) |
| `tests/` | builder tests with an in-memory source (no Bloomberg) |

## Install and run

```
pip install -r marketdata/requirements.txt      (blpapi from Bloomberg's index, pandas, pyarrow, PyYAML)
cd marketdata
python -m mdbuild                               (Bloomberg terminal running on this machine)
python -m unittest discover -s tests
```

A run updates the history (only the missing days; a new ticker gets the whole window) and writes
`live/<yyyymmdd>/<hhmm>/` with:

| File | Content | Source |
|---|---|---|
| `eq_spot.csv` | `ticker, ccy, spot, repo, div_yield` (repo and yield empty) | `PX_LAST`, `CRNCY` |
| `eq_dividends.csv` | `ticker, ccy, ex_date, pay_date, amount`, future ex-dates only, pay date = ex-date | `BDVD_ALL_PROJECTIONS` |
| `fx_spot.csv` | `base, quote, spot` (1 base = spot quote) | `PX_LAST` |
| `ir_fixings.csv` | `index, ccy, fixing_date, value` (decimal), `bloomberg` | `PX_LAST`, `LAST_UPDATE_DT` |
| `ir_quotes.csv` | `ccy, curve, tenor, instrument, quote, bloomberg, last_update`: members of the ICVS curves of `universe.yaml` (`curves:`), mid, decimal; `fixing`, `fra`, `ois`, `swap`. tQuant bootstraps the curves on load | `CURVE_TENOR_RATES` |
| `ir_curves.csv`, `ir_curves_jacobian.csv`, `ir_calibration.csv` | the curves bootstrapped from `ir_quotes.csv` right after the download (zero rates at the pillars), `dNPV_i / dr_j`, quotes and `dNPV_i / dq_i` for the par sensitivities: written when tensorquant is installed (optional), otherwise tQuant bootstraps on load | tQuant `bootstrap_snapshot` |
| `correlations.csv` | `factor_1, factor_2, corr`: daily log returns, last 252 common dates; factors `EQ:<ccy>:<ticker>`, `FX:<quote>:<pair>` | history |
| `snapshot_info.json` | time, universe, fields, row counts, what is missing | |

The tables follow `MarketEnvironment.from_folder`; equity tickers are the Bloomberg ones without the yellow key,
spaces to `_` (`ENI IM Equity` -> `ENI_IM`).

**Not yet in the real snapshots**: equity / rate / FX vols (internal service, `mdbuild/internal.py`), repo.

Bloomberg data are licensed: keep `live/` and `history/` out of git and check the terms before sharing them through
the web app.
