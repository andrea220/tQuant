# TensorQuant

**TensorQuant (tQuant)** is a Python quant-finance library built on **TensorFlow**. Every price is a TensorFlow
graph from the market data to a number, so the sensitivities come from automatic differentiation (AAD): one backward
pass gives delta, vega per volatility node, rho per curve pillar and the dividend sensitivities, also through the
model calibrations. **tqstudio** is the web application on top of it: market data, calibrations and pricing in the
browser, with the engine running on one machine and the pages open to the colleagues on the same network.

This page is an overview. The details are in the technical documents (see [Documentation](#documentation)).

---

## At a glance

- **Market data**: real snapshots from Bloomberg (equities and indices, dividends, listed options, FX, rate curve
  quotes) or synthetic ones; curves bootstrapped from the quotes, equity vol surfaces and repo curves built from the
  listed options.
- **Instruments**: deposits, FRAs, swaps, OIS, caps / floors / collars, swaptions; vanilla (European, American),
  digital, barrier and autocallable equity options.
- **Pricers**: discounting and closed forms (Black-Scholes, Bjerksund-Stensland, Reiner-Rubinstein, Bachelier,
  Black-76) and Monte Carlo.
- **Models**: local volatility, displaced diffusion, Heston, Heston stochastic local volatility; Hull-White and G2++
  for rates.
- **Calibration**: equity models calibrated on a snapshot and stored as calibration sets, with the Jacobian that maps
  the model sensitivities to the market; Hull-White / G2++ to caplets and swaptions.
- **Greeks**: AAD to every market risk factor, in float64.

---

## Architecture

```
  Bloomberg / synthetic            marketdata/  (snapshots, calibration sets)
            |                                 |
            v                                 v
  MarketEnvironment  -- one access point to the market: curves, vol surfaces, spots, dividends, repo, FX, correlations
            |              (every input a float64 tf.Variable: the risk factors)
            v
  Product  +  Pricer  ( + model: StochasticProcess / calibrated EquityModel )
            |
            v
  price (TensorFlow graph)  -->  tape.gradient(price, market risk factors)  =  greeks
```

- **Market data on disk** (`marketdata/`): a snapshot is a folder of CSV tables per date and time. `mdbuild` builds the
  real ones from the Bloomberg Desktop API and, right after the download, bootstraps the rate curves and builds the
  equity vol surfaces from the option quotes.
- **`MarketEnvironment`**: the only access point to the market, a dictionary of market objects under keys such as
  `IR:EUR:ESTR:SPOT` or `EQ:EUR:SX5E:VOL`. Their numbers are `tf.Variable`s with a stable identity, the risk factors
  of every price. `get_equity(ticker)` gives a view of one underlying: spot, forward, dividends, repo, vol surface.
- **Instruments** (`Product`): the description of a trade only (dates, strikes, schedules, underlying or rate index),
  no market data and no pricing logic.
- **Pricers** (`Pricer`): `pricer.price(product, market_env, autodiff=True)` builds the price inside a gradient tape;
  `PricerAssignment` picks the default pricer of a product.
- **Models** (`StochasticProcess`, `EquityModel`): the Monte Carlo pricers simulate a process on the market forwards. An
  equity model is used either calibrated inside the pricing (greeks through the calibration) or calibrated once,
  stored and held (fast pricing, greeks mapped through the stored Jacobian).
- **Calibration**: equity vols from listed options, model calibration sets with their fit and drift, rates
  calibration.

| Package | Content |
|---|---|
| `tensorquant/timehandles` | dates, calendars (TARGET, US, UK), day counts, schedules |
| `tensorquant/numericalhandles` | interpolation (linear, monotone convex, cubic splines), Newton, implicit-function derivatives |
| `tensorquant/markethandles` | `MarketEnvironment`, curves, vol surfaces, dividends, `Equity` view, bootstrap, I/O |
| `tensorquant/index`, `flows`, `instruments` | rate indices, coupons and legs, the products |
| `tensorquant/pricers` | closed-form, discounting and Monte Carlo pricers |
| `tensorquant/models` | stochastic processes and equity model instances |
| `tensorquant/calibration` | equity vols from options, calibration sets, caplet stripping, short-rate calibration |
| `tensorquant/analytics` | swap exposures (EE, ENE, PFE) |
| `tqstudio` | the web application (FastAPI) |
| `marketdata` | snapshots, `mdbuild` (Bloomberg), configuration, calibration sets |

---

## tqstudio

- **Market data**: one underlying at a glance: spot, dividends and repo, forwards, the implied vol surface with the
  market quotes, correlations; rate curves and FX.
- **Calibration**: the calibration sets (fit per model and underlying, smiles against the quotes, model tests), the
  universe of the models to calibrate, and when to recalibrate Heston SLV.
- **Tradepad**: define a trade and price it with the model of your choice: price first, then the greeks.

```
python -m tqstudio                   # real snapshots (marketdata/live)
python -m tqstudio --data synthetic  # synthetic snapshots
```

---

## Documentation

| Document | Content |
|---|---|
| **Architecture** (`study/docs/architecture.tex`) | how the library is built: market data environment, instruments, pricers, models, calibration, the planned bridge with Murex |
| **Models and Pricing** (`study/docs/models_and_pricing.tex`) | the models as implemented: market handles (curves, vol surfaces), dynamics, calibration, closed forms, simulation, sensitivities |
| **Exercises** (`study/docs/exercises.tex`) | questions and exercises with the notebooks of `study/notebooks` |
| `ROADMAP.md` / `DEVELOPMENT_LOG.md` | open work / completed developments with the numbers that verify them |

The PDFs are built with `latexmk` in `study/docs` (into `study/docs/build`) and opened from the tqstudio home page.
The old ReadTheDocs documentation (`docs/`) is deprecated.

---

## Getting started

```
python -m venv tqenv && tqenv\Scripts\activate          # Python 3.12
pip install -r requirements.txt -e .
python -m unittest discover -s tests                     # fast suite (TQ_SLOW=1 for the full one)
```

```python
import datetime as dt
import tensorquant as tq

mkt = tq.MarketEnvironment.from_folder("marketdata")     # latest snapshot
tq.Settings.evaluation_date = mkt.evaluation_date

today = mkt.evaluation_date
option = tq.VanillaOption(tq.Currency.EUR, today, today + dt.timedelta(days=365), tq.OptionType.Put,
                          strike=float(mkt.get_eq_spot("TICKER1")), underlying="TICKER1")
pricer = tq.PricerAssignment.create(option)              # Black-Scholes
pricer.price(option, mkt, autodiff=True)
greeks = pricer.tape.gradient(option.price, mkt.risk_factors())   # delta, vega per node, rho per pillar, ...
```

More in `examples/` and `study/notebooks/`.

---

## License and contact

GPL-3.0, see [LICENSE](LICENSE). Contact: [carapelliandrea@gmail.com](mailto:carapelliandrea@gmail.com).
