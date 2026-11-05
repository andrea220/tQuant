# TensorQuant (tQuant)

Python quant-finance library on **TensorFlow**. Core idea: prices are TF graphs, so greeks come
from `tf.GradientTape` (AAD) instead of separate analytic code.

- Package `tensorquant/` (import as `import tensorquant as tq`; everything is re-exported via `*`).
- `setup.py` only (no `pyproject.toml`): loose bounds there, exact verified pins in `requirements.txt`.
- `docs/` (Sphinx/ReadTheDocs) is **deprecated**: out of date, do not update or rebuild it.
- Study material in `study/` (own `CLAUDE.md`).
- Code, docstrings and comments in English.

## Working rules

- Do **only** what is asked. Anything extra (installs, reinstalls, builds, refactors, cleanups, extra files,
  follow-up fixes) → ask first, don't do it.
- Commit directly on `main`; commit and push only when asked.

## Environment & commands

- Venv: `C:\dev\virtual_envs\tqenv` (Python 3.12), set up with `pip install -r requirements.txt -e .`.
- Tests: `python -m unittest discover -s tests -v`
- Examples: `python examples/<script>.py` — most read `SOURCE_MKT_DATA`: `'Generated'` uses the synthetic
  market in `examples/market_data.py`; `'Real'` needs the MDM datastore on `W:/`, which **no longer exists** (do not look for it; see roadmap step 5).
- Set `TF_ENABLE_ONEDNN_OPTS=0` for reproducible numbers.

## Module map

| Package | Main classes |
|---|---|
| `timehandles/` | `Settings` (global `evaluation_date`), `DayCounter` + `DayCounterConvention`, `Calendar`, `TARGET`, `ScheduleGenerator`, `DateGrid`, enums `TimeUnit`, `BusinessDayConvention`, `CompoundingType`, `Frequency` |
| `numericalhandles/` | `LinearInterp`, `newton`, `newton_1d`, `DTYPE` + `as_variable` (dtype of market data, risk-factor variables) |
| `markethandles/` | `MarketEnvironment`, `RiskFactor`, `MarketDataType`, `MarketMapValidator`, `RateCurve` / `FlatCurve` / `DefaultCurve`, `VolatilitySurface` / `BlackConstantVolatility`, `DividendCurve`, `Equity` (underlying view), `CorrelationMatrix`, `InterestRate`, `CurveBootstrap`; `utils.py`: `Currency`, `OptionType`, `ExerciseType`, `PayoffType`, `SwapType`, `Position` |
| `index/` | `Index` (ABC), `OvernightIndex`, `IborIndex`, `InflationIndex` |
| `flows/` | `CashFlow`, `Coupon` (ABC), `FixedCoupon`/`FixedRateLeg`, `FloatingCoupon`/`FloatingRateLeg` |
| `instruments/` | `Product` (ABC), `Deposit`, `Fra`, `Swap`, `Ois`, `Option`/`VanillaOption`, `AutocallableOption`, `Cap`/`Floor`/`Collar`, generators in `helpers.py` |
| `models/` | `StochasticProcess` (ABC), `GeometricBrownianMotion`, `ArithmeticBrownianMotion`, `OrnsteinUhlenbeckProcess`, `HullWhiteProcess`, `G2PlusPlusProcess`, `LocalVolatilityModel` (Dupire), `DisplacedDiffusionModel` |
| `pricers/` | `Pricer` (ABC), `BlackScholesPricer` (EU BS + American Bjerksund–Stensland; `dividend_model` discrete/continuous), `VanillaMCPricer`, `AutocallableMCPricer`, `SwapPricer`, `OisPricer`, `FraPricer`, `DepositPricer`, coupon/leg discounting, `PricerAssignment` (factory) |
| `analytics/` | `GaussianPathGenerator`, `HullWhiteShortRateGenerator`, `SwapExposureGenerator` |

## Key patterns

- **Product → Pricer.** Instruments subclass `Product` (`ccy`, `start_date`, `end_date`, `price`).
  Pricers subclass `Pricer` and implement `calculate_price(product, market_env)`.
  `Pricer.price(product, market_env, autodiff=True)` runs it under a `GradientTape`, stores the price on
  `product.price` and the tape on `pricer.tape`; greeks via `pricer.tape.gradient(product.price, <tf.Variable>)`.
- **Risk factors live in market objects** as float64 `tf.Variable`s with stable identity (update with `assign`):
  `RateCurve.variables` (one per pillar; `FlatCurve` has one), `VolatilitySurface.volatility_matrix`
  (`[n_maturities, n_strikes]`), and EQ `SPOT|REPO|DIVYIELD` (wrapped by `MarketEnvironment`, one shared
  variable per key). `market_env.risk_factors()` returns `{market_key: [variables]}`, so one
  `tape.gradient(price, market_env.risk_factors())` gives bucketed rho, vega per node and delta.
  Pricers must not create their own variables from market data nor detach with `.numpy()`/`float()`; the only
  pricer-local variable allowed is time to maturity (theta), exposed on the product (`product.time_to_maturity`).
- **MarketEnvironment** is the only access point to market data. Flat dict with keys
  `RISKFACTOR:CCY:TICKER:TYPE`, e.g. `IR:EUR:ESTR:SPOT` → `RateCurve`, `EQ:EUR:SX5E:VOL` → `VolatilitySurface`,
  `EQ:EUR:X:SPOT|REPO|DIVYIELD` → float64 `tf.Variable` (a float is accepted as input and wrapped), `EQ:EUR:X:DIV` → `DividendCurve`,
  `FX:USD:EURUSD:SPOT` → float64 `tf.Variable` (1 EUR = x USD), `CORR:ALL:GLOBAL:SPOT` → `CorrelationMatrix`
  (factor names = market-map key prefix, e.g. `EQ:EUR:TICKER1`; one float64 `[n, n]` variable).
  Typed accessors: `get_ir_curve`, `get_eq_spot`, `get_eq_vol_surface`, `get_eq_repo`, `get_eq_div_yield`,
  `get_eq_dividends`, `get_currency`, `get_fx_spot(base, quote)` (direct, inverse or cross via a common
  currency; only stored pairs are risk factors), `get_correlation_matrix`, `get_correlation(names)`. Build from a snapshot with
  `MarketEnvironment.from_folder("marketdata", evaluation_date, calibration_set=None)` (latest snapshot of the day
  when `None`); `from_data_path(...)` reads the old MDM layout (deprecated, `W:/` gone).
- **Equity underlying**: `market_env.get_equity(ticker)` returns an `Equity` **view** (references, no copies) with
  spot, repo, dividends, vol surface and the currency discount curve; `forward(T)`, `pv_dividends(T)`,
  `implied_div_yield(T)`, `risk_factors()`, `summary(T)`. Dividend model from the data: `DIV` present → discrete
  (`DIVYIELD` ignored); otherwise `DIVYIELD` as continuous yield. Single place for the equity forward.
- **Evaluation date** is global: set `tq.Settings.evaluation_date` before building/pricing.

## Development rule

**Before adding anything, check whether existing infrastructure already covers it, even partially.**

1. New instrument → subclass `Product`; reuse `flows/` for cash flows.
2. New pricer → subclass `Pricer`; read market data only via `MarketEnvironment` accessors; register it in
   `PricerAssignment` if it should be auto-selected.
3. New model → subclass `StochasticProcess`; check `analytics/GaussianPathGenerator` first for Gaussian rate models.
4. New market data type → extend `RiskFactor` / `MarketDataType` + `MarketMapValidator`, then add an accessor.
5. New curve/surface → check `RateCurve`, `VolatilitySurface`, `DividendCurve` first; propose the structure before coding.
6. Day count / calendars → extend the existing enums, no parallel classes.
7. Interpolation → add to `numericalhandles/interpolation.py`, never inline in pricers.

If a needed piece is missing, **propose first** (which class/ABC/enum, where it lives, how it plugs into
`MarketEnvironment` / `Pricer` / `Product`) and only then implement.

## Code conventions

- Market inputs and calibrated params are float64 `tf.Variable`/`tf.Tensor`; scalar equity data (spot, repo,
  div yield) may be passed as `float`, `MarketEnvironment` wraps them into `tf.Variable`s.
- Google-style docstrings (Args/Returns/Raises); type hints everywhere; relative imports inside the package.
- No access to `MarketEnvironment._market` outside `MarketEnvironment`.
- Verify numbers against an independent check (closed form, MC, or the examples) when touching pricing code.

## Testing

Regression safety comes from the unit tests in `tests/`, **not** from running `examples/` or `study/` notebooks.
Run examples/notebooks only when changing them or when asked.

- Run: `python -m unittest discover -s tests -v` (in `tqenv`, with `TF_ENABLE_ONEDNN_OPTS=0`).
- Before refactoring an area, add tests that pin the current numbers (prices, greeks) with explicit tolerances,
  then refactor until they are green. Add independent checks where possible: closed form, put–call parity,
  finite differences vs AAD gradients, MC within a few standard errors.
- Fixtures are self-contained and synthetic (no `W:/` datastore, no imports from `examples/`); fixed seeds; float64.
- Keep the coverage table below up to date when adding tests.

| Area | Test file | Covered |
|---|---|---|
| `MarketEnvironment` accessors | `tests/test_marketenvironment.py` | spot, vol, repo, div yield, dividends, currency, default curve |
| Risk factors | `tests/test_risk_factors.py` | float64 variables in curves/surfaces/EQ scalars, `risk_factors()`, gradients vs finite differences, bilinear weights |
| `RateCurve` / `FlatCurve` / `DividendCurve` | `tests/test_rate_curve.py` | discount/zero/forward/inst_fwd vs numpy reference, `from_zcb` round trip, PV dividends, float64 tensor outputs, inst_fwd gradient vs finite differences, Hull-White values that use inst_fwd (pinned) |
| `Equity` view | `tests/test_equity.py` | references to market objects, dividend model selection, PV/forward vs closed form, forward vs `BlackScholesPricer`, implied yield, gradients (spot, repo, pillars vs FD), summary |
| `BlackScholesPricer` | `tests/test_black_scholes.py` | EU/AM prices and greeks pinned from the float32 version, EU vs closed form, put–call parity, forward = `Equity.forward`, delta/rho per pillar/vega per node vs bumps, dividend models (discrete, continuous sticky yield, DIVYIELD-only), float64 |
| FX / `CorrelationMatrix` | — | postponed (see roadmap step 5) |
| `timehandles/` | — | — |
| `numericalhandles/dtypes.py` | `tests/test_dtypes.py` | `DTYPE`, `as_variable` (reuse, casts, no float32 round trip) |
| `numericalhandles/` (interpolation, newton) | — | — |
| Equity pricers/models | — | — |
| Rate pricers/flows/bootstrap | — | — |

## Refactor roadmap

Order: **1. market & time handles → 2. equity (analytic) → 3. interest rates (analytic) → 4. stochastic processes &
Monte Carlo → 5. market data I/O**. Finish a step before starting the next: unit tests for the area added first (see Testing), then all
tests green.

### 1. Market & time handles (`markethandles/`, `timehandles/`, `numericalhandles/`)

- [x] Risk factors in market objects: float64 `variables`, `MarketEnvironment.risk_factors()`.
- [x] dtype policy for market objects: float64 variables and float64 tensor outputs (`discount(0)` no longer returns `int`).
- [x] `RateCurve.inst_fwd` returns a differentiable float64 tensor (was `math.log` → float, no gradient).
- [x] `pandas` stays capped `<3` (only used for CSV loading and display tables; revisit in step 5).

### 2. Equity (analytic pricers)

- [x] `Equity` underlying view (`MarketEnvironment.get_equity`) with forward, PV dividends, implied yield, risk
      factors; example `study/notebooks/03_equity_underlying.ipynb`.
- [x] `BlackScholesPricer` on `Equity` and market risk factors, float64 (kernels dtype-generic; fixed a float32
      `sqrt(2)` in `_norm_cdf`); no pricer-local market variables; `examples/3_options.py` greeks via AAD;
      example `study/notebooks/04_black_scholes.ipynb`.
- [ ] `DividendCurve` amounts as risk factors.
- [ ] Examples 2-4 default to real tickers (ISP/SX5E): use `TICKER1` with `'Generated'`.
- [x] `Option.strike` is a float64 `tf.Variable` (was float32: strikes like 389.24 lost ~1e-5, parity error ~1e-5).

### 3. Interest rates (analytic pricers)

- [ ] dtype policy in pricers: float64 (same as equity).
- [ ] Swap/OIS/FRA/Deposit pricers: add tests that gradients reach `RateCurve.variables`.
- [ ] `CurveBootstrap` writes into `market_env._market` directly.
- [ ] `instruments/capfloor.py` is not exported and has no pricer.
- [ ] `examples/swap_pricing.py` never sets `tq.Settings.evaluation_date` (defaults to `date.today()`), so it fails
      once past fixings are needed.
- [ ] `PricerAssignment` uses `type(...) ==` and covers only Deposit/Ois/Fra/Swap → extend once equity and rate
      pricers are settled.

### 4. Stochastic processes & Monte Carlo (`models/`, MC pricers, `analytics/`)

- [ ] dtype policy: MC may stay float32 via `tf.cast`, analytic parts float64.
- [ ] `StochasticProcess.evolve(t0, x0, dt, dw)` vs subclasses `evolve(t_grid, dw)`; GBM `drift(dt)` differs too.
- [ ] `HullWhiteProcess.x0` calls `.numpy()` on the numpy value returned by `OrnsteinUhlenbeckProcess.x0`: `AttributeError`.
- [ ] `VanillaMCPricer` + `LocalVolatilityModel`: check the gradient reaches `VolatilitySurface.volatility_matrix`.
- [ ] `AutocallableMCPricer` returns floats: no AAD greeks.
- [ ] `SwapExposureGenerator` passes a plain `dict` to `SwapPricer.price` (expects a `MarketEnvironment`): likely broken.
- [ ] Unused, not exported: `pricers/localvolmc.py`, `models/localvol.py` (superseded by `models/localvolatility.py`).
- [ ] `examples/local_vol.py` draws `tf.random.normal` before `tf.random.set_seed` (path stats not reproducible).

### 5. Market data I/O

- [x] New market data layout, replacing the MDM CSV layout read by `MarketEnvironment.from_data_path`
      (`curve.csv`, `spot.csv`, `vol.csv`, `dividends.csv`). Long-format CSV, ISO dates, decimal units, snake_case:
      ```
      marketdata/                 (repo root; synthetic data, committed)
        <yyyymmdd>/<calibration_set>/   (several snapshots per day, e.g. morning/evening)
          ir_curves.csv     ccy, curve, pillar, value, quote (df|zero), daycount, interp
          eq_spot.csv       ticker, ccy, spot, repo, div_yield   (empty cell = absent)
          eq_vol.csv        ticker, ccy, expiry, strike, vol
          eq_dividends.csv  ticker, ccy, ex_date, pay_date, amount
          correlations.csv  factor_1, factor_2, corr   (upper triangle, names like EQ:EUR:TICKER1)
          fx_spot.csv       base, quote, spot   (1 base = spot quote)
      ```
      Synthetic snapshots from `marketdata/generate_synthetic.py` (seed 42), written to `<yyyymmdd>/<HHMM>/` of the
      generation time (evaluation date = generation date, all dates relative to it): 50 tickers
      (TICKER1-30 EUR, 31-40 GBP, 41-50 USD), curves ESTR/SONIA/SOFR, EURUSD/GBPUSD, correlations 0.30-0.80
      (one-factor model, PSD by construction).
      Conventions come from the data (no hardcoded ESTR/EUR6M, day count, interp); market keys derived from the
      columns. Reader `MarketEnvironment.from_folder` done (default overnight curve per ccy: ESTR, SOFR, SONIA);
      writer `to_folder` not yet. Recheck pandas 3 here.
- [ ] The `W:/` MDM datastore no longer exists: `from_data_path` and the `'Real'` branch of the examples
      (`1_market_handles`, `2_forwards`, `3_options`, `4_montecarlo`, `local_vol_mdm`) to be replaced by `marketdata/`.
- [x] FX spots and `CorrelationMatrix` in `MarketEnvironment` (keys, validation, risk factors, accessors).
- [ ] Unit tests for FX (`get_fx_spot` inverse/cross and gradients) and `CorrelationMatrix` (validation,
      `submatrix`, `cholesky`): postponed, only checked by a script so far.
- [ ] FX forward from the curves (`S * DF_base / DF_quote`); FX vol surfaces.
- [ ] Synthetic market data generation modules in the library (today only `examples/market_data.py`), usable by
      tests, examples and `study/` notebooks.
