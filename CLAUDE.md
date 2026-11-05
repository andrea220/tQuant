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
  market in `examples/market_data.py`; `'Real'` needs the MDM datastore on `W:/` (not always available).
- Set `TF_ENABLE_ONEDNN_OPTS=0` for reproducible numbers.

## Module map

| Package | Main classes |
|---|---|
| `timehandles/` | `Settings` (global `evaluation_date`), `DayCounter` + `DayCounterConvention`, `Calendar`, `TARGET`, `ScheduleGenerator`, `DateGrid`, enums `TimeUnit`, `BusinessDayConvention`, `CompoundingType`, `Frequency` |
| `numericalhandles/` | `LinearInterp`, `newton`, `newton_1d`, `DTYPE` + `as_variable` (dtype of market data, risk-factor variables) |
| `markethandles/` | `MarketEnvironment`, `RiskFactor`, `MarketDataType`, `MarketMapValidator`, `RateCurve` / `FlatCurve` / `DefaultCurve`, `VolatilitySurface` / `BlackConstantVolatility`, `DividendCurve`, `InterestRate`, `CurveBootstrap`; `utils.py`: `Currency`, `OptionType`, `ExerciseType`, `PayoffType`, `SwapType`, `Position` |
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
  Pricers must not create their own variables from market data nor detach with `.numpy()`/`float()`.
  Migration in progress: `BlackScholesPricer` still creates local variables (`pricer._sigma`, `_r`, ...)
  — equity step of the refactor roadmap.
- **MarketEnvironment** is the only access point to market data. Flat dict with keys
  `RISKFACTOR:CCY:TICKER:TYPE`, e.g. `IR:EUR:ESTR:SPOT` → `RateCurve`, `EQ:EUR:SX5E:VOL` → `VolatilitySurface`,
  `EQ:EUR:X:SPOT|REPO|DIVYIELD` → float64 `tf.Variable` (a float is accepted as input and wrapped), `EQ:EUR:X:DIV` → `DividendCurve`.
  Typed accessors: `get_ir_curve`, `get_eq_spot`, `get_eq_vol_surface`, `get_eq_repo`, `get_eq_div_yield`,
  `get_eq_dividends`, `get_currency`. Build from CSV with `MarketEnvironment.from_data_path(...)`.
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
| `timehandles/` | — | — |
| `numericalhandles/dtypes.py` | `tests/test_dtypes.py` | `DTYPE`, `as_variable` (reuse, casts, no float32 round trip) |
| `numericalhandles/` (interpolation, newton) | — | — |
| Equity pricers/models | — | — |
| Rate pricers/flows/bootstrap | — | — |

## Refactor roadmap

Order: **1. shared handles → 2. equity → 3. interest rates**. Finish a step before starting the next:
unit tests for the area added first (see Testing), then all tests green.

### 1. Shared handles (`markethandles/`, `timehandles/`, `numericalhandles/`, base ABCs)

- [x] Risk factors in market objects: float64 `variables`, `MarketEnvironment.risk_factors()`.
- [x] dtype policy for market objects: float64 variables and float64 tensor outputs (`discount(0)` no longer returns `int`).
- [ ] dtype policy in pricers/models: float64 for analytic pricers, MC may stay float32 via `tf.cast`; today most
      are still `tf.float32` with ad-hoc casts → fixed pricer by pricer in the equity and interest-rate steps.
- [x] `RateCurve.inst_fwd` returns a differentiable float64 tensor (was `math.log` → float, no gradient).
- [ ] `StochasticProcess.evolve(t0, x0, dt, dw)` vs subclasses `evolve(t_grid, dw)`; GBM `drift(dt)` differs too.
- [ ] `PricerAssignment` uses `type(...) ==` and covers only Deposit/Ois/Fra/Swap.
- [ ] `pandas` capped `<3` in `setup.py` until `MarketEnvironment.from_data_path` is checked on pandas 3.

### 2. Equity

- [ ] `BlackScholesPricer`: still creates local variables (`pricer._sigma`, `_r`, ...) and detaches with `.numpy()`;
      move to market risk factors (no aliases), update `examples/3_options.py` and `study/notebooks/02_dividends.ipynb`.
- [ ] `VanillaMCPricer` + `LocalVolatilityModel`: check the gradient reaches `VolatilitySurface.volatility_matrix`.
- [ ] `AutocallableMCPricer` returns floats: no AAD greeks.
- [ ] `DividendCurve` amounts as risk factors.
- [ ] Unused, not exported: `pricers/localvolmc.py`, `models/localvol.py` (superseded by `models/localvolatility.py`).
- [ ] `examples/local_vol.py` draws `tf.random.normal` before `tf.random.set_seed` (path stats not reproducible).
      Examples 2-4 default to real tickers (ISP/SX5E): use `TICKER1` with `'Generated'`.

### 3. Interest rates

- [ ] `CurveBootstrap` writes into `market_env._market` directly.
- [ ] `SwapExposureGenerator` passes a plain `dict` to `SwapPricer.price` (expects a `MarketEnvironment`): likely broken.
- [ ] Swap/OIS/FRA/Deposit pricers: add tests that gradients reach `RateCurve.variables`.
- [ ] `instruments/capfloor.py` is not exported and has no pricer.
- [ ] `HullWhiteProcess.x0` calls `.numpy()` on the numpy value returned by `OrnsteinUhlenbeckProcess.x0`: `AttributeError`.
- [ ] `examples/swap_pricing.py` never sets `tq.Settings.evaluation_date` (defaults to `date.today()`), so it fails
      once past fixings are needed.
