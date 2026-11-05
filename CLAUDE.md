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
| `numericalhandles/` | `LinearInterp`, `newton`, `newton_1d` |
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
  — refactor phase 2.
- **MarketEnvironment** is the only access point to market data. Flat dict with keys
  `RISKFACTOR:CCY:TICKER:TYPE`, e.g. `IR:EUR:ESTR:SPOT` → `RateCurve`, `EQ:EUR:SX5E:VOL` → `VolatilitySurface`,
  `EQ:EUR:X:SPOT|REPO|DIVYIELD` → float, `EQ:EUR:X:DIV` → `DividendCurve`.
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

- Market inputs and calibrated params are `tf.Variable`/`tf.Tensor`; scalar equity data (spot, repo,
  div yield) may be `float` in the market dict.
- Google-style docstrings (Args/Returns/Raises); type hints everywhere; relative imports inside the package.
- No access to `MarketEnvironment._market` outside `MarketEnvironment`.
- Verify numbers against an independent check (closed form, MC, or the examples) when touching pricing code.

## Known issues (refactor backlog)

- `pandas` capped `<3` in `setup.py` until the CSV loading in `MarketEnvironment.from_data_path` is checked on pandas 3.
- `examples/swap_pricing.py` never sets `tq.Settings.evaluation_date` (defaults to `date.today()`), so it fails once
  past fixings are needed. Examples 2-4 default to real tickers (ISP/SX5E): use `TICKER1` with `'Generated'`.
- dtypes are mixed: market objects are float64, most pricers/models still `tf.float32` (explicit casts needed).
  Target: float64 for market data and analytic pricers; MC may stay float32 via `tf.cast`.
- `RateCurve.inst_fwd` uses `math.log`, which converts to float and breaks the gradient.
- `SwapExposureGenerator` passes a plain `dict` to `SwapPricer.price` (expects a `MarketEnvironment`): likely broken.
- `examples/local_vol.py` draws `tf.random.normal` before `tf.random.set_seed`, so its path stats are not reproducible.
- `DividendCurve` amounts are not risk factors yet (planned).
- `StochasticProcess.evolve(t0, x0, dt, dw)` vs subclasses `evolve(t_grid, dw)`; GBM `drift(dt)` signature differs too.
- `PricerAssignment` uses `type(...) ==` and covers only Deposit/Ois/Fra/Swap.
- Not exported / unused: `instruments/capfloor.py`, `pricers/localvolmc.py`, `models/localvol.py`
  (superseded by `models/localvolatility.py`).
- `CurveBootstrap` writes into `market_env._market` directly.
- Tests cover only `MarketEnvironment` (`tests/test_marketenvironment.py`).
