# TensorQuant (tQuant)

Python quant-finance library on **TensorFlow**. Core idea: prices are TF graphs, so greeks come
from `tf.GradientTape` (AAD) instead of separate analytic code.

- Package `tensorquant/` (import as `import tensorquant as tq`; everything is re-exported via `*`).
- `setup.py` only (no `pyproject.toml`): loose bounds there, exact verified pins in `requirements.txt`.
- `docs/` (Sphinx/ReadTheDocs) is **deprecated**: out of date, do not update or rebuild it.
- Study material in `study/` (own `CLAUDE.md`).
- Code, docstrings and comments in English.

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
  Pricers create market inputs as `tf.Variable`s inside `calculate_price` and expose them
  (e.g. `pricer._sigma`, `product.spot`) so they can be differentiated.
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
- dtypes are mixed: mostly `tf.float32`, `float64` in `DividendCurve` and a few places (explicit casts needed).
- `StochasticProcess.evolve(t0, x0, dt, dw)` vs subclasses `evolve(t_grid, dw)`; GBM `drift(dt)` signature differs too.
- `PricerAssignment` uses `type(...) ==` and covers only Deposit/Ois/Fra/Swap.
- Not exported / unused: `instruments/capfloor.py`, `pricers/localvolmc.py`, `models/localvol.py`
  (superseded by `models/localvolatility.py`).
- `CurveBootstrap` writes into `market_env._market` directly.
- Tests cover only `MarketEnvironment` (`tests/test_marketenvironment.py`).
