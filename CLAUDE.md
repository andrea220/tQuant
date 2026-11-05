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
- Examples: `python examples/<script>.py`. Examples 1-4 and `local_vol_mdm` load the latest `marketdata/` snapshot
  (`SNAPSHOT_DATE` / `CALIBRATION_SET` at the top select another one). The MDM datastore on `W:/` **no longer
  exists** (do not look for it).
- Set `TF_ENABLE_ONEDNN_OPTS=0` for reproducible numbers.

## Module map

| Package | Main classes |
|---|---|
| `timehandles/` | `Settings` (global `evaluation_date`), `DayCounter` + `DayCounterConvention`, `Calendar`, `TARGET`, `ScheduleGenerator`, `DateGrid`, enums `TimeUnit`, `BusinessDayConvention`, `CompoundingType`, `Frequency` |
| `numericalhandles/` | `LinearInterp`, `natural_cubic_spline` (TF, value/d1/d2, linear wings, differentiable in knots and values), `newton`, `newton_1d`, `DTYPE` + `as_variable` (dtype of market data, risk-factor variables) |
| `markethandles/` | `MarketEnvironment`, `RiskFactor`, `MarketDataType`, `MarketMapValidator`, `RateCurve` / `FlatCurve` / `DefaultCurve`, `VolatilitySurface` / `BlackConstantVolatility`, `DividendCurve`, `Equity` (underlying view), `CorrelationMatrix`, `InterestRate`, `CurveBootstrap`; `utils.py`: `Currency`, `OptionType`, `ExerciseType`, `PayoffType`, `SwapType`, `Position` |
| `index/` | `Index` (ABC), `OvernightIndex`, `IborIndex`, `InflationIndex` |
| `flows/` | `CashFlow`, `Coupon` (ABC), `FixedCoupon`/`FixedRateLeg`, `FloatingCoupon`/`FloatingRateLeg` |
| `instruments/` | `Product` (ABC), `Deposit`, `Fra`, `Swap`, `Ois`, `Option`/`VanillaOption`, `AutocallableOption`, `Cap`/`Floor`/`Collar`, generators in `helpers.py` |
| `models/` | `StochasticProcess` (ABC), `GeometricBrownianMotion`, `ArithmeticBrownianMotion`, `OrnsteinUhlenbeckProcess`, `HullWhiteProcess`, `G2PlusPlusProcess`, `LocalVolatilityModel` (Dupire), `DisplacedDiffusionModel` |
| `pricers/` | `Pricer` (ABC), `BlackScholesPricer` (EU BS + American Bjerksund–Stensland; `dividend_model` discrete/continuous), `VanillaMCPricer` (model class with `from_equity`: GBM / local vol built from the market), `AutocallableMCPricer` (investor's side, local vol default, digitals smoothed: `barrier_smoothing`), `SwapPricer`, `OisPricer`, `FraPricer`, `DepositPricer`, coupon/leg discounting, `PricerAssignment` (factory, `isinstance`: Deposit/Ois/Fra/Swap/VanillaOption) |
| `analytics/` | `GaussianPathGenerator`, `HullWhiteShortRateGenerator`, `SwapExposureGenerator` |

## Key patterns

- **Product → Pricer.** Instruments subclass `Product` (`ccy`, `start_date`, `end_date`, `price`).
  Pricers subclass `Pricer` and implement `calculate_price(product, market_env)`.
  `Pricer.price(product, market_env, autodiff=True)` runs it under a `GradientTape`, stores the price on
  `product.price` and the tape on `pricer.tape`; greeks via `pricer.tape.gradient(product.price, <tf.Variable>)`.
- **Risk factors live in market objects** as float64 `tf.Variable`s with stable identity (update with `assign`):
  `RateCurve.variables` (one per pillar; `FlatCurve` has one), `VolatilitySurface.volatility_matrix`
  (`[n_maturities, n_strikes]`), `DividendCurve.variables` (amounts `[n]`, ex-dates fixed), and EQ `SPOT|REPO|DIVYIELD` (wrapped by `MarketEnvironment`, one shared
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
  when `None`).
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
| `RateCurve` / `FlatCurve` / `DividendCurve` | `tests/test_rate_curve.py` | discount/zero/forward/inst_fwd vs numpy reference, `from_zcb` round trip, PV dividends, amounts as float64 variable (PV gradient = discount factor, `assign`), float64 tensor outputs, inst_fwd gradient vs finite differences, Hull-White values that use inst_fwd (pinned) |
| `Equity` view | `tests/test_equity.py` | references to market objects, dividend model selection, PV/forward vs closed form, forward vs `BlackScholesPricer`, implied yield, gradients (spot, repo, pillars vs FD, dividend amounts vs closed form), summary |
| `BlackScholesPricer` | `tests/test_black_scholes.py` | EU/AM prices and greeks pinned from the float32 version, EU vs closed form, put–call parity, forward = `Equity.forward`, delta/rho per pillar/vega per node vs bumps, dividend models (discrete, continuous sticky yield, DIVYIELD-only), dPrice/dD_i vs bumps (both models) and = -delta·P(0,ex_i) (discrete), float64 |
| FX / `CorrelationMatrix` | — | postponed (see roadmap step 5) |
| `timehandles/` | — | — |
| `numericalhandles/dtypes.py` | `tests/test_dtypes.py` | `DTYPE`, `as_variable` (reuse, casts, no float32 round trip) |
| `numericalhandles/` (interpolation, newton) | — | — |
| `VanillaMCPricer` (GBM, local vol) | `tests/test_vanilla_mc.py` | GBM vs BS within 3 SE (discrete/continuous, call/put), float64, exact maturity, stateless seed, n_steps-invariance, AAD greeks reach market factors, delta/vega vs BS, delta vs common-random-numbers bump, dV/dD_i = −delta·P(ex_i); LV: flat surface = BS within 3 SE, vega reaches nodes, node vega vs CRN bump; forward mode (`from_equity`): MC mean = `Equity.forward` at every step (discrete dividends), arbitrage-free SSVI on a narrow grid repriced at the 1Y/3Y nodes (OTM implied vols within 0.6/1.0 vol points), arbitrage warning, delta and node vega vs CRN bumps; `natural_cubic_spline` vs scipy and gradients vs FD |
| `AutocallableMCPricer` | `tests/test_autocallable_mc.py` | hard payoff = numpy replica on the same paths (memory on/off), consistent with the old pricer (coupons, put within 4 SE), closed-form limits (never called, called at the first date), smoothing → hard payoff, digitals invisible to AAD when hard, delta/vega vs CRN bumps (GBM, local vol), LV flat = GBM |
| Rate pricers / flows | `tests/test_rate_pricers.py` | Deposit/FRA/Swap/OIS prices pinned, float64 tensors, gradients per pillar of both curves vs bumps, deposit closed form (ACT/365 and ACT/360 curves), FRA standard formula, fixed leg = sum amount·P(pay), single-curve floating legs telescope to N(P(s)−P(e)), NPV = 0 at the fair rate |
| `CurveBootstrap`, `PricerAssignment` | `tests/test_bootstrap.py` | ESTR (deposit + OIS) and 6M (FRA + swaps) stripped, instruments reprice to 0, curves registered via `set_ir_curve`, float64 variables; factory incl. `VanillaOption` |

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
- [x] `DividendCurve` amounts as risk factors (`EQ:<ccy>:<ticker>:DIV` in `risk_factors()`, dense gradient);
      notebook 03 shows dF/dD_i.
- [x] Examples 2-4 default to `TICKER1` (marketdata/ snapshot).
- [x] `Option.strike` is a float64 `tf.Variable` (was float32: strikes like 389.24 lost ~1e-5, parity error ~1e-5).

### 3. Interest rates (analytic pricers)

- [x] dtype policy in pricers: float64 tensors everywhere (paid/empty legs return a float64 zero, not `0`).
- [x] Swap/OIS/FRA/Deposit: tests that gradients reach `RateCurve.variables` (per pillar vs bumps).
- [x] Discounting by *date* (`curve.discount(payment_date)`, curve day count) in fixed/floating/OIS coupons, FRA and
      deposit: coupons were discounted at the coupon day-count year fraction (ACT/360 on an ACT/365 curve: −0.11%
      on a 5Y fixed leg). OIS forecast rate `(P(s)/P(e) − 1)/tau` on the coupon's own day count (was the curve's).
- [x] FRA: standard `N tau (F − K) P(0,T1)/(1 + F tau)` with F on the product dates (was `P(0,T2)` and index dates).
- [x] `CurveBootstrap` registers curves with `MarketEnvironment.set_ir_curve` (no `_market` writes).
- [x] `examples/swap_pricing.py` sets `tq.Settings.evaluation_date`, uses `curve.variables`.
- [x] `PricerAssignment` with `isinstance`, adds `VanillaOption` → `BlackScholesPricer`.
- [ ] `instruments/capfloor.py` is not exported and has no pricer (Black/Bachelier cap/floor: later).

### 4. Stochastic processes & Monte Carlo (`models/`, MC pricers, `analytics/`)

- [ ] dtype policy: **float64 everywhere** (decided; no float32 casts). Done for GBM/ABM, `LocalVolatilityModel`,
      `DisplacedDiffusionModel`, `VanillaMCPricer` and the local vol / DD examples; rate models (4c) still to check.
- [x] 4a `VanillaMCPricer(model=<class>)` builds the model from the market at each price via
      `model.from_equity(equity, maturity, strike)` (`GeometricBrownianMotion`, `LocalVolatilityModel`): one
      `tape.gradient` gives pathwise delta, vega per node, rho, dividend sensitivities. Grid ends exactly at maturity
      (was the nearest point of a grid up to the last vol maturity), stateless Gaussian draws (CRN bumps),
      `product.standard_error`. Discrete dividends: escrowed `S − PV`, carry repo (as BS discrete).
- [ ] `DisplacedDiffusionModel`: float64 now, but calibrated in numpy from `float()` market data (no AAD to market);
      add `from_equity` and a TF calibration.
- [x] `LocalVolatilityModel.evolve`: the loop is a module-level `tf.function` on tensors (was a `tf.function` on a
      bound method: retraced per model and, with a temporary model, `self` garbage-collected → `None._C`).
- [ ] `StochasticProcess.evolve(t0, x0, dt, dw)` vs subclasses `evolve(t_grid, dw)`; GBM `drift(dt)` differs too.
- [ ] `HullWhiteProcess.x0` calls `.numpy()` on the numpy value returned by `OrnsteinUhlenbeckProcess.x0`: `AttributeError`.
- [x] `VanillaMCPricer` + `LocalVolatilityModel`: the gradient reaches `VolatilitySurface.volatility_matrix` (tested).
- [x] 4b `AutocallableMCPricer`: model class built from the market (`from_equity`, local vol default), payoff in TF
      (was numpy + `tf.constant`: no greeks), investor's side `coupons − put` (the put was added), stateless seed,
      grid = future fixings ∪ uniform step (vol-surface day count; was ACT/360 vs annual vol). Digitals (coupon,
      autocall, put barrier) smoothed with a sigmoid of width `barrier_smoothing * K` (default 1%): pathwise AAD
      greeks; 0 = hard payoff. TICKER1 5Y phoenix: bias of eps = 1% +0.04% of notional (< 1 SE).
- [x] Local vol AAD speed: the simulation loop gathered from the captured local vol grid; the sparse gradients
      were accumulated by concatenation in the while loop (backward quadratic in the steps: 16 s at 260 steps).
      Rows per step are now interpolated outside the loop and read from a `TensorArray`: 2 s, same numbers (1e-15).
- [x] `evolve(t_grid, dw, observation_times=None)` (GBM, ABM, local vol): returns only the observed columns; in
      local vol forward mode `Equity.forward` (and the martingale correction) only there (was every step: 260 calls).
      Pricers pass the maturity / the fixings. Autocallable example: price 1.0 s, all greeks 2.2 s (was 1.6 + 20 s
      before both fixes), script 16 s incl. ~8 s TF import; same numbers.
- [ ] Same `gather`-in-loop pattern in the legacy `_simulate_log_euler` (`LocalVolatilityModel(C, ...)` /
      `from_implied_vol`).
- [ ] `SwapExposureGenerator` passes a plain `dict` to `SwapPricer.price` (expects a `MarketEnvironment`): likely broken.
- [ ] Unused, not exported: `pricers/localvolmc.py`, `models/localvol.py` (superseded by `models/localvolatility.py`).
- [x] `examples/local_vol.py` reproducible (stateless draws, seed 42).
- [x] `LocalVolatilityModel.from_equity` (forward mode): simulates `X = S/F(t)` on the market forward
      (`Equity.forward`; discrete dividends as proportional drops) with Gatheral's local vol in total implied variance
      and forward moneyness `y = ln(K/F(T))` (`gatheral_local_vol`): per maturity a TF natural cubic spline of
      `w = iv²T` in y through the strike nodes, Lee-linear wings (|dw/dy| ≤ 1.9), total variance linear in T;
      derivatives from the spline (the old call-price finite differences collapsed at short maturities and far
      wings: local vol → 0 at 1M above 120%). Differentiable w.r.t. vol nodes, spot, repo, dividends, curve
      (knots move with the forward). Martingale correction (MC forward = F(t) exactly). Static-arbitrage
      diagnostics on `model.arbitrage` + warning where the surface has data. Defaults `n_y=300`, `time_step=1/96`
      (converged: 1/365 gives the same numbers); short maturities need ≥ 100 Euler steps.
      History: the market grid (60%-140%) first cut the skew (5Y ATM −4.7, 60% −15.6 vol points); the
      "+4 vol points" of the old `local_vol_mdm` was std(log S_T)/√T, not an implied vol.
      Results (OTM implied vols at the nodes, MC − market): `local_vol` static market (arbitrage-free) within the MC
      noise (5 seeds at 3Y: mean ≈ 0, ±0.5 ATM / ±1 at 170% with 100k paths); `local_vol_vanilla` LV 7.2933 vs
      BS 7.2930; TICKER1 on the SSVI snapshot: all nodes 1M-5Y within ~0.3 vol points (1M/3M: mean over 5 seeds
      within ±0.2), except the 1M 140% call (~3 sd OTM, not measurable with 50k paths).
- [x] Synthetic vol surfaces are SSVI (`marketdata/generate_synthetic.py`, snapshot `20261105/2130`): power-law
      `phi = eta/sqrt(theta(1+theta))`, `eta(1+|rho|) <= 2`, increasing ATM total variance, forward moneyness with
      the dividend forward (= `Equity.forward` to 1e-16): free of static arbitrage (min g +0.24 over the 50
      tickers). Same random draws as before: spots, repos, dividends, curves, FX, correlations unchanged. The
      previous surfaces (snapshot `20261105/1516`, removed) had butterfly arbitrage from 1Y (g −0.36 at 5Y).
- [ ] Local vol MC noise is large in the wings and at long maturities (OTM call at 170%, 3Y: ±1 vol point with
      100k paths): variance reduction (antithetics, control variate on the forward) not done.
- [x] `examples/local_vol.py` / `local_vol_mdm.py`: one analysis (`analyse_local_vol` in `local_vol.py`, reused by
      `local_vol_mdm.py`): `from_equity` per maturity, MC forward vs `Equity.forward`, MC implied vols of OTM
      options vs market at every node with SE (removed `std(log S_T)/√T` "ATM vol" and `BS(σ_loc)` "reprice error").

### 5. Market data I/O

- [x] New market data layout, replacing the old MDM CSV layout (`from_data_path`, removed)
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
- [x] Examples (`1_market_handles`, `2_forwards`, `3_options`, `4_montecarlo`, `local_vol_mdm`) and `study/`
      notebooks 01-04 load `marketdata/` via `from_folder` (no date: latest snapshot; `env.evaluation_date`).
- [x] Removed `examples/market_data.py` and `MarketEnvironment.from_data_path` (old MDM layout).
- [x] FX spots and `CorrelationMatrix` in `MarketEnvironment` (keys, validation, risk factors, accessors).
- [ ] Unit tests for FX (`get_fx_spot` inverse/cross and gradients) and `CorrelationMatrix` (validation,
      `submatrix`, `cholesky`): postponed, only checked by a script so far.
- [ ] FX forward from the curves (`S * DF_base / DF_quote`); FX vol surfaces.
- [ ] Synthetic curves: the 5bp random noise on pillar zero rates gives jagged instantaneous forwards (e.g. SONIA
      4.0% -> 4.8% between 2Y and 3Y) with linear interpolation on zero rates. To review.
- [ ] Synthetic market data generation modules in the library (today `marketdata/generate_synthetic.py`), usable by
      tests, examples and `study/` notebooks.
