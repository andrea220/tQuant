# TensorQuant (tQuant)

Python quant-finance library on **TensorFlow**. Core idea: prices are TF graphs, so greeks come
from `tf.GradientTape` (AAD) instead of separate analytic code.

- Package `tensorquant/` (import as `import tensorquant as tq`; everything is re-exported via `*`).
- `setup.py` only (no `pyproject.toml`): loose bounds there, exact verified pins in `requirements.txt`.
- `docs/` (Sphinx/ReadTheDocs) is **deprecated**: out of date, do not update or rebuild it.
- Study material in `study/` (own `CLAUDE.md`).
- Reference books and papers in `resources/` (local only, git-ignored): when in doubt on a model, formula, numerical
  method or market convention, check them first (e.g. Brigo-Mercurio for rates, Gatheral / Derman for vol
  surfaces, Savine for AAD and simulation, Oosterlee-Grzelak for Heston / SLV, with Python codes). They are a
  reference only: never copy-paste their code; implement in the library architecture (TF, float64, market objects,
  `Product` / `Pricer` / `StochasticProcess`, AAD to the risk factors).
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
| `markethandles/` | `MarketEnvironment`, `RiskFactor`, `MarketDataType`, `MarketMapValidator`, `RateCurve` / `FlatCurve` / `DefaultCurve`, `VolatilitySurface` / `BlackConstantVolatility`, `DividendCurve`, `Equity` (underlying view), `CorrelationMatrix`, `SwaptionVolCube` (expiry × tenor × strike offset), `InterestRate`, `CurveBootstrap`, `generate_market_data` / `generate_market` (`synthetic.py`; EUR 6M caplet vols and swaption cube ATM from one G2++, SABR smile); `utils.py`: `Currency`, `OptionType`, `ExerciseType`, `PayoffType`, `SwapType`, `Position` |
| `index/` | `Index` (ABC), `OvernightIndex`, `IborIndex`, `InflationIndex` |
| `flows/` | `CashFlow`, `Coupon` (ABC), `FixedCoupon`/`FixedRateLeg`, `FloatingCoupon`/`FloatingRateLeg` |
| `instruments/` | `Product` (ABC), `Deposit`, `Fra`, `Swap`, `Ois`, `Option`/`VanillaOption`, `AutocallableOption`, `CapFloor` (cap = `OptionType.Call`, floor = `Put`) / `Collar`, `Swaption` (European, physical), generators in `helpers.py` (incl. `CapFloorGenerator`, `SwaptionGenerator`) |
| `models/` | `StochasticProcess` (ABC: `size`, `initial_values`, `evolve(t_grid, dw, observation_times=None)`, `noise_size()` normals per step, default 1), `GeometricBrownianMotion`, `ArithmeticBrownianMotion`, `GaussianShortRateModel` (n-factor G1++/G2++ on the market curve, constant or piecewise constant vols (`sigma_times`): exact simulation of factors + integrals, closed-form bonds and caplets, swaptions by the Gaussian swap-rate approximation or exact (Jamshidian / Brigo-Mercurio), AAD to the pillars), `LocalVolatilityModel` (Gatheral on the market forward, `from_equity`), `DisplacedDiffusionModel` (shifted lognormal on the market forward, `from_equity`: slice fit of beta/sigma, forward pieces by skew averaging, differentiable calibration by the implicit function theorem, exact MC steps), `HestonModel` (on the market forward, `from_equity`: COS closed form, calibration to the whole surface with greeks through it (implicit function theorem), QE-M Monte Carlo, `noise_size() = 2`), `HestonSLVModel` (stochastic local vol: calibrated Heston (vol of vol x `mixing`) + leverage `sigma_LV / sqrt(E[v | X])` built on the pricing paths by a particle method (tent-kernel smoother); greeks with the Heston parameters held, through the local vol and the forward) |
| `pricers/` | `Pricer` (ABC), `BlackScholesPricer` (EU BS + American Bjerksund–Stensland; `dividend_model` discrete/continuous), `VanillaMCPricer` (model class with `from_equity`: GBM / local vol / displaced diffusion / Heston / Heston SLV; draws `model.noise_size()` normals per step), `AutocallableMCPricer` (investor's side, local vol default, digitals smoothed: `barrier_smoothing`), `SwapPricer`, `OisPricer`, `FraPricer`, `DepositPricer`, `CapFloorPricer` (Bachelier on normal caplet vols, Black on lognormal), `SwaptionPricer` (annuity × Bachelier / Black-76 on the cube), coupon/leg discounting, `PricerAssignment` (factory, `isinstance`: Deposit/Ois/Fra/Swap/VanillaOption/CapFloor/Collar/Swaption) |
| `analytics/` | `SwapExposureGenerator` (swap MtM/EE/ENE/PFE under `GaussianShortRateModel`) |
| `calibration/` | `strip_caplet_vols` / `cap_flat_vols` (caplet vols ↔ flat cap quotes), `calibrate_gaussian_short_rate` (Hull-White / G2++ to caplet and / or swaption vols: closed-form model caplets, batched approximate or exact swaptions, Hull-White sigma(t) (`sigma_times`), `fixed` params, least squares with AAD Jacobian, bounded params) |

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
  `IR:EUR:6M:VOL` → `VolatilitySurface` of caplet vols (`vol_type` `normal` | `lognormal`), `IR:EUR:6M:SWVOL` →
  `SwaptionVolCube`,
  `FX:USD:EURUSD:SPOT` → float64 `tf.Variable` (1 EUR = x USD), `FX:USD:EURUSD:VOL` → `VolatilitySurface`,
  `CORR:ALL:GLOBAL:SPOT` → `CorrelationMatrix`
  (factor names = market-map key prefix, e.g. `EQ:EUR:TICKER1`; one float64 `[n, n]` variable).
  Typed accessors: `get_ir_curve`, `get_eq_spot`, `get_eq_vol_surface`, `get_eq_repo`, `get_eq_div_yield`,
  `get_eq_dividends`, `get_currency`, `get_fx_spot(base, quote)` (direct, inverse or cross via a common
  currency; only stored pairs are risk factors), `get_ir_vol_surface(ccy, ticker)` / `set_ir_vol_surface`, `get_swaption_vol_cube` / `set_swaption_vol_cube`, `get_fx_forward(base, quote, T)` (covered interest parity on the
  default curves), `get_fx_vol_surface(base, quote)` (stored pairs), `get_correlation_matrix`, `get_correlation(names)`.
  I/O: `from_folder("marketdata", evaluation_date=None, calibration_set=None)` (latest snapshot when `None`) =
  CSV → `from_frames(frames, evaluation_date)`; inverse `to_frames()` / `to_folder(root, calibration_set)` (exact round
  trip). Synthetic data: `tq.generate_market(evaluation_date, seed)` (in memory) / `generate_market_data` (tables).
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
3. New model → subclass `StochasticProcess` (`evolve(t_grid, dw, observation_times=None)`, draws made by the caller);
   Gaussian rate models: extend `GaussianShortRateModel`; equity models: a `from_equity` factory for the MC pricers.
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
| FX, `CorrelationMatrix`, market data I/O, synthetic generator | `tests/test_market_data.py` | FX spot direct/inverse/cross + gradients, FX forward = covered interest parity + gradients on both curves, FX vol surface; correlation from_pairs/submatrix/cholesky/validation, pair gradient on both entries; `from_frames`/`to_frames` and `to_folder`/`from_folder` exact round trip (risk factors, dividend and vol dates, BS price), latest snapshot; generator deterministic/seeded, layout, correlation range, SSVI arbitrage-free, smooth curves |
| `timehandles/` | — | — |
| `numericalhandles/dtypes.py` | `tests/test_dtypes.py` | `DTYPE`, `as_variable` (reuse, casts, no float32 round trip) |
| `numericalhandles/` (interpolation, newton) | — | — |
| `VanillaMCPricer` (GBM, local vol) | `tests/test_vanilla_mc.py` | GBM vs BS within 3 SE (discrete/continuous, call/put), float64, exact maturity, stateless seed, n_steps-invariance, AAD greeks reach market factors, delta/vega vs BS, delta vs common-random-numbers bump, dV/dD_i = −delta·P(ex_i); LV: flat surface = BS within 3 SE, vega reaches nodes, node vega vs CRN bump; forward mode (`from_equity`): MC mean = `Equity.forward` at every step (discrete dividends), arbitrage-free SSVI on a narrow grid repriced at the 1Y/3Y nodes (OTM implied vols within 0.6/1.0 vol points), arbitrage warning, delta and node vega vs CRN bumps; `natural_cubic_spline` vs scipy and gradients vs FD |
| `DisplacedDiffusionModel` | `tests/test_displaced_diffusion.py` | closed form: beta = 1 Black, beta -> 0 Bachelier, parity, shifted strike <= 0, skew vs beta (normal limit -sigma/2); skew averaging round trip; calibration recovers the generating forward parameters (1e-6), vega per node and delta through the calibration vs bump-and-recalibrate (interior and beta on its bound), steep SSVI skew out of reach (beta at bound, ATM < 1 vol pt); MC: constant beta = closed form within 3 SE, time-dependent beta within 0.3 vol pts at the nodes, grids across the pieces, forward = `Equity.forward` (discrete dividends), `VanillaMCPricer` price, delta and node vega vs CRN bumps; autocallable on a flat smile (beta = 1) = GBM, greeks vs bumps |
| `AutocallableMCPricer` | `tests/test_autocallable_mc.py` | hard payoff = numpy replica on the same paths (memory on/off), consistent with the old pricer (coupons, put within 4 SE), closed-form limits (never called, called at the first date), smoothing → hard payoff, digitals invisible to AAD when hard, delta/vega vs CRN bumps (GBM, local vol), LV flat = GBM |
| `HestonModel` | `tests/test_heston.py` | characteristic function = numpy, martingale, cumulants vs finite differences; COS = Lewis quadrature (1e-12, 1W-10Y), Black-Scholes limit, parity, negative skew; calibration recovers the generating parameters (1e-6), vega per node and delta through the calibration vs bump-and-recalibrate (1e-6, exact fit and SSVI residuals), SSVI fit (ATM < 1.5 vol pts, wings < 1.5 from 1Y); QE-M = closed form within 3 SE (Feller violated), MC forward = `Equity.forward` (discrete dividends), E[v_t], `VanillaMCPricer` price and AAD delta / vega = closed-form greeks (CRN bumps are no reference for QE); autocallable on a flat smile = GBM |
| `HestonSLVModel` | `tests/test_heston_slv.py` | SSVI smile repriced at 1Y/3Y within 0.6 vol pts (mixing 1 and 0.5); Heston market: leverage = 1 (median within 3%), smile = Heston within 0.5 vol pts; mixing 0.05 autocallable = local vol, mixing 1 far from it; MC forward = `Equity.forward` (discrete dividends); vanilla AAD delta / vega = CRN bumps with Heston held (1%) and Black-Scholes sticky-strike greeks (3%); autocallable greeks finite and of the local vol order; `VanillaMCPricer` put = Black-Scholes; validation |
| Rates MC: `GaussianShortRateModel`, `SwapExposureGenerator` | `tests/test_rates_mc.py` | piecewise vol: equal pieces = constant model, variances vs quadrature, MC martingales / Var x / caplet, gradient per piece; bonds at 0 = curve, G1++/G2++ = legacy `HullWhiteProcess`/`G2PlusPlusProcess` closed forms (and `HullWhiteProcess.x0`), martingales E[D]=P(0,t) and E[D P(t,T)]=P(0,T), factor variances and G2++ correlation, seed, rho by AAD = dP(0,T); exposure: today = `SwapPricer`, E[D V(t)] = PV(0) of the remaining flows (HW, G2++), EE/ENE/PFE identities, zero after maturity, discounted EPE gradient vs CRN bump |
| Caps / floors, rate vols | `tests/test_capfloor.py` | Bachelier closed form per caplet (cap, floor), cap − floor = Σ N τ P (F − K), zero vol = intrinsic, lognormal vols → Black, collar = cap − floor, first period excluded, fixed caplet = intrinsic, vega per node and rho per pillar (both curves) vs bumps; generated EUR 6M curve (spread over ESTR) and normal SABR caplet vols (decreasing, convex caplet prices), `ir_vol.csv` round trip, `vol_type` validation |
| Calibration | `tests/test_calibration.py` | stripping: stripped caplet vols reprice every cap quote (1e-10), within 3bp of a smooth SABR surface, flat quotes → flat caplets, invalid grid / vol type, `set_ir_vol_surface` validation; model caplets (HW, G2++) = MC within 3.5 SE, cap − floor parity, curve gradients; calibration recovers HW params from HW caplet vols, G2++ fits its own caplets (< 0.05bp), ATM fit on the synthetic (G2++) market (HW rms < 10bp, G2++ < 0.3bp); HW sigma(t) on the caplet fixings reprices every ATM caplet (< 0.001bp), recovers piecewise vols |
| Swaptions | `tests/test_swaption.py` | `SwaptionVolCube` nodes / trilinear / flat extrapolation / gradient / shape; `SwaptionPricer` = annuity × Bachelier closed form, payer − receiver = A (F − K), forward rate zeroes the swap, lognormal → Black, vega per node and rho per pillar vs bumps; cube round trip, arbitrage-free synthetic cube; G2++/HW Gaussian swap-rate vols vs MC (< 0.5bp + 2 SE ATM); exact swaptions (HW, piecewise HW, G2++) vs MC within 3 SE at three strikes, parity, implied vol, AAD vs FD; G2++ recovers its own parameters from caplet + swaption vols; joint calibration on the synthetic market (< 0.3bp, generating G2++ parameters within 5%) |
| Rate pricers / flows | `tests/test_rate_pricers.py` | Deposit/FRA/Swap/OIS prices pinned, float64 tensors, gradients per pillar of both curves vs bumps, deposit closed form (ACT/365 and ACT/360 curves), FRA standard formula, fixed leg = sum amount·P(pay), single-curve floating legs telescope to N(P(s)−P(e)), NPV = 0 at the fair rate |
| `CurveBootstrap`, `PricerAssignment` | `tests/test_bootstrap.py` | ESTR (deposit + OIS) and 6M (FRA + swaps) stripped, instruments reprice to 0, curves registered via `set_ir_curve`, float64 variables; factory incl. `VanillaOption` |

## Roadmap and development log

- Completed developments: `DEVELOPMENT_LOG.md`. Open and future work: `ROADMAP.md`. Read them for history and
  context before working on an area.
- When a roadmap item is done, move it (with what was done and the numbers that verify it) from `ROADMAP.md` to
  `DEVELOPMENT_LOG.md`; add new findings to `ROADMAP.md`.
