# Development log

Completed developments of the refactor (moved from `CLAUDE.md`). Open and future work: `ROADMAP.md`.
Order of the refactor: 1. market & time handles → 2. equity (analytic) → 3. interest rates (analytic) →
4. stochastic processes & Monte Carlo → 5. market data I/O.

## 1. Market & time handles (`markethandles/`, `timehandles/`, `numericalhandles/`)

- [x] Risk factors in market objects: float64 `variables`, `MarketEnvironment.risk_factors()`.
- [x] dtype policy for market objects: float64 variables and float64 tensor outputs (`discount(0)` no longer returns `int`).
- [x] `RateCurve.inst_fwd` returns a differentiable float64 tensor (was `math.log` → float, no gradient).
- [x] `pandas` (CSV loading and display tables): cap `<3` removed in `setup.py` after the full test suite and examples
      1-4, `swap_pricing`, `autocallable`, `local_vol_mdm` passed with pandas 3.1.1 (`FutureWarning` as error);
      `requirements.txt` still pins the installed 2.2.2.

## 2. Equity (analytic pricers)

- [x] `Equity` underlying view (`MarketEnvironment.get_equity`) with forward, PV dividends, implied yield, risk
      factors; example `study/notebooks/03_equity_underlying.ipynb`.
- [x] `BlackScholesPricer` on `Equity` and market risk factors, float64 (kernels dtype-generic; fixed a float32
      `sqrt(2)` in `_norm_cdf`); no pricer-local market variables; `examples/3_options.py` greeks via AAD;
      example `study/notebooks/04_black_scholes.ipynb`.
- [x] `DividendCurve` amounts as risk factors (`EQ:<ccy>:<ticker>:DIV` in `risk_factors()`, dense gradient);
      notebook 03 shows dF/dD_i.
- [x] Examples 2-4 default to `TICKER1` (marketdata/ snapshot).
- [x] `Option.strike` is a float64 `tf.Variable` (was float32: strikes like 389.24 lost ~1e-5, parity error ~1e-5).

## 3. Interest rates (analytic pricers)

- [x] dtype policy in pricers: float64 tensors everywhere (paid/empty legs return a float64 zero, not `0`).
- [x] Swap/OIS/FRA/Deposit: tests that gradients reach `RateCurve.variables` (per pillar vs bumps).
- [x] Discounting by *date* (`curve.discount(payment_date)`, curve day count) in fixed/floating/OIS coupons, FRA and
      deposit: coupons were discounted at the coupon day-count year fraction (ACT/360 on an ACT/365 curve: −0.11%
      on a 5Y fixed leg). OIS forecast rate `(P(s)/P(e) − 1)/tau` on the coupon's own day count (was the curve's).
- [x] FRA: standard `N tau (F − K) P(0,T1)/(1 + F tau)` with F on the product dates (was `P(0,T2)` and index dates).
- [x] `CurveBootstrap` registers curves with `MarketEnvironment.set_ir_curve` (no `_market` writes).
- [x] `examples/swap_pricing.py` sets `tq.Settings.evaluation_date`, uses `curve.variables`.
- [x] `PricerAssignment` with `isinstance`, adds `VanillaOption` → `BlackScholesPricer`.

- [x] Caps / floors / collars (`instruments/capfloor.py`, was a deprecated stub): `CapFloor` (strip of caplets on a
      `FloatingRateLeg`, cap = `OptionType.Call`, floor = `Put`), `Collar` (long cap, short floor), `CapFloorGenerator`
      (spot-start, first period excluded). `CapFloorPricer`: caplet by caplet, forward on the index curve (as
      `SwapPricer`), discounting on the default curve, Bachelier on normal caplet vols / Black on lognormal; fixed
      caplets at intrinsic; registered in `PricerAssignment`. Rate vols: `IR:<ccy>:<index>:VOL` caplet vol surfaces
      (`VolatilitySurface.vol_type`), `ir_vol.csv`, `get_ir_vol_surface`. Synthetic data: EUR 6M Euribor curve (ESTR +
      Nelson-Siegel spread, 30-40bp) and EUR 6M normal caplet vols from a normal SABR (beta = 0) smile (70-110bp); added
      to snapshot `20261105/2240` (other files unchanged). 5Y cap on 1M: K=2% 50,748, K=3% 21,455.

## 4. Stochastic processes & Monte Carlo (`models/`, MC pricers, `analytics/`)

- [x] dtype policy: **float64 everywhere** (decided; no float32 casts): GBM/ABM, `LocalVolatilityModel`,
      `DisplacedDiffusionModel`, `VanillaMCPricer`, `AutocallableMCPricer`, `GaussianShortRateModel`, examples.
- [x] 4a `VanillaMCPricer(model=<class>)` builds the model from the market at each price via
      `model.from_equity(equity, maturity, strike)` (`GeometricBrownianMotion`, `LocalVolatilityModel`): one
      `tape.gradient` gives pathwise delta, vega per node, rho, dividend sensitivities. Grid ends exactly at maturity
      (was the nearest point of a grid up to the last vol maturity), stateless Gaussian draws (CRN bumps),
      `product.standard_error`. Discrete dividends: escrowed `S − PV`, carry repo (as BS discrete).
- [x] `LocalVolatilityModel.evolve`: the loop is a module-level `tf.function` on tensors (was a `tf.function` on a
      bound method: retraced per model and, with a temporary model, `self` garbage-collected → `None._C`).
- [x] 4d `StochasticProcess` reduced to `size`, `initial_values`, `evolve(t_grid, dw, observation_times=None)` (was an
      Euler `evolve(t0, x0, dt, dw)` + drift/diffusion/expectation no pricer used, overridden with other signatures);
      DD and `GaussianShortRateModel` (`evolve` = factors, `simulate` = draws + discount) on it.
- [x] `HullWhiteProcess.x0` (was `.numpy()` on a numpy value: `AttributeError`).
- [x] 4c `GaussianShortRateModel` (`hull_white`, `g2pp`; any n factors): `r = sum x_i + phi(t)` fitted to the market
      curve; exact joint simulation of `(x_i, int x_i)` (no time-step error, stateless seed, float64); bank-account
      discount `D(0,t) = P(0,t) exp(-sum I - V/2)`; Brigo-Mercurio closed-form bonds. The curve enters only through
      `P(0, .)`: rho by AAD. Checked against the legacy `HullWhiteProcess` / `G2PlusPlusProcess` closed forms.
- [x] Removed the legacy `HullWhiteProcess`, `G2PlusPlusProcess`, `OrnsteinUhlenbeckProcess`, `GaussianPathGenerator`,
      `HullWhiteShortRateGenerator` (superseded by `GaussianShortRateModel`); their closed-form values are pinned in
      `test_rate_curve` / `test_rates_mc`. The pinning caught a float32 round trip of numeric parameters in
      `GaussianShortRateModel` (a = 0.03 → 0.0299999993): fixed.
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
- [x] `SwapExposureGenerator` rewritten (was broken: `SwapPricer(market_map)` + `dict` market): vectorized swap MtM
      per path/date from the model bonds, index curve via a deterministic multiplicative basis to the discount
      curve, past-fixed coupons from the path fixing; EE, ENE, PFE, discounted EE; today = `SwapPricer` exactly,
      E[D V(t)] = PV(0) of the remaining flows. 5Y swap, 23 dates, 20k paths: ~5 s.
- [x] Removed `pricers/localvolmc.py`, `models/localvol.py` (unused) and the legacy local vol path (`LocalVolatilityModel(C, ...)`,
      `from_implied_vol`, Dupire on call prices, `_simulate_log_euler`): only the forward/Gatheral model remains.
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
- [x] `examples/local_vol.py` / `local_vol_mdm.py`: one analysis (`analyse_local_vol` in `local_vol.py`, reused by
      `local_vol_mdm.py`): `from_equity` per maturity, MC forward vs `Equity.forward`, MC implied vols of OTM
      options vs market at every node with SE (removed `std(log S_T)/√T` "ATM vol" and `BS(σ_loc)` "reprice error").

## 5. Market data I/O

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
      columns. Reader `from_folder` / `from_frames`, writer `to_frames` / `to_folder` (default overnight curve per ccy:
      ESTR, SOFR, SONIA). Also `fx_vol.csv  base, quote, expiry, strike, vol`.
- [x] Examples (`1_market_handles`, `2_forwards`, `3_options`, `4_montecarlo`, `local_vol_mdm`) and `study/`
      notebooks 01-04 load `marketdata/` via `from_folder` (no date: latest snapshot; `env.evaluation_date`).
- [x] Removed `examples/market_data.py` and `MarketEnvironment.from_data_path` (old MDM layout).
- [x] FX spots and `CorrelationMatrix` in `MarketEnvironment` (keys, validation, risk factors, accessors).
- [x] Unit tests for FX and `CorrelationMatrix` (`tests/test_market_data.py`); `submatrix` now has a dense gradient
      (was `tf.gather` on the variable → `IndexedSlices`).
- [x] FX forward (`get_fx_forward`, covered interest parity) and FX vol surfaces (`FX:<quote>:<pair>:VOL`, `fx_vol.csv`).
- [x] Synthetic curves: Nelson-Siegel per currency (was 5bp random noise per pillar: inst forward jumps up to 80bp;
      now ≤ 18bp, from the linear interpolation of zero rates).
- [x] Synthetic generator in the library (`tensorquant/markethandles/synthetic.py`): `generate_market_data` (tables) /
      `generate_market` (`MarketEnvironment`); independent random stream per block (curves, equities, FX,
      correlations); SSVI on the library forwards (`Equity.forward`, `get_fx_forward`). `marketdata/generate_synthetic.py`
      is a thin wrapper; snapshot `20261105/2240` (the previous `20261105/2130` removed).
