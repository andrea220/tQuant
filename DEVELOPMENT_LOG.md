# Development log

Completed developments of the refactor (moved from `CLAUDE.md`). Open and future work: `ROADMAP.md`.
Order of the refactor: 1. market & time handles → 2. equity (analytic) → 3. interest rates (analytic) →
4. stochastic processes & Monte Carlo → 5. market data I/O.

## 1. Market & time handles (`markethandles/`, `timehandles/`, `numericalhandles/`)

- [x] Risk factors in market objects: float64 `variables`, `MarketEnvironment.risk_factors()`.
- [x] dtype policy for market objects: float64 variables and float64 tensor outputs (`discount(0)` no longer returns `int`).
- [x] `RateCurve.inst_fwd` returns a differentiable float64 tensor (was `math.log` → float, no gradient).
- [x] Guided review, notebook 01 (market handles): `RateCurve.zero_rate` is the continuously compounded `-ln P / T` (the
      stored pillar rates; was the simple rate `(1 - P) / (T P)`, unused outside the tests); `inst_fwd` exact
      `r + T r'` through `LinearInterp.derivative` (right value at a pillar, flat outside; was a central difference
      with step 0.01, which straddled the pillars and evaluated negative times near 0): Hull-White `phi` at a pillar
      repinned (test curve: alpha(1) 0.0238735 → 0.0230485, alpha(2) of the rates MC curve 0.0268411 → 0.0255145),
      unchanged inside the segments; English error message in `forward_rate`. Notebook 01 rewritten (curves table,
      zero / DF / instantaneous and 6M forwards with the pillars, EUR 6M - ESTR basis, equities, smiles and ATM term
      structure; the old one plotted the 6M curve three times by taking the `IR:*:VOL` / `SWVOL` keys); study docs:
      Architecture (market data), Models and Pricing (curves, implied vol surface), Exercises (3 questions).
- [x] `pandas` (CSV loading and display tables): cap `<3` removed in `setup.py` after the full test suite and examples
      1-4, `swap_pricing`, `autocallable`, `local_vol_mdm` passed with pandas 3.1.1 (`FutureWarning` as error);
      `requirements.txt` still pins the installed 2.2.2.

## 2. Equity (analytic pricers)

- [x] `DigitalOption` (`instruments/digital.py`, European cash- / asset-or-nothing, `PayoffType`, `cash_amount`): Black-Scholes
      closed form in `BlackScholesPricer` (`digital_calc`, vol at the strike), factory entry; `VanillaMCPricer` prices it
      too, `digital_smoothing` (sigmoid of width eps K, default 1%, 0 = hard: unbiased, no AAD delta). Tests vs closed
      form, parities, call-spread limit (flat: = digital; skewed: = digital - vega dsigma/dK), MC. Notebook
      `digital_options`: TICKER1 1Y ATM BS 0.431, call spread on the smile 0.500 (skew term +0.070), local vol MC 0.493
      (smooth spline smile vs the piecewise-linear surface: the call-spread digital jumps at the strike nodes);
      super / sub replication bounds; smoothing vs price bias and delta; greeks against the spot at 1M / 3M / 1Y (delta
      spike, gamma and vega flip near the strike) and how to hedge (static over-replicating call spread).
- [x] Notebook `vanilla_options`: BS vs MC (GBM, local vol) prices, greeks against the spot (delta, gamma, vega, theta),
      delta-hedging simulation (vectorised AAD delta through `blackscholes_calc`): realised = implied gives mean P&L ~0 and
      a hedging error std x sqrt(n) ~ 23 constant; gamma trading, mean P&L = vega x (realised - implied) (±4.5 for ±5
      vol points), path by path P&L vs realised vol. Docs: Delta hedging and gamma trading (Models and Pricing), 2
      questions (Exercises).
- [x] Guided review, notebook `monte_carlo_gbm`: GBM from the market (= BS discrete), convergence 1/sqrt(N) (20 seeds),
      stateless seeds, steps irrelevant for the exact GBM (the 252-step default of `VanillaMCPricer` costs 35x a 1-step
      run for the same law), antithetic (s.e. 0.151 -> 0.124) and forward control variate (-> 0.064) on `model.evolve`,
      AAD greeks vs BS (delta 0.5591 / 0.5593), CRN vs independent-seed bumps. Docs: Monte Carlo framework (Models and
      Pricing), Models (Architecture), 3 questions (Exercises, incl. the empty "greeks via AD").

- [x] Guided review, notebook 02 (dividends): Bjerksund-Stensland 1993 trigger fixed in `pricers/black.py`
      (`h(T) = -(bT + 2 sigma sqrt T) B_0 / (B_inf - B_0)`; it was `B_0 B_inf / ((B_inf - B_0) K)`): the early-exercise
      premium was 3-4x too small (TICKER1 1Y ATM: discrete put 0.022 → 0.435, continuous call 0.184 → 0.614; CRR tree
      0.615 / 0.818). Now within 1.2% of a 2000-step CRR tree on all strikes, = Haug's example (5.2704); American pins
      repinned, new test vs the tree. Notebook 02: time-weighted sigma* for the escrowed vs cash-drop comparison (24.40 vs
      MC 24.46; the flat rule gave 25.16). Study docs: dividend models in Models and Pricing, how the pricer consumes
      the dividends in Architecture, 3 questions in Exercises. `examples/3_options.py`, `option_pricing.py` price
      Americans: not rerun.
- [x] Guided review, notebooks `equity_underlying` and `black_scholes`: `DividendCurve.pv_dividends` discounts each
      dividend at its payment date (when given; the ex-date still decides the horizon): TICKER1 1Y PV 11.8653 → 11.8551
      (payment 14 days after the ex-date); test added. `equity_underlying` shows everything the market holds on an
      equity (keys and risk factors, full dividend schedule, vol surface table and ATM-forward vols, correlations, forward,
      implied yield, AAD sensitivities); `black_scholes` gained European vs American (premium by strike, model and
      maturity, with jumps at the ex-dates; American put delta). Docs: forward and Black--Scholes in Models and Pricing,
      `Equity` members and pricers in Architecture, 4 questions in Exercises (incl. the empty "gamma close to expiry").

- [x] `Equity` underlying view (`MarketEnvironment.get_equity`) with forward, PV dividends, implied yield, risk
      factors; example `study/notebooks/equity_underlying.ipynb`.
- [x] `BlackScholesPricer` on `Equity` and market risk factors, float64 (kernels dtype-generic; fixed a float32
      `sqrt(2)` in `_norm_cdf`); no pricer-local market variables; `examples/3_options.py` greeks via AAD;
      example `study/notebooks/black_scholes.ipynb`.
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

- [x] Caplet vol stripping (`tensorquant/calibration/capletstripping.py`): `strip_caplet_vols` (per strike, caps by
      increasing maturity, one vol for the caplets each cap adds, Brent root so the cap reprices its flat-vol quote)
      and the inverse `cap_flat_vols`; `MarketEnvironment.set_ir_vol_surface`. Snapshot: flat cap vols 71-99bp; the
      stripped vols reprice every cap with `CapFloorPricer` to 4e-12 and stay within 2.2bp of the SABR caplet vols.

- [x] Swaptions: `SwaptionVolCube` (expiry × tenor × strike offset from ATM, one float64 variable, trilinear, flat
      outside), key `IR:<ccy>:<index>:SWVOL`, `ir_swaption_vol.csv`, `get/set_swaption_vol_cube`; `Swaption`
      (European, physical) + `SwaptionGenerator`; `SwaptionPricer`: annuity × Bachelier (normal) / Black-76
      (lognormal) on the two-curve forward swap rate, AAD to both curves and every cube node. Synthetic EUR 6M cube:
      normal SABR per (expiry, tenor) in K − F (7 × 5 × 9, ATM 1Y×5Y 78bp), added to snapshot `20261105/2240`.
      The cube interpolation found a float32 round trip of Python numbers (`tf.cast(float)`), fixed (1e-12).
- [x] G2++ (and HW) swaptions by the Gaussian swap-rate approximation (`GaussianShortRateModel.swaption`,
      `swaption_from_arrays`, batched `swaption_normal_vols`: sensitivity of the swap rate to the factors analytic,
      variance g^T Cov(x(te)) g): ≤ 0.4bp ATM / 0.8bp at ±1% vs MC. Joint calibration caplets + swaptions in
      `calibrate_gaussian_short_rate` (batched, 50 s → 5 s); recovers the G2++ parameters (rho included) from
      model-generated caplet + swaption vols.

- [x] Synthetic EUR 6M caplet vols and swaption cube from one G2++ (parameters drawn per seed): ATM caplets by the
      closed form, ATM swaptions by the exact formula, normal SABR smile on top (vol(K) = ATM · sabr(K)/sabr(ATM));
      caplet expiries every 6M (on the fixings of the caplets). Joint calibration on the snapshot: caplets and
      swaptions within 0.1bp (exact) / 0.3bp (approx), the generating parameters back within ~5% (seed 42: a
      0.0210 vs 0.0207, rho -0.621 vs -0.624; the residue is conventions: dates, lags). Snapshot `20261105/2240`:
      only `ir_vol.csv` and `ir_swaption_vol.csv` rewritten.
- [x] Exact swaptions in `GaussianShortRateModel` (`swaption(..., method="exact")`, batched `swaption_exact_values`):
      bond portfolio at expiry under the expiry-forward measure, Jamshidian for one factor, Brigo-Mercurio
      (Gauss-Hermite on the first factor, closed form on the second) for G2++; roots by bisection outside the tape
      (the value is stationary in the root, so AAD stays exact: gradient = FD to 1e-10). Vs MC (400k paths) within
      2.6 SE at -1%/ATM/+1%, parity exact; `swaption_method="exact"` in the calibration (≈ 5 s for 19 caplets + 12
      swaptions, ~3 s of it the one-off graph build; Jacobian by a while loop of backward passes, 2x faster to
      build than pfor). Generator and exposures kept fast: swaptions batched in one call, discount factors cached
      per date, variance kernels computed only when needed (`generate_market` 0.5 s, exposure tests 38 → 27 s).
- [x] Piecewise constant vols (`sigma_times`) in `GaussianShortRateModel`: every variance / covariance as exact
      integrals over the pieces (expm1, more accurate than before for small speeds: 2.5e-5 → 7e-8 relative at
      a = 1e-4); constant case unchanged (1e-16). Hull-White sigma(t) in `calibrate_gaussian_short_rate`
      (`sigma_times="caplets"`: one vol per caplet, `fixed={"a": ...}`): every ATM caplet repriced (< 0.001bp,
      constant HW: rms 3bp).

## 4. Stochastic processes & Monte Carlo (`models/`, MC pricers, `analytics/`)

- [x] dtype policy: **float64 everywhere** (decided; no float32 casts): GBM/ABM, `LocalVolatilityModel`,
      `DisplacedDiffusionModel`, `VanillaMCPricer`, `AutocallableMCPricer`, `GaussianShortRateModel`, examples.
- [x] 4a `VanillaMCPricer(model=<class>)` builds the model from the market at each price via
      `model.from_equity(equity, maturity, strike)` (`GeometricBrownianMotion`, `LocalVolatilityModel`): one
      `tape.gradient` gives pathwise delta, vega per node, rho, dividend sensitivities. Grid ends exactly at maturity
      (was the nearest point of a grid up to the last vol maturity), stateless Gaussian draws (CRN bumps),
      `product.standard_error`. Discrete dividends: escrowed `S − PV`, carry repo (as BS discrete).
- [x] `HestonSLVModel` (`models/hestonslv.py`): `dX / X = L(t, X) sqrt(v) dW_1`, Heston variance from
      `HestonModel.from_equity` (vol of vol x `mixing` in (0, 1]), `L = sigma_LV / sqrt(E[v | X])` with the local vol of
      `LocalVolatilityModel.from_equity`. Particle method in one pass inside `evolve` (as Oosterlee-Grzelak ch. 10):
      `E[v | ln X]` per step by a tent-kernel smoother on 100 nodes over the 0.1-99.9% quantiles (sparse nodes shrunk
      to the mean, floor 1e-4); QE variance step, log step with the leverage frozen on the step and the exact
      martingale correction per particle. Smile repriced within 0.6 vol pts (LV-grid / time-step bias, the local vol
      model alone is ~0.3 off); on a Heston smile the leverage is 1 (median within 3%). Greeks: Heston parameters
      held (SLV risk convention), leverage moving through the local vol and the forward: vanilla delta / vega =
      CRN bumps (Heston held) within 1%, = Black-Scholes sticky strike within 3%. Rejected: AAD through E[v | X] and
      the Heston calibration: equal-count bins miss the membership change (vanilla delta +4%); with the smoother, the
      derivative w.r.t. a particle's own position (slope between noisy nodes) blew up over 260 steps (5Y autocallable
      delta 1e18); freezing E alone left the Heston channel uncompensated (vanilla delta 16% off). Autocallable
      (synthetic, 5Y): SLV mixing 1 ~ Heston (0.034) far from local vol (0.019), mixing 0.05 = local vol.
- [x] `HestonModel` (`models/heston.py`) on the market forward (`S = F(t) X`, discrete dividends as proportional
      drops). Closed form: characteristic function in the stable form, COS (1024 terms, ±30 sd; the variance
      cumulant from the CF: the Fang-Oosterlee c2 formula was 5-7% off) = Lewis quadrature to 1e-12 from 1W to 10Y
      (very heavy tails, xi 1.5 / rho -0.9, still 1e-4 at 5Y). `from_equity`: least squares on the whole surface
      (each slice scaled by its ATM vega), search on a cheap COS, polish on the full one inside the bounds,
      guarded Newton refinement; greeks through the calibration with a value-preserving implicit step (the value
      stays the bounded optimum, the derivative is -H^{-1} dgrad/dm): vega per node and delta = bump-and-recalibrate
      to 1e-9 (exact fit) / 1e-6. 0.4 s per calibration (compiled residuals, gradient and Hessian traced once per
      session). Synthetic SSVI: ATM within ~1.3 vol pts, short-term wings off (5-parameter Heston), Feller violated.
      MC: Andersen QE with the exact martingale correction (QE-M), = closed form within 2 SE from 12 steps a year.
      Pathwise AAD greeks = closed-form greeks; CRN bump-and-reprice is no reference for QE (the branch switch per
      path makes the estimator jump: FD does not converge as the bump shrinks). `StochasticProcess.noise_size()`;
      `VanillaMCPricer` / `AutocallableMCPricer` draw `[n_paths, n_steps, k]` normals for k > 1 (unchanged for 1).
- [x] `DisplacedDiffusionModel` rewritten on the market forward (was built from floats S0 / flat r / q, numpy
      calibration, not usable by the MC pricers): `S = F(t) X`, `dX = sigma(t) (beta(t) X + 1 - beta(t)) dW`,
      `from_equity` (VanillaMCPricer / AutocallableMCPricer). Slice fit of the closed form (beta, sigma) per
      maturity on vega-weighted prices; forward pieces by inverting Piterbarg's skew averaging (exact integral
      weights); calibration differentiable by one Newton step at the optimum (implicit function theorem, exact
      Hessian, bound parameters held): vega per node and delta = bump-and-recalibrate to 1e-7. MC with exact
      shifted-lognormal steps (a step across pieces keeps the ATM variance; first version summed beta^2 sigma^2:
      up to +68% variance, caught by a grid-across-pieces test). `from_equity` 0.2 s (Hessian rows instead of
      batch_jacobian: 1.15 s → 0.2 s; report lazy). Skew limit documented (normal skew -sigma/2: synthetic SSVI
      smiles not reached at short maturities). Example `displ_diffusion.py` on the snapshot.
- [x] `LocalVolatilityModel.evolve`: the loop is a module-level `tf.function` on tensors (was a `tf.function` on a
      bound method: retraced per model and, with a temporary model, `self` garbage-collected → `None._C`).
- [x] 4d `StochasticProcess` reduced to `size`, `initial_values`, `evolve(t_grid, dw, observation_times=None)` (was an
      Euler `evolve(t0, x0, dt, dw)` + drift/diffusion/expectation no pricer used, overridden with other signatures);
      DD and `GaussianShortRateModel` (`evolve` = factors, `simulate` = draws + discount) on it.
- [x] `HullWhiteProcess.x0` (was `.numpy()` on a numpy value: `AttributeError`).
- [x] Calibration of `GaussianShortRateModel` to caplet vols (`tensorquant/calibration/shortrate.py`,
      `calibrate_gaussian_short_rate`): model caplet in closed form (`caplet` / `exchange_option_values`: exchange option
      between two bonds with the deterministic index basis, = MC within 2 SE), least squares on vega-scaled residuals
      with the Jacobian by AAD, compiled with `tf.function` (G2++ 47 s → 1 s), bounded natural parameters with
      `at_bound` report. Recovers HW params from HW-generated vols; snapshot ATM caplets: HW (a, sigma) rms 3.5bp
      (a at the bound: increasing vols), G2++ max 0.25bp (rho at the bound: under-determined without swaptions).
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
      the study notebooks load `marketdata/` via `from_folder` (no date: latest snapshot; `env.evaluation_date`).
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
