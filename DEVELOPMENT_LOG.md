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

- [x] `BarrierOption(Option)` (`instruments/barrier.py`, `BarrierType` DownIn/DownOut/UpIn/UpOut in
      `markethandles/utils.py`; continuous or discrete `monitoring_dates`, no rebate). `BlackScholesPricer`
      (`barrier_calc`): Reiner-Rubinstein for continuous monitoring, exact vanilla + cash digital decomposition for a
      barrier observed at expiry only, Broadie-Glasserman-Kou shift for n dates; factory maps it to Black-Scholes.
      `VanillaMCPricer`: continuous monitoring by the Brownian-bridge survival between grid points (surface vol at
      `(H, T)`; continuous in the path, AAD without smoothing), discrete dates on the grid (indicator smoothed by
      `digital_smoothing`). Verified (`tests/test_barrier.py`): in + out = vanilla (BS 1e-10, MC path by path), MC GBM
      = Reiner-Rubinstein within 3 SE in all 8 cases, at-expiry MC = exact, monthly MC = BGK within 3 SE + 1%,
      ordering at expiry < daily < continuous, AAD delta / vega vs bumps (BS), bridge AAD delta vs CRN bump.
      Notebooks `digital_strip` (five annual digitals: BS = MC GBM, MC LV = call spread on the surface = the
      autocallable template with the autocall off, 11.54% vs 11.56% ± 0.03%; BS 10.06% misses the skew) and
      `down_and_in_put` (autocallable put = `BarrierOption` at expiry, 23.25% vs 23.28% under LV; annual / monthly /
      continuous monitoring 23.9 / 24.4 / 24.7%, vanilla 24.8%). `autocallable` notebook ported to the library
      (local vol, funding leg on the pricer paths).
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

- [x] Par sensitivities (`par_sensitivities(market_env, zero_greeks)`, `markethandles/bootstrapping.py`): sensitivity
      to the quotes the curves were bootstrapped from, `dV/dq = dV/dz (-J^-1) diag(dNPV_i/dq_i)` from the zero
      sensitivities (AAD) and the stored bootstrap Jacobian and quote DV01s; an Ibor curve discounted on an OIS curve is
      carried to the OIS pillars first (`dz_ibor/dz_ois = -J_ibor^-1 dNPV_ibor/dz_ois`, the calibration instruments
      rebuilt from the stored quotes and repriced with AAD, cached on the curve). Tests (`test_par_sensitivities.py`):
      a calibration instrument (ESTR OIS 5Y, 6M swap 10Y, FRA, 6M fixing) has minus its quote DV01 on its own quote and
      zero (1e-6 relative) on every other quote, ESTR quotes included for the 6M instruments; a 1Yx4Y 6M swap: sum of
      par sensitivities x 1bp = PV change under a parallel 1bp move of every quote with both curves rebootstrapped
      (within 2%, slow). In the Tradepad (rates products): rho table pillars x (Par | Zero) x curves, total first, Par
      n/a for curves without a Jacobian. Real 5Y EUR 6M swap at the market par rate (2026-11-13): all its par
      sensitivity on the 5Y swap quote (4563.6 per bp on 10M), zero on the ESTR quotes; first call per snapshot about
      8 s (the 6M instruments repriced against ESTR), then 0.3 s.
- [x] Tradepad swap screen on generators (`swap_generator(ccy, curve)`: the conventions of the bootstrap,
      `QUOTE_CONVENTIONS`; the bootstrap uses it too): EUR 6M Euribor IRS (fixed 1Y 30/360, 6M Act/360, discounted on
      ESTR), EUR ESTR / USD SOFR / GBP SONIA OIS; forward start and a floating spread (`Swap` / `Ois` `spread`,
      generators `forward`, `spread`); the two legs with their flows, index, projection and discount curves. The old
      screen had an Act/360 fixed leg and TARGET for every currency: the par rates now equal the Bloomberg quotes
      (EUR 6M 2Y / 5Y / 10Y 3.382 / 3.387 / 3.433, ESTR 5Y 3.096, SOFR 5Y 4.671). Tests: spread = N s sum tau P, forward
      start (`test_rate_pricers.py`), legs' PV = pricer PV in 5 cases (`test_tqstudio_tradepad_swap.py`).

- [x] Index dividends implied by the options with the repo held (`implied_index_dividends`, in
      `calibrate_snapshot_vols`): projections kept to 2 years, repo held at the level of the last listed expiry within
      them, beyond each listed forward gives `PV_div(T) = S - F P e^{q0 T}`, the increase spread with the weekly
      pattern of the projections, constant yield past the last expiry to 10 years; `eq_dividends.csv` with `source`
      download / implied / extrapolated, the download kept in `eq_dividends_download.csv` (reruns read it). Market
      practice: index dividends from the market, the repo a smooth financing level (the constant-yield extrapolation
      of the projections made the SX5E repo fall from -0.93% to -1.53% at 6Y: dividends about 35% above the options').
      Test: a true market (quarterly 0.8, flat repo 0.4%), options to 5 years, projections +30% beyond 2 years and
      missing beyond 3: the true dividends recovered (1e-6) and the repo flat at every expiry. Snapshot 2026-11-13
      rebuilt (85 s): SX5E repo flat -0.82% from 2028 to 2032, implied dividends 168 / 156 / 157 / 154 points a year
      in 2029-2032 (constant yield then 2.26%), forward 10Y 7405; SPX flat -0.72%, implied dividends falling to 55
      points in 2031 (open in the roadmap).

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
- [x] **Equity models as instances, separate from the market** (part 1 of the `tqstudio` design in `ROADMAP.md`):
      `models/equitymodel.py` `EquityModel` (settings + calibrated parameters as float64 `tf.Variable`s, no `Equity`):
      `Model.calibrate(equity)` (parameters, report, ticker, date, calibration Jacobian `dp/d(market risk factors)` by
      `tape.jacobian` through the differentiable calibration), `Model(**parameters)` by hand (no Jacobian),
      `model.process(equity)` → bound process, `Model.from_equity` = process calibrated inside the caller's tape
      (unchanged behaviour), `par_greeks`, `to_frames` / `from_frames` (exact round trip). The former model classes
      are the processes (`LocalVolatilityProcess`, `HestonProcess`, `DisplacedDiffusionProcess`, `HestonSLVProcess`,
      same numerical code); new `LocalVolatilityModel` (settings only, vegas direct), `HestonModel`,
      `DisplacedDiffusionModel` (slice parameters as settings), `HestonSLVModel` (a `HestonModel` + a
      `LocalVolatilityModel`, Heston held). `VanillaMCPricer` / `AutocallableMCPricer` take a class (as before) or an
      instance (`bind`: ticker checked). GBM unchanged (option-specific factory). Checks: `tests/test_model_pins.py`
      (prices, delta, vega of the four classes and the autocallable, calibrated Heston / DD parameters, pinned before
      the change) unchanged to rtol 1e-8; `tests/test_model_instances.py`: instance price = class price (Heston 1e-12),
      LV / SLV greeks identical, Heston and DD par greeks through the stored Jacobian = AAD through the calibration
      (rtol 1e-6), serialization, by-hand models, argument checks. Full suite 265 tests green. Notebook `heston_slv`
      on the instances (SLV family for the barrier: one Heston calibration, `HestonSLVModel(heston_model, mixing=e)`).
- [x] **SLV vega per node (stored `E[v | X]`, adjoint of the leverage calibration)**: the AAD greeks held
      `E[v | X]` out of the tape, which misplaced the vega of a vanilla (snapshot, TICKER1, 1Y 80% put, per vol point:
      0.45 on its node vs 0.57 by bump-and-rebuild of the leverage with common random numbers on all 72 nodes, the rest
      on other strikes of the same expiry; 5Y put: total 1.09 vs 1.47). Now `E[v | X]` is calibrated on its own
      particles (`calibrate_expectation`, `HestonSLVModel.calibrate_leverage`: 41 nodes per row, 50 rows a year,
      normals seeded per row, end nodes 1% beyond the quantiles) and stored as the parameter `expectation`; the
      pricing reads it (differentiable in the path position and the table); `par_greeks(model, ..., equity=eq)` maps
      `dV/dE` by the adjoint of the calibration (`EquityModel.map_model_greeks` hook, Savine ch. 13); the class path
      calibrates inside the pricing tape. Found on the way: a node on a particle (the quantile end nodes) is a kink at
      the calibration point (one-sided AAD slope, 11% off finite differences on the first row; padded). Smoother bias
      vs adjoint stability: 41 nodes bias the repricing (snapshot, 3Y 80/100/120%: -0.38/-0.61/-0.53 vol points, 1Y ATM
      -0.32); 101 nodes or the Richardson combination `(4 E_h - E_2h) / 3` (`bias_correction`) remove most of it
      (-0.13/-0.23/-0.22, 1Y ATM -0.08) but their adjoint blows up (5Y parallel vega 131x the bumps; 101 nodes still
      wrong at 200k particles: 0.43 vs 0.60); a stop-gradient on the correction inside one table is wrong too (1Y
      parallel 0.75-0.93x). Adopted: smoothing for risk (Savine), two tables on the same nodes and particles, the
      corrected one for the value (outside the tape, `leverage_table["price_values"]`), the 41-node one for the
      derivative (`X_risk + stop_gradient(X_price - X_risk)`). Checks (`tests/test_slv_vega.py`, synthetic market,
      20k particles): adjoint par vega = central bumps of the two-stage corrected model (both tables recalibrated)
      within 3% on the node and its neighbour, 5% on the parallel shift (1Y 1.03), 1Y and 5Y puts; off the node up to
      0.05 per vol point (1Y, strike below: 0.02 vs 0.07), not tested; class path one pass = instance direct + adjoint
      (rtol 1e-6). Snapshot (default settings): 1Y 80% put vega 0.52
      on its node, total 0.69 (Black-Scholes 0.69); 5Y 1.03 on the node, total 1.50 (prototype vs bumps: 1.471 vs
      1.470, node 1.005 vs 1.014; with the two tables: 1Y total 0.687). Times: Heston 3 s now and then, leverage
      3.2 s (3Y, one table) / 3.5 s (3Y, two tables) / 5.4 s (10Y, two tables), the corrected price run adds ~40% to
      a pricing (1Y put, default pricer, price + AAD 28-38 s -> 48-52 s, timings noisy), adjoint 3-9 s (1Y) / 50 s
      (5Y). Check-pointing (Savine ch. 13): the calibration and pricing steps are wrapped in `tf.recompute_grad`
      (the tape keeps only `(v, ln X)` per step, the step is recomputed in the backward pass); reverse mode had stored
      every intermediate of every step for every path (memory bound: the forward under the tape took 13 s instead of
      1 s). Same numbers (1e-15). Snapshot, default settings (50k paths, 252 steps): 1Y 80% put price + AAD 25-40 s
      -> 8-9 s, adjoint 9 s -> 2-4 s; 5Y put price + AAD 100-150 s -> 9 s, adjoint 50 s -> 7-9 s; price only 3 s;
      leverage calibration 4-6 s (both tables, to 5Y). SLV pin repinned; notebooks `heston_slv` (step 3 on the stored table,
      section 6 by calibration particles, greeks) and `equity_models` updated; Models and Pricing, Architecture.
- [x] **Faster calibrations (implicit step precomputed)**: the implicit-function derivatives of the Heston and
      displaced diffusion calibrations are computed once at the optimum, eagerly, as numbers (`numericalhandles/
      implicit.py`: `optimality_derivatives` = g, H, dg/dz with one backward pass per parameter; `newton_step`;
      `implicit_sensitivities` = -H^{-1} dg/dz with bounded parameters fixed; `implicit_solution` = parameters with a
      custom gradient). Before, Heston traced a `tf.function` with the Hessian and differentiated it inside the
      caller's tape (11.6 s of tracing on first use), and the DD Jacobian walked the second-order eager graph;
      `EquityModel._record_calibration` takes one Jacobian per parameter to all the risk factors (was one per market
      key). TICKER1 (snapshot): Heston 16.2 s -> 3.2 s first call (2.0 s once the COS residuals are compiled), DD
      7.6 s -> 2.0 s. Checks: new calibration Jacobian pins (`TestCalibrationJacobianPins`: Heston on the pins market,
      DD on a market with interior betas and non-zero residuals, sum and abs sum per parameter / key / variable,
      pinned with the old code) unchanged to rtol 1e-8; parameter, price and greek pins unchanged to 1e-8 except the
      SLV vanilla (1e-6: the Heston parameters move by ~1e-15 relative, eager vs compiled, and the particle leverage
      amplifies it to ~4e-7, also in the old code); bump-and-recalibrate tests and par greeks = AAD unchanged. Only
      first order through the calibration (as before, the Hessian was held constant).
- [x] **Forward-starting autocallable**: `AutocallableOption(strike_date=...)` (strike `None` for a future strike date,
      fixings checked to follow it; `put_strike` in % of the initial level, default 100); `AutocallableMCPricer`
      simulates the strike date with the fixings and uses `K = S(t0)` path by path for every barrier, the put strike and
      the smoothing width; a past strike date without the fixed level raises (live trades: roadmap, tqstudio item 7).
      Checks (`tests/test_autocallable_mc.py`): strike date today = the spot-starting price (1e-12), numpy replica on the
      same paths with `K = S(t0)` (1e-12), flat GBM forward start = `P(0, t0)` x the spot-starting autocallable within
      3 SE, AAD delta below 1e-10 under GBM (paths and strike scale together), schedule validation. Tradepad: strike
      date in the deal parameters, schedules after it. TICKER1, local vol, 3Y after a 1Y strike date: -5.00 per 100
      with delta 0.05 (cash dividends), against -5.66 and delta 0.22 for the trade struck today.

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
- [x] Monotone convex interpolation of the curves (Hagan-West 2006, `numericalhandles.MonotoneConvex`,
      `RateCurve(..., "MONOTONE_CONVEX")`), default of `bootstrap_from_quotes`: discrete forwards and pillar zero
      rates preserved, continuous instantaneous forward (linear zero rates gave jumps at every pillar; on the
      2026-11-12 Bloomberg curves the largest move on a 0.01y grid is now 1.9bp), flat forward after the last
      pillar, no positivity collar. Coefficients computed once (vectorised) and cached; `curve.pinned()` reuses them
      under one tape; without a tape the values come from numpy (equity market view 26 s -> 0.9 s).
- [x] Bootstrap speed: one persistent tape for all the instruments of a Newton iteration (under `curve.pinned()`),
      discount factors of a leg in one vectorised call (`RateCurve.discounts`, `PrefetchedCurve` in the leg
      pricers). ESTR test curve: linear 4.2 s -> 2.8 s, monotone convex 90 s -> 5.2 s; the four real curves of the
      2026-11-12 snapshot (122 instruments to 50Y) 71.6 s with monotone convex (66 s before with linear), once at
      download. A compiled `tf.function` objective was tried and dropped (tracing too slow on the large graphs).
- [x] tqstudio market page: the equity view shows the currency only (no discount curve panel).
- [x] Repo term structure: `EQ:<ccy>:<ticker>:REPO` is a scalar or a `RateCurve` of zero repo rates (linear,
      ACT/365); `Equity.repo_rate(T)` is the rate to a maturity, used by `Equity.forward`, `BlackScholesPricer` and
      the GBM (the other MC models read `equity.forward(t)`); repo sensitivities per pillar (AAD = finite
      differences); optional `eq_repo.csv` (written only for curves, exact round trip). Tests `tests/test_repo_curve.py`.
- [x] Equity vols from listed options (`calibration/equityvol.py`, after the Bloomberg "Equity Implied Volatility
      Surface" methodology in `resources/`): `implied_volatility` (European / American Bjerksund-Stensland,
      vectorised: grid bracket + Illinois steps, compiled price functions, 1e-15); `implied_forward` (put-call parity
      at the two strikes around the forward, de-Americanization on the model of `BlackScholesPricer`, secant steps;
      on noisy quotes jumping between two pairs: mean of the two, `converged = False`), `implied_repo_curve`;
      `LognormalMixture` (4 lognormals + default mass, fit to the out-of-the-money prices, vega-weighted);
      `build_equity_vol` (filters, settlement prices at weight 0.25, PCHIP in total variance between expiries,
      Heston-like ATM level beyond the last one, Lee wings; grid 1M-6Y x 50%-150% of the spot; expiries under 14
      days out of the repo curve); `calibrate_snapshot_vols` (`eq_vol.csv`, `eq_repo.csv`, `eq_vol_calibration.csv`
      with the mixture parameters, `eq_vol_quotes.csv` with bid / mid / ask vols). Synthetic checks: forward and repo
      recovered to 1e-10 (European) / 1e-9 (American), repo curve reprices the forwards to 1e-12, mixture smile
      recovered to 2e-4 in vol; 12 s an underlying on a 16-expiry American chain (121 s before compiling).
- [x] mdbuild listed options (`mdbuild/options.py`): `OPT_CHAIN` cached a week (`marketdata/cache/`, git-ignored),
      main root (`option_roots` overrides: ENEL `ENE IM`, RACE `RAC IM`), first 3 then quarterly expiries to 6.2Y,
      the nearest out-of-the-money strike to 17 forward moneyness plus 2 parity strikes a side, four fields
      (`PX_BID`, `PX_ASK`, `PX_LAST`, `LAST_UPDATE_BID_RT`: a time is today, a date that session; `LAST_UPDATE_DT`
      is the last trade, days old on live IDEM quotes), reference spot = close of the quote session when not today.
      Declared dividends (`DVD_HIST_ALL`, cash, future ex-date, pay dates) replace the BDVD projections within 30
      days; special dividends excluded (listed options adjusted). Snapshot 2026-11-12 16:24: 2450 options of 10
      underlyings (about 9800 data points), curves 74 s, vols 210 s, 128 expiries, no errors, repos -6% / +3%, the
      model within bid / ask on 95-100% of the market quotes.
- [x] tqstudio market page: equity smile vs market (expiry selectable, default the nearest to 9 months): bid / ask
      vol bars, settlement and filtered quotes, calibrated mixture, stored surface; fit summary.
- [x] Option chains by expiry (Bloomberg's advice for large chains: ``OPT_CHAIN`` truncates at 8000 options, SPX
      stopped at 4 expiries): ``CHAIN_EXPIRATION_DATES`` then ``CHAIN_TICKERS`` per selected expiry (calls and puts,
      strikes 40%-200% of the spot, unadjusted contracts, main exchange); expiries: the first 2 listed (IDEM weekly
      roots ``2ENI`` / ``4ENI``, ``SPXW``), monthlies to 3 months, quarterlies to 18 months, then one a year to 6.2
      years; cache refreshed weekly or when its first expiry has passed. Equity indices (``equity_indices`` in
      ``universe.yaml``: SX5E, SPX) with weekly projected dividends in index points (``BDVD_PROJ_DIV_INDX_PTS``,
      ``PERIODICITY_OVERRIDE = W``, about 3 years). Stale bid / ask kept as settlement on the last price. Spot read in
      the same request as the option quotes; parity on 4 strikes a side. ``implied_spot`` (SPX: delayed price on this
      terminal): the spot implied by the options (forward of the nearest expiry from 3 days, plus the dividends,
      discounted), written to ``eq_spot.csv``; Bloomberg's ``IMPLIED_FORWARD_PRICE`` matches our implied forwards
      within 2-5bp. Snapshot 2026-11-13 16:19: 2079 options of 6 underlyings, SPX to Dec 2031, the model within bid /
      ask on 97-100% of the quotes, repo curves -1% / +0.3% (ISP short end -5%).
- [x] Equity vol surface on the listed expiries (then one point a year to 6 years); repo curve from 30 days.
- [x] Pricing speed: ``Pricer.price`` pins the market curves (``MarketEnvironment.pinned``: monotone convex
      coefficients once inside the tape; autocallable 6.5 s -> 2.4 s at 5k paths); the local vol grid built up to the
      product maturity (``process(equity, maturity)``: same values and vega, zero vega after); dividend PVs with one
      vectorised discount call (``DividendCurve.pv_dividends``: SX5E 97 weekly dividends, 4Y autocallable at 20k paths
      under the tape 8.2 s -> 1.9 s for the price, greeks 7.7 s -> 2.1 s). A compiled local vol step loop was tried:
      no gain (it was already a ``tf.function``).
- [x] tqstudio: market page (strikes under the moneyness, listed expiry labels and dates, copy surface to Excel,
      implied repo plot, repo 1Y, correlation matrix of picked underlyings); Tradepad: price first, greeks next (one
      computation, the tape kept for ``/api/price/greeks``); autocallable defaults (strike date +10 days, 3 years, 2%
      coupon 80% semiannual, put 80 / 100 / 100%, autocall 100% semiannual from 1Y).
- [x] Calibration sets (`calibration/models.py`, `python -m tqstudio.calibrate`, `marketdata/calibrations/<source>/`,
      git-ignored): the equity models of a snapshot calibrated apart from the market data (Heston and displaced
      diffusion on every underlying, Heston SLV on request), stored with `to_frames`, fit per node, quality on the
      meaningful region (maturities from 1 month, strikes within 2 standard deviations: good <= 1 vol pt rms, fair <=
      2.5, else poor), drift of stored parameters on a newer market (to recalibrate above 1 vol pt). First real set
      (2026-11-13 16:19, 6 underlyings, 292 s): Heston 0.7-1.3 vol pts rms, displaced diffusion 0.8-2.5 (ENI: skew
      averaging fails, reported). Model test: vanillas, down-and-in puts observed at expiry, cash digitals on shared
      paths (20k, about 1 s a model) against Black and local vol (SX5E 1Y down-and-in 60%: Black 0.07, local vol 1.80,
      Heston 1.45, displaced diffusion 0.45 % of spot). tqstudio Calibration tab (list, set grid, underlying: parameters,
      report, drift, errors map, smiles with quotes, ATM term structure, model test; new sets in a separate process),
      link from Market data, Heston / displaced diffusion / Heston SLV in the Tradepad from the latest set.
- [x] Calibration universe: `marketdata/config/calibration.yaml` (flags per underlying and model, by asset class, defaults
      per class: equity Heston and displaced diffusion on, SLV off; rates Hull-White / G2++ listed, off, not in the sets
      yet), edited by the Calibration tab > Universe (one click saves); `python -m tqstudio.calibrate` and New
      calibration set calibrate the flagged models (`calibrate_snapshot_models(plan=...)`; unsupported flags reported as
      skipped).
- [x] Heston SLV, when to recalibrate (Calibration > Universe): per underlying its latest SLV calibration against the
      latest snapshot: age, spot move, surface change (rms at 3M-3Y, 80-120% of the spot, sticky moneyness), ATM 1Y vol,
      drift of the Heston part (closed form) and, on request, the Monte Carlo fit of the stored SLV on the latest market;
      recalibrate above 7 days, 5% spot, 1 vol pt surface change or 1 vol pt rms fit. SX5E (16:19 calibration) on the
      16:09 snapshot: spot +0.11%, surface 0.05 vol pt, fit 0.34 vol pt: OK.
- [x] Real market data, moved from the roadmap: rate curves from Bloomberg (the snapshot stores the ICVS curve members,
      `ir_quotes.csv`: EUR ESTR S514, EUR 6M S45, USD SOFR S490, GBP SONIA S141, mid quotes; every instrument
      repriced to 1e-15 on 2026-11-12); curves bootstrapped right after the download (`bootstrap_snapshot`:
      `ir_curves.csv`, `ir_curves_jacobian.csv`, `ir_calibration.csv`; loading a snapshot 0.5 s instead of 63 s;
      calendars `UnitedStates`, `UnitedKingdom`); tQuant / tqstudio reading the live snapshots (curves, equity vols,
      repo curves).
- [x] Smile mixture without point masses: `LognormalMixture.fit` keeps the vols of the lognormals above
      `MIN_VOL_FRACTION` = 1/4 of the at-the-money vol of the expiry (and 0.5%; argument `min_vol_fraction`, 0 = the old
      0.5% bound). Without it the least squares, freer than the quotes need where the strikes are sparse, put a
      lognormal on 0.5% vol, an almost point mass (2026-11-13: 12 of 25 expiries of AAPL / SX5E, up to 18.8% of
      probability on AAPL 5 days, 12.2% on SX5E Dec-2032; the spike of the AAPL Dec-2027 density at 1.26 F) for
      hundredths of a vol point. Variants compared on every AAPL / SX5E expiry: floor 1/4, 1/3, 1/2 of the ATM vol,
      three lognormals (no fix on SX5E 2031-32, worse short expiries): 1/4 removes every lognormal below 3% vol, rms
      up by < 0.01 vol pts on most expiries (at most +0.04), the same quotes within bid / ask; largest increase of
      P(S_T < K) over 1% of F on the long expiries 0.02-0.04 -> 0.011-0.014; probabilities at 60-80% of F move by up
      to 2.3 pts (SX5E 2031 P(S_T < 60%) 12.6% -> 14.9%). Whole snapshot recalibrated (copy): grid nodes beyond 30 days
      within 0.2 vol pts between 70% and 130% (ISP 1.0), up to 4 in the 50% / 150% wings; large moves only in the
      unquoted wings of the 3-7 day expiries (ISP 7 days at 150%: 98.6% -> 21.8%, the old one fitted a 54.5%
      settlement with a point mass); two-sided quotes within bid / ask ISP 99.1% -> 100%, SPX 96.6% -> 96.0%, others
      unchanged. Test: smooth skewed smile with 0.2 vol pt noise (free fit 15% of probability on a 0.6% vol lognormal,
      jump of 5% in P(S_T < K); with the floor none below 1/4 ATM, smooth, rms within 0.05 vol pts). The notebook data
      set (`study/notebooks/equity_vol_calibration_data.csv`) recalibrated, notebook and Models and Pricing 1.4
      updated (the AAPL Dec-2027 scenarios change, the smile does not: 80% put 30.0%). The live snapshots keep the old
      surfaces until their vols are recalibrated.
- [x] Equity vol grid in the wings (Beta 1, item 1, first part): `GRID_MONEYNESS` 30%-200% of the spot, every 5%
      below 95%, then 97.5%-150%, 175%, 200% (26 strikes, were 17 from 50% to 150%). Beyond the outermost quotes of
      an expiry the total variance is straight in log-moneyness, with the slope of the mixture at the junction within
      Lee's bounds and never decreasing away from the money (`_wing_spec`, `_smile_w`); the junction moves outward
      when the straight wing would make the density negative (Gatheral's g on the wing, out to 10% / 400% of the
      forward), no junction: the mixture's tail, unreliable. Calendar step before the PCHIP in time: an expiry raised
      to the earlier ones at the same forward moneyness, a reliable point (quoted or straight wing) only by earlier
      reliable points; after the last expiry the edges carried in ln(K/F)/sqrt(t), never below the last expiry.
      Report columns `k_lo` / `k_hi`. Alternatives measured on 2026-11-13 and dropped: straight wings from the edge
      with a full calendar maximum (6 butterflies beyond the quotes), the mixture's tail when the wing fails with a
      full maximum (unreliable tails of 3-5 day expiries raised the next expiries' quotes: AAPL max error 28 vol
      pts), anchor points in the fit (SPX smile within bid / ask 96% -> 89%). Result, snapshot 2026-11-13
      recalibrated (copy): two-sided quotes within bid / ask on the stored surface AAPL 98.5%, ENI 100%, ISP 98.2% ->
      100%, MSFT 100%, SPX 79.1% -> 79.5%, SX5E 95.4%; rms error on the surface ENI 0.16 -> 0.07 vol pts, ISP 0.60 ->
      0.29, others equal or lower; butterfly arbitrage on the grid 5 -> 0; calendar arbitrage 0 on SPX / SX5E / ISP,
      left only in the unreliable tails of 3-5 day expiries (AAPL, MSFT) and residuals of 5e-4 (ENI 2031-2032). AAPL
      at one year: 39.2% at 50%, 48.4% at 30% (the mixture's tail bent down below 40%). Tests: no butterfly, no
      calendar arbitrage over the whole grid, straight wings with Lee slopes (European and American cases). Notebook
      data set and Models and Pricing 1.4 updated; the live snapshots keep their surfaces until recalibrated.
- [x] Settlement prices and expiries with a wing free of quotes (Beta 1, item 1). Settlement prices (`last`, no bid /
      ask) only for an underlying without a bid / ask market, Bloomberg's rule (an expiry has a good market with 3
      two-sided quotes and 10% of its quotes; 3 such expiries or 10% of them: settlements dropped): forwards, repo,
      smiles and index dividends from the quoted expiries only, the repo flat beyond the last one, index dividends
      projected at a constant yield with the repo held (SX5E 2026-11-13: quoted to Dec-2028, then 3.10%, about
      202-210 points a year, against 155-168 implied by the 2029-2032 settlement prices; ENI's Dec-2028 settlement
      gave a repo of +0.21% between -0.9% and -0.7%). Expiries with a wing free of quotes, quoted only within 90%-110%
      of the forward (a few days) or out of the money on one side only (ISP Dec-2029: 5 puts), fitted again with the
      smile of the next complete expiry (the previous one for the last expiries) carried to their maturity
      (Bloomberg's short-term extrapolation: scenario vols kept, ln xi scaled by sqrt(t/T); for the narrow ones scaled
      to their at-the-money vol) as anchors beyond their quotes (weight 0.1) and start; report column `smile_from`.
      Replacing the smile outright (Bloomberg's choice for expiries with 1-2 options) cost 1-4 vol pts on the 5-13
      quotes of those expiries; dropping one-sided expiries lost ISP's 2029 puts (5 vol pts). Snapshot 2026-11-13
      recalibrated (copy): quotes within bid / ask on the stored surface AAPL 98.5%, ENI / ISP / MSFT 100%, SPX 79.5%,
      SX5E 95.4%; max grid vol MSFT 436% -> 174%, ENI 155% -> 90%; calendar arbitrage 0 on ENI, MSFT, SPX, SX5E; no
      butterfly arbitrage. Download: moneyness targets 30%-200% of the forward (were 50%-150%), chains 25%-250% of the
      spot (were 40%-200%), chain cache version 4. Tests: settlements dropped with a market and used without, a
      narrow expiry guided by the next one (own quotes within 0.2 vol pts, wings within 2 of the guide's shape).
- [x] Step-down autocall (Beta 1, section 2 item 1): `AutocallableOption(autocall_step=, autocall_floor=)`, the
      autocall barrier lowered by the step at each autocall observation after the first and bounded by the floor,
      `max(B - j step, floor)` (stored in `autocall_barrier`, one per observation; negative step or floor above the
      first barrier rejected). Tradepad: "Step-down per observation (%)" (default 0) and "Step-down floor (%)"
      (default 80, used only with a step) in the early-redemption group, the barriers in the schedule and the
      summary; "GBM (debug)" among the autocallable models, not the default (the Black-Scholes model of one expiry,
      wrong at the intermediate fixings: for debugging), with its Monte Carlo standard error. SX5E 2026-11-13, 3Y,
      6M observations from 1Y, local vol: -1.78% of the notional flat, -1.72% with a 2% step (barriers 100-92%).
      Tests: barriers with step and floor, same price as the same barriers given one by one, validation (19 green).
- [x] Quanto (section 2 item 3, first part). `Product.payment_ccy` (default `ccy`, the underlying's currency) and
      `is_quanto`, taken by `VanillaOption` and `AutocallableOption`: paid in the payment currency at a fixed rate.
      `MarketEnvironment.get_quanto(ticker, ccy, payment_ccy, tenor)`: correlation of the underlying with X (one unit
      of `ccy` in `payment_ccy`: the stored pair, or minus its correlation for the inverse) and the vol of the pair
      at its spot, tensors. `BlackScholesPricer`: payment-currency discount rate, carry
      `q + (r_pay - r) + rho sigma sigma_X` (European vanilla, digital, barrier; American rejected). Monte Carlo
      (`pricers/quanto.py`, `apply_quanto`): `StochasticProcess.set_quanto(rho sigma_X)` (GBM: `mu - rho sigma_X
      sigma`; local vol: `-rho sigma_X sigma_loc(t, S)` on `ln X` each step, the paths no longer rescaled to the
      forward; the other processes raise), payments discounted on the payment curve (`VanillaMCPricer`,
      `AutocallableMCPricer`). Market data: historical FX vols (`mdbuild.history.historical_fx_vols`, `fx_vol` block of
      the settings: 52 weekly log returns, annualised) as flat surfaces in `fx_vol.csv` (1M-10Y x 50%-200% of the
      spot), built by every download and written into the 2026-11-13 snapshot from the price history: EURUSD 5.8%,
      GBPUSD 5.9%, EURGBP 3.2%, EURCHF 3.8%, USDJPY 8.1%. Tradepad: "Payment currency (quanto)" on the vanilla and the
      autocallable, the info rows (correlation with the FX rate, FX vol, FX vega per vol point, correlation
      sensitivity per +0.01), the payment curve in the rho. 2026-11-13: SPX 90% put 1Y (Black-Scholes) 233.69 paid
      in USD, 230.90 in EUR (rho(S, X) = -0.31: quanto forward 7858 against 7826; FX vega -1.14, correlation -0.21
      per +0.01); SX5E 90% put 214.52 in USD against 213.78 in EUR (rho +0.23: forward 6412 against 6429); SPX
      autocallable (local vol) -0.70% of the notional in USD, -0.56% in EUR. Tests (`test_quanto.py`, 10): quanto
      inputs (direct / inverse pair, missing pair), Black-Scholes closed form `DF_d Black(F e^{-rho sigma sigma_X T})`,
      own currency = plain price, zero correlation = plain forward discounted in the payment currency, AAD
      correlation and FX vol greeks vs bumps, American rejected, GBM and local vol vs the closed form within 3 SE on a
      flat smile, Heston rejected, GBM autocallable quanto = plain one with the repo shifted by
      `rho sigma sigma_X` (same draws, equal rates), local vol autocallable price ordered by the correlation;
      marketdata 26 (historical FX vols = direct weekly std, grid). Models and Pricing (quanto subsection),
      Architecture (FX data, products, pricers).
- [x] Composite and price currency (section 2 item 3). `Product.fx_mode` `"quanto"` (default) or `"composite"`
      (`is_composite`), taken by `VanillaOption` and `AutocallableOption`: the payoff on `Y = S X` (X one unit of the
      underlying's currency in the payment one), strikes and levels in the payment currency. Black-Scholes on Y:
      spot `S X0`, forward `F_S F_X(T)`, vol `sqrt(sigma_S^2 + sigma_X^2 + 2 rho sigma_S sigma_X)` (`sigma_S` at
      `K / F_X`), discounted in the payment currency (vanilla, digital, barrier; American rejected). Monte Carlo
      (`composite_fx_paths`): S with the quanto drift, `X_t = F_X(t) exp(-sigma_X^2 t / 2 + sigma_X W_X)`,
      `W_X = rho W_S + sqrt(1 - rho^2) W_Z` on the underlying's normals and an independent stream, paths `S X`
      (vanilla, digital, autocallable; composite barriers rejected; local vol and GBM). `Pricer.price(...,
      price_currency=)`: the price converted from the payment currency at the FX spot inside the tape (FX delta by
      AAD), the standard error too, `product.price_currency`. Tradepad: "FX treatment" (Quanto / Composite) and "Price
      currency" on the vanilla and the autocallable (composite strikes in % of the spot in the payment currency; the
      % of the price on a reference in the price currency; FX spots among the greeks: "FX delta"). 2026-11-13, SPX 90%
      put 1Y paid in EUR: quanto 230.90, composite 207.53 EUR (Black-Scholes), FX delta (per 1 of EURUSD) 1415
      Black-Scholes, 1401 GBM, 834 local vol (40k paths: the skew lowers the probability below the strike); the same
      put priced in EUR 211.50 = 233.69 / 1.1049. A forward-start composite autocallable has no FX delta (its strike
      is S X at the strike date). Tests (`test_quanto.py`, 15): composite flags and validation, Black-Scholes closed
      form on Y, GBM and local vol Monte Carlo within 3 SE of it, composite autocallable without FX risk (FX vol ~0,
      equal rates) = the plain one on the same draws; price currency = price / EURUSD, AAD delta to the FX spot
      `-price / EURUSD^2`, standard error converted, same currency unchanged.
- [x] Worst-of autocallable under local vol and GBM (section 2 item 2). `AutocallableOption` on several underlyings
      (`underlying` a list, `strike` one initial level each or `None` for a forward start; `is_worst_of`,
      `underlyings`, `strikes`; composite rejected): every level on the worst performance `min_k S_k / K_k`.
      `MultiAssetProcess` (`models/multiasset.py`): one process per underlying (one normal per step: local vol, GBM),
      independent normals `[n_paths, n_steps, n]` times the Cholesky factor of `market_env.get_correlation` (AAD to
      the correlation). `AutocallableMCPricer`: model class or dict ticker -> instance; the payoff written once on the
      performance (one underlying: `S / K`, same numbers: model pins at 1e-8); an underlying in another currency is
      quanto (its drift, `product.quanto`), payments on the product's curve; `pricer.performance`. Tradepad:
      "Underlying 2 / 3 (worst-of)" (local vol or GBM), strikes on each spot, delta and vega per underlying, one vega
      map with rows by underlying and expiry, correlation and sensitivity per pair, quanto rows. 2026-11-13, 3Y
      phoenix (coupon 2% at 80%, autocall 100%, put 80%), local vol 10k paths: SX5E 0.33% of the notional, SX5E / SPX
      -1.44%, SX5E / SPX / ISP -12.14% (ISP, the most volatile, carries the delta: 12.2 per EUR; correlation
      sensitivities SX5E-ISP +0.86, SX5E-SPX +0.23 per +0.01). Tests (`test_worst_of.py`, 8): flags and validation,
      hard payoff = numpy replica on the same paths (spot and forward start, memory), marginals = forwards within
      3 SE and the simulated correlation (GBM and local vol), Heston rejected, price increasing with the correlation
      and below each single-underlying product, quanto underlying = repo shift on the same draws, AAD delta per
      underlying and correlation sensitivity vs bumps (3%), local vol on flat smiles = GBM (slow). Found on the way:
      the local vol delta on the real surfaces is spiky (ROADMAP section 2 item 2).
- [x] Local vol delta on the real surfaces: density margin on the wings, floor on g (section 2 item 2). Found with
      the worst-of: on 2026-11-13 the SX5E 3Y autocallable had a local vol delta per 1% of 0.35 by a 1% bump, 0.40 by
      0.1%, 4.2 by 1e-5 and 6.2 by AAD (exact, the derivative of a spiky estimator). Cause: the straight wing beyond
      the quotes starts with the slope of the mixture at its last quote, often its steepest; on SX5E 2.09Y from 68% of
      the forward, slope -0.34, Gatheral's g 0.02 (g = density over the Black density of the same total variance): an
      almost empty band, where the spline of the local vol rings to g -0.13 and sigma_loc^2 = dw/dT / g reaches 183%
      (the band moves with the forward, so with the spot). The nodes were butterfly-free: the check on the wings only
      asked g >= 0. (A) `_wing_spec`: the wing must keep `g >= WING_G_MIN` (0.25) out to 10% / 400% of the forward;
      candidates at the edge, then inside the quotes up to `WING_INWARD` (0.2 in log moneyness, the straight line within
      `WING_QUOTE_TOL` = 0.25 vol pts of the smile it replaces), then beyond the edge; none reaches the margin: the
      candidate with the largest density among those with g >= 0 (never thinner than before; a first version without
      this fallback dropped to the mixture tails and made the deep wings worse). (B) local vol `G_FLOOR` 0.05 instead
      of 1e-3 (`gatheral_local_vol`, `LocalVolatilityModel`); diagnostics: `arbitrage_fraction` is now g < 0 or
      dw/dT <= 0, `floored_fraction` the cells below the floor. Live snapshot recalibrated. 2026-11-13, before / after:
      SX5E node smiles with g < 0.05 after 3M 4 / 0, local vol max at 50-165% of the forward after 3M 113% / 101%,
      ISP capped cells in the wings 3422 / 650; quotes within bid / ask unchanged on AAPL (97.9%), ENI, MSFT (100%),
      ISP (97.3%), SPX 78.5% / 77.8%, SX5E 96.0% / 94.2% (rms 0.057 / 0.075 vol pts, max 0.38 both); SX5E 3Y
      autocallable price -1.712 / -1.702, delta per 1% AAD 0.370, bumps 1% 0.369, 1e-5 0.370. Synthetic surfaces of
      the tests unchanged (model pins at 1e-8). Tests: `test_equity_vol.py` (junction moved inside the quotes with the
      margin and within the tolerance, fallback never thinner than the edge wing), `test_vanilla_mc.py` (a steep
      straight wing: local vol at the 200% cap with the former floor, bounded by sqrt(dw/dT / 0.05) now, floored cells
      reported). Left: thin densities inside the quotes (ROADMAP section 2 item 2).
- [x] Live (seasoned) autocallables (section 2 item 4). Market: `eq_fixings.csv` (`ticker, ccy, date, value`, the
      closes `PX_LAST` before the snapshot date, `mdbuild.history.equity_fixings`, written by every download; written
      into the 2026-11-13 snapshot from the price history, 2513-2567 dates a ticker) -> `EQ:<ccy>:<ticker>:FIXINGS`
      (`Fixings`, not a risk factor), `get_eq_fixings`, `get_eq_fixing` (raises on a missing date), optional in
      `from_frames` / `to_frames`. Product: `AutocallableOption.fixings` (`{date: value}`, a list per date for a
      worst-of), winning over the market's. `AutocallableMCPricer` (one underlying and worst-of): the strike date and
      the observations up to the evaluation date are known levels (a fixing on the evaluation date not in the data is
      the spot), run through the same payoff with exact indicators before the simulated dates: strike, coupons paid
      or missed (memory), autocall already triggered; payments before the evaluation date left out; nothing simulated
      when every observation is fixed. `trade_state` (strikes, per past observation levels, worst performance,
      coupon, autocall; `called_on`, `unpaid_coupons`). Also fixed: an observation on the evaluation date was dropped
      (neither simulated nor known); now the spot. Tradepad: a strike date before the pricing date is a live trade
      (strike from the fixings), schedule with the fixing (% of the strike) and the outcome of the past dates, status
      in the summary. 2026-11-13: SX5E 3Y memory phoenix struck 2025-11-13 at 5742.79: May 2026 at 102.1% (coupon
      paid), Nov 2026 at 111.1% (today, the spot): called, price 2.00 = the coupon paid today, no delta; SX5E / SPX
      worst-of struck 2025-05-13: called on 2025-11-17, price 0. Tests (`test_live_autocallable.py`, 10): numpy
      replica with past fixings (memory carried into the simulated dates), called in the past (the pending coupon
      exactly, then 0), seasoned trade = new trade on the remaining dates plus the pending coupon (same draws), trade
      fixings win and a missing one raises, fixing today = spot, no delta when every date is fixed, trade state,
      worst-of replica and fixings on the trade, market round trip; `marketdata` test of `equity_fixings`.
- [x] Tradepad on the autocallable: underlyings together in the deal (one by default, "+ Add underlying" up to 5, the
      spot under each), one currency (EUR by default, the flows paid and the price shown in it; the underlyings in
      another currency quanto, listed in the summary), no spot / vol editing and no single-underlying market block on
      the left; a Market Data tab next to Deal, filled by "Build trade" (per underlying spot, level at the strike date:
      forward / spot / historical fixing, forward and repo to maturity, dividends; the discount curves with their
      pillars); a Projections tab next to Risk after the price, computed on request from the paths kept with the
      pricing job (`AutocallableMCPricer.projections`, about 0.4 s): expected cash flows per payment date (coupon,
      redemption, put as a negative flow, % of the notional; above the chart the PV of the coupons and of the put), autocall probability per observation and cumulated, maturity (reached,
      knock-in, loss given knock-in, expected life), mean simulated levels with their 10%-90% band vs the market
      forwards (the worst underlying at maturity is computed by `projections` but not shown: the bands show the
      dispersion). 2026-11-13, SX5E / SPX / ISP 3Y
      worst-of: expected life 2.11 years, ISP the worst in 85% of the paths at maturity (83% of the knock-ins), SPX
      11%, SX5E 4%. SX5E / SPX 3Y worst-of
      in EUR, 20k paths: called at the first autocall 55%, by maturity 76%, knock-in 17.6% with a 50% loss; SPX
      (quanto, rho -0.31) mean 7845 vs forward 7832 after one year, SX5E on its forward. Test: expected flows
      discounted = the price (hard payoff), probabilities add up to one, GBM means = forwards within 3 SE, expected
      life = sum of probability x time, worst shares add up to one (the higher-vol underlying the worst more often).
- [x] Universe with an active flag, Create new snapshot (selected or all active, optional calibration). `universe.yaml`:
      `instruments` (ticker, `active`, optional `equity_index`, `implied_spot`, `option_root`, `models` = the
      calibration flags of the equity, moved from `calibration.yaml`, which keeps the defaults) and `curves` (name,
      ICVS ticker, `active`); `mdbuild.universe` reads it (and the former format: commented = inactive;
      `convert_universe`, the repository's file converted). Full download: the active ones; `python -m mdbuild --select
      "<equities>"`: those (active or not) with every active rate fixing, FX pair and curve, a complete snapshot of its
      own (`selected` in `snapshot_info.json`). tqstudio Home: the universe (Equity / Rates / FX; Select, Active, the
      model flags of the equities, status in the latest snapshot, Add ticker) and Create new snapshot (popup: only the
      selected equities or all active; Calibrate: `DownloadRuns.start(select, then)` starts the calibration of the new
      snapshot after a successful download); the universe view left the Market data page. A first version copied the
      latest snapshot of the day and the models of the other underlyings (dropped: every snapshot complete and
      independent). Real run on 2026-11-25 (Bloomberg, with that first version): ENI alone -> 20261125/1756 with FX,
      fixings and the 4 curves (curves 76 s, vols 16 s). Tests: `marketdata` (format, conversion, selection, selected
      snapshot recorded), `test_tqstudio_universe.py` (add / activate, deactivate, active flag of instruments and
      curves, the step after a successful download only).
- [x] Home: Documents first, then Reference data (the universe: Select, Active, Last download = the latest snapshot
      with the instrument, model flags; no status, spot or option columns); Heston off by default
      (`calibration.yaml`). Add ticker checks a new ticker on Bloomberg first (`python -m mdbuild.check`, with the
      download's Python: name, currency, security type; `verify_ticker`): not found, an equity index added under
      Rates or a fixing added as an equity index are rejected and the file is untouched, the terminal not reachable
      too (it cannot be checked); 2026-11-25: UCG IM Equity -> UNICREDIT SPA, EUR; ENII IM Equity -> not found.
- [x] Equity vols, very short expiries. Expiries less than 7 days away (`SHORT_DAYS`) keep only their quotes within
      90%-110% of the forward (`NARROW_RANGE`): narrow, their wings guided by the next complete expiry. The guide is
      scaled to the expiry's at-the-money vol by the vols of its scenarios, and by the spread of their forwards too
      when that spread alone is above it (SPX 3 days on 2026-11-13: 15.8% vs 15.2%, the guide was skipped before). A
      grid point with no implied vol in a smile (price 0) is extended along the wing before the PCHIP in time (it
      stopped the SOXX surface on 2026-11-25: `y` must contain only finite values). Real snapshots: SOXX 2-day row
      max vol 647% -> 202%, SPX short rows 180% -> 168% (3 expiries guided instead of 0), quotes within bid / ask
      unchanged (SPX 96.0% -> 95.9%, one quote), AAPL, ENI, ISP, MSFT, SX5E identical. Tests (`test_equity_vol.py`):
      a 5-day expiry quoted 60%-140% uses 90%-110% only and is guided, a later one keeps all its quotes; NaN far
      points of a short expiry (forced) give a finite surface, the same as without it after one year.
- [x] Calibration page: the Universe tab removed (the flags are in Reference data on the Home; the Heston SLV monitor
      kept in its own tab); the model test only on request (Run test), the stored results otherwise. Local vol in the
      calibration sets as a check (`LocalVol`, default on): `LocalVolatilityModel.calibrate` stores the diagnostics of
      the local vol built on the snapshot (arbitrage, thin density, largest local vol, share at the 200% cap) and
      `model_fit` reprices the nodes by Monte Carlo (as Heston SLV, about 3 s); the pricers still rebuild it on their
      market. Tradepad: the snapshot time next to the pricing date (every call on that snapshot). 2026-11-25 SOXX:
      local vol fit poor (rms 3.2 vol pts, max 7.7; arbitrage on 0.7% of the grid, 2.1% at the cap), displaced
      diffusion fair (1.1). Test: on a flat surface no arbitrage, floor or cap, nodes repriced within 0.5 vol pts,
      report round trip.
- [x] Local vol fit and the Monte Carlo grid. The fit of the calibration sets simulated weekly steps: a local vol
      builds a short smile only through small steps, so the 1-3 month wings came out low (SPX 2026-11-25, 50% strike:
      1 month -14.6 vol pts, 1.7 months -6.7, 3 months -4.9; daily steps: -8.3, -1.5, -0.2). `simulation_grid`
      (`models/brownian.py`): daily steps up to 3 months, weekly after, used by the fit and by
      `AutocallableMCPricer` (`fine_time_step`, `fine_horizon`; 0 = the former uniform grid, kept by the pins).
      Nodes with fewer than 30 paths expected beyond the strike are not measured (`MIN_PATHS_BEYOND`: NaN, grey in
      the page, out of the rms; e.g. SX5E 200% calls within a year at a zero vol). Set 20261125/1153: local vol
      SPX rms 0.50 -> 0.16 (max 2.6 -> 0.5), SX5E 0.30 -> 0.28 (1.4 -> 0.8), Heston SLV 0.56 -> 0.45. Left: expiries
      of a few days (SPX 2 days, 90%: -50 vol pts with daily steps and 100k paths), local vols beyond the 200% cap
      (Berestycki-Busca-Florent). SX5E 3Y quarterly phoenix: 1.175 -> 1.456 (SE 0.11, other draws), same time.
- [x] Tradepad vega: next to the map a table of the vega per expiry (sum of the row: every strike of that expiry moved
      together) and its running total.
- [x] Vega by expiry and skew (`analytics/vega.py`: `vega_by_expiry`, `skew_weights`): from the node vegas,
      level (row sum), cumulative, skew (the smile of the expiry rotated around its forward, +1 vol point per 10% of
      strike below it, -1 above, held beyond 50%-150%). Tradepad: Skew column and total next to the vega map.
      Notebook `autocallable_skew_sensy.ipynb` (3Y phoenix on TICKER1, local vol, 20k paths, same paths for the
      bumps): whole-surface level AAD -0.6734 vs central bump -0.6733 (1 pt), -0.6746 (3 pts); skew AAD -1.0206 vs
      central -1.0256 (0.5 pt), -1.0389 (1 pt), -1.1085 (2 pts), second-order part 0.10 at 1 pt; level by expiry
      within 8% at 1 pt (exact as the bump shrinks); skew of one expiry not linear (2Y: -0.571 at 1 pt, -0.637 at
      0.25, -0.737 at 0.05 vs AAD -0.759): one row rotated bends the surface in time. Test: Black-Scholes put, level
      and skew = central bumps (1e-4).
- [x] Tradepad equity vega: the map replaced by a table, one row per expiry of the surface (tenor and date) with its
      vega, then the total vega and the skew sensitivity (rate vol maps unchanged).
- [x] Tradepad equity vega by standard tenor (3M, 6M, 1Y, 2Y, ...): `vega_by_tenor` splits the vega of each expiry
      of the surface linearly in time between its two nearest tenors (total kept); cells shaded blue / red by sign.
- [x] Autocallable swap and the side of the autocallables. `side` on `AutocallableOption` (`SwapType.Receiver` client:
      receives the coupons, sells the put, the default; `Payer` issuer: the other sign, projected flows too).
      `AutocallableSwap` (`autocallable.py`): the autocallable plus a funding leg on `funding_curve` + `funding_margin`
      (bp), ACT/360 periods from the strike date to each coupon payment date. `AutocallableSwapPricer`: the rate not
      simulated, each period `(F + m) tau P(0, pay)` times the probability of being alive at its start (same paths and
      smoothed indicators: delta / vega / skew include the termination), funding flows in `projections()`. Tradepad:
      Autocallable Swap screen (funding index among the market curves, margin), Side on both, the legs in the info
      and the projections, the funding curve in Market Data and Rho. Tests (`test_autocallable_swap.py`): no autocall
      = the floating leg of a swap (`FloatingLegDiscounting`, 1.6e-15), called at the first date = the periods up to
      it, swap = autocallable - funding on the same paths, payer = -receiver (autocallable too), projections (funding
      flows discounted = funding leg, mirrored for the payer), rho on a 6M and an ESTR pillar (1e-4) and delta (3%) vs
      bumps. SX5E 3Y on 2026-11-25 (EUR 6M + 30 bp, local vol, 20k paths): client -9.42% = structured -2.36% +
      funding -7.06%.
- [x] Tradepad autocallable: the first autocall observation is the first schedule date from strike date + first
      observation on, on unadjusted dates (a 1Y date rolled from Sunday to Monday was dropped against the adjusted
      bound: 1Y -> 1.5Y with the defaults of 2026-11-25).
- [x] Tradepad correlations: the Market Data tab of the autocallables shows the underlyings, then the correlation
      matrix of the deal (`deal_correlations`: the underlyings and, for those in another currency than the payment
      one, the stored FX pair), editable for the pricing of the view only (an edit invalidates the price, not the
      trade): `_correlations_overridden` sets both entries of the matrix variable, checks the edited matrix is
      positive semi-definite and restores it (also on an error); the correlation sensitivities still from the tape.
      Dividends and curves out of that tab (in the Market Data page). Test (`test_tqstudio_tradepad_quanto.py`):
      worst-of matrix as stored, none with one factor, an override moves the price and leaves the matrix unchanged,
      out of (-1, 1) and not positive semi-definite raise.
- [x] Tradepad Market Data tab: the forward at maturity in the underlying's currency and, for a quanto one, the quanto
      forward in the payment currency's measure, F exp(-rho sigma_ATM sigma_X T), recomputed when the correlation is
      edited (SPX paid in EUR on 2026-11-25, 3Y: forward 8075.07, quanto 8160.50, local vol Monte Carlo mean 8161.23).
