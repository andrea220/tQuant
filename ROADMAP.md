# Roadmap

Open and future developments, in order of priority. Completed work: `DEVELOPMENT_LOG.md`.

## Order of work

1. **Market data: equity vols and dividends** (now). Index dividends from dividend futures, SPX dividends, the
   first download with the wider strike range, SPX implied spot, single-stock dividends and repo, the rest of the
   vol surface.
2. **Autocallable pricing** (Beta 1: local vol autocallables on the real snapshots). Terms as traded, worst-of,
   quanto, live trades, validation.
3. **After the beta: equity.** Guided review of the pricers and models, barriers, forward start / cliquets,
   variance swaps, total return swaps, multi-asset stochastic vol.
4. **tqstudio portfolio engine.** Model instances, Heston `eps(t)`, SLV table refresh, multi-asset process, the
   batch pipeline, persistence, trades and portfolios.
5. **Cross-asset simulation.**
6. **Future development.** Interest-rate market data, sensitivities, equity model issues, rates models.
7. **Bridge with Murex.**

## 1. Market data: equity vols and dividends (now)

Done (`DEVELOPMENT_LOG.md`, section 5): `marketdata/mdbuild` (Bloomberg universe edited from the app, price history,
snapshots with spots, dividends, listed options, FX, fixings, bootstrapped curves, correlations of weekly returns),
equity vol surfaces from the listed options (lognormal mixture per expiry with a vol floor, grid 30%-200% of the spot,
straight wings beyond the quotes, calendar step, settlement prices only without a bid / ask market, guided expiries
with a wing free of quotes), repo curves, index dividends implied with the repo held.

1. [ ] **Index dividends from dividend futures.** The Bloomberg projections (`BDVD_PROJ_DIV_INDX_PTS`) cover about
   3 years and the listed options with a bid / ask market about 2 (SX5E on 2026-11-13: quoted to Dec-2028); beyond,
   the dividends are projected at a constant yield with the repo held (SX5E 3.10%, about 202-210 points a year, where
   the settlement prices of the 2029-2032 options gave 155-168). Eurex dividend futures (SX5E, yearly to about 10
   years) give the market dividends of each year: download them, use them for the years they quote (repo implied
   from the options with those dividends), the constant yield only beyond.
2. [ ] **SPX implied dividends.** On 2026-11-13 the dividends implied by the listed forwards fell to about 55 points
   in 2031 (a financing spread growing with the maturity left in the dividends?). Check again with the settlement
   prices dropped (forwards from the quoted expiries only, constant yield beyond) and against the dividend futures.
3. [ ] **First download with the wider range** (targets 30%-200% of the forward, chains 25%-250% of the spot, chain
   cache version 4): how many quotes reach the wings, the wings and long expiries before / after, the Fit view.
4. [ ] **SPX implied spot.** The anchor expiry has zero repo by construction; a futures-based spot (ES) or a repo
   level from another source would fix the level of the curve.
5. [ ] **Single stocks: dividends and long repo.** The Bloomberg projections are used as downloaded and the repo
   absorbs the rest of the carry: check that the implied repo of the long quoted expiries is plausible (ENI and ISP
   to 2029, AAPL and MSFT to Jan-2029) and what a projection error does to it.
6. [ ] **Vol surface, the rest.**
   - Interpolation between the nodes: bilinear in strike, it cuts the curvature of the short index expiries (SPX
     79.5% of the two-sided quotes within bid / ask on the surface against 96% on the smile, 2026-11-13; the gap is
     near the money, on expiries under a month): total variance along the log-moneyness (natural cubic, as the local
     vol), or more nodes between 90% and 110%.
   - Calendar arbitrage left: ISP between the 7- and 14-day expiries at 30%-60% of the forward (up to 0.16 of total
     variance, the far wing of a guided 7-day smile), AAPL at 175%-200% between 2 and 4 months (0.02-0.03); far
     from anything priced. Bloomberg damps the default intensity below the quotes on short expiries (its
     14.1.1-14.1.2), not done.
7. [ ] **Option quotes of different times against one spot.** Each underlying has one quote date (the most frequent bid update date) and one reference spot (read in the same request as the options, or the close of the quote date): every option is converted to a vol against it, whatever the time of its own quote. Bid / ask of an earlier session within `stale_days` (3) enter the fit with full weight; older ones are dropped (settlement prices are used only without a bid / ask market); within the session an option whose bid has not moved for hours counts as live. Error: a stale price read on today's spot and the expiry's implied forward is off by about delta x spot move / vega in vol (small in the wings, up to a few vol points near the money on a 1-2% day); an expiry stale as a whole keeps consistent vols in moneyness (its parity forward is stale too) and puts the gap into the repo curve. On 2026-11-13 it is minor: the only earlier-session quotes in the fit are settlements (ISP 17, SX5E 1, spot moves of -0.5% / -0.3%), and the smile reprices the two-sided quotes within bid / ask (Fit view: SX5E, ENI 100%). Done: the time of the last bid update kept in `eq_option_quotes.csv` (`last_update_time`, from the next download) and shown in the Fit view (Quote Time). To decide on the measured ages: drop or down-weight the quotes older than a threshold when the expiry has fresher ones; a spot per quote (close of its session, strike rescaled in moneyness) only if earlier-session two-sided quotes become frequent; intraday spots per quote time (Bloomberg intraday bars) judged overkill.
8. [ ] Equity vols: a reference (OTC) surface for the long end and the wings when the internal service has one (the
   Bloomberg methodology blends it in; now the Heston-like ATM level); Kalman smoothing of the implied forwards
   across snapshots if intraday snapshots come (one a day now); calendar-arbitrage check of the stored grid
   (bilinear in vol between the tenors).
9. [ ] `BlackScholesPricer` / `bjerksund_stensland_calc`: NaN below about 1.5% vol and with a tiny positive carry
   (overflow), negative put prices at very low vols (immediate-exercise branch of the put symmetry). The option
   calibration avoids these regions (vols 2%-300%, zero-carry calls priced European); fix the pricer.

## 2. Autocallable pricing (Beta 1)

Goal: autocallables priced under local vol on the real snapshots, single underlying and worst-of, in their own or
another currency, new and live trades, with stable AAD greeks.

1. [ ] **Autocallable terms as traded.** Step-down autocall done (DEVELOPMENT_LOG). Left: fixed coupons (standard
   and memory done); final redemption variants (put without barrier, capital protection level), postponed: the
   standard payoff (put knocked in at expiry, participation) for now. And `GeometricBrownianMotion.from_equity` on multi-date
   products: it is the Black-Scholes model of one expiry (vol at `(K, T)`, spot escrowed of all the dividends up to
   `T`), so the path is the escrowed spot at the intermediate dates and `AutocallableMCPricer(model=GBM)` and
   discretely monitored barriers are wrong there (digital strip: first annual coupon 1.11% vs 2.62% Black-Scholes).
   GBM kept as an option for debugging, not the default (Tradepad "GBM (debug)"); later a GBM on the market
   forward (`S = F(t) X`) with a term-structure vol.
2. [ ] **Basket options** on the multi-asset Monte Carlo (worst-of done: DEVELOPMENT_LOG). Stochastic vol per asset:
   section 3.
3. [ ] **Quanto: what is left.** Done (DEVELOPMENT_LOG): `payment_ccy` on the products, quanto drift and payment
   curve for Black-Scholes (vanilla, digital, barrier), GBM and local vol (vanilla and autocallable Monte Carlo),
   historical FX vols in the snapshot, AAD greeks to the FX vols and the correlation, composite, price currency,
   Tradepad. Left:
   - FX implied vols from Bloomberg when the terminal is open (ATM / risk reversal by tenor, or BVOL, within the
     licence), instead of the historical ones; the correlation stays historical.
   - Quanto drift and composite in Heston, displaced diffusion and SLV (they raise now); composite barrier options.
   - Cross pairs (neither the pair nor its inverse stored: e.g. a GBP-paid product on a CHF underlying).
4. [ ] **Live trades: what is left.** Autocallables done (DEVELOPMENT_LOG: market fixings, trade state, Tradepad).
   Left: discretely monitored barrier options with past observations (knocked in / out already); composite products
   with past fixings (the FX fixings); the trade's confirmation as the source of the fixings when the contract fixes
   on something else than the close (average, auction price): today `fixings` on the trade.
5. [ ] **Validation of local vol and the autocallable.** The `local_vol` and `autocallable` steps of the guided review
   (the full plan: section 3). Local vol Monte Carlo noise is large in the wings and at long maturities (OTM call
   at 170%, 3Y: +-1 vol point with 100k paths): no variance reduction yet. For the beta: antithetic variates and a
   control variate on the forward if the greeks of a 5Y autocallable are not stable at 20k-50k paths.
   Also: autocallable prices and vegas on the new surfaces
   (grid 30%-200%) against the old ones, on SX5E and a single stock; local vol arbitrage warnings on the wider grid
   (SX5E, 1.7% of the local vol grid on 2026-11-19); a new calibration set on the recalibrated snapshot.

## 3. After the beta: equity

Equity (in order):

- [ ] **Guided review of the equity pricers and models, together, step by step** (before the new products): for each
      piece, walk through the code and its tests, fix what the review finds, and write the study material
      (`study/`, rules in `study/CLAUDE.md`): one notebook on the `marketdata/` snapshot, `\Theory` / `\Practice` Q/A
      in `study/docs/chapters/equity.tex`, 1-3 exercises, and the matching sections of the *Models and Pricing*
      (functional) and *Architecture* (technical) documents (`study/docs/models/`, `study/docs/architecture/`).
      Done (`DEVELOPMENT_LOG.md`): market handles, dividends, equity underlying, `black_scholes`, `monte_carlo_gbm`
      (and the digital / vanilla notebooks). Left, in order: `local_vol` (Gatheral, SSVI, forward mode: in section 2), `displaced_diffusion` (slice fit, skew averaging, implicit-function greeks, skew limit),
      `heston` (characteristic function, COS, calibration, QE-M), `heston_slv` (leverage, particle method,
      risk convention), `autocallable` (payoff, barrier smoothing, model comparison).
- [ ] **Barrier options, rest** (`BarrierOption` done, see `DEVELOPMENT_LOG.md`): rebates; Brownian-bridge vol from
      the model (now the surface vol at `(H, T)`, exact only for a flat-vol GBM); barrier prices under Heston / SLV
      compared with local vol (same smile, different barrier values); a `barrier` notebook with the greeks.
- [ ] **Forward-start and cliquet**: forward-start options (strike fixed at a future date as % of the spot then) and
      cliquets (sum of capped / floored periodic returns, local and global caps / floors). Forward smile products:
      Black-Scholes closed form for the forward start as the reference, prices under LV / Heston / SLV compared.
- [ ] **Variance swaps and Heston pricer**: variance swap fair strike by static replication on the smile (OTM strip,
      log-contract) with AAD vega per node; MC check (realised variance) under LV / Heston (Heston closed form for the
      expected variance). `HestonPricer` for `VanillaOption` (COS, calibrated Heston from the market) in the factory.
- [ ] **Total return swaps**: `TotalReturnSwap(Product)` (equity leg: price return + dividends, periodic resets,
      notional reset or fixed; funding leg: floating index + spread, reusing `flows/`), `TotalReturnSwapPricer`
      (equity leg from the market forwards / dividends / repo, funding leg as a floating leg on the curves). Tests:
      at-inception fair spread = repo vs funding basis, MtM after a fixing, deltas to spot / dividends / repo /
      pillars vs bumps.
- [ ] **Multi-asset with stochastic vol** (after the local vol worst-of, section 2): the `2n x 2n` correlation of spots and variances has to be completed
      from the asset-asset block and the per-asset spot-vol `rho_i` (Kahl-Günther: `<W_i, Z_j> = rho_ij rho_j`,
      `<Z_i, Z_j> = rho_ij rho_i rho_j`, positive semi-definite by construction but low variance-variance correlation;
      "perfect" completion `<Z_i, Z_j> = 1`, not always PSD; modified perfect with negative eigenvalues floored and
      renormalised). Reference: the IMI internal report `resources/HestonModel.pdf`, sec. 1.3.

## 4. tqstudio portfolio engine (the app runs; the portfolio engine is not started)

Goal: reprice multi-asset products and whole portfolios on a joint simulation of the risk factors, with greeks, in a
tractable way. Design agreed in the review of the equity models (Heston, SLV, the IMI Heston report
`resources/HestonModel.pdf`, literature on SLV in production); to start when the equity review is done.

**Principle.** `tensorquant` stays a stateless pricing library: it receives market data and calibrated objects and
returns prices and greeks; it never stores anything nor decides when to recalibrate. `tqstudio/` (new package in the
repo root, depends on `tensorquant`, never the reverse, with its own tests) is the orchestrator: universe of risk
factors, calibration policy, joint simulation, portfolio pricing, aggregation, persistence.

**1. Model instances separate from the market.** Done for the equity Monte Carlo models (`EquityModel`: `calibrate`,
`process(equity)`, Jacobian, `par_greeks`, `to_frames`; see `DEVELOPMENT_LOG.md`, section 4), and stored: the
calibration sets (`calibrate_snapshot_models`, `marketdata/calibrations/<source>/<yyyymmdd>/<hhmm>/`, apart from the
snapshots by design; Calibration tab). Left for `tqstudio`:
- the same split for `GaussianShortRateModel` (parameters + curve read at pricing) and a GBM on the market forward
  (see section 2, item 1);
- greeks at portfolio level: sum `dV/dp` over the trades of one underlying, one product with the stored Jacobian.

**2. Heston with `eps(t)` and two-speed calibration** (as the IMI model, sec. 1.2 and 4 of the report).
- `dX / X = eps(t) sqrt(v) dW`, `eps` piecewise constant on the maturities: ATM repriced exactly per maturity.
  Characteristic function with piecewise-constant coefficients (backward recursion of the Riccati solutions, report
  sec. 2.4); COS unchanged on the new `phi`; QE steps aligned with the `eps` nodes.
- Optional objective weighted by the Black vega of each option (`sum ((p_H - p_mkt) / vega_i)^2`, report eq. 21)
  besides the current ATM-vega weights.
- Two speeds: Heston parameters calibrated rarely (weekly or on a trigger: fit error above a threshold, structural
  move), `eps(t)` re-fitted before every pricing (one-dimensional root finds on ATM closed forms). Stable parameters,
  no spurious P&L from parameter jumps. Greeks consistent with the policy: model greeks through `eps` only; skew /
  wing risk separately (par greeks through the Jacobian, or bump and recalibrate).

**3. SLV with a stored conditional expectation.** Done for the single underlying (`DEVELOPMENT_LOG.md`: `E[v | X]`
calibrated on its own particles and stored, read at pricing, market greeks by the adjoint of its calibration), stored
in the calibration sets with its Heston parameters and mixing, with the recalibration rules of the Calibration tab
(age 7 days, spot 5%, surface 1 vol pt, Monte Carlo fit 1 vol pt). Left:
- refreshing the table alone (Heston parameters held) in a calibration set: today a recalibration redoes the whole
  model;
- estimation of `E` by the Fokker-Planck PDE of the joint density (Göttker-Schnetmann / Spanderen, QuantLib; van der
  Stoep et al.): also an independent check of the open short-dated SLV error (see Future development, Equity).

**4. Multi-asset process.**
- `MultiAssetProcess(StochasticProcess)`: N per-asset models (GBM, local vol, Heston, SLV with stored `E`), each with
  its `noise_size()`, driven by one correlation matrix of all the noises: spot-spot from the market
  (`CorrelationMatrix`), spot-variance `rho_i` from each Heston, variance-variance by completion (Kahl-Günther,
  perfect, modified perfect: see "Multi-asset with stochastic vol" above). Variance-variance correlations matter for worst-of
  autocallables (De Col and Kuppinger, Risk).
- QE: the spot shock contains a non-Gaussian part from the variance increment, so the inter-asset correlation imposed
  on the independent normals is not exactly the target one: measure simulated vs target correlation, correct if needed.
- Later: local correlation calibrated to basket / index smiles (Guyon, particle method) for the correlation skew.
- Tests: one asset = the single-asset model; independence and perfect-correlation limits; basket mean = weighted
  forwards; correlation gradients vs bumps.

**5. The `tqstudio` pipeline (per batch / date).** Partly there in the app: the universes (market data with add /
remove and the download from the app; models to calibrate per underlying), the snapshots loaded by date, the
calibration sets with their recalibration rules. Left: the batch itself (steps 4-6) and the portfolio.
1. Load the universe and configuration (underlyings, models per underlying, simulation dates, portfolio).
2. Load the market snapshot of the date (`MarketEnvironment.from_folder`).
3. Calibrations: load the valid ones or recalibrate (policy: age, fit error, structural moves; the fast parts such as
   `eps(t)` and local vol always rebuilt).
4. Simulate the risk factors jointly (`MultiAssetProcess`, stateless seeds).
5. Price the portfolio on the paths (closed forms where they exist, Monte Carlo / American Monte Carlo regression for
   future values); greeks by AAD to the model parameters per trade, summed over the portfolio, mapped to par greeks
   with the stored Jacobians (one product per underlying).
6. Aggregate (prices, greeks, exposures) and persist.

**6. Persistence.**
- Per batch: market snapshot id, calibrations used (with hash and version), configuration and seeds, aggregated
  results. Not the simulated cubes (paths x dates x factors: tens of GB on a large portfolio): the stateless seeds
  reproduce them exactly; save a cube only on request (audit), binary (parquet / zarr) in a git-ignored `runs/`.

**7. Trades and portfolios** (replicate the Murex portfolios: tens of trades).
- One Pydantic model per instrument template, on a common header (`trade_id`, `murex_id`, portfolio, book,
  template, status, trade date, currency, notional, buy / sell) plus the template's fields, nested where natural (the
  autocallable's coupon / final redemption / autocall sections and schedule, a swap's legs): a JSON document parses
  straight into a validated object. Each model builds the existing `tensorquant` `Product` (`build_product`), so the
  pricers do not change.
- Storage behind a small `TradeStore` (`load`, `save`, `get`, `query`), the rest of the code unaware of it: one JSON
  file per template (`{trade_id: trade}`, sorted keys, indented: readable git diffs), atomic writes (temporary file +
  rename) under the server lock, since all writes go through the `tqstudio` process. Versions (amendments,
  cancellations) as a history per template or through git. Move to SQLite (indexed header columns + JSON payload)
  only if volumes or concurrent writers grow: same models, another `TradeStore`.
- Tradepad: the form generated from the models' JSON schema (replacing the hand-written field lists), save / load /
  amend trades from the page, portfolio views on the store.
- Import from Murex: an extract (datamart report / CSV) mapped onto the templates; CSV only as the exchange format.
- Live (seasoned) trades: see section 2, item 4.

**Order of work.** 1 (calibration objects, pinned by tests) -> 2 (Heston `eps(t)`) -> 4 on GBM / local vol / Heston
-> 5-6 minimal `tqstudio` on two underlyings -> 3 (SLV with stored `E`, after the short-dated SLV error is
understood) -> local correlation.

## 5. Cross-asset (not started)

- [ ] **Cross-asset simulation model**: equities, rates and FX simulated jointly under one measure (the domestic
      risk-neutral one of a base currency), the generalisation of the equity "Multi-asset process" of the tqstudio
      engine below (item 4).
      - Rates: a `GaussianShortRateModel` (Hull-White / G2++) per currency on its curve; FX: lognormal or local vol with
        stochastic domestic / foreign rates (drift `r_d - r_f`), the foreign short rate under the domestic measure with
        the quanto drift `-rho(r_f, X) sigma_r sigma_X`; equities: the existing models on their forwards, with
        stochastic discounting in their currency and the quanto drift when foreign.
      - One correlation matrix of all the noises (equity-equity, equity-FX, rate-rate, rate-FX, equity-rate): from the
        historical log returns (`correlations.csv`, rate changes to add) or implied (quanto prices) where available;
        completed to positive semi-definite.
      - Consistency checks: every discounted asset a martingale under the domestic measure (bonds, FX forwards, equity
        forwards in domestic terms), FX forwards = covered interest parity, zero correlations = the single-asset models.
      - Uses: quanto and composite products in Monte Carlo, hybrids (equity / rates), portfolio exposures (EE, PFE) on
        mixed portfolios (the swap exposure generator extended), the tqstudio portfolio pipeline.

## 6. Future development (not prioritised)

Equity market data:

- [ ] **Equity vols: thin densities inside the quotes** (not a priority: the local vol floor covers them). Wing margin
  and local vol floor done (DEVELOPMENT_LOG). Left: a mixture with two separate bodies gives a thin density inside the
  quotes, which no wing rule touches (2026-11-13:
  ISP 3.1Y quoted only at 63-90% of the forward, `g` 0.01; ENI 0.84Y `g` 0.008 at 50%; a synthetic 2% crash body at
  30% of the forward keeps a spiky local vol delta up to a floor of about 0.2). To do: a density floor in the
  mixture fit (penalty on `g` below a margin over the quoted range), then measure the quotes within bid / ask.
  Short expiries: small calendar arbitrage (below 1e-4 in total variance) except ISP 1-2 weeks (-0.07).

App:

- [ ] Tradepad: the first price after the app starts is 2-3 s slower (TensorFlow warm-up, snapshot loading).

Interest-rate market data (after the equity and dividend work):

- [ ] Check the bootstrapped curves against the internal service once available.
- [ ] **Short end of the Ibor projection curves (synthetic deposits).** EUR 6M has no quote before its first pillar (the 6M fixing as a spot FRA 0x6, 2027-05-17 = 0.507Y on 2026-11-13): the first segment [0, 6M] is pure monotone convex interpolation, the instantaneous forward rises from 2.83% to a hump of 3.43-3.55% at 0.5-0.55Y and drops to 3.25% at 0.6Y (ESTR, quoted from 1W, has none). The quoted FRAs (1x7 to 12x18, monthly) are repriced exactly, so the 6M forwards at the FRA start dates are the market's; broken-date 6M fixings and the forward plot depend on the interpolation. Plan: synthetic deposits on the 6M curve at 1W, 1M, 2M, 3M, 4M, 5M with rate = ESTR simple rate on the same period + spread, spread = 6M fixing - ESTR 6M simple rate (about 57bp on 2026-11-13); not the 1M / 3M Euribor fixings (other indices, tenor basis 3M/6M of 10-15bp would put a step at 6M). Marked `synthetic` in `ir_calibration.csv`; their rates as functions of the ESTR curve and the 6M fixing, so that the par sensitivities map their risk to real quotes (ESTR members and the fixing), not to quotes that do not exist. Tests: quoted instruments still repriced to 0, no hump in the instantaneous forward below 1Y, par sensitivities of a short 6M-indexed product only on real quotes and equal to bump-and-rebootstrap.
- [ ] Par sensitivities in the portfolio views (done in the library and the Tradepad: see `DEVELOPMENT_LOG.md`).
- [ ] Vol surfaces of caps / swaptions and FX from the internal service (`mdbuild/internal.py`): no BVOL (equity
      vols and repo: from the listed options, done).

Sensitivities:

- [ ] **Calibration Jacobians, rest**: done for Heston and displaced diffusion (`EquityModel.jacobian`,
      `par_greeks`, equal to the AAD through the calibration: see `DEVELOPMENT_LOG.md`). Left: Hull-White / G2++
      (`calibrate_gaussian_short_rate`), the portfolio mapping (sum of `dV/dp` per underlying, one product), and the
      conditioning of the least-squares Jacobian (par greeks are a projection, not an exact hedge, and can be noisy when
      the Hessian is ill-conditioned, Heston `kappa` / `theta` / `xi`): optional regularisation.
- [ ] **Heston market vega by structure (IMI style)**: the par vega of a Heston price maps the model greeks through the
      whole-surface calibration, so a vanilla gets vega on every node (`equity_models`: 1Y 80% put, 0.04 on its own
      node of 0.69, spread over all maturities, -0.10 at 5Y), dependent on the fit weights and the option set: right
      for the model, misleading for a vanilla's risk. The IMI Heston (`resources/HestonModel.pdf`, sections 1.2 and 5)
      splits the parameters: a piecewise-constant `epsilon(t)` fitted exactly to the ATM vol of each expiry (moves only
      with that ATM vol) and time-independent `kappa, xi, rho, ...` recalibrated rarely and held (model greeks only, no
      market reading). Option: a Heston with that split, market vega only to the ATM vols up to the product's expiry.

Equity:

- [ ] **Degenerate Heston calibration on real data (ENI)**: in the calibration sets of 2026-11-13 / 16 (snapshot 20261113 1619) the Heston of ENI_IM ends with kappa = 0.016 and theta = 1.0, on its upper bound (long-run vol 100%), Feller violated: with kappa near 0 the variance hardly mean-reverts and only kappa * theta is identified, so the parameters are meaningless one by one even when the fit on the nodes looks acceptable. To check: the fit region and weights (short expiries, wings), the bounds and the start, the conditioning of the least squares (also behind noisy par greeks, see Calibration Jacobians); a flag in the calibration sets when a parameter ends on a bound (the rates calibration already reports `at_bound`).
- [ ] Heston SLV does not reprice the short-dated downside of the surface: on the snapshot (TICKER1, mixing 1) the 6M
      smile is too high by about +0.7 vol points at 60% and +0.4 at 70% (+0.5 at 60% at 1Y, gone at 3Y), against
      local vol on the same draws at ~0. Systematic, not Monte Carlo noise: unchanged from 80k to 320k particles,
      with 400 smoother nodes instead of 100, without the shrinkage, with 200 steps a year instead of 50, without the
      leverage cap and the variance floor; it disappears as the mixing goes to 0 (SLV = local vol exactly), so the
      log step and the leverage are applied correctly and the error comes with the stochastic variance. Cause not
      identified (candidates: the leverage frozen at the step start while `E[v | X]` moves fast on the short
      downside, the QE log step with a per-particle leverage, the interplay with Feller strongly violated). Probably
      also behind the SLV vs LV gaps in `digital_strip` and `down_and_in_put`. Study in `study/notebooks/heston_slv.ipynb`
      (section 6).
- [ ] **SLV greeks by smoothing for risk (interim, to discuss)**: the price comes from the Richardson-corrected
      `E[v | X]` table, the greeks from the 41-node one (the adjoint of the corrected or of a finer estimator blows
      up: `DEVELOPMENT_LOG.md`). The greeks are those of a slightly smoother model: within 3-5% of the bumps on the
      node and the parallel vega, up to 0.05 per vol point off the node, and a second path simulation per price
      (~+40%). Options: the leverage by the Fokker-Planck PDE (deterministic and smooth calibration: exact adjoint,
      no bias / noise trade-off; see part 3 of `tqstudio`), or bump-and-recalibrate with common random numbers by
      maturity bucket (desk practice). To decide after studying the model.
- [ ] **SLV 5Y vega off the node**: the adjoint par vega of a 5Y 80% put has +-0.1-0.2 per vol point on the 3Y-5Y
      wings (130%-140%) besides 1.03 on its node (total 1.50): not checked against bumps node by node (the tests check
      the node, a neighbour and the parallel shift). Bump those nodes to tell noise from model.
- [ ] Heston COS truncation range from the cumulants (`heston_cumulants`, mean +/- 30 sd, `abs(c2)` as a guard): with
      Feller strongly violated the second cumulant can come out negative or inaccurate (Oosterlee-Grzelak, ch. 6, use a
      fixed range `+/- L sqrt(T)` for this reason). Fine on the snapshot (Feller -0.48, `c2 = 1.12` at 5Y, prices = the
      book's puts + parity to 1e-6), but not checked on extreme parameters: test a range of parameter sets against a
      fixed-range COS / Lewis quadrature, fall back to the fixed range if `c2` is not positive.

Interest rates:

- [ ] Caps on overnight rates (compounded SOFR / SONIA / €STR caplets) and other Ibor indices: only EUR 6M now.
- [ ] Gaussian models give a (nearly) flat normal smile: the synthetic SABR smile on top of the G2++ ATM vols is not
      fitted (only ATM). A smile needs another model (e.g. SABR / Cheyette with local vol).
- [ ] Exact swaptions for one or two factors only (Jamshidian / Brigo-Mercurio); piecewise sigma(t) for Hull-White only
      in the calibration (the model accepts it for any number of factors).
- [ ] Swaptions: physical settlement only (EUR market swaptions are cash-settled with the IRR annuity); no Bermudans.
      Cap quotes are not part of the snapshot (only caplet vols); a market data file for them if needed.

## 7. Bridge with Murex (to be defined, not started)

Link tQuant to Murex to check the front-office system day by day and reprice its portfolios independently. Data
flow Murex -> tQuant only (extracts, datamart reports); tQuant never writes into Murex. Documented as a placeholder in
`study/docs/architecture/murex.tex`.

- [ ] **Daily market data checks**: Murex market data of the day (curves, equity spots, dividends, repo, FX) against
      the tQuant snapshot built from Bloomberg, tolerances per data type, report of the differences.
- [ ] **Vol surfaces**: Murex equity and rates vol surfaces against the tQuant surfaces (listed options, rate vols),
      node by node in vol points at fixed tenors and moneyness.
- [ ] **Calibrations**: Murex model parameters (Heston, local vol, Hull-White, ...) against the tQuant calibration
      sets, and the fit of each on the same surface.
- [ ] **Portfolio repricing**: Murex trades (extract mapped onto the tQuant instrument templates, see the trades and
      portfolios item of the tqstudio engine above) repriced on the same market; prices and greeks compared trade by
      trade and by book, differences explained by market data, model or conventions.
- To define: the Murex extracts available (format, frequency, scope), the mapping of the Murex templates and market
  data keys onto the `MarketEnvironment` keys, the tolerances, where the daily reports run and are published
  (tqstudio).
