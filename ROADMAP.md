# Roadmap

Open and future developments (moved from `CLAUDE.md`). Completed work: `DEVELOPMENT_LOG.md`.

Current focus: equity.

## Equity (next, in order)

- [ ] **Guided review of the equity pricers and models, together, step by step** (before the new products): for each
      piece, walk through the code and its tests, fix what the review finds, and write the study material
      (`study/`, rules in `study/CLAUDE.md`): one notebook on the `marketdata/` snapshot, `\Theory` / `\Practice` Q/A
      in `study/docs/chapters/equity.tex`, 1-3 exercises, and the matching sections of the *Models and Pricing*
      (functional) and *Architecture* (technical) documents (`study/docs/models/`, `study/docs/architecture/`).
      Proposed order (after `black_scholes`):
      `monte_carlo_gbm` (`VanillaMCPricer`, AAD greeks, common random numbers), `local_vol` (Gatheral, SSVI,
      forward mode), `displaced_diffusion` (slice fit, skew averaging, implicit-function greeks, skew limit),
      `heston` (characteristic function, COS, calibration, QE-M), `heston_slv` (leverage, particle method,
      risk convention), `autocallable` (payoff, barrier smoothing, model comparison).
- [ ] **Barrier options, rest** (`BarrierOption` done, see `DEVELOPMENT_LOG.md`): rebates; Brownian-bridge vol from
      the model (now the surface vol at `(H, T)`, exact only for a flat-vol GBM); barrier prices under Heston / SLV
      compared with local vol (same smile, different barrier values); a `barrier` notebook with the greeks.
- [ ] **`GeometricBrownianMotion.from_equity` on multi-date products**: it is the Black-Scholes model of one expiry
      (vol at `(K, T)`, spot escrowed of all the dividends up to `T`), so the path is the escrowed spot at the
      intermediate dates: `AutocallableMCPricer(model=GeometricBrownianMotion)` and discretely monitored barriers are
      wrong there (digital strip: first annual coupon 1.11% vs 2.62% Black-Scholes). Fix: a GBM on the market forward
      (`S = F(t) X`, like the other models) with a term-structure vol, or reject GBM for multi-date products.
- [ ] **Multi-asset Monte Carlo**: correlated underlyings on the market forwards (GBM / local vol per asset, Cholesky of
      `market_env.get_correlation`), a multi-asset `evolve`; basket options and the worst-of autocallable
      (`AutocallableOption` on several underlyings). Greeks by AAD per underlying (delta, vega per node) and on the
      correlation pairs. Tests: one asset = single-asset pricers, perfect correlation / independence limits, basket
      mean = weighted forwards, correlation gradient vs bumps.
      Stochastic vol per asset (Heston / SLV): the `2n x 2n` correlation of spots and variances has to be completed
      from the asset-asset block and the per-asset spot-vol `rho_i` (Kahl-Günther: `<W_i, Z_j> = rho_ij rho_j`,
      `<Z_i, Z_j> = rho_ij rho_i rho_j`, positive semi-definite by construction but low variance-variance correlation;
      "perfect" completion `<Z_i, Z_j> = 1`, not always PSD; modified perfect with negative eigenvalues floored and
      renormalised). Reference: the IMI internal report `resources/HestonModel.pdf`, sec. 1.3.
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

## tqstudio: cross-asset risk engine (design, not started)

Goal: reprice multi-asset products and whole portfolios on a joint simulation of the risk factors, with greeks, in a
tractable way. Design agreed in the review of the equity models (Heston, SLV, the IMI Heston report
`resources/HestonModel.pdf`, literature on SLV in production); to start when the equity review is done.

**Principle.** `tensorquant` stays a stateless pricing library: it receives market data and calibrated objects and
returns prices and greeks; it never stores anything nor decides when to recalibrate. `tqstudio/` (new package in the
repo root, depends on `tensorquant`, never the reverse, with its own tests) is the orchestrator: universe of risk
factors, calibration policy, joint simulation, portfolio pricing, aggregation, persistence.

**1. Model instances separate from the market.** Done for the equity Monte Carlo models (`EquityModel`: `calibrate`,
`process(equity)`, Jacobian, `par_greeks`, `to_frames`; see `DEVELOPMENT_LOG.md`, section 4). Left for `tqstudio`:
- the calibrated models in the snapshot layout (`MarketEnvironment.to_folder` / `from_folder`: one more folder per
  date and `calibration_set`, written from `to_frames`);
- the same split for `GaussianShortRateModel` (parameters + curve read at pricing) and a GBM on the market forward
  (see the GBM item above);
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
calibrated on its own particles and stored, read at pricing, market greeks by the adjoint of its calibration). Left:
- the table in the snapshots (with the Heston parameters and the mixing it was estimated with), refreshed daily or
  on a trigger;
- estimation of `E` by the Fokker-Planck PDE of the joint density (Göttker-Schnetmann / Spanderen, QuantLib; van der
  Stoep et al.): also an independent check of the open short-dated SLV error (see Future development, Equity).

**4. Multi-asset process.**
- `MultiAssetProcess(StochasticProcess)`: N per-asset models (GBM, local vol, Heston, SLV with stored `E`), each with
  its `noise_size()`, driven by one correlation matrix of all the noises: spot-spot from the market
  (`CorrelationMatrix`), spot-variance `rho_i` from each Heston, variance-variance by completion (Kahl-Günther,
  perfect, modified perfect: see "Multi-asset Monte Carlo" above). Variance-variance correlations matter for worst-of
  autocallables (De Col and Kuppinger, Risk).
- QE: the spot shock contains a non-Gaussian part from the variance increment, so the inter-asset correlation imposed
  on the independent normals is not exactly the target one: measure simulated vs target correlation, correct if needed.
- Later: local correlation calibrated to basket / index smiles (Guyon, particle method) for the correlation skew.
- Tests: one asset = the single-asset model; independence and perfect-correlation limits; basket mean = weighted
  forwards; correlation gradients vs bumps.

**5. The `tqstudio` pipeline (per batch / date).**
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
- Live (seasoned) trades: historical fixings (underlying, date, value) and the trade state (initial fixing, past
  coupon observations, memory of the unpaid coupons, autocalls already triggered) used by the pricers; today they
  price every trade as if it started on the pricing date.

**Order of work.** 1 (calibration objects, pinned by tests) -> 2 (Heston `eps(t)`) -> 4 on GBM / local vol / Heston
-> 5-6 minimal `tqstudio` on two underlyings -> 3 (SLV with stored `E`, after the short-dated SLV error is
understood) -> local correlation.

## Real market data (`marketdata/`, started)

Done: `marketdata/mdbuild` (self-contained, own `requirements.txt`): universe of Bloomberg tickers
(`config/universe.yaml`), Desktop API source, 10-year split-adjusted price history in Parquet (incremental),
snapshots under `marketdata/live/` (git-ignored) with equity spots, Bloomberg dividend projections, FX spots, rate
fixings and correlations of daily log returns. Left:
- [x] Rate curves from Bloomberg: the snapshot stores the ICVS curve members (`ir_quotes.csv`: EUR ESTR S514,
      EUR 6M S45, USD SOFR S490, GBP SONIA S141, mid quotes) and tQuant bootstraps them on load
      (`bootstrap_from_quotes`, market conventions; every instrument repriced to 1e-15 on 2026-11-12).
- [x] Curves bootstrapped right after the download (`mdbuild` calls tQuant `bootstrap_snapshot` when installed):
      `ir_curves.csv` plus the Jacobian (`ir_curves_jacobian.csv`) and the quote DV01s (`ir_calibration.csv`) for the
      par sensitivities; loading a snapshot 0.5 s instead of 63 s. Calendars `UnitedStates` (Fed) and
      `UnitedKingdom` for SOFR / SONIA.
- [ ] Check the bootstrapped curves against the internal service once available.
- [ ] Par sensitivities from the stored Jacobian: `dV/dq = dV/dr . (-J^-1) . diag(dNPV_i/dq_i)` per curve, in the
      Tradepad and the portfolio views.
- [ ] Vol surfaces (equity, caps / swaptions, FX) from the internal service (`mdbuild/internal.py`): no BVOL.
- [ ] Repo: no source yet (empty, i.e. zero); dividend pay dates: the projections have none (pay date = ex-date).
- [ ] Correlations: daily closes of different markets are asynchronous (Milan vs New York closes): same-day
      returns understate the cross-market correlation; weekly returns or lagged terms as an option. Window choice:
      on the first snapshot (2026-11-12) ENI is negatively correlated with the Italian banks over the last year
      (-0.3) after ten years at +0.3 / +0.8: a regime the 252-day window takes in full.
- [ ] tQuant / tqstudio reading the live snapshots once curves and vols are there (they read the synthetic ones now).

## Future development (not prioritised)

Sensitivities:

- [ ] **Par sensitivities of the curves**: `CurveBootstrap.strip` stores `dNPV_i/dz_j` (`curve.jacobian`) and
      `dNPV_k/dq_k` (`curve.calib_quote_dv01`) but nothing combines them. Par sensy `s_q = s_z dz/dq` with
      `dz/dq = -(dNPV/dz)^-1 dNPV/dq` from the zero sensy `s_z` (rho per pillar, AAD). Test: par sensy of a calibration
      instrument = its own quote DV01 on itself and 0 on the others; sum of par sensies vs a parallel quote bump.
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

- [ ] Local vol MC noise is large in the wings and at long maturities (OTM call at 170%, 3Y: ±1 vol point with
      100k paths): variance reduction (antithetics, control variate on the forward) not done.
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
