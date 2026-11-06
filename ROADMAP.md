# Roadmap

Open and future developments (moved from `CLAUDE.md`). Completed work: `DEVELOPMENT_LOG.md`.

Current focus: equity.

## Equity (next, in order)

- [ ] **Guided review of the equity pricers and models, together, step by step** (before the new products): for each
      piece, walk through the code and its tests, fix what the review finds, and write the study material
      (`study/`, rules in `study/CLAUDE.md`): one notebook on the `marketdata/` snapshot, `\Theory` / `\Practice` Q/A
      in `study/docs/chapters/equity.tex`, 1-3 exercises. Proposed order (notebooks after `04_black_scholes`):
      `05_monte_carlo_gbm` (`VanillaMCPricer`, AAD greeks, common random numbers), `06_local_vol` (Gatheral, SSVI,
      forward mode), `07_displaced_diffusion` (slice fit, skew averaging, implicit-function greeks, skew limit),
      `08_heston` (characteristic function, COS, calibration, QE-M), `09_heston_slv` (leverage, particle method,
      risk convention), `10_autocallable` (payoff, barrier smoothing, model comparison).
- [ ] **Barrier options**: `BarrierOption(Product)` (up/down, in/out, rebate, discrete monitoring dates or
      continuous), priced by the MC pricers with a Brownian-bridge crossing probability between steps (continuous
      monitoring, smooth: AAD-friendly) or on the monitoring dates; closed form under Black-Scholes (Reiner-Rubinstein,
      Broadie-Glasserman-Kou shift for discrete monitoring) as the test reference and an analytic pricer for flat vol.
      Tests: in + out = vanilla, MC (GBM) = closed form, monitoring frequency → continuous, greeks vs bumps, prices
      under LV / Heston / SLV (same smile, different barrier values).
- [ ] **Multi-asset Monte Carlo**: correlated underlyings on the market forwards (GBM / local vol per asset, Cholesky of
      `market_env.get_correlation`), a multi-asset `evolve`; basket options and the worst-of autocallable
      (`AutocallableOption` on several underlyings). Greeks by AAD per underlying (delta, vega per node) and on the
      correlation pairs. Tests: one asset = single-asset pricers, perfect correlation / independence limits, basket
      mean = weighted forwards, correlation gradient vs bumps.
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

## Future development (not prioritised)

Equity:

- [ ] Local vol MC noise is large in the wings and at long maturities (OTM call at 170%, 3Y: ±1 vol point with
      100k paths): variance reduction (antithetics, control variate on the forward) not done.

Interest rates:

- [ ] Caps on overnight rates (compounded SOFR / SONIA / €STR caplets) and other Ibor indices: only EUR 6M now.
- [ ] Gaussian models give a (nearly) flat normal smile: the synthetic SABR smile on top of the G2++ ATM vols is not
      fitted (only ATM). A smile needs another model (e.g. SABR / Cheyette with local vol).
- [ ] Exact swaptions for one or two factors only (Jamshidian / Brigo-Mercurio); piecewise sigma(t) for Hull-White only
      in the calibration (the model accepts it for any number of factors).
- [ ] Swaptions: physical settlement only (EUR market swaptions are cash-settled with the IRR annuity); no Bermudans.
      Cap quotes are not part of the snapshot (only caplet vols); a market data file for them if needed.
