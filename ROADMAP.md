# Roadmap

Open and future developments (moved from `CLAUDE.md`). Completed work: `DEVELOPMENT_LOG.md`.

## 3. Interest rates (analytic pricers)

- [ ] Caplet vol stripping from market cap quotes (flat cap vols by maturity × strike): today the caplet vols are
      stored directly (`IR:EUR:6M:VOL`).
- [ ] Caps on overnight rates (compounded SOFR / SONIA / €STR caplets) and other Ibor indices: only EUR 6M now.

## 4. Stochastic processes & Monte Carlo (`models/`, MC pricers, `analytics/`)

- [ ] Calibration of the Gaussian short rate models to rate vols (swaptions/caps): later (no swaption pricer).
- [ ] Local vol MC noise is large in the wings and at long maturities (OTM call at 170%, 3Y: ±1 vol point with
      100k paths): variance reduction (antithetics, control variate on the forward) not done.

## Future development (not prioritised)

- [ ] `DisplacedDiffusionModel` (not needed for now): float64, common `evolve`, but not wired to the market: built from floats (S0, flat r, q)
      and calibrated in numpy/scipy (per-maturity root finding of beta, sigma on the smile), so no `from_equity`, no
      AAD to the market, not usable by the MC pricers. To do: `from_equity` on the market forward (as local vol) and a
      differentiable calibration (implicit function theorem on the calibration equations, or a TF solver).
