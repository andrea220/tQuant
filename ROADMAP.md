# Roadmap

Open and future developments (moved from `CLAUDE.md`). Completed work: `DEVELOPMENT_LOG.md`.

## 3. Interest rates (analytic pricers)

- [ ] Caps on overnight rates (compounded SOFR / SONIA / €STR caplets) and other Ibor indices: only EUR 6M now.

## 4. Stochastic processes & Monte Carlo (`models/`, MC pricers, `analytics/`)

- [ ] Gaussian short rate calibration to caplets only is under-determined for G2++ (factor correlation ends at the
      bound, a at 1e-4 on the synthetic market): swaptions (pricer + vols) would pin the correlation. Hull-White with
      constant (a, sigma) fits only the ATM term structure (±6.5bp on the SABR market); time-dependent sigma(t) would
      fit it exactly. Cap quotes are not part of the snapshot (only caplet vols); a market data file for them if needed.
- [ ] Local vol MC noise is large in the wings and at long maturities (OTM call at 170%, 3Y: ±1 vol point with
      100k paths): variance reduction (antithetics, control variate on the forward) not done.

## Future development (not prioritised)

- [ ] `DisplacedDiffusionModel` (not needed for now): float64, common `evolve`, but not wired to the market: built from floats (S0, flat r, q)
      and calibrated in numpy/scipy (per-maturity root finding of beta, sigma on the smile), so no `from_equity`, no
      AAD to the market, not usable by the MC pricers. To do: `from_equity` on the market forward (as local vol) and a
      differentiable calibration (implicit function theorem on the calibration equations, or a TF solver).
