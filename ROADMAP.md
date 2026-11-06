# Roadmap

Open and future developments (moved from `CLAUDE.md`). Completed work: `DEVELOPMENT_LOG.md`.

Current focus: equity.

## 4. Stochastic processes & Monte Carlo (`models/`, MC pricers, `analytics/`)

- [ ] `DisplacedDiffusionModel` cannot reach equity skews steeper than the normal one (about -sigma/2 per unit
      log-moneyness): on the synthetic SSVI smiles beta ends on its lower bound up to ~1Y and the wings are off
      (13 vol points at 1M, < 1 at 5Y; ATM within ~1.7). With time-dependent beta the MC reprices the slices up to
      the skew averaging (~0.2 vol points). A steeper-skew model (stochastic vol, SLV) if needed; local vol for smiles.
- [ ] Local vol MC noise is large in the wings and at long maturities (OTM call at 170%, 3Y: ±1 vol point with
      100k paths): variance reduction (antithetics, control variate on the forward) not done.

## Future development (not prioritised)

Interest rates:

- [ ] Caps on overnight rates (compounded SOFR / SONIA / €STR caplets) and other Ibor indices: only EUR 6M now.
- [ ] Gaussian models give a (nearly) flat normal smile: the synthetic SABR smile on top of the G2++ ATM vols is not
      fitted (only ATM). A smile needs another model (e.g. SABR / Cheyette with local vol).
- [ ] Exact swaptions for one or two factors only (Jamshidian / Brigo-Mercurio); piecewise sigma(t) for Hull-White only
      in the calibration (the model accepts it for any number of factors).
- [ ] Swaptions: physical settlement only (EUR market swaptions are cash-settled with the IRR annuity); no Bermudans.
      Cap quotes are not part of the snapshot (only caplet vols); a market data file for them if needed.
