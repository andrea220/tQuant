# Roadmap

Open and future developments (moved from `CLAUDE.md`). Completed work: `DEVELOPMENT_LOG.md`.

Current focus: equity.

## Equity models (next)

- [ ] **Heston SLV** (stochastic local vol): `dX / X = L(t, X) sqrt(v) dW_1` with the Heston variance (vol of vol
      possibly scaled by a mixing fraction) and the leverage `L^2(t, x) = sigma_LV^2(t, x) / E[v | X = x]`
      (`sigma_LV` from `gatheral_local_vol`, conditional expectation by binning / regression in a calibration MC
      pass, leverage on the local vol grid). Reprices the vanilla smile like local vol, Heston dynamics for
      forward-start / path-dependent payoffs. Tests: smile repriced by MC, limits (L = 1 → Heston, xi → 0 → local
      vol), autocallable vs local vol / Heston.

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
