# Roadmap

Open and future developments (moved from `CLAUDE.md`). Completed work: `DEVELOPMENT_LOG.md`.

Current focus: equity.

## Equity models (next)

- [ ] **Heston** (`models/heston.py`, `HestonModel(StochasticProcess)`), on the market forward like local vol and
      displaced diffusion: `S = F(t) X`, `dX / X = sqrt(v) dW_1`, `dv = kappa (theta - v) dt + xi sqrt(v) dW_2`,
      `d<W_1, W_2> = rho dt` (discrete dividends as proportional drops).
  - closed form: characteristic function in the stable ("little trap") form, European prices by a Fourier method
    in TF (COS or Lewis), float64, differentiable w.r.t. the parameters, forward and discount;
  - calibration to the equity vol surface (`from_equity`): least squares on vega-weighted prices, AAD Jacobian,
    bounds, Feller condition reported; sensitivities to the vol nodes through the calibration (implicit function
    theorem, as the displaced diffusion);
  - Monte Carlo: Andersen QE for the variance (+ martingale correction for X), two Gaussian draws per step:
    `VanillaMCPricer` / `AutocallableMCPricer` to draw `[n_paths, n_steps, n_factors]` noise for 2-factor models;
  - tests: Black-Scholes limit (xi -> 0), put-call parity, closed form vs MC, own parameters recovered, greeks vs
    bumps, MC forward = market forward.
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
