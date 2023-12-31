# TensorQuant

TensorQuant is a *work in progress* Python financial library, it leverages on Tensor arrays for dealing with intensive risk management computations and algorithmic differentiation. 


## Dependencies

- TensorFlow

## Examples

* black_scholes.ipynb: Montecarlo performances vs QuantLib.

* legs.ipynb: Pricing of single fixed and floating legs. 

* simple_shortrate.ipynb: a simplified version of Tensor-Montecarlo simulation for the Hull and White model.

* swap_exposure.ipynb: **under construction**, Tensor-pricing of multiple fixed and floating legs.

<!-- ROADMAP -->
## Roadmap

- [x] Add simple fixed leg handles
- [x] Add simple index and floating leg handles
- [ ] Add calendars and daycounting
- [ ] Include convexity adjustments for interest rate indices
- [ ] Fine tuning of "simple" objects.
- [ ] Add curve bootstrapping
    - [ ] curve instrument helpers
    - [ ] market/zero rates Jacobian
- [ ] Add instrument templates and common pricing engines
- [ ] Add CrossAsset simulation kernel



