# tQuant

tQuant is a *work in progress* Python financial library, it leverages on Tensor arrays for dealing with intensive risk management computations and algorithmic differentiation. 


## Dependencies

- TensorFlow
- Pandas

## Examples

* black_scholes.ipynb: Montecarlo performances vs QuantLib.

* legs.ipynb: Pricing of single fixed and floating legs. 

* simple_shortrate.ipynb: a simplified version of Tensor-Montecarlo simulation for the Hull and White model.

* swap_exposure.ipynb: **under construction**, Tensor-pricing of multiple fixed and floating legs.

<!-- ROADMAP -->
## Roadmap
### 2024
- [x] Add simple fixed leg handles (01/2024)
- [x] Add simple index and floating leg handles (01/2024)
- [x] Add Swap Instrument (02/2024)
- [x] Add calendar class (02/2024)
- [ ] Add daycounting class
- [x] Include ibor indices (9/02/2024)
- [x] Create market data handles (10/02/2024)
- [x] Create portfolio data handles (10/02/2024)
- [ ] Add curve bootstrapping
    - [ ] curve instrument helpers
    - [ ] market/zero rates Jacobian
- [ ] Architecture design (pricing engines, data, indices)
- [ ] Intermediate pricing examples
    - [ ] swap portfolio pricing 
    - [ ] swap AAD zero-sensitivity
    - [ ] Validation and fine tuning vs QuantLib 
- [ ] Fine tuning of "simple" objects and documentation.
- [ ] Include cds template for pricing
- [ ] Include inflation swap an zc inflation prices
- [ ] Include convexity adjustments for interest rate indices
- [ ] Include pricing engines for all vanilla products
- [ ] Final pricing examples
    - [ ] portfolio pricing 
    - [ ] AAD zero-sensitivity
    - [ ] Validation and fine tuning vs QuantLib / Production




