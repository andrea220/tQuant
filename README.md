# tQuant

**tQuant** is a *work-in-progress* Python financial library that leverages Tensor arrays for intensive risk management computations and algorithmic differentiation.

---

## 📋 Dependencies

- **TensorFlow**
- **Pandas**

---

## 🛠️ Examples

- **`black_scholes.ipynb`**: Monte Carlo performances vs QuantLib.
- **`legs.ipynb`**: Pricing of single fixed and floating legs.
- **`simple_shortrate.ipynb`**: A simplified version of Tensor-Monte Carlo simulation for the Hull and White model.
- **`swap_exposure.ipynb`**: **Under Construction** - Tensor-pricing of multiple fixed and floating legs.

---

## 🛤️ Roadmap

### 2024

- [x] **Add simple fixed leg handles** (01/2024)
- [x] **Add simple index and floating leg handles** (01/2024)
- [x] **Add Swap Instrument** (02/2024)
- [x] **Add calendar class** (02/2024)
- [ ] Add daycounting class
- [x] Include ibor indices (09/02/2024)
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
- [ ] Fine tuning of "simple" objects and documentation
- [ ] Include cds template for pricing
- [ ] Include inflation swap and zc inflation prices
- [ ] Include convexity adjustments for interest rate indices
- [ ] Include pricing engines for all vanilla products
- [ ] Final pricing examples
  - [ ] portfolio pricing
  - [ ] AAD zero-sensitivity
  - [ ] Validation and fine tuning vs QuantLib / Production
