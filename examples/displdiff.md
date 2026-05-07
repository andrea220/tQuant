# Implementazione e calibrazione del modello Displaced Diffusion

## 1. Obiettivo del modello

Il modello **Displaced Diffusion** o **Shifted Lognormal** è un modello diffusivo pensato per incorporare in modo semplice lo **skew di volatilità implicita** osservato sul mercato.

L’idea principale è sostituire il processo lognormale standard di Black-Scholes con un processo lognormale applicato non direttamente al sottostante, ma a una variabile traslata:

\[
S_t + a_t e^{rt}
\]

dove \(a_t\) è il parametro di displacement/shift.

Il modello è utile perché:

- genera skew implicito in modo parsimonioso;
- mantiene una formula analitica per opzioni vanilla europee;
- può essere calibrato rapidamente;
- può essere esteso al caso multi-asset;
- è pratico per pricing Monte Carlo di payoff esotici o basket.

---

## 2. Dinamica del modello

### 2.1 Processo driftless

Si parte dal processo:

\[
dX_t = [\beta_t X_t + (1-\beta_t)X_0]\sigma_t dZ_t
\]

dove:

- \(X_t\) è il sottostante driftless;
- \(\beta_t\) è il parametro di skewness;
- \(\sigma_t\) è la volatilità istantanea;
- \(dZ_t\) è un moto browniano.

La dinamica può essere riscritta come:

\[
dX_t = (X_t + a_t)\sigma_t^a dZ_t
\]

con:

\[
a_t = X_0 \frac{1-\beta_t}{\beta_t}
\]

e

\[
\sigma_t^a = \sigma_t \beta_t
\]

Quindi la variabile traslata \(X_t + a_t\) segue approssimativamente una dinamica lognormale:

\[
d \ln(X_t + a_t)
\approx
-\frac{1}{2}(\sigma_t^a)^2 dt + \sigma_t^a dZ_t
\]

Nel caso \(a_t = 0\), il modello torna al caso lognormale standard. Per valori elevati dello shift, la distribuzione dei ritorni tende invece verso una distribuzione più vicina alla normale.

---

## 3. Dinamica con drift risk-neutral

Per il sottostante con drift risk-neutral, si considera:

\[
S_t = X_t e^{rt}
\]

La dinamica diventa:

\[
d \ln(S_t + a_t e^{rt})
\approx
r dt - \frac{1}{2}(\sigma_t^a)^2 dt + \sigma_t^a dZ_t
\]

Questa è la dinamica da usare per pricing risk-neutral.

---

## 4. Formula di pricing vanilla

Per una call europea con scadenza \(T\), strike \(K\), spot iniziale \(S_0\), tasso risk-free \(r\), displacement \(a\) e volatilità shifted \(\sigma^a\), il prezzo è:

\[
C^{DD}
=
e^{-rT}
\left[
(S_0 e^{rT} + a e^{rT})N(d_+^a)
-
(K + a e^{rT})N(d_-^a)
\right]
\]

dove:

\[
d_{\pm}^a
=
\frac{
\ln\left(
\frac{S_0 e^{rT} + a e^{rT}}
{K + a e^{rT}}
\right)
\pm
\frac{1}{2}(\sigma^a)^2 T
}
{\sigma^a \sqrt{T}}
\]

In pratica, il modello equivale a Black-Scholes applicato a:

\[
F_0^{shifted} = S_0 e^{rT} + a e^{rT}
\]

e

\[
K^{shifted} = K + a e^{rT}
\]

con volatilità:

\[
\sigma^a = \sigma \beta
\]

---

## 5. Parametri del modello

Per ogni maturity \(T\), i parametri principali sono:

| Parametro | Significato |
|---|---|
| \(S_0\) | Spot corrente |
| \(r\) | Tasso risk-free continuo |
| \(T\) | Scadenza dell’opzione |
| \(\sigma_T\) | Volatilità di modello |
| \(\beta_T\) | Parametro di skew |
| \(a_T\) | Displacement |
| \(\sigma_T^a\) | Volatilità del processo shifted |

Il legame tra i parametri è:

\[
a_T = S_0 \frac{1-\beta_T}{\beta_T}
\]

e

\[
\sigma_T^a = \sigma_T \beta_T
\]

Nell’implementazione conviene calibrare direttamente \(\beta_T\), perché è il parametro che governa lo skew.

---

## 6. Input necessari per la calibrazione

Per una singola maturity servono:

- spot \(S_0\);
- curva dei tassi o discount factor;
- maturity \(T\);
- lista di strike \(K_i\);
- volatilità implicite di mercato \(\sigma^{BS}_{mkt}(K_i,T)\);
- opzionalmente dividendi o repo/forward;
- pesi \(w_i\) per dare più o meno importanza ad alcuni strike.

Gli input di mercato vengono prima convertiti in prezzi vanilla Black-Scholes:

\[
C^{mkt}_i = C^{BS}(S_0, K_i, r, T, \sigma^{BS}_{mkt}(K_i,T))
\]

Poi si calibra il modello DD minimizzando la differenza tra prezzi di mercato e prezzi DD.

---

## 7. Calibrazione di un singolo smile

### 7.1 Funzione obiettivo

Per una maturity \(T\), si cerca il parametro \(\beta_T\) che minimizza il residuo:

\[
R(\beta_T)
=
\sum_i
w_i
\left(
1 -
\frac{C^{mkt}(K_i)}
{C^{DD}(K_i;\beta_T)}
\right)^2
\]

In alternativa, si può usare un errore sui prezzi:

\[
R(\beta_T)
=
\sum_i
w_i
\left(
C^{DD}(K_i;\beta_T)
-
C^{mkt}(K_i)
\right)^2
\]

oppure un errore sulle volatilità implicite:

\[
R(\beta_T)
=
\sum_i
w_i
\left(
\sigma^{DD}_{imp}(K_i)
-
\sigma^{mkt}_{imp}(K_i)
\right)^2
\]

La scelta più stabile, in genere, è calibrare sui prezzi o su volatilità implicite con pesi vega-adjusted.

---

## 8. Correzione della volatilità ATM

Una volta scelto \(\beta_T\), la volatilità di modello \(\sigma_T\) può essere collegata alla volatilità ATM Black-Scholes \(\sigma_T^{ATM}\).

Il documento propone la correzione:

\[
\sigma_T
=
\sigma_T^{ATM}
\left[
\frac{
1 - (\sigma_T^{ATM})^2 T/24
}
{
1 - (\beta_T \sigma_T^{ATM})^2 T/24
}
\right]
\]

Questa correzione serve a rendere compatibile il prezzo ATM prodotto dal modello DD con il prezzo ATM Black-Scholes, almeno in approssimazione.

Dopo aver ottenuto \(\sigma_T\), si calcola:

\[
\sigma_T^a = \beta_T \sigma_T
\]

e

\[
a_T = S_0 \frac{1-\beta_T}{\beta_T}
\]

---

## 9. Procedura pratica di calibrazione per una maturity

### Step 1 — Preparazione dati

Per ogni maturity \(T\):

1. Prendere gli strike \(K_i\).
2. Prendere le volatilità implicite di mercato \(\sigma^{mkt}(K_i,T)\).
3. Calcolare i prezzi Black-Scholes di mercato:

\[
C_i^{mkt}
=
C^{BS}(S_0,K_i,r,T,\sigma^{mkt}(K_i,T))
\]

4. Identificare la volatilità ATM \(\sigma_T^{ATM}\).

---

### Step 2 — Definire il parametro libero

Il parametro principale da calibrare è:

\[
\beta_T
\]

Per ogni valore candidato di \(\beta_T\):

\[
a_T = S_0 \frac{1-\beta_T}{\beta_T}
\]

\[
\sigma_T =
\sigma_T^{ATM}
\left[
\frac{
1 - (\sigma_T^{ATM})^2 T/24
}
{
1 - (\beta_T \sigma_T^{ATM})^2 T/24
}
\right]
\]

\[
\sigma_T^a = \beta_T \sigma_T
\]

---

### Step 3 — Pricing DD per ogni strike

Per ogni strike \(K_i\), calcolare:

\[
C_i^{DD}(\beta_T)
\]

usando la formula displaced diffusion.

---

### Step 4 — Minimizzazione

Minimizzare:

\[
R(\beta_T)
=
\sum_i
w_i
\left(
1 -
\frac{C_i^{mkt}}
{C_i^{DD}(\beta_T)}
\right)^2
\]

oppure una funzione alternativa più stabile:

\[
R(\beta_T)
=
\sum_i
w_i
\left(
C_i^{DD}(\beta_T)
-
C_i^{mkt}
\right)^2
\]

L’algoritmo può essere:

- Levenberg-Marquardt;
- Nelder-Mead;
- BFGS;
- least squares;
- grid search iniziale + ottimizzazione locale.

---

## 10. Pseudocodice per calibrazione single maturity

```python
def calibrate_dd_single_maturity(
    S0,
    strikes,
    market_vols,
    T,
    r,
    weights=None
):
    if weights is None:
        weights = np.ones_like(strikes)

    market_prices = [
        black_scholes_call(S0, K, r, T, vol)
        for K, vol in zip(strikes, market_vols)
    ]

    atm_vol = interpolate_atm_vol(strikes, market_vols, S0)

    def objective(beta):
        sigma = atm_vol * (
            (1 - atm_vol**2 * T / 24)
            /
            (1 - (beta * atm_vol)**2 * T / 24)
        )

        a = S0 * (1 - beta) / beta
        sigma_a = sigma * beta

        dd_prices = [
            displaced_diffusion_call(S0, K, r, T, a, sigma_a)
            for K in strikes
        ]

        residuals = [
            w * (dd - mkt)**2
            for w, dd, mkt in zip(weights, dd_prices, market_prices)
        ]

        return sum(residuals)

    result = minimize(objective, x0=initial_beta_guess)

    beta_star = result.x[0]

    sigma_star = atm_vol * (
        (1 - atm_vol**2 * T / 24)
        /
        (1 - (beta_star * atm_vol)**2 * T / 24)
    )

    a_star = S0 * (1 - beta_star) / beta_star
    sigma_a_star = beta_star * sigma_star

    return {
        "beta": beta_star,
        "sigma": sigma_star,
        "a": a_star,
        "sigma_a": sigma_a_star,
        "calibration_error": result.fun
    }


## 11. Calibrazione di una superficie di volatilità

Se si dispone di una superficie:

\[
\sigma^{mkt}(K_i,T_j)
\]

si calibra ogni slice di maturity separatamente.

Per ogni \(T_j\):

1. prendere lo smile relativo a \(T_j\);
2. calibrare \(\beta_{T_j}\);
3. ottenere \(\sigma_{T_j}\);
4. salvare il bucket:

\[
\{T_j, \beta_{T_j}, \sigma_{T_j}, a_{T_j}, \sigma^a_{T_j}\}
\]

Il risultato è una term structure di parametri spot:

\[
\beta_{T_1}, \beta_{T_2}, ..., \beta_{T_n}
\]

e

\[
\sigma_{T_1}, \sigma_{T_2}, ..., \sigma_{T_n}
\]

---

## 12. Pseudocodice per calibrazione della superficie

```python
def calibrate_dd_surface(
    S0,
    maturities,
    strikes_by_maturity,
    vols_by_maturity,
    r_curve
):
    calibrated_params = []

    for T, strikes, vols in zip(
        maturities,
        strikes_by_maturity,
        vols_by_maturity
    ):
        r = r_curve(T)

        params = calibrate_dd_single_maturity(
            S0=S0,
            strikes=strikes,
            market_vols=vols,
            T=T,
            r=r
        )

        params["T"] = T
        calibrated_params.append(params)

    return calibrated_params
```

---

## 13. Forward skew calibration

La calibrazione slice-by-slice produce parametri spot:

\[
\beta_T
\]

ma per simulazioni Monte Carlo su più date può essere utile ricostruire una term structure forward:

\[
\beta_{t_1}, \beta_{t_2}, ..., \beta_{t_n}
\]

Il documento propone di usare una formula di skew averaging:

\[
\beta_T
=
\frac{
\int_0^T
\beta_t |\sigma_t|^2
\left(
\int_0^t |\sigma_\tau|^2 d\tau
\right)
dt
}
{
\int_0^T
|\sigma_t|^2
\left(
\int_0^t |\sigma_\tau|^2 d\tau
\right)
dt
}
\]

In forma discreta:

\[
\beta_{T_n}
=
\frac{
\sum_{j=1}^n
\beta_{t_j}
|\sigma_{t_j}|^2
\left(
\sum_{i=1}^j |\sigma_{t_i}|^2 \Delta t_i
\right)
\Delta t_j
}
{
\sum_{j=1}^n
|\sigma_{t_j}|^2
\left(
\sum_{i=1}^j |\sigma_{t_i}|^2 \Delta t_i
\right)
\Delta t_j
}
\]

Questa relazione permette di ricostruire i parametri forward \(\beta_{t_j}\) a partire dai parametri spot \(\beta_{T_j}\).

---

## 14. Algoritmo per forward skews

Per ogni intervallo temporale \(t_{j-1} \to t_j\):

1. Calcolare la forward volatility ATM:

\[
\sigma_{t_j}^{fwd}
=
\sqrt{
\frac{
\sigma^2_{ATM}(T_j)T_j
-
\sigma^2_{ATM}(T_{j-1})T_{j-1}
}
{
T_j - T_{j-1}
}
}
\]

2. Trovare \(\beta_{t_j}\) tale che la formula di averaging riproduca \(\beta_{T_j}\).

3. Salvare:

\[
\beta_{t_j}, \sigma_{t_j}^{fwd}
\]

---

## 15. Pseudocodice per forward calibration

```python
def calibrate_forward_betas(
    spot_betas,
    atm_vols,
    maturities
):
    forward_betas = []
    forward_vols = []

    for j, T_j in enumerate(maturities):
        if j == 0:
            sigma_fwd = atm_vols[j]
        else:
            T_prev = maturities[j - 1]
            sigma_fwd = np.sqrt(
                (
                    atm_vols[j]**2 * T_j
                    -
                    atm_vols[j - 1]**2 * T_prev
                )
                /
                (T_j - T_prev)
            )

        forward_vols.append(sigma_fwd)

        def objective(beta_j):
            trial_betas = forward_betas + [beta_j]

            beta_avg = compute_discrete_skew_average(
                trial_betas,
                forward_vols,
                maturities[:j+1]
            )

            return (beta_avg - spot_betas[j])**2

        result = minimize(objective, x0=spot_betas[j])
        forward_betas.append(result.x[0])

    return forward_betas, forward_vols
```

---

## 16. Simulazione Monte Carlo

Per payoff path-dependent o multi-asset, si può simulare il processo.

La discretizzazione single asset è:

\[
S_{t_i}
=
(S_{t_{i-1}} + a_{t_{i-1}})
\exp
\left[
r_i \Delta t_i
-
\frac{1}{2}
(\sigma^a_{t_i})^2
\Delta t_i
+
\sigma^a_{t_i}
\Delta W_i
\right]
-
a_{t_{i-1}} e^{r_i \Delta t_i}
\]

con:

\[
a_{t_{i-1}}
=
S_{t_{i-1}}
\frac{1-\beta_{t_{i-1}}}
{\beta_{t_{i-1}}}
\]

e

\[
\sigma^a_{t_i}
=
\sigma_{t_{i-1}}\beta_{t_{i-1}}
\]

---

## 17. Pseudocodice Monte Carlo single asset

```python
def simulate_dd_paths(
    S0,
    times,
    r_curve,
    beta_curve,
    sigma_curve,
    n_paths
):
    n_steps = len(times)
    paths = np.zeros((n_paths, n_steps))
    paths[:, 0] = S0

    for i in range(1, n_steps):
        t_prev = times[i - 1]
        t = times[i]
        dt = t - t_prev

        r = r_curve(t_prev)
        beta = beta_curve(t_prev)
        sigma = sigma_curve(t_prev)

        S_prev = paths[:, i - 1]

        a = S_prev * (1 - beta) / beta
        sigma_a = sigma * beta

        Z = np.random.normal(size=n_paths)

        paths[:, i] = (
            (S_prev + a)
            *
            np.exp(
                r * dt
                - 0.5 * sigma_a**2 * dt
                + sigma_a * np.sqrt(dt) * Z
            )
            -
            a * np.exp(r * dt)
        )

    return paths
```

---

## 18. Estensione multi-asset

Per \(N\) asset, il modello diventa:

\[
d \ln(S_i(t) + a_i(t)e^{r_i t})
\approx
r_i dt
-
\frac{1}{2}
[\sigma_i^a(t)]^2 dt
+
\beta_i(t)
\sum_{j=1}^N
\sigma_{ij}(t)dZ_j(t)
\]

dove:

\[
\sigma_i^a(t) = \sigma_i(t)\beta_i(t)
\]

e la dipendenza tra asset è governata dalla matrice di correlazione:

\[
\rho_{ij}
\]

In pratica:

1. si calibra ogni asset singolarmente alla propria superficie di volatilità;
2. si ottiene \(\beta_i(t)\) e \(\sigma_i(t)\) per ogni asset;
3. si specifica o calibra la matrice di correlazione;
4. si genera un vettore di shock gaussiani correlati tramite decomposizione di Cholesky.

---

## 19. Pseudocodice Monte Carlo multi-asset

```python
def simulate_multi_asset_dd(
    S0_vector,
    times,
    r_curves,
    beta_curves,
    sigma_curves,
    corr_matrix,
    n_paths
):
    n_assets = len(S0_vector)
    n_steps = len(times)

    paths = np.zeros((n_paths, n_steps, n_assets))
    paths[:, 0, :] = S0_vector

    chol = np.linalg.cholesky(corr_matrix)

    for i in range(1, n_steps):
        t_prev = times[i - 1]
        dt = times[i] - times[i - 1]

        Z_uncorr = np.random.normal(size=(n_paths, n_assets))
        Z_corr = Z_uncorr @ chol.T

        for k in range(n_assets):
            S_prev = paths[:, i - 1, k]

            r = r_curves[k](t_prev)
            beta = beta_curves[k](t_prev)
            sigma = sigma_curves[k](t_prev)

            a = S_prev * (1 - beta) / beta
            sigma_a = sigma * beta

            paths[:, i, k] = (
                (S_prev + a)
                *
                np.exp(
                    r * dt
                    - 0.5 * sigma_a**2 * dt
                    + sigma_a * np.sqrt(dt) * Z_corr[:, k]
                )
                -
                a * np.exp(r * dt)
            )

    return paths
```

---

## 20. Controlli di calibrazione

Dopo la calibrazione bisogna controllare:

### 20.1 Fit dello smile

Confrontare:

\[
\sigma^{mkt}_{imp}(K_i,T)
\]

con:

\[
\sigma^{DD}_{imp}(K_i,T)
\]

dove \(\sigma^{DD}_{imp}\) si ottiene invertendo Black-Scholes sul prezzo DD.

---

### 20.2 Errore ATM

Controllare che il prezzo ATM DD sia vicino al prezzo ATM Black-Scholes.

---

### 20.3 Stabilità dei parametri

Controllare che la term structure di \(\beta_T\) non abbia salti irragionevoli.

---

### 20.4 Qualità del fit sulle ali

Il modello DD cattura bene lo skew, ma non necessariamente lo smile puro. Quindi può fallire nelle ali molto lontane dall’ATM.

---

## 21. Limiti del modello

Il modello è semplice e veloce, ma ha alcuni limiti:

1. **Riproduce principalmente lo skew, non lo smile completo.**

   Se il mercato presenta forte convexity dello smile, il modello può sottostimare o sovrastimare le ali.

2. **La calibrazione può essere instabile per strike molto lontani dall’ATM.**

3. **Il parametro \(\beta\) può assumere valori difficili da interpretare economicamente.**

4. **La dinamica forward richiede assunzioni aggiuntive.**

5. **Nel caso multi-asset, la correlazione resta esogena o deve essere calibrata separatamente.**


---

## Nota pratica

Per un’implementazione robusta conviene:

- calibrare sui prezzi, non direttamente sulle volatilità;
- usare pesi vega-adjusted o concentrati nella regione liquida dello smile;
- escludere strike illiquidi o troppo profondi nelle ali;
- usare una grid iniziale su \(\beta\);
- imporre bounds ragionevoli a \(\beta\);
- controllare sempre che \(S_0 + a > 0\) e \(K + a e^{rT} > 0\);
- verificare il fit in implied volatility, anche se la calibrazione è fatta sui prezzi.

---

## Output atteso della calibrazione

La calibrazione dovrebbe produrre una tabella del tipo:

| Maturity | Beta | Sigma | Displacement | Sigma shifted | Error |
|---|---:|---:|---:|---:|---:|
| 1M | ... | ... | ... | ... | ... |
| 3M | ... | ... | ... | ... | ... |
| 6M | ... | ... | ... | ... | ... |
| 1Y | ... | ... | ... | ... | ... |

Questa tabella rappresenta la parametrizzazione completa del modello per lo smile o la superficie considerata.