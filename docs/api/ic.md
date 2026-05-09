# Information-criterion selection

Pick the best λ from a fitted `*PathRegressor` by AIC, BIC, or EBIC.
Single free function — no per-estimator wrapper explosion.

The criteria use the negative log-likelihood at each λ on the
training data plus a complexity penalty:

- **AIC** = 2k + 2·NLL
- **BIC** = log(n)·k + 2·NLL
- **EBIC** = BIC + 2γ·log C(p, k), with `gamma_ebic ∈ [0, 1]`
  (default 0.5; matches `ncvreg::BIC`'s high-dim recommendation).

Effective df is the number of nonzero coefficients per λ — the
Zou-Hastie-Tibshirani unbiased estimator and the standard
`ncvreg`/`glmnet` convention.

`select_by_ic` dispatches the per-family NLL by sniffing the path
estimator's class name. The five families currently supported:

- **LS** (`MCPPathRegressor`, `SCADPathRegressor`, `*Group*PathRegressor`):
  `NLL = (n/2) · log(RSS/n)`.
- **Logistic** (`Logistic*PathRegressor`):
  `NLL = Σ softplus(η) − y·η`.
- **Poisson** (`Poisson*PathRegressor`):
  `NLL = Σ exp(η) − y·η`.
- **Cox PH** (`Cox*PathRegressor`):
  Breslow per-sample partial NLL × `n`, read from the path's
  `info_["final_losses"]`.
- **Multinomial** (`Multinomial*PathClassifier`):
  per-λ `Σ_i (logsumexp(η_i) − η_{i, y_i})`. Effective df is
  the **row-grouped** active-feature count (a feature is "active"
  if any of its K class coefficients is nonzero), the analog of
  the Zou-Hastie-Tibshirani df for row-grouped lasso.

```{eval-rst}
.. autofunction:: skein_glm.ic.select_by_ic
```
