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

::: skein_glm.ic.select_by_ic
