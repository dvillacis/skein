# Debiased / desparsified lasso

Confidence intervals and Wald p-values for high-dimensional penalized
fits. Van de Geer–Bühlmann–Ritov (VBR 2014) for least squares; the
GLM extension uses the canonical-link score plus the Fisher
information as the surrogate Gram.

The penalized lasso estimator `β̂` is biased toward zero — the
penalty trades variance for bias. The **debiased estimator** corrects
the bias by walking back along an approximate score direction:

$$
\hat\beta_d \;=\; \hat\beta \;+\; \frac{1}{n}\,\hat\Theta\,X^\top\!\bigl(y - \hat\mu(\hat\beta)\bigr)
$$

with `Θ̂ ≈ Σ⁻¹` (LS) or `Θ̂ ≈ J⁻¹` (GLM, with `J = (1/n) Xᵀ W X` the
empirical Fisher), built **nodewise** — one lasso per column. Under
sparsity + restricted-eigenvalue regularity:

$$
\sqrt{n}\,(\hat\beta_d - \beta) \;\rightsquigarrow\; \mathcal N\!\bigl(0,\;\sigma^2 \hat\Theta\hat\Sigma\hat\Theta^\top\bigr)
$$

for LS (no `σ²` for GLMs — the noise is in `W`). Diagonal of that
matrix gives the per-coordinate standard error, and the rest is a
two-sided Wald test.

## Why

`glmnet` / `ncvreg` / `grpreg` ship the penalized fit but **no
inference**. The canonical R interface for this is
`hdi::lasso.proj`; skein matches that semantics — free function on
`(X, y)` returning a result dataclass, plus a thin sklearn-style
estimator wrapper for pipeline use.

## LS

```{eval-rst}
.. autofunction:: skein_glm.debiased.debiased_lasso

.. autoclass:: skein_glm.debiased.DebiasedLassoResult
   :members:

.. autoclass:: skein_glm.debiased.DebiasedLassoRegressor
   :members:
```

## Logistic + Poisson

For canonical-link GLMs (binomial logit, Poisson log), the score is
`Xᵀ(y − μ̂)` and the Fisher information uses working weights
`W = diag(μ̂(1−μ̂))` (binomial) or `diag(μ̂)` (Poisson). `Θ̂` is
built nodewise on the **weighted design** `X̃ = W^{1/2} X` — the
same lasso primitive as the LS case, applied to a re-weighted matrix.

```{eval-rst}
.. autofunction:: skein_glm.debiased.debiased_logistic_lasso

.. autofunction:: skein_glm.debiased.debiased_poisson_lasso

.. autoclass:: skein_glm.debiased.DebiasedGLMResult
   :members:

.. autoclass:: skein_glm.debiased.DebiasedLogisticLassoRegressor
   :members:

.. autoclass:: skein_glm.debiased.DebiasedPoissonLassoRegressor
   :members:
```

## Tuning

- `lambda_` (main fit): default theoretical scale `√(2 log p / n)` on
  standardized features. For best inference quality, pass a CV-tuned
  λ from a prior `ElasticNetPathCV` (LS) or `LogisticLassoPathCV` /
  `PoissonLassoPathCV` (GLM) fit.
- `lambda_nodewise`: per-column λ for the nodewise lassos that build
  `Θ̂`. Scalar or array of length `p`. Default also `√(2 log p / n)`
  uniformly.
- `alpha` (CI level): default 0.05 (95% intervals).
- `n_jobs`: joblib parallelism over the `p` nodewise lassos.
  Materially faster for `p ≳ 50` since each nodewise lasso releases
  the GIL during compute.
- `standardize` (default True): the dimensionless `lambda_nodewise`
  default is calibrated to unit-variance columns. Turning standardize
  off without supplying a column-specific `lambda_nodewise` usually
  gives the wrong scale on the CIs.

## Diagnostics

`DebiasedLassoResult` / `DebiasedGLMResult` expose the underlying
penalized fit (`coef_lasso` / `coef_glm`), `Theta` for inspection or
reuse, `lambda_main` and `lambda_nodewise`, and (for GLMs)
`mu_fitted`. The GLM working weights are floored at `1e-8` to
prevent degenerate columns in `X̃` near boundary fitted
probabilities — inspect `mu_fitted` if CIs look implausibly narrow.

## Example

```python
import numpy as np
from skein_glm import DebiasedLogisticLassoRegressor

rng = np.random.default_rng(0)
n, p = 200, 30
X = rng.standard_normal((n, p))
beta = np.zeros(p); beta[:3] = [1.5, -1.0, 0.8]
y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-X @ beta))).astype(float)

est = DebiasedLogisticLassoRegressor(n_jobs=-1, random_state=0).fit(X, y)
significant = np.where(est.pvalues_ < 0.05)[0]
print("significant features:", significant)
print("CI for β_0:", (est.ci_lower_[0], est.ci_upper_[0]))
```

## References

- Van de Geer, S., Bühlmann, P., Ritov, Y., Dezeure, R. (2014). *On
  asymptotically optimal confidence regions and tests for
  high-dimensional models.* Annals of Statistics 42(3): 1166–1202.
- Zhang, C.-H., Zhang, S. (2014). *Confidence intervals for low-
  dimensional parameters in high-dimensional linear models.* JRSS B
  76(1): 217–242.
- Reid, S., Tibshirani, R., Friedman, J. (2016). *A study of error
  variance estimation in lasso regression.* Statistica Sinica 26:
  35–67.
