# Porting from glmnet

`glmnet` (Friedman, Hastie & Tibshirani) is the dominant R package
for L1 / elastic-net regularized GLMs. If you're moving from R to
Python and want to keep your `cv.glmnet`-based workflow, this page
maps `glmnet`'s API onto `skein`.

`skein` doesn't ship a native lasso or elastic net (M6 roadmap), so
"lasso" in this guide means **MCP at large γ** — numerically
indistinguishable from lasso but solved through `skein`'s nonconvex
machinery. The advantages: better statistical properties (less
biased estimates of large coefficients), the same code path lets you
turn the nonconvexity dial back when you want it, and the entire
`skein` weight-axis story is available.

## The three top-line translations

| `glmnet` (R)                                 | `skein` (Python)                               |
|----------------------------------------------|------------------------------------------------|
| `glmnet(x, y, family = "gaussian")`          | `MCPPathRegressor(gamma=1e6).fit(X, y)`        |
| `cv.glmnet(x, y, family = "gaussian")`       | `MCPPathCV(gamma=1e6, cv=10).fit(X, y)`        |
| `coef(cv_fit, s = "lambda.min")`             | `cv_fit.coef_`, `cv_fit.intercept_`            |

In R you write:

```r
library(glmnet)
fit <- cv.glmnet(x, y, family = "gaussian", nfolds = 10)
beta_hat <- as.numeric(coef(fit, s = "lambda.min"))[-1]   # drop intercept
alpha_hat <- as.numeric(coef(fit, s = "lambda.min"))[1]
```

In Python:

```python
import skein
cv_fit = skein.MCPPathCV(gamma=1e6, cv=10).fit(X, y)
beta_hat = cv_fit.coef_
alpha_hat = cv_fit.intercept_
```

## Family map

| `glmnet`         | `skein` estimator base    | Notes                                        |
|------------------|---------------------------|----------------------------------------------|
| `"gaussian"`     | `MCP*` / `SCAD*`          | Default. LS datafit.                         |
| `"binomial"`     | `LogisticMCP*` / `LogisticSCAD*` | Two-class only. v0.1 doesn't support multinomial. |
| `"poisson"`      | `PoissonMCP*` / `PoissonSCAD*` | Log link. y ≥ 0 required.                    |
| `"cox"`          | `CoxMCP*` / `CoxSCAD*`    | Right-censored survival, fit signature is `fit(X, time, event)`. |
| `"multinomial"`  | (not in v0.1)             | M3.6 roadmap. Use one-vs-rest manually for now. |
| `"mgaussian"`    | (not in v0.1)             | M7 multi-task roadmap.                       |

## Per-argument translation

### Most-used arguments

| `glmnet` arg            | `skein` arg                       | Notes                                                |
|-------------------------|-----------------------------------|------------------------------------------------------|
| `x`                     | `X` (positional)                  | numpy array, scipy.sparse, MmapDesignF64/32, ChunkedDesignF64/32. |
| `y`                     | `y` (positional)                  | For Cox: `fit(X, time, event)`.                      |
| `family`                | (choose estimator class)          | See family map above.                                |
| `alpha`                 | `alpha` on `ElasticNet*Regressor` | `skein.ElasticNet*Regressor(alpha=...)` matches `glmnet`'s `alpha ∈ [0, 1]` exactly. `α=1` is lasso, `α=0` is ridge. |
| `lambda`                | `lambdas`                         | numpy array. Pass `None` to auto-compute.            |
| `nlambda`               | `n_lambdas`                       | Default 100 (matches glmnet).                        |
| `lambda.min.ratio`      | `lambda_min_ratio`                | Default 1e-3 if `n > p`, 1e-2 if `n < p` (glmnet); skein defaults to 1e-3 always. |
| `weights`               | `sample_weight` (in `fit()`)      | Per-sample weights. Identical semantics.             |
| `penalty.factor`        | `weights` (in constructor)        | Per-feature penalty weights. `penalty.factor[j]=0` → unpenalized; same in skein. |
| `intercept`             | `fit_intercept`                   | Default `True`.                                      |
| `standardize`           | `standardize`                     | Default `False` (glmnet defaults to `True`!).        |
| `thresh`                | `tol`                             | Default `1e-6` (glmnet `1e-7`).                      |
| `maxit`                 | `max_iter`                        | Default `100`.                                       |
| `nfolds` (cv.glmnet)    | `cv`                              | Pass an int or any sklearn CV splitter.              |
| `type.measure`          | (auto-selected by family)         | Family-appropriate metric. See "type.measure" below. |

### Defaults that differ

Two glmnet defaults that bite people moving to skein:

1. **`standardize`**: glmnet defaults to `TRUE`, skein defaults to
   `False`. If your features have heterogeneous scales, pass
   `standardize=True` explicitly.
2. **`thresh`**: glmnet defaults to `1e-7`, skein defaults to
   `1e-6`. Tighten with `tol=1e-8` or smaller for numerically
   delicate problems (e.g. matching reference fits exactly).

### `type.measure` map

`glmnet`'s `type.measure` selects the CV scoring metric. In skein, the
metric is auto-selected by the GLM family, but you can override via
the `*PathCV` mixin's scorer attribute (advanced; see API ref).

| `glmnet` `type.measure` | `skein` family default scorer                    |
|--------------------------|--------------------------------------------------|
| `"mse"` (gaussian)       | mean squared error (lower-better)                |
| `"deviance"` (binomial)  | binomial deviance (lower-better)                 |
| `"class"` (binomial)     | (not default; can override)                      |
| `"auc"` (binomial)       | (not default; can override)                      |
| `"deviance"` (poisson)   | Poisson deviance (lower-better)                  |
| `"deviance"` (cox)       | Harrell's c-index (higher-better) — actually skein uses concordance, not deviance, by default. Note this difference: `glmnet`'s default for Cox is partial-likelihood deviance. |

## Workflow translations

### Basic CV fit and predict

```r
# R
library(glmnet)
fit <- cv.glmnet(x, y, family = "gaussian")
y_hat <- predict(fit, newx = x_new, s = "lambda.min")
beta <- coef(fit, s = "lambda.min")
```

```python
# Python
import skein
fit = skein.MCPPathCV(gamma=1e6, cv=10).fit(X, y)
y_hat = fit.predict(X_new)
beta = fit.coef_
intercept = fit.intercept_
```

### Logistic with class weights

```r
# R
fit <- cv.glmnet(x, y, family = "binomial", weights = w)
prob <- predict(fit, newx = x_new, type = "response", s = "lambda.min")
```

```python
# Python
fit = skein.LogisticMCPPathCV(gamma=1e6, cv=10).fit(X, y, sample_weight=w)
# v0.1: LogisticMCPPathCV picks lambda_best_ at fit time and refits;
# `fit.predict_proba(X_new)` returns a 1D probability vector.
prob = fit.predict_proba(X_new)
labels = fit.predict(X_new)
```

For path inspection (every λ at once), use the `*PathRegressor`
instead of `*PathCV`:

```python
path = skein.LogisticMCPPathRegressor(gamma=1e6, n_lambdas=50).fit(X, y)
prob_path = path.predict_proba(X_new)   # shape (n_new, n_lambdas)
```

### Cox PH

```r
# R
fit <- cv.glmnet(x, Surv(time, event), family = "cox")
risk <- predict(fit, newx = x_new, s = "lambda.min")
```

```python
# Python
fit = skein.CoxMCPPathCV(gamma=1e6, cv=10).fit(X, time, event)
risk = fit.predict(X_new)   # the prognostic index η = Xβ
```

skein's Cox uses Breslow ties by default; glmnet uses Efron. For
moderate tie rates the difference is negligible. Efron is on the
M3.7 roadmap.

### Adaptive lasso (two-stage)

`glmnet` doesn't have a built-in adaptive lasso, but the canonical
recipe is straightforward:

```r
# R
fit_init <- cv.glmnet(x, y, family = "gaussian")
beta_init <- as.numeric(coef(fit_init, s = "lambda.min"))[-1]
penalty <- 1 / (abs(beta_init) + 1e-3)
fit_adaptive <- cv.glmnet(x, y, family = "gaussian", penalty.factor = penalty)
```

```python
# Python — same two-stage recipe.
import numpy as np

# Stage 1: coarse fit.
init = skein.MCPPathCV(gamma=1e6).fit(X, y)
beta_init = init.coef_
weights = 1.0 / (np.abs(beta_init) + 1e-3)

# Stage 2: refit with adaptive weights.
adaptive = skein.MCPPathCV(gamma=1e6, weights=weights).fit(X, y)
```

The M5.x roadmap promotes this two-stage idiom to a one-shot
`AdaptiveLasso` / `AdaptiveMCP` estimator.

## Sparse design matrices

glmnet accepts `Matrix::sparseMatrix` (CSC) seamlessly; skein does
the same with `scipy.sparse.csc_matrix`:

```r
# R
library(Matrix)
x_sp <- as(x, "CsparseMatrix")
fit <- cv.glmnet(x_sp, y, family = "binomial")
```

```python
# Python
import scipy.sparse as sp
X_sp = sp.csc_matrix(X)
fit = skein.LogisticMCPPathCV(gamma=1e6).fit(X_sp, y)
```

CSR inputs are converted to CSC at the boundary in skein. Group and
sparse-group penalties also accept sparse inputs.

## Things glmnet does that skein doesn't yet

- **Multinomial** (`family = "multinomial"`): M3.6 roadmap. Use
  one-vs-rest manually for now.
- **Multi-response Gaussian** (`family = "mgaussian"`): M7 multi-task
  roadmap.
- **`relax = TRUE`** (relaxed lasso): not in v0.1.
- **Offset terms**: not in v0.1 (M3.7 roadmap).

## Things skein does that glmnet doesn't

- **Nonconvex penalties** (MCP / SCAD with γ < ∞): nearly unbiased
  estimates of truly active features. `glmnet` is L1 / L2 only.
- **Group MCP / group SCAD via LLA** + Rayon-parallel block CD:
  `grpreg` has the penalties but is single-threaded R.
- **Three weight axes** (per-sample, per-feature, per-group)
  composable on every estimator. `glmnet` does per-sample +
  per-feature; per-group is awkward.
- **Memory-mapped and chunked design matrices** out of the box —
  fits problems too big to load into RAM.
- **Sparse-group penalties** convex and nonconvex.
- **The dual extension surface** (Python ABCs mirroring Rust traits)
  for prototyping custom penalties / datafits.

## See also

- [Porting from ncvreg](ncvreg.md) — for users coming from the
  MCP / SCAD-focused R package.
- [Porting from grpreg](grpreg.md) — for users coming from the
  group-penalty R package.
- [Concepts: Penalties](../concepts/penalties.md) — when to use MCP
  vs SCAD vs group vs sparse-group.
