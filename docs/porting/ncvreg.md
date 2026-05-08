# Porting from ncvreg

`ncvreg` (Patrick Breheny) is the canonical R package for nonconvex
penalized regression — MCP and SCAD with rigorous statistical
treatment. If you came to skein because you wanted MCP/SCAD support
in Python, this is your guide.

`skein`'s scalar MCP/SCAD path is a direct port of `ncvreg`'s
algorithmic stack: coordinate descent with strong-rule screening,
gap-safe screening (skein's addition), warm-started λ-paths, and
the same glmnet-style standardization conventions. Coefficients
should match `ncvreg` to ~1e-6 on canonical problems (the M8
numerical regression suite, pending, will pin this).

## The three top-line translations

| `ncvreg` (R)                                  | `skein` (Python)                            |
|-----------------------------------------------|---------------------------------------------|
| `ncvreg(X, y, penalty = "MCP")`               | `MCPPathRegressor(gamma=3.0).fit(X, y)`     |
| `cv.ncvreg(X, y, penalty = "MCP")`            | `MCPPathCV(gamma=3.0, cv=10).fit(X, y)`     |
| `coef(cvfit)` (at `lambda.min`)               | `cv.coef_`, `cv.intercept_`                 |

R:

```r
library(ncvreg)
fit <- cv.ncvreg(X, y, penalty = "MCP", gamma = 3, nfolds = 10)
beta <- coef(fit)[-1]   # drop intercept
alpha <- coef(fit)[1]
```

Python:

```python
import skein_glm
fit = skein_glm.MCPPathCV(gamma=3.0, cv=10).fit(X, y)
beta = fit.coef_
alpha = fit.intercept_
```

## Family map

| `ncvreg` family             | `skein` estimator base         |
|------------------------------|--------------------------------|
| `family = "gaussian"` (default) | `MCP*` / `SCAD*` (LS)        |
| `family = "binomial"`        | `LogisticMCP*` / `LogisticSCAD*` |
| `family = "poisson"`         | `PoissonMCP*` / `PoissonSCAD*`  |
| `ncvsurv` (separate function) | `CoxMCP*` / `CoxSCAD*`         |

For the binomial family, `ncvreg` uses an iteratively reweighted
least-squares (IRLS) outer loop; `skein` uses prox-Newton (which is
mathematically the same scheme, with the inner LS solved by CD
instead of QR). Convergence behavior should be similar.

## Penalty + γ map

| `ncvreg`                        | `skein`                                            |
|---------------------------------|----------------------------------------------------|
| `penalty = "MCP", gamma = 3`    | `MCPPathRegressor(gamma=3.0)`                      |
| `penalty = "SCAD", gamma = 3.7` | `SCADPathRegressor(a=3.7)` ← **note `a` not `gamma`** |
| `penalty = "lasso"`             | `MCPPathRegressor(gamma=1e6)` (≈ lasso)            |

`ncvreg` uses `gamma` for the SCAD shape parameter (which is the `a`
in the SCAD literature — Fan & Li 2001). `skein` keeps the literature
naming: `gamma` for MCP, `a` for SCAD. Default `a=3.7` matches
`ncvreg`.

## Per-argument map

| `ncvreg` arg              | `skein` arg                       | Notes                                    |
|---------------------------|-----------------------------------|------------------------------------------|
| `X`                       | `X` (positional)                  | Same conventions.                        |
| `y`                       | `y` (positional)                  | For Cox (ncvsurv): `fit(X, time, event)`. |
| `penalty`                 | (choose `MCP*` vs `SCAD*` class)  | See above.                               |
| `gamma`                   | `gamma` (MCP) or `a` (SCAD)       | Same numerical meaning.                  |
| `family`                  | (choose family-prefixed class)    | See family map.                          |
| `lambda`                  | `lambdas`                         | numpy array.                             |
| `nlambda`                 | `n_lambdas`                       | Default 100 (matches ncvreg).            |
| `lambda.min`              | `lambda_min_ratio`                | Same semantic.                           |
| `eps`                     | `tol`                             |                                          |
| `max.iter`                | `max_iter`                        |                                          |
| `convex`                  | (handled internally)              | ncvreg's `convex = TRUE` mode is equivalent to skein's LLA outer loop, which is always on for nonconvex penalties. |
| `dfmax`                   | (not directly; use working set + KKT) | skein's default working-set strategy already screens; no explicit df cap. |
| `penalty.factor`          | `weights` (constructor)           | Per-feature penalty weights. Identical semantics. |
| `nfolds` (cv.ncvreg)      | `cv` (PathCV)                     | int or sklearn splitter.                 |
| `seed` (cv.ncvreg)        | `random_state`                    |                                          |

## ncvreg-specific recipes in skein

### Standard MCP path

```r
# R
fit <- ncvreg(X, y, penalty = "MCP", gamma = 3)
plot(fit)
coef(fit, lambda = 0.1)    # coefficients at a specific λ
```

```python
# Python
fit = skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=100).fit(X, y)
# fit.coefs_ is (100, p), fit.lambdas_ is (100,)
# Pick a specific λ:
import numpy as np
idx = np.argmin(np.abs(fit.lambdas_ - 0.1))
beta_at_0p1 = fit.coefs_[idx]
intercept_at_0p1 = fit.intercepts_[idx]
```

### CV with custom λ grid

```r
# R
my_lambda <- 10^seq(0, -3, length.out = 50)
fit <- cv.ncvreg(X, y, penalty = "MCP", lambda = my_lambda, nfolds = 10)
```

```python
# Python
import numpy as np
my_lambda = 10 ** np.linspace(0, -3, 50)
fit = skein_glm.MCPPathCV(gamma=3.0, lambdas=my_lambda, cv=10).fit(X, y)
```

### BIC selection

`ncvreg` ships `BIC.ncvreg` for information-criterion selection.
`skein` has `select_by_ic`:

```r
# R
fit <- ncvreg(X, y, penalty = "MCP")
sel <- summary(fit, lambda = "BIC")
```

```python
# Python
path = skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=100).fit(X, y)
best_idx, scores = skein_glm.select_by_ic(path, X, y, criterion="bic")
beta_best = path.coefs_[best_idx]
```

`select_by_ic` supports `"aic"`, `"bic"`, and `"ebic"` (with
`ebic_gamma=0.5` matching `ncvreg::BIC`'s high-dim recommendation).

### Survival analysis

```r
# R
library(ncvreg)
fit <- ncvsurv(X, y = Surv(time, event), penalty = "MCP", gamma = 3)
risk <- predict(fit, X_new, lambda = lambda_min, type = "link")
```

```python
# Python
fit = skein_glm.CoxMCPPathRegressor(gamma=3.0, n_lambdas=100).fit(X, time, event)
risk = fit.predict(X_new)   # shape (n_new, n_lambdas), η = Xβ

# Or with CV-selected λ:
cv = skein_glm.CoxMCPPathCV(gamma=3.0, cv=10).fit(X, time, event)
risk_at_lambda_min = cv.predict(X_new)   # 1D, at the best λ
```

skein's Cox uses Breslow ties (matches `ncvsurv`'s default).

## Differences worth knowing

### Standardization

`ncvreg` defaults to `standardize = FALSE` for the X matrix; skein
also defaults to `standardize=False`. So the "compatibility" mode
is the default. If you want centering+scaling, pass `standardize=True`
on both sides.

### lambda.min behavior under CV

`ncvreg::cv.ncvreg` returns both `lambda.min` and `lambda.1se`
(one-standard-error rule). `skein_glm.MCPPathCV` only returns
`lambda_best_` (= the equivalent of `lambda.min`). The 1se rule is
on the M5.x roadmap; for now you can compute it manually:

```python
cv = skein_glm.MCPPathCV(gamma=3.0, cv=10).fit(X, y)
mean_scores = cv.cv_mean_scores_   # shape (n_lambdas,)
std_scores = cv.cv_std_scores_     # shape (n_lambdas,)

best_score = mean_scores.min()
best_idx = mean_scores.argmin()
# lambda.1se: largest λ with mean_score within 1 SE of best.
threshold = best_score + std_scores[best_idx]
candidates = np.where(mean_scores <= threshold)[0]
# largest λ → smallest index (lambdas are decreasing).
lambda_1se_idx = candidates.min()
lambda_1se = cv.lambdas_[lambda_1se_idx]
```

### Convergence threshold

`ncvreg`'s `eps = 1e-4` is its default (relative change in β); skein's
`tol = 1e-6` is its default (max-coord-update L1, KKT-based). Both
are reasonable; if you want to match `ncvreg` numerics exactly, pass
`tol=1e-4` and bump `max_iter`.

## Sparse design matrices

`ncvreg` doesn't natively support sparse X (you'd convert to dense
first). `skein` does:

```python
import scipy.sparse as sp
X_sp = sp.csc_matrix(X)
fit = skein_glm.MCPPathCV(gamma=3.0).fit(X_sp, y)
```

This alone can be a major speedup if your X is mostly zero.

## What skein adds vs ncvreg

- **Group penalties** (group MCP / SCAD via LLA) with Rayon-parallel
  block CD. `ncvreg` is scalar-only; for groups you'd switch to
  `grpreg` (single-threaded R).
- **Sparse-group penalties** (convex and nonconvex).
- **Three weight axes** including per-group, all composable.
- **Memory-mapped / chunked design matrices** for n in the
  hundreds of millions.
- **`scipy.sparse.csc_matrix` support** transparently.
- **The dual extension surface** for prototyping new penalties.

## See also

- [Porting from glmnet](glmnet.md) — for L1 / elastic-net users.
- [Porting from grpreg](grpreg.md) — for group penalty users.
- [Concepts: Penalties](../concepts/penalties.md) — MCP and SCAD
  shapes in detail.
