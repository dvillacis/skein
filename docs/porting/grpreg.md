# Porting from grpreg

`grpreg` (Patrick Breheny) is the canonical R package for **group**
penalized regression — group lasso, group MCP, group SCAD, and
group bridge. If you're moving group-penalty workflows from R to
Python, this is your guide.

`skein`'s group MCP / SCAD path implements the same Local Linear
Approximation outer loop that `grpreg::grpreg` uses, but the inner
solver is parallelized across groups via Rayon. For overlapping
groups (which `grpreg` doesn't natively support), see the M6
roadmap.

## The three top-line translations

| `grpreg` (R)                                    | `skein` (Python)                                  |
|------------------------------------------------|---------------------------------------------------|
| `grpreg(X, y, group, penalty = "grLasso")`     | `GroupLassoPathRegressor(groups=g).fit(X, y)`     |
| `grpreg(X, y, group, penalty = "grMCP")`       | `GroupMCPPathRegressor(groups=g, gamma=3.0).fit(X, y)` |
| `cv.grpreg(X, y, group, penalty = "grLasso")`  | `GroupLassoPathCV(groups=g, cv=10).fit(X, y)`     |

R:

```r
library(grpreg)
fit <- cv.grpreg(X, y, group, penalty = "grMCP", gamma = 3, nfolds = 10)
beta <- coef(fit)[-1]
alpha <- coef(fit)[1]
```

Python:

```python
import skein_glm
fit = skein_glm.GroupMCPPathCV(groups=g, gamma=3.0, cv=10).fit(X, y)
beta = fit.coef_
alpha = fit.intercept_
```

## Group specification

Both libraries accept a length-`p` integer vector mapping each
feature to its group:

```r
# R: factor or integer; group levels needn't be contiguous.
group <- c(1, 1, 2, 2, 2, 3, 3)
```

```python
# Python: numpy int64 array; same semantics. Labels can be any
# integers; skein normalizes them internally to contiguous group
# indices.
import numpy as np
g = np.array([1, 1, 2, 2, 2, 3, 3], dtype=np.int64)
```

For "5 features per group, 10 groups":

```python
g = np.repeat(np.arange(10), 5)   # [0, 0, 0, 0, 0, 1, 1, ..., 9, 9]
```

For singleton groups (each feature its own group, recovering scalar
behavior):

```python
g = np.arange(p)
```

## Penalty map

| `grpreg` `penalty`        | `skein` estimator base                      |
|---------------------------|---------------------------------------------|
| `"grLasso"`               | `GroupLassoPathRegressor`                   |
| `"grMCP"`                 | `GroupMCPPathRegressor` (`gamma=3.0`) — native block-CD on the group MCP prox per Breheny & Huang 2015 §3 (v0.8); skein measures **1.20× faster than grpreg** on the canonical `medium / dense` cell. |
| `"grSCAD"`                | (M3.7 wiring; M2.7 surrogate exists)        |
| `"cMCP"` (concave MCP)    | `GroupMCPPathRegressor` — same path solver as `grMCP`. The "concave" framing is grpreg's name for the non-convex penalty; skein's solver is non-convex direct BCD (no LLA wrapper for LS). |
| `"gel"` (group exp lasso) | (not in v0.1)                               |
| `"gBridge"` (group bridge)| (M6 roadmap)                                |

For sparse-group penalties:

| `grpreg`                      | `skein`                                                  |
|-------------------------------|----------------------------------------------------------|
| (not in `grpreg` natively)    | `SparseGroupLassoPathRegressor(groups=g, alpha=0.5)`     |
| (M-step from cMCP)            | `SparseGroupMCPPathRegressor(groups=g, gamma=3.0, alpha=0.5)` |

## Family map

| `grpreg` `family`         | `skein` estimator base                  |
|---------------------------|-----------------------------------------|
| `"gaussian"` (default)    | `Group*Regressor` (LS family)           |
| `"binomial"`              | `LogisticGroup*Regressor`               |
| `"poisson"`               | `PoissonGroup*Regressor`                |
| `grpsurv` (separate fn)   | `CoxGroup*Regressor`                    |

## Per-argument map

| `grpreg` arg            | `skein` arg                       | Notes                                         |
|-------------------------|-----------------------------------|-----------------------------------------------|
| `X`                     | `X` (positional)                  | Same.                                         |
| `y`                     | `y` (positional)                  | Cox: `fit(X, time, event)`.                   |
| `group`                 | `groups` (constructor)            | int label vector.                             |
| `penalty`               | (choose estimator class)          | See penalty map.                              |
| `family`                | (choose family-prefixed class)    | See family map.                               |
| `gamma`                 | `gamma`                           | Same numerical meaning for MCP.               |
| `alpha`                 | `alpha`                           | For sparse-group only; not in plain group.    |
| `lambda`                | `lambdas`                         | numpy array.                                  |
| `nlambda`               | `n_lambdas`                       |                                               |
| `lambda.min`            | `lambda_min_ratio`                |                                               |
| `eps`                   | `tol`                             |                                               |
| `max.iter`              | `max_iter`                        |                                               |
| `group.multiplier`      | `weights` (constructor, on group estimators) | **Per-group penalty weights.** Defaults to `sqrt(group_size)` in `grpreg`; **defaults to ones in skein**. See "group multiplier" below. |
| `nfolds` (cv.grpreg)    | `cv` (PathCV)                     |                                               |

## Group multiplier — important default difference

`grpreg::grpreg` defaults `group.multiplier = sqrt(group sizes)` —
the canonical group-size correction that prevents larger groups
from being more aggressively penalized. **`skein` does not apply
this correction by default.** If you're matching `grpreg` numerics,
pass it explicitly:

```python
import numpy as np
g = np.array([0, 0, 1, 1, 1, 2])   # groups of sizes 2, 3, 1
group_sizes = np.bincount(g)
weights = np.sqrt(group_sizes.astype(float))   # [√2, √3, 1]

fit = skein_glm.GroupLassoPathRegressor(
    groups=g, weights=weights, n_lambdas=50,
).fit(X, y)
```

This is also the default `weights=` you'd want for
`GroupMCPPathRegressor`, `SparseGroupLasso*`, etc. when you want
`grpreg`-compatible behavior.

The reason `skein` defaults to ones rather than `sqrt(|g|)`: the
correction is an opinionated choice (good for groups of features
representing the same "thing", arguably wrong when a single large
group is meaningfully one feature), so we leave it explicit. The
M5.x adaptive weights work will eventually expose a one-shot
helper for the canonical correction.

## Workflow translations

### Group MCP + CV

```r
# R
fit <- cv.grpreg(X, y, group, penalty = "grMCP", gamma = 3, nfolds = 10)
beta <- coef(fit)[-1]
print(predict(fit, type = "groups"))   # which groups are active
```

```python
# Python
fit = skein_glm.GroupMCPPathCV(groups=g, gamma=3.0, cv=10).fit(X, y)
beta = fit.coef_
intercept = fit.intercept_

# Which groups are active (any non-zero coef in the group)?
group_ids = np.unique(g)
active_groups = [
    int(gid) for gid in group_ids
    if np.any(beta[g == gid] != 0)
]
```

### Logistic + group lasso

```r
# R
fit <- grpreg(X, y, group, family = "binomial", penalty = "grLasso")
prob <- predict(fit, X_new, type = "response", lambda = 0.05)
```

```python
# Python
fit = skein_glm.LogisticGroupLassoPathRegressor(
    groups=g, n_lambdas=100,
).fit(X, y_bin)
# At a specific λ:
idx = np.argmin(np.abs(fit.lambdas_ - 0.05))
prob_at_0p05 = fit.predict_proba(X_new)[:, idx]
```

### Sparse-group lasso (skein-only)

```python
# Mixed across- and within-group sparsity.
fit = skein_glm.SparseGroupLassoPathRegressor(
    groups=g, alpha=0.5, n_lambdas=50,
).fit(X, y)
```

`alpha=1` recovers scalar lasso; `alpha=0` recovers group lasso;
intermediate values give "some groups are partially active."

### Survival with group penalty

```r
# R
fit <- grpsurv(X, y = Surv(time, event), group, penalty = "grMCP")
```

```python
# Python
fit = skein_glm.CoxGroupMCPPathRegressor(
    groups=g, gamma=3.0, n_lambdas=50,
).fit(X, time, event)
risk = fit.predict(X_new)
```

## Things grpreg does that skein doesn't yet

- **Group bridge** (`gBridge`): M6 roadmap.
- **Group exponential lasso** (`gel`): not in v0.1.
- **`returnX = TRUE`** (return the standardized X used internally):
  not in v0.1, but you can recover this from `Standardized<D>` if
  needed.

## Things skein does that grpreg doesn't

- **Sparse-group penalties** (convex and nonconvex). `grpreg` has
  composite MCP (`cMCP`) which is related but not the same as
  sparse-group MCP.
- **Rayon-parallel block CD** across groups. `grpreg` is single-
  threaded.
- **Three weight axes** (per-sample, per-feature, per-group)
  composable.
- **`scipy.sparse.csc_matrix` design matrices** native.
- **Memory-mapped / chunked** design matrices for very large `n`.
- **The dual extension surface** for prototyping custom group
  penalties.

## See also

- [Porting from glmnet](glmnet.md) — for scalar-penalty users.
- [Porting from ncvreg](ncvreg.md) — for nonconvex scalar users.
- [Concepts: Penalties](../concepts/penalties.md) — group, sparse-
  group, and the LLA outer loop.
- [Concepts: Weights](../concepts/weights.md) — the per-group weight
  axis in detail.
