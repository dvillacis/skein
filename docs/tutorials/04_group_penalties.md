# 4. Group penalties

When features come in **natural groups** — dummy-encoded categoricals,
gene pathways, polynomial expansions, time-frequency bands — selecting
features one at a time can produce nonsensical results: keep three of
five dummies for the same factor and you've effectively kept all five.
Group penalties fix this by treating an entire group as the unit of
selection.

This tutorial walks through the three flavors of group penalty in
skein and when each is the right choice.

## Building a `groups` vector

`groups` is a length-`n_features` integer vector: feature `j` belongs
to group `groups[j]`. Group labels must form a contiguous range
`{0, 1, …, n_groups - 1}`.

```python
import numpy as np

# 30 features in 6 groups of 5.
groups = np.repeat(np.arange(6), 5).astype(np.int64)
# array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, ...])
```

For a one-hot encoding pattern with mixed group sizes, build the
labels explicitly — `np.repeat([0, 1, 2], [3, 4, 2])` gives
`[0, 0, 0, 1, 1, 1, 1, 2, 2]` for groups of size 3, 4, 2.

## Three penalties, three behaviors

### `GroupLasso` — convex baseline

The L1-of-L2 penalty: `λ Σ_g w_g ‖β_g‖_2`. Convex, easy to fit, but
**biased** on truly active groups (the L2-norm shrinkage applies even
when the group is unambiguously active).

```python
import skein_glm

rng = np.random.default_rng(0)
n, p = 200, 30
X = rng.standard_normal((n, p))
true_beta = np.zeros(p)
true_beta[0:5] = [1.5, -1.0, 0.8, -0.6, 1.2]   # group 0 fully active
true_beta[10:15] = [0.7, -0.5, 0.4, 0.3, -0.6] # group 2 fully active
y = X @ true_beta + 0.3 * rng.standard_normal(n)

groups = np.repeat(np.arange(6), 5).astype(np.int64)

m = skein_glm.GroupLassoPathRegressor(
    groups=groups, n_lambdas=30, lambda_min_ratio=1e-2,
).fit(X, y)

# Active groups: features 0-4 and 10-14 should have nonzero β.
last = m.coefs_[-1]
group_norms = np.array([
    np.linalg.norm(last[groups == g]) for g in range(6)
])
print(group_norms.round(3))  # large at g=0 and g=2, small elsewhere
```

Use this when you want a fast, convex baseline and don't mind the
attenuation bias.

### `GroupMCP` — unbiased nonconvex

Same L2-of-coefficients aggregation but with **MCP shrinkage** at the
group level: shrinks small group norms toward zero, leaves large group
norms alone. Solved via Local Linear Approximation (LLA) — a sequence
of weighted group-lasso problems.

```python
m = skein_glm.GroupMCPPathRegressor(
    groups=groups, gamma=3.0,
    n_lambdas=30, lambda_min_ratio=1e-2,
).fit(X, y)

# `gamma` controls the kink: smaller = more aggressive sparsification.
# Default gamma=3.0 is the ncvreg recommendation.
```

Use this when you want unbiased estimates on the active groups —
which is usually what you want for downstream interpretation.

### `SparseGroupMCP` — hybrid

Combines group-level and **within-group** sparsity: a feature can be
zero even when its group is active. Useful when you expect that only
some features within each "active" group actually contribute.

```python
# Sparse-truth: feature 1 of group 0 is the only active feature there.
true_beta = np.zeros(p)
true_beta[1] = 1.5            # group 0, feature 1 only
true_beta[12] = 0.8            # group 2, feature 12 only
y = X @ true_beta + 0.3 * rng.standard_normal(n)

m = skein_glm.SparseGroupMCPPathRegressor(
    groups=groups, gamma=3.0, alpha=0.5,  # α: within-group L1 vs L2 mix
    n_lambdas=30, lambda_min_ratio=1e-2,
).fit(X, y)

active = np.flatnonzero(np.abs(m.coefs_[-1]) > 1e-6)
print(active)  # [1, 12] — exact within-group sparsity
```

`alpha` controls the mix: `alpha=1` is pure within-group L1,
`alpha=0` is pure group L2. The default `0.5` is a balanced
compromise.

## CV picks λ the same way

```python
cv = skein_glm.GroupMCPPathCV(
    groups=groups, gamma=3.0, cv=5, random_state=0,
    n_lambdas=30, lambda_min_ratio=1e-2,
).fit(X, y)

cv.lambda_best_       # selected λ
cv.coef_              # final-refit β
cv.predict(X[:5])     # standard predict surface
```

## Group penalties for GLMs

Every group-penalty regressor has logistic, Poisson, and Cox
counterparts. Same workflow, different datafit:

```python
clf = skein_glm.LogisticGroupMCPPathRegressor(
    groups=groups, gamma=3.0, n_lambdas=30,
).fit(X, y_binary)

cox = skein_glm.CoxGroupLassoPathRegressor(
    groups=groups, n_lambdas=30,
).fit(X, time, event)
```

## Per-group weights

If you know some groups are *a priori* more important, pass per-group
weights (length `n_groups`):

```python
n_groups = 6
group_w = np.ones(n_groups)
group_w[0] = 0.0       # don't penalize group 0
group_w[1] = 0.5       # half-penalize group 1

m = skein_glm.GroupMCPPathRegressor(
    groups=groups, gamma=3.0, weights=group_w,
).fit(X, y)
```

Per-group weights are the **per-group axis** of skein's three weight
axes (the others are per-sample and per-feature; see
[concepts/weights](../concepts/weights.md)).

## Recap

| Penalty | Convex? | Use for |
|---|---|---|
| `GroupLasso` | yes | fast baseline; entire groups in/out |
| `GroupMCP` | no (LLA) | unbiased estimates on active groups |
| `GroupSCAD` | no (LLA) | alternative to MCP, less aggressive |
| `GroupElasticNet` | yes | correlated groups, want stability |
| `SparseGroupLasso` | yes | groups + within-group sparsity, convex |
| `SparseGroupMCP` | no (LLA) | groups + within-group, unbiased |
| `SparseGroupSCAD` | no (LLA) | as above, SCAD-flavored |

## Next

→ [5. Sparse and standardize](05_sparse_and_standardize.md) — scipy.sparse
input and the standardization story.
