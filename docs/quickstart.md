# Quick start

A tour of the headline features through worked snippets. Every
example assumes you've imported `numpy as np` and `skein`, plus
have a small synthetic problem set up:

```python
import numpy as np
import skein

rng = np.random.default_rng(0)
n, p = 200, 50
X = rng.standard_normal((n, p))
true_beta = np.zeros(p)
true_beta[:3] = [1.5, -2.0, 0.8]
y = X @ true_beta + 0.1 * rng.standard_normal(n)
```

## Single-λ MCP regression

```python
model = skein.MCPRegressor(lambda_=0.05, gamma=3.0).fit(X, y)
print(model.coef_[:5])        # near [1.5, -2.0, 0.8, 0, 0]
print(model.intercept_)       # near 0
print(model.score(X, y))      # R² on training data
```

`gamma` controls the MCP nonconvexity. `gamma=∞` is lasso; smaller
`gamma` is more aggressive sparsification. `gamma >= 3` is a common
default; `gamma=1e6` is a numerically convex stand-in for lasso.

## Full λ-path with warm starts

`*PathRegressor` solves the entire regularization path in one call,
reusing each λ's β as the warm-start for the next. This is what you
want for any analysis that picks λ post-hoc (cross-validation,
information criteria, plotting the path).

```python
model = skein.MCPPathRegressor(
    gamma=3.0, n_lambdas=50, lambda_min_ratio=1e-3,
    standardize=True,
).fit(X, y)

print(model.coefs_.shape)         # (50, 50): n_lambdas × n_features
print(model.intercepts_.shape)    # (50,)
print(model.lambdas_[0:3])        # [λ_max, λ_max·r, λ_max·r²], r = ratio**(1/(K-1))
```

Use `lambdas=` to supply an explicit grid; otherwise `skein` derives
λ_max from the KKT conditions at β=0 and builds a geometric grid.

## Cross-validation

```python
cv = skein.MCPPathCV(gamma=3.0, n_lambdas=50, cv=5, random_state=0).fit(X, y)
print(cv.lambda_best_)            # CV-selected λ
print(cv.coef_, cv.intercept_)    # refit β at λ_best on the full data
print(cv.cv_mean_scores_.shape)   # (50,) — mean test MSE per λ
```

`cv` accepts an integer (KFold) or any sklearn cross-validator. Higher-
is-better scorers (binomial deviance, c-index) are auto-detected for
GLM / Cox families.

## Information criteria (AIC / BIC / EBIC)

```python
path = skein.MCPPathRegressor(gamma=3.0, n_lambdas=50).fit(X, y)
best_idx, scores = skein.select_by_ic(path, X, y, criterion="bic")
print(best_idx, path.lambdas_[best_idx])
print(path.coefs_[best_idx])
```

Works with all four GLM families. EBIC uses `gamma_ebic=0.5` by
default (the high-dimensional recommendation from `ncvreg::BIC`).

## Logistic regression with a group penalty

```python
groups = np.repeat(np.arange(p // 5), 5)   # 10 groups of 5 features each
y_bin = (X[:, :3].sum(axis=1) > 0).astype(float)

clf = skein.LogisticGroupMCPPathRegressor(
    groups=groups, gamma=3.0, n_lambdas=20,
).fit(X, y_bin)

proba = clf.predict_proba(X)               # shape (n, n_lambdas)
labels = clf.predict(X)                    # shape (n, n_lambdas), values in {0, 1}
print(clf.coefs_[-1].nonzero())            # which features are active at smallest λ
```

The path solver runs Local Linear Approximation outer iterations
around a group block-CD inner solver — the headline algorithm. Only
**groups** of features turn on/off; within an active group, every
coefficient is non-zero.

## Cox proportional hazards

```python
time = rng.exponential(1.0 / np.exp(X[:, :3].sum(axis=1)))
event = (rng.uniform(size=n) < 0.7).astype(float)

cox = skein.CoxMCPPathRegressor(gamma=3.0, n_lambdas=50).fit(X, time, event)
risk = cox.predict(X)                       # shape (n, n_lambdas)
                                            # prognostic index η = Xβ;
                                            # higher → shorter survival
print(cox.coefs_[-1].nonzero())
```

No intercept (the baseline hazard absorbs it). Right-censored data;
Breslow ties handling.

## Sparse design matrices

`scipy.sparse.csc_matrix` flows through every estimator transparently
— no separate API:

```python
import scipy.sparse as sp

# Inflate the column scales so standardize=True actually does work.
X_dense = rng.standard_normal((n, p))
X_dense[X_dense < 0.5] = 0.0
X_sparse = sp.csc_matrix(X_dense)

model = skein.MCPPathRegressor(
    gamma=3.0, n_lambdas=20, standardize=True,
).fit(X_sparse, y)

# Same coefficients (up to numerical tolerance) as fitting on X_dense.
```

Group penalties, GLMs, and Cox all dispatch to the same `_sparse`
PyO3 entry points without user-visible API differences.

## Memory-mapped (out-of-RAM) inputs

For problems where `X` is too large to load into memory:

```python
# Write a column-major f64 file. NOTE: np.tofile() always writes in
# C order regardless of array flags, so we use tobytes(order='F'):
X_disk = rng.standard_normal((100_000, 1_000))
buf = np.ascontiguousarray(X_disk, dtype=np.float64).tobytes(order="F")
with open("X.bin", "wb") as f:
    f.write(buf)

design = skein.MmapDesignF64("X.bin", n_rows=100_000, n_cols=1_000)
y_big = rng.standard_normal(100_000)

model = skein.MCPPathRegressor(gamma=3.0, n_lambdas=20).fit(design, y_big)
```

For half the disk footprint (and half the page-cache pressure),
write the matrix as `f32` and use `MmapDesignF32`. The solver stays
f64; each column is cast `f32 → f64` on read.

## Row-block-chunked inputs

For problems where even one mmap'd file is unwieldy (multi-shard
data pipelines, files larger than your filesystem's per-file limit,
etc.):

```python
chunks = [
    ("chunk_0.bin", 10_000_000),
    ("chunk_1.bin", 10_000_000),
    ("chunk_2.bin",  7_345_678),
]
design = skein.ChunkedDesignF64(chunks, n_cols=50_000)
model = skein.MCPPathRegressor(...).fit(design, y_big)
```

Each chunk is a separate column-major file with the same `n_cols`.
The solver streams chunk-by-chunk; total `n` is the sum.
`ChunkedDesignF32` is the f32 sibling.

## Standardization

`standardize=True` divides each column by its glmnet-style standard
deviation (`s_j = sqrt((‖X[:,j]‖² − n·x̄_j²)/n)`) before fitting,
returning `coef_` in original-feature scale. Important for any
design where columns have wildly different magnitudes — without it,
the L1 / group-L2 penalty falls disproportionately on small-scale
columns:

```python
X_inflated = rng.standard_normal((n, p))
X_inflated[:, 0] *= 100.0     # column 0 is on a different scale

# With standardize=True, all columns are penalized comparably.
model = skein.MCPPathRegressor(
    gamma=3.0, n_lambdas=50, standardize=True,
).fit(X_inflated, y)
```

Works for dense, sparse, mmap, and chunked backends, across every
estimator family. See [Concepts: Backends](concepts/backends.md) for
how the lazy `Standardized<D>` wrapper is composed under the hood.

## Where next

- **[Concepts](concepts/index.md)** for the conceptual model behind
  these knobs.
- **Porting from glmnet/ncvreg/grpreg** (coming in commit 2) for
  side-by-side R → Python translation tables.
- **Extending skein** (coming in commit 2) for implementing custom
  penalties or datafits against the trait surface.
