# 5. Sparse and standardize

Real data is rarely a dense float64 matrix. This tutorial covers two
practical concerns: **sparse input** (scipy.sparse CSC) and
**standardization** (per-column scaling).

## Sparse input

skein dispatches transparently on `scipy.sparse.issparse(X)`. Pass a
CSC matrix and the path solver routes through the sparse backend; the
estimator interface is identical.

```python
import numpy as np
import skein_glm
from scipy import sparse

rng = np.random.default_rng(0)
n, p = 500, 100

# Build a sparse design with ~10% nonzero entries.
X_dense = rng.standard_normal((n, p))
X_dense[rng.uniform(size=X_dense.shape) > 0.1] = 0.0

X_csc = sparse.csc_matrix(X_dense)
print(f"density = {X_csc.nnz / X_csc.size:.2%}")

true_beta = np.zeros(p)
true_beta[[0, 5, 12, 30]] = [1.5, -1.0, 0.8, -1.2]
y = X_csc @ true_beta + 0.3 * rng.standard_normal(n)

m_sparse = skein_glm.MCPPathRegressor(
    gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-2,
).fit(X_csc, y)

# predict() also accepts sparse input.
m_sparse.predict(X_csc).shape   # (n_samples, n_lambdas)
```

What you can pass to `fit`:

- `numpy.ndarray` (dense float64).
- `scipy.sparse.csc_matrix` (compressed sparse column — the layout
  the solver needs).
- `scipy.sparse.csr_matrix`, `coo_matrix`, etc. — converted to CSC
  internally.
- `MmapDesignF64` / `MmapDesignF32` — memory-mapped column-major
  files (covered in tutorial 9 / advanced).
- `ChunkedDesignF64` / `ChunkedDesignF32` — row-block streaming.

## Standardization

Per-column scaling matters when features have wildly different
variances. A column with stddev 1000 dominates the L1 / L2 penalty
unless you scale it.

```python
# Inflate one feature's scale by 50×.
X_scaled = X_dense.copy()
X_scaled[:, 1] *= 50.0

# Without standardization, feature 1's penalty is too small relative
# to its scale — it'll get over-selected.
m_no_std = skein_glm.MCPPathRegressor(
    gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-2,
).fit(X_scaled, y)

# With standardization, the solver scales internally and returns
# coefficients in original-feature scale.
m_std = skein_glm.MCPPathRegressor(
    gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-2, standardize=True,
).fit(X_scaled, y)

print("no std β[1] last λ:  ", m_no_std.coefs_[-1, 1].round(4))
print("with std β[1] last λ:", m_std.coefs_[-1, 1].round(4))
```

Skein uses **glmnet's standardization convention**:

- `s_j = sqrt((‖X[:,j]‖² − n·x̄_j²) / n)` per column.
- The solver runs in scaled-β-space; the returned `coef_` and
  `intercept_` are converted back to the original feature scale.
- Constant columns (`s_j ≈ 0`) clamp to `s_j = 1` and are left
  untouched.

The bottom line: **`coef_` and `intercept_` are always in
original-feature units**. You don't need to re-scale after fitting.

## The dense ↔ sparse equivalence

A subtle point worth knowing: skein's dense and sparse paths use
**different intercept conventions** internally (centering for dense,
column augmentation for sparse), but they reach the same minimizer at
the same λ-grid. That equivalence is verified by tests in the suite.

```python
# Same problem, dense and sparse — get the same coefficients.
m_dense = skein_glm.MCPPathRegressor(
    gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-2,
).fit(X_dense, y)

m_sparse = skein_glm.MCPPathRegressor(
    gamma=3.0, lambdas=m_dense.lambdas_,  # share the λ-grid
).fit(X_csc, y)

np.testing.assert_allclose(m_dense.coefs_, m_sparse.coefs_, atol=1e-7)
```

(Why share the λ-grid? Auto-grid generation uses the residual at
β=0, which is computed slightly differently for dense vs sparse. With
an explicit grid, the two backends produce identical coefficients
within solver tolerance.)

## Sparse + standardize

Combining the two: sparse + `standardize=True`. Internally this uses
a **lazy column-scaling wrapper** so the matrix never densifies.

```python
m = skein_glm.MCPPathRegressor(
    gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-2, standardize=True,
).fit(X_csc, y)
```

Per-column scales are computed from CSC arrays in O(nnz) without
materializing the dense matrix. The solver wraps the CSC backend in
`Standardized<SparseCSC>` for the duration of the fit.

## Per-feature weights

A third axis for weighting: per-feature penalty weights. Useful when
some features should be regularized harder than others (or not at all
— set the weight to 0):

```python
# Always include feature 0 (no penalty on it).
weights = np.ones(p)
weights[0] = 0.0

m = skein_glm.MCPPathRegressor(
    gamma=3.0, weights=weights,
    n_lambdas=20, lambda_min_ratio=1e-2,
).fit(X_dense, y)

# Feature 0 will be active at every λ — never penalized.
```

Per-feature weights compose with `standardize=True` correctly: skein
rescales them by `1/s_j` internally so the standardized-space penalty
matches the original-scale one.

## Recap

| Concern | API |
|---|---|
| Sparse design matrix | pass `scipy.sparse` directly to `fit` |
| Per-column standardization | `standardize=True` |
| Per-feature weights | `weights=array_of_length_p` |
| Memory-mapped huge X | `MmapDesignF64(path, n_rows, n_cols)` |
| Row-chunked streaming | `ChunkedDesignF64([(path, n_rows), ...], n_cols)` |

`coef_` and `intercept_` are always in original-feature units, no
matter which backend you used.

## Next

→ [6. Counts and rates](06_counts_and_rates.md) — Poisson regression with
exposure offsets.
