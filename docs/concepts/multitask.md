# Multi-task / multi-response

Multi-task LS fits `K` related linear regressions jointly under a
penalty that selects features at the **row level** of the
coefficient matrix `B ∈ ℝ^(p × K)`. Whereas running `K` independent
scalar fits would let each task pick a different active set, the
multi-task formulation lets each *feature* be either active in
every task or inactive in every task — useful when you believe the
underlying support is shared.

The data is `X ∈ ℝ^(n × p)` and `Y ∈ ℝ^(n × K)`. The objective is

$$
\min_{B} \;\; \frac{1}{2nK}\, \|Y - XB\|_F^2 \;+\; \lambda \sum_{j=1}^{p} w_j \, \|B[j, :]\|_2
$$

for the multi-task lasso (the convex case). Reading off the
penalty: `B[j, :]` is the row of feature `j` across the K tasks,
and we penalize its L2 norm. If the row goes to exactly zero, the
feature is dropped from every task simultaneously.

## The four shapes

| Estimator                          | Penalty           | Convex?        | Use when                           |
|------------------------------------|-------------------|----------------|------------------------------------|
| `MultiTaskLassoRegressor`          | row-L2            | yes            | baseline; matches sklearn          |
| `MultiTaskMCPRegressor`            | row-MCP via LLA   | non-convex     | want unbiased active features      |
| `MultiTaskSCADRegressor`           | row-SCAD via LLA  | non-convex     | alternative to MCP, less aggressive |
| `MultiTaskElasticNetRegressor`     | row-L2 + row-L2²  | yes            | correlated features, want stability |

Each has a `*PathRegressor` sibling (full λ-path with warm starts)
and a `*PathCV` wrapper (K-fold CV scoring by mean per-task MSE).
The estimators live in `skein_glm.multitask` and are also re-exported
at the package root, so `skein_glm.MultiTaskLassoRegressor` works.

## How it works: stack-and-reuse

Multi-task LS reduces *exactly* to a group-lasso problem on a virtual
design — no new solver code in skein. The trick: lay `B` out
row-major as a length-`pK` vector `bvec[jK + k] = B[j, k]`, stack
`Y` task-outer as `yvec[k·n + i] = Y[i, k]`, and define a virtual
design `X̃ ∈ ℝ^(nK × pK)` where column `jK + k` holds the base
column `X[:, j]` lifted into row block `k` (rows `k·n..(k+1)·n`)
and zeros elsewhere.

With this layout the row-grouping of `B` becomes the standard
contiguous-block grouping `{jK, jK+1, …, jK+K-1}` for each feature.
The block-CD machinery from M2 handles the resulting group-lasso
problem unchanged. The wrapper that materializes `X̃` lazily is
called `MultiTaskDesign<D>` in the Rust core; it carries no state
beyond the inner design and the task count `K`, and every
`DesignMatrix` op (matvec, col_dot, columns block) is `O(n)` —
the same cost as a scalar column op on the base.

The same wrapper composes with `Augmented` (for sparse intercept
augmentation) and `Standardized` (for per-feature scaling), so
`MultiTaskDesign<Standardized<Augmented<SparseCSC>>>` is a
real (and supported) backend stack.

## Convention vs sklearn

skein and sklearn use **different objective scalings**:

| Library  | Data fidelity factor | Penalty constant      |
|----------|----------------------|-----------------------|
| skein    | `1/(2nK)`            | `λ`                   |
| sklearn  | `1/(2n)`             | `α`                   |

The same minimizer is reached at `λ_skein = α_sklearn / K`. So
when porting code from sklearn:

```python
# sklearn
sklearn.linear_model.MultiTaskLasso(alpha=0.05).fit(X, Y)

# skein equivalent (Y has K columns)
K = Y.shape[1]
skein_glm.MultiTaskLassoRegressor(lambda_=0.05 / K).fit(X, Y)
```

skein keeps the natural `1/(2nK)` convention so the MCP/SCAD shape
parameters (`gamma`, `a`) have the same meaning across the rest of
the library — those control where the per-row penalty kink falls
on the L2 norm scale, which is dimensionally consistent with the
natural formulation.

The pytest suite has a regression test that fits both sklearn's
`MultiTaskLasso` and skein's `MultiTaskLassoRegressor` at the
matching `(alpha, lambda)` pair and asserts coefficient agreement
to `1e-5`.

## Sparse and standardize

Both work for multi-task. The Python estimators dispatch on
`scipy.sparse.issparse(X)`; the underlying Rust path then routes
through `MultiTaskDesign<SparseCSC>`. The intercept is handled
two different ways depending on the input:

- **Dense**: per-task center `Y` and per-feature center `X`
  (physically), fit a no-intercept model, recover per-task
  intercepts via $\alpha_k = \bar{y}_k - \sum_j \bar{x}_j B[j, k]$.
- **Sparse**: append a single `1`s column to the inner CSC (the
  `MultiTaskDesign` wrapper replicates it `K` times into `K`
  virtual intercept columns living in disjoint row blocks), give
  the resulting row-group weight `0`, and read each per-task
  intercept directly off `bvec[p·K + k]`.

Both are equivalent at convergence — the dense pytest suite
includes a parity test on a problem with `standardize=True`
turned on for both backends.

## Limitations

skein v0.2 ships **multi-task LS only** — no multi-task GLM
(logistic, Poisson, Cox) and no multinomial logit yet. The
algebraic reduction works the same way for those, but the
prox-Newton outer loop needs adjustment; that's M7.3.

`coord_weights` (per-task L1 weights independent of the per-row L2
weights) aren't exposed yet; the per-feature `weights=` argument
is uniform across tasks. The Rust core could support this via the
existing `SparseGroupLasso::with_coord_weights` machinery, but the
sklearn-facing API for it isn't designed yet.

## See also

- [Multi-task estimators](../api/estimators-multitask.md) —
  full API reference.
- [Cross-validation](../api/cv.md) — `MultiTaskLassoPathCV` etc.
- [Penalties](penalties.md) — the row-L2, row-MCP, etc.
  penalty shapes that the multi-task family applies to `B[j, :]`.
