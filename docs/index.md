# skein

**Weighted structured nonconvex sparse models. Rust core, Python API.**

`skein` targets a niche that R fills well (`grpreg`, `ncvreg`) but
Python doesn't: nonconvex group-structured penalties (group MCP,
group SCAD, sparse-group nonconvex) with first-class support for
**weights along three axes** — per-sample, per-feature, and per-group
— and design-matrix backends that go beyond "fits in RAM."

## Why skein

When someone asks "why not just `skglm`, `glmnet`, or `ncvreg`?":

1. **Three weight axes, first class.** Per-sample, per-feature,
   per-group weights wired through every solver. R packages support
   some, none support all. `skglm` partially.
2. **Nonconvex group penalties at scale.** Group MCP, group SCAD,
   sparse-group MCP/SCAD via Local Linear Approximation + parallel
   block coordinate descent. `grpreg` has the penalties but is
   single-threaded R; `skglm` has the parallelism but not the
   nonconvex group penalties.
3. **Design-matrix abstraction.** Dense, sparse CSC, memory-mapped
   (f64 + f32), row-block-chunked (f64 + f32), standardized-on-the-fly,
   intercept-augmented — all behind one trait. Algorithm code never
   sees the backend; competitors hard-code dense + CSC.
4. **Rust core, Python sklearn API, extension surface in both.**
   Downstream researchers can prototype a custom penalty in Python
   against the same ABCs the Rust traits mirror, then port hot ones
   to Rust without re-architecting.

## What's in v0.1

| Family    | Datafits                          | Penalties                                          | Estimators           |
|-----------|-----------------------------------|----------------------------------------------------|----------------------|
| Gaussian  | Least squares                     | MCP, SCAD, group lasso, group MCP,<br>sparse-group lasso, sparse-group MCP | 12 sklearn classes |
| Binomial  | Logistic (with prox-Newton)       | Same as above                                      | 12 sklearn classes   |
| Poisson   | Log-link                          | Same as above                                      | 12 sklearn classes   |
| Cox PH    | Breslow ties                      | Same as above                                      | 12 sklearn classes   |

48 estimators total. Plus 24 `*PathCV` cross-validation wrappers, plus
`select_by_ic` for AIC/BIC/EBIC across all four GLM families.

## Quick taste

```python
import numpy as np
from skein_glm import MCPPathRegressor, LogisticGroupMCPPathRegressor, CoxMCPRegressor

# Nonconvex sparse least squares with a λ-path.
rng = np.random.default_rng(0)
n, p = 200, 50
X = rng.standard_normal((n, p))
y = X[:, :3] @ np.array([1.5, -2.0, 0.8]) + 0.1 * rng.standard_normal(n)
model = MCPPathRegressor(gamma=3.0, n_lambdas=50, standardize=True).fit(X, y)
print(model.coefs_[-1, :5], model.intercepts_[-1])

# Logistic + group MCP via LLA, with sklearn-style predict/predict_proba.
groups = np.repeat(np.arange(p // 5), 5)  # 5 features per group
y_bin = (X[:, :3].sum(axis=1) > 0).astype(float)
clf = LogisticGroupMCPPathRegressor(groups=groups, gamma=3.0, n_lambdas=20).fit(X, y_bin)
proba = clf.predict_proba(X)  # shape (n, n_lambdas)

# Cox PH with right-censored survival data.
time = rng.exponential(1.0 / np.exp(X[:, :3].sum(axis=1)))
event = rng.uniform(size=n) < 0.7
cox = CoxMCPRegressor(lambda_=0.01, gamma=3.0).fit(X, time, event.astype(float))
risk = cox.predict(X)  # prognostic index η
```

Every regressor follows the same `(datafit) × (penalty)` × `({,Path}Regressor)`
naming scheme. The path variants warm-start across λ; their `coefs_` /
`intercepts_` are 2D arrays indexed by λ.

## Where to next

- **[Installation](installation.md)** — pip + from source.
- **[Quick start](quickstart.md)** — worked snippets covering paths,
  CV, IC selection, sparse, and memory-mapped inputs.
- **[Concepts](concepts/index.md)** — the conceptual model: penalties,
  datafits, weights, and design-matrix backends.
- **[Roadmap](roadmap.md)** — what's in v0.1, what's coming next, and
  the differentiator pitch.

## Status

v0.1 is a complete, tested implementation: **199 cargo + 138 pytest
tests, all green** at last snapshot. Sparse + dense + mmap + chunked
backends all interoperate; every datafit × penalty combination is
wired end-to-end with sklearn-style `fit` / `predict` /
`predict_proba` / `score`. Wheels are built via `cibuildwheel` for
Linux (x86_64 + aarch64), macOS (x86_64 + arm64), and Windows
(AMD64).

What's not yet in: multinomial / multi-task / elastic net (M3.6, M6,
M7), comparison benchmarks vs `glmnet`/`ncvreg`/`grpreg`/`skglm`
(M8), numerical regression tests against R reference fits. See the
[roadmap](roadmap.md) for the full picture.
