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

## What's in v0.3

| Family       | Datafits                          | Penalties                                          | Estimators           |
|--------------|-----------------------------------|----------------------------------------------------|----------------------|
| Gaussian     | Least squares                     | MCP, SCAD, **elastic net**, **bridge `\|β\|^q`**, group lasso, group MCP, **group elastic net**, sparse-group lasso, sparse-group MCP, **sparse-group SCAD** | 18 sklearn classes |
| Multi-task   | Multi-response least squares      | Multi-task lasso / MCP / SCAD / elastic net (row-grouped, dense + sparse, ±standardize) | 8 sklearn classes |
| Binomial     | Logistic (with prox-Newton)       | MCP, SCAD, group lasso, group MCP, sparse-group lasso, sparse-group MCP, **sparse-group SCAD** | 14 sklearn classes   |
| Multinomial  | Softmax (K classes, prox-Newton + Böhning bound) | Row-grouped lasso / MCP / SCAD / elastic net (dense + sparse, ±standardize) | 12 sklearn classes |
| Poisson      | Log-link, **offset support**      | Same as binomial                                   | 14 sklearn classes   |
| Cox PH       | **Breslow + Efron** ties           | Same as binomial                                   | 14 sklearn classes   |

108 estimators total (incl. 28 adaptive variants spanning LS, group,
logistic, Poisson, and Cox families). Plus 51 `*PathCV` cross-
validation wrappers, plus `select_by_ic` for AIC/BIC/EBIC across all
five GLM families.

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
- **[Tutorials](tutorials/index.md)** — nine guided walkthroughs in
  three tiers (basics, structure, advanced). Read in order or skip
  to the tier that matches what you already know.
- **[Concepts](concepts/index.md)** — the conceptual model: penalties,
  datafits, weights, and design-matrix backends.
- **[Roadmap](roadmap.md)** — what's in v0.1, what's coming next, and
  the differentiator pitch.

## Status

v0.3 is a complete, tested implementation: **265 cargo + 279 pytest
tests, all green** at last snapshot. Sparse + dense + mmap + chunked
+ multi-task backends all interoperate; every datafit × penalty
combination is wired end-to-end with sklearn-style `fit` / `predict` /
`predict_proba` / `score`. Wheels are built via `cibuildwheel` for
Linux (x86_64 + aarch64), macOS (x86_64 + arm64), and Windows
(AMD64).

What's not yet in: multi-response GLMs for Poisson / Cox (M7.3) and
comparison benchmarks vs `glmnet`/`ncvreg`/`grpreg`/`skglm` (M8).
See the [roadmap](roadmap.md) for the full picture.

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
tutorials/01_first_fit
tutorials/02_picking_lambda
tutorials/03_logistic_and_cox
tutorials/04_group_penalties
tutorials/05_sparse_and_standardize
tutorials/06_counts_and_rates
tutorials/07_stability_selection
tutorials/08_adaptive_estimators
tutorials/09_multinomial_and_multitask
```

```{toctree}
:hidden:
:caption: Concepts

concepts/index
concepts/penalties
concepts/datafits
concepts/weights
concepts/backends
concepts/multitask
concepts/multinomial
```

```{toctree}
:hidden:
:caption: Porting from R

porting/glmnet
porting/ncvreg
porting/grpreg
```

```{toctree}
:hidden:
:caption: Extending

extending/penalty
extending/datafit
extending/backend
extending/rust-api
```

```{toctree}
:hidden:
:caption: Examples

examples/genomics
examples/nlp
examples/survival
```

```{toctree}
:hidden:
:caption: API reference

api/index
api/estimators-ls
api/estimators-multitask
api/estimators-adaptive
api/estimators-logistic
api/estimators-multinomial
api/estimators-poisson
api/estimators-cox
api/cv
api/ic
api/stability
api/design
api/abcs
```

```{toctree}
:hidden:
:caption: Performance

perf/lasso_ls_profile
perf/celer_skglm_study
```

```{toctree}
:hidden:
:caption: Project

roadmap
```
