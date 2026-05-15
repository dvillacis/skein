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

## What's in v0.8

| Family       | Datafits                          | Penalties                                          | Estimators           |
|--------------|-----------------------------------|----------------------------------------------------|----------------------|
| Gaussian     | Least squares                     | MCP, SCAD, elastic net, lasso (α=1 facade), bridge `\|β\|^q`, group lasso, group MCP, group elastic net, sparse-group lasso, sparse-group MCP, sparse-group SCAD | 18 sklearn classes |
| Multi-task   | Multi-response least squares      | Multi-task lasso / MCP / SCAD / elastic net (row-grouped, dense + sparse, ±standardize) | 8 sklearn classes |
| Binomial     | Logistic (with prox-Newton)       | MCP, SCAD, elastic net, lasso (first-class convex L1, not the MCP-at-γ=1e9 approximation), group lasso, group MCP, sparse-group lasso, sparse-group MCP, sparse-group SCAD | 18 sklearn classes   |
| Multinomial  | Softmax (K classes, prox-Newton + Böhning bound) | Row-grouped lasso / MCP / SCAD / elastic net (dense + sparse, ±standardize) | 12 sklearn classes |
| Poisson      | Log-link, offset support          | Same as binomial                                   | 18 sklearn classes   |
| Cox PH       | Breslow + Efron ties              | MCP, SCAD, group lasso, group MCP, sparse-group lasso, sparse-group MCP, sparse-group SCAD | 14 sklearn classes   |
| Graphical models | Sparse precision `Θ = Σ⁻¹`    | L1, MCP, SCAD on edges; per-edge weights; joint estimation across `K` populations (Danaher–Wang–Witten group form via ADMM); EBIC tuner | 5 sklearn-style classes |
| Inference    | VBR debiased lasso (LS + logistic + Poisson) | Wald CIs / p-values for high-dimensional penalized fits; nodewise inverse-Gram / Fisher approximation | 3 free functions + 3 sklearn-style wrappers |
| Edge stability | Bootstrap edge stability for graphical models | MB-style stability selection on edges + bootnet-style non-parametric bootstrap | 2 sklearn-style classes |

122 regression / classification estimators + 5 graphical-model
estimators + 5 inference / edge-stability classes. Plus 36 `*PathCV`
wrappers, `select_by_ic` for AIC/BIC/EBIC across all four GLM
families, `ebic_path` / `joint_ebic_path` for graphical models, and
**every** CV class supports `n_jobs` for threaded fold parallelism
(real parallelism — every Rust path solver releases the GIL during
compute). 28 adaptive variants span LS, group, logistic, Poisson, and
Cox families.

**v0.8 is a hardening + performance release.** No new estimators or
penalties; instead the focus is on closing the M12 audit punch list
and shipping two M13 perf wins. The headlines:

1. **Native group-MCP block-CD for LS** (M13.4b) — `solve_group_mcp_ls_path`
   no longer routes through an LLA outer loop wrapping
   `GroupLasso(weighted)`; it now calls block-CD directly on
   `GroupMcp::prox_group` (Breheny & Huang 2015 §3 closed-form prox).
   On the canonical `medium / dense` `ls_group_mcp` cell (n=10k, p=1k,
   group_size=5, 100 λs, γ=3.0): **wall drops 36.2 s → 10.5 s
   (-3.46×)**. Compared against grpreg's 12.56 s on the same cell,
   skein flips from "3.34× slower" to **"1.20× faster than grpreg"**.
   `max_outer` / `outer_tol` parameters are kept on the Python
   estimators for backward compat (kwargs still accept them) but are
   now ignored. Logistic / Poisson / Cox group-MCP variants are
   unchanged (still on LLA).
2. **Cross-λ gradient cache in `solve_path`** (M13.2) — the warm-start
   residual at λ_{k+1} is the post-CD residual at λ_k, so the
   gradient `compute_outer_state` computed at the end of λ_k is the
   gradient `priority_rule_screen` needs at the start of λ_{k+1}.
   Caching it skips one O(np) matvec per λ. **Medium Lasso wall
   2.847 s → 2.552 s (-10.4 %)**, iter count + KKT passes unchanged.
   Plus an env-var-gated `PhaseTimings` instrumentation
   (`SKEIN_PROFILE_PATH=1`) attributing per-λ time to setup /
   screening / lipschitz / inner_cd / outer_state / dual_extrap /
   bookkeeping — kept as permanent observability for future perf work.

The M12 hardening close-out: penalty + datafit numerical guards
centralized (`W_FLOOR`, `ETA_CLAMP`); pre-flight tight-tol screening
test promoted to a fail-fast CI step with a 2-min timeout;
`Groups::has_overlap()` + parallel block-CD overlap detection with
serial Gauss-Seidel fallback (overlap-misuse no longer ships
silently); criterion bench tree expanded with new microbenches for
LLA outer-loop, prox-Newton GLM, and glasso ADMM scaling; and
`crates/skein-py/src/lib.rs` split from a 10 628-line monolith into
seven focused modules (one per datafit family — `glasso.rs`,
`glm.rs`, `ls.rs`, `mmap_chunked.rs`, `multinomial.rs`,
`multitask.rs` — plus a 275-line `lib.rs` entry point).

Full per-feature changelog: [`CHANGELOG.md`](https://github.com/dvillacis/skein/blob/main/CHANGELOG.md).

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

# Logistic + group MCP (native non-convex BCD), with sklearn-style predict/predict_proba.
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
- **[Tutorials](tutorials/index.md)** — eleven guided walkthroughs
  (basics, structure, advanced, plus the two graphical-model
  tutorials). Read in order or skip to the tier that matches what
  you already know.
- **[Concepts](concepts/index.md)** — the conceptual model: penalties,
  datafits, weights, and design-matrix backends.
- **[Roadmap](roadmap.md)** — what's in v0.1, what's coming next, and
  the differentiator pitch.

## Status

v0.8 is a complete, tested implementation. Sparse + dense + mmap +
chunked + multi-task backends all interoperate; every datafit ×
penalty combination is wired end-to-end with sklearn-style `fit` /
`predict` / `predict_proba` / `score`. The graphical-model family
ships with a gram-form CD inner solver (for single-population glasso)
and an ADMM kernel (for joint glasso). Every Rust path solver
releases the GIL during compute, so Python-side `joblib` parallelism
is real rather than a no-op. Wheels are built via `cibuildwheel` for
Linux (x86_64 + aarch64), macOS (x86_64 + arm64), and Windows
(AMD64). **M12 hardening is fully done** (penalty + datafit unit
tests, integration test directory, R-fixture CI gate, PyO3 smoke job,
fail-fast pre-flight CI step, parallel block-CD overlap detection,
centralized numerical guards, criterion bench expansion, and the
PyO3 facade split into per-datafit modules). **M13 perf is in
progress** — M13.1 / M13.2 / M13.4 (Phase 2.3 + 4b) shipped; M13.5
MCP one-outer-iter short-circuit and a follow-up to extend native
group-MCP BCD to the GLM variants remain open. Test count: **355
cargo + 412 pytest, all green**.

What's not yet in: multi-response GLMs for Poisson / Cox (M7.3),
polychoric / polyserial correlation helpers for ordinal Likert data,
an R facade. See the [roadmap](roadmap.md) for the full picture.

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
tutorials/10_graphical_lasso
tutorials/11_joint_networks
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
concepts/graphical_models
concepts/graph_inference
concepts/polychoric
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
examples/psychometrics
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
api/estimators-graphical
api/cv
api/ic
api/stability
api/graph_selection
api/graph_stability
api/debiased
api/design
api/abcs
```

```{toctree}
:hidden:
:caption: Benchmarks

benchmarks/index
benchmarks/lasso_ls_correctness
benchmarks/mcp_ls
benchmarks/scad_ls
```

```{toctree}
:hidden:
:caption: Performance

perf/lasso_ls_profile
perf/celer_skglm_study
perf/m13_4_profile
```

```{toctree}
:hidden:
:caption: Project

roadmap
```
