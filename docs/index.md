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

## What's in v0.9

| Family       | Datafits                          | Penalties                                          | Estimators           |
|--------------|-----------------------------------|----------------------------------------------------|----------------------|
| Gaussian     | Least squares                     | MCP, SCAD, elastic net, lasso (α=1 facade), bridge `\|β\|^q`, group lasso, group MCP, group elastic net, sparse-group lasso, sparse-group MCP, sparse-group SCAD | 18 sklearn classes |
| Multi-task   | Multi-response least squares      | Multi-task lasso / MCP / SCAD / elastic net (row-grouped, dense + sparse, ±standardize) | 8 sklearn classes |
| Binomial     | Logistic (with prox-Newton)       | MCP, SCAD, elastic net, lasso (first-class convex L1, not the MCP-at-γ=1e9 approximation), group lasso, group MCP, sparse-group lasso, sparse-group MCP, sparse-group SCAD | 18 sklearn classes   |
| Multinomial  | Softmax (K classes, prox-Newton + Böhning bound) | Row-grouped lasso / MCP / SCAD / elastic net (dense + sparse, ±standardize) | 12 sklearn classes |
| Poisson      | Log-link, offset support          | Same as binomial                                   | 18 sklearn classes   |
| Cox PH       | Breslow + Efron ties              | MCP, SCAD, group lasso, group MCP, sparse-group lasso, sparse-group MCP, sparse-group SCAD | 14 sklearn classes   |
| Graphical models | Sparse precision `Θ = Σ⁻¹`    | L1, MCP, SCAD on edges; per-edge weights; joint estimation across `K` populations (Danaher–Wang–Witten group form via ADMM); EBIC tuner | 5 sklearn-style classes |
| Inference (debiased)    | VBR debiased lasso (LS + logistic + Poisson + **Cox** PH) | Wald CIs / p-values for high-dimensional penalized fits; nodewise inverse-Gram / Fisher approximation. Cox uses the partial-likelihood Fisher diagonal (new in v0.9). | 4 free functions + 4 sklearn-style wrappers |
| Edge inference | Bootstrap edge stability + **multiple-testing correction** on edges | MB-style stability selection on edges, bootnet-style non-parametric bootstrap, **Benjamini–Hochberg FDR**, **Bonferroni / Holm FWER**, **Meinshausen–Bühlmann closed-form bound** (new in v0.9). | 2 sklearn-style classes + 4 helper functions |
| Preprocessing  | **Polychoric / polyserial correlations** (Olsson 1979 / Olsson-Drasgow-Dorans 1982 two-step ML, new in v0.9) | Latent-Gaussian correlation for ordinal Likert / mixed-type data. Output feeds directly into `GraphicalLasso(cov=…)` and friends. | 3 helper functions |

123 regression / classification estimators + 5 graphical-model
estimators + 6 inference / edge-stability classes + 11 helper /
preprocessing functions. Plus 36 `*PathCV` wrappers,
`select_by_ic` for AIC/BIC/EBIC across all four GLM families,
`ebic_path` / `joint_ebic_path` for graphical models, and **every**
CV class supports `n_jobs` for threaded fold parallelism (real
parallelism — every Rust path solver releases the GIL during
compute). 28 adaptive variants span LS, group, logistic, Poisson,
and Cox families.

**v0.9 is the *research-grade* release.** Closes the inference axis
across all four GLM families, fills the methodological gaps that
separated `skein` from a citation-grade methods package in its
target application domains (network psychometrics, genomics,
survival), and ships the remaining M13 / M14 perf items so the
algorithm surface is now uniformly native — no LLA wrappers
underneath any GLM × group penalty. The headlines:

### Inference & applications (M14a)

1. **Cox debiased lasso** — `DebiasedCoxLassoRegressor` extends the
   Van de Geer / Cai-Wang construction to Cox PH, reusing the
   partial-likelihood Fisher diagonal from
   `CoxPH::surrogate_at` via a 16-line PyO3 binding. No mainstream
   Python package has this. Empirical 95 % CI coverage ≥ 80 % on
   inactive coordinates over 40 replications.
2. **Edge-level error control on graphical models** —
   `GraphicalBootstrap.fdr_threshold(0.1)` / `.fwer_threshold(0.05)`
   apply BH / Bonferroni / Holm to the per-edge bootstrap
   distribution; `GraphicalStabilitySelection.mb_threshold(EV)`
   inverts the Meinshausen–Bühlmann closed-form bound to a
   stability threshold. No other graphical-models package controls
   error rates at the edge level.
3. **Polychoric / polyserial preprocessing** —
   `polychoric_correlation(X)`,
   `polyserial_correlation(X_ord, Y)`, and
   `polychoric_covariance_matrix(X)` (auto-detecting mixed-type)
   recover the latent-Gaussian correlation underlying ordinal
   Likert data via Olsson's two-step ML. Closes the M11.1
   psychometrics-replication exit criterion that was deferred since
   v0.7 — `docs/examples/psychometrics.md` is now an end-to-end
   `polychoric → GraphicalMCP → bootstrap-FDR` pipeline.

### Performance & correctness closeout (M13.4c + M13.8 + M14c)

4. **Native group-MCP block-CD for logistic / Poisson / Cox**
   (M13.4c) — extends M13.4b across GLM families. The
   `prox_newton_block_solve_path` outer loop now hands a
   β-independent `GroupMcp::with_weights(λ, γ, w)` factory directly,
   dropping the LLA layer underneath prox-Newton. On the
   `logistic group-MCP medium` cell (n=4 000, p=400, γ=3.0): **wall
   drops 226.7 s → 106.8 s (-2.12×)** with min support Jaccard 0.97
   vs LLA and identical final-λ objective.

4½. **Celer-style gap-safe screening on the GLM prox-Newton
   surrogate** (M13.8) — closes the F-series infrastructure that
   M10 wave F left LS-only. `LeastSquares::lasso_dual_obj` now
   handles the weighted case (`Σwᵢrᵢ²` replaces `‖r‖²`), and
   `GlmDatafit` gains `glm_dual_obj` / `glm_per_sample_loss_grad`
   with closed-form impls for `BinomialLogit` (sigmoid Fenchel)
   and `PoissonLog` (Bregman, offset-aware). The new
   `prox_newton_solve_screened` runs the same per-λ KKT loop as
   `solve_path` does for LS — gap-safe sphere screening,
   Anderson dual extrapolation on `(β, r)` pairs, adaptive inner
   tol `= max(tol, 0.3 × prev_outer_pgd)`, M13.1-style saturation
   bypass — on the weighted-LS surrogate. Same wiring in
   `prox_newton_block_solve_path` via the generalised
   `block_gap_safe_screen`. Wall-clock on bench v2 `logistic_lasso`
   (host `3c43bb844695`): **8.2× small-sparse, 3.1× small-deep,
   7.2× medium-sparse, 2.2× medium-deep** (the last with 95 %
   saturation, where the win comes from Anderson dual extrapolation
   + adaptive inner tol rather than from screening itself). The
   unweighted-LS path is unchanged (`max(w) → 1.0` collapses to
   the original FGS 2015 formula).
5. **Native sparse-group MCP for logistic / Poisson / Cox** (M14c.2)
   — sibling of M13.4c for the sparse-group penalty. New Rust
   `SparseGroupMcp` penalty implements the Breheny & Huang (2015)
   Proposition 1 closed-form (per-coord scalar MCP + per-group
   block MCP); 6 PyO3 closures swapped. Drops the last LLA wrapper
   in the non-convex GLM × group family.
6. **Scalar LLA weight short-circuit** (M14c.1) — ports the
   M13.4 Phase 2.3 fix from `block_path_lla.rs` to scalar
   `path_lla.rs`. Affects bridge `|β|^q`, adaptive lasso, and
   multi-task LLA paths. Average outer iters per λ at convergence
   drops to ~1.2 on a representative bridge q=0.5 problem.
7. **At-scale R-fixture tier** (M14c.3) — `tests/fixtures/generate.R`
   gains an n=500, p=100 mid tier for three representative penalty
   / family combinations, with looser tolerances. Catches
   scale-dependent regressions that the n=200 small tier would
   miss.

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

v0.9 is a complete, tested implementation. Sparse + dense + mmap +
chunked + multi-task backends all interoperate; every datafit ×
penalty combination is wired end-to-end with sklearn-style `fit` /
`predict` / `predict_proba` / `score`. The graphical-model family
ships with a gram-form CD inner solver (for single-population
glasso) and an ADMM kernel (for joint glasso). Every Rust path
solver releases the GIL during compute, so Python-side `joblib`
parallelism is real rather than a no-op. Wheels are built via
`cibuildwheel` for Linux (x86_64 + aarch64), macOS (x86_64 +
arm64), and Windows (AMD64). **M12 hardening done**, **M13 perf
work done** through M13.4c (native group-MCP for all GLM
families) and **M13.8** (celer-style gap-safe screening on the
GLM prox-Newton surrogate, 3–8× wall on `logistic_lasso` v2
cells), and **M14 inference + applications + perf closeout
done** through M14a / M14c — Cox debiasing, edge-level FDR/FWER/MB,
polychoric preprocessing, scalar LLA short-circuit, native
sparse-group MCP for GLMs, and an at-scale R-fixture tier.
Test count: **358 cargo lib + 8 integration + 455 pytest, all
green**.

**Coming next**: M14b — run the full `benches/v2` GLM + graphical
headline matrix and draft the JMLR-MLOSS / JOSS software-paper
manuscript from the 10 figures + 5 tables that already
auto-generate. Smaller open items: multi-response GLMs for Poisson
/ Cox (M7.3), an R facade, n=5000 / p=2000 at-scale fixtures
(needs an artifact-server pipeline). See the [roadmap](roadmap.md)
for the full picture.

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
concepts/inference
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
benchmarks/speed
benchmarks/lasso_ls_correctness
benchmarks/mcp_ls
benchmarks/scad_ls
benchmarks/at_scale
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
