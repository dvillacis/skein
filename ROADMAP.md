# skein Roadmap

The goal: own the niche held by `grpreg` / `ncvreg` (R) and partially by
`skglm` (Python) — nonconvex group-structured sparse models with weights —
and push past it on scale, parallelism, and extensibility. R packages set
the statistical bar; `skglm` sets the Python bar; we beat both on
throughput, design-matrix flexibility, and weighted formulations.

Each milestone is shippable on its own. Headline algorithm (M2) is the
load-bearing piece; everything after stacks on top of it.

## Status snapshot

| Milestone | Status | Notes |
|-----------|--------|-------|
| M0 — Scaffold | ✅ done | trait surface + smoke solver |
| M1 — Production CD core | ✅ done | path solver, screening, Anderson, KKT-stop, standardization |
| M2 — LLA + group block-CD + parallel | ✅ done | inner CD, working set, LLA outer, path, Rayon, op-norm Lipschitz, sparse-group, gap-safe, PyO3, criterion benches |
| M3 — GLM datafits | ⏳ partial | M3.1 trait refactor + M3.2 logistic + M3.3 logistic×group + M3.4 Poisson + M3.5 Cox PH Breslow (Rust + PyO3 + estimators) done; multinomial + Efron ties pending |
| M4 — Design-matrix backends | ⏳ partial | M4.1 SparseCSC core + M4.2 sparse PyO3 (LS + GLM × scalar + group, all 24 path functions) + M4.3 lazy `Standardized<D>` for LS and GLMs (dense + sparse, all 36 GLM estimators) done; mmap + chunked + mixed precision + GPU pending |
| M5 — Model selection & inference | ⏳ partial | M5.1 CV (24 `*PathCV` estimators) + M5.2 information criteria (`select_by_ic` for AIC/BIC/EBIC across all four GLMs) done; stability selection + adaptive + debiased + Rayon-parallel folds pending |
| M6 — Penalty zoo | ⏳ partial | sparse-group already done in M2.7 |
| M7 — Multi-task | ⏳ | multi-response GLMs |
| M8 — Distribution & DX | ⏳ | wheels, CI, docs, comparison benches |

Test count at this snapshot: **175 cargo + 122 pytest, all green.**

---

## ✅ M0 — Scaffold (done, v0.1)

- Workspace + crate layout, MIT license, README.
- Trait surface: `DesignMatrix`, `Datafit`, `Penalty`, `GroupPenalty`, `Groups`.
- Concrete: `DenseMatrix`, `LeastSquares`, `Mcp`, `Scad`, `GroupLasso`, `GroupMcp`.
- Smoke solver: cyclic CD over separable penalties + LS.
- PyO3 bindings: `solve_mcp_ls`, `solve_scad_ls`.
- sklearn estimators: `MCPRegressor`, `SCADRegressor` (single-fit, no path).
- pytest smoke tests; cargo unit tests on prox + CD.

---

## ✅ M1 — Production CD core for separable penalties

Foundational quality bar before any group work.

- ✅ **Standardization & intercept**: column centering / scaling done once
  (glmnet/ncvreg convention: scale relative to column mean even when not
  centering), intercept recovered as `α = ȳ − Σ_j β_j · x̄_j`. Per-feature
  weight rescaling helper for the standardized-space ↔ original-scale
  bijection.
- ✅ **Path solver**: `solve_path(...)` with `lambda_max` from KKT at zero,
  geometric grid, warm starts. Reports per-λ working-set sizes and KKT
  pass counts.
- ✅ **Working set / strong rules**: Tibshirani sequential strong rule +
  KKT verification cycle. Active features always retained (the textbook
  rule wrongly drops them in MCP's saturated regime).
- ✅ **Gap-safe screening** for LS + convex separable penalties
  (Fercoq–Gramfort–Salmon sphere). Selectable per fit via
  `Screening { Off, Strong, GapSafe }`.
- ✅ **Anderson acceleration** on the iterate sequence (Type II,
  Tikhonov-regularized normal equations, obj-decrease safeguard). Off by
  default in `cd_solve_subset` empty-feature case; otherwise
  `acceleration: Some(5)` default.
- ✅ **Convergence**: KKT-based stopping (max coord-update L1 in coefficient
  space), not relative-objective. The old criterion plateaued on objective
  while β was still moving — found via test that obj-tol `1e-8` corresponded
  to coord-residual `7·10⁻⁵`.
- ✅ **PyO3 surfacing**: `solve_mcp_ls_path` / `solve_scad_ls_path` for the
  path; `MCPRegressor` / `SCADRegressor` rebuilt on the path with
  `intercept_`; new `MCPPathRegressor` / `SCADPathRegressor` for the full
  λ-path.

Public surface (frozen at end of M1):
- `solver::{cd_solve, cd_solve_warm, cd_solve_subset, solve_path, lambda_max, lambda_grid, CdConfig, CdReport, PathConfig, PathReport, Screening}`
- `standardize::{standardize, destandardize, destandardize_path, rescale_weights_for_standardize, StandardizeConfig, StandardizationStats}`

Known limitations carried forward:
- `lambda_max` and the path's gradient computation hardcode LS scaling
  (`∂_j L = X_jᵀr/n`); becomes datafit-agnostic when M3's GLM datafits
  introduce a `coord_grad_at_zero` accessor on `Datafit`.
- Anderson is correct but problem-dependent — accepts cleanly when it
  helps, rejects when it doesn't, but doesn't dramatically reduce iter
  counts on the sparse problems we benchmarked.
- Reproducibility: no randomized steps yet; matters for M5 stability
  selection.

---

## ✅ M2 — Headline algorithm: LLA + group block-CD with working set (parallel)

The reason this library exists. Folds nonconvex group penalties (group MCP,
group SCAD, sparse-group MCP/SCAD) into a sequence of weighted convex group
problems via Local Linear Approximation, each solved by group block-CD over
a working set, with groups dispatched across Rayon threads.

Decomposed into shippable sub-milestones; each closes one stack layer.

### ✅ M2.1 — Group block-CD inner solver

- ✅ `block_cd_solve(design, datafit, group_penalty, groups, config)` —
  per-group block prox-gradient step with incremental residual update.
- Initially used Frobenius Lipschitz bound; tightened to operator-norm
  in M2.6.
- Convergence: max-block-L₂ coefficient-space tolerance.
- Singleton-group equivalence test against scalar `cd_solve` validates
  the reduction.

### ✅ M2.2 — Working set over groups

- ✅ `block_strong_rule_screen` and `block_find_kkt_violators` (private,
  used by M2.4 path solver).
- ✅ `block_cd_solve_subset` mirroring M1's `cd_solve_subset`.
- ✅ Always-keep-active convention: groups with any non-zero β
  coordinate are retained regardless of the strong rule's verdict.

### ✅ M2.3 — LLA outer loop

- ✅ Generic `lla_solve(design, datafit, groups, init_β, λ, update_weights, …)`
  — closure-based; user supplies a `Fn(β, &Groups) → Array1<f64>` that
  computes per-group surrogate weights from the current iterate.
- ✅ `surrogate_weights_group_mcp` helper: returns
  `max(0, w_base − ‖β_g‖/(λγ))` per group.
- Outer loop terminates on max block change. Typically 2–5 outer
  iterations.
- *Pending*: `surrogate_weights_group_scad`, estimators
  `GroupMCPRegressor` / `GroupSCADRegressor`.

### ✅ M2.4 — Path solver for groups

- ✅ `solve_block_path(...)` analogous to M1's `solve_path`, threading β
  across decreasing λ with strong-rule + KKT cycle.
- ✅ `BlockPathConfig`, `BlockPathReport` with per-λ
  working-set sizes and KKT passes.
- ✅ `block_lambda_max(...)` for the auto-grid.
- ✅ `solve_block_path_lla(...)` — wraps the path with LLA outer loop, so
  users fit a non-convex group-MCP/SCAD path through one entry point.
  Surrogate weights closure is `Fn(β, &Groups, λ) → Array1<f64>`.
- *Pending*: block-level gap-safe screening (`Screening::GapSafe`
  currently falls back to `Strong` for blocks).

### ✅ M2.5 — Rayon parallelism over groups

- ✅ `block_cd_solve_subset_parallel` — Jacobi-style sweeps via
  `rayon::par_iter`, snapshot β + r at sweep start, serial apply phase
  reduces deltas into β and r.
- ✅ `BlockPathConfig.parallel: bool` flag dispatches the path solver
  through the parallel inner.
- *Caveat documented inline*: per-group Frobenius/operator-norm
  Lipschitz is correct for serial Gauss-Seidel; for Jacobi it's correct
  when off-diagonal `X_gᵀ X_{g'}` coupling is small (uncorrelated
  groups). Overlapping groups also fall back to serial in spirit (no
  explicit overlap detection yet).
- *Pending*: criterion benchmark to quantify the wall-clock speedup at
  `n_groups ≫ n_threads`.

### ✅ M2.6 — Tight per-group Lipschitz

- ✅ Power iteration on `X_gᵀ X_g` (30 iters) to compute
  `‖X_g‖_op² / n`. Singleton groups short-circuit to `col_sq_norm/n`.
- ✅ Replaces the Frobenius bound at both `block_cd_solve_subset`
  callsites (serial + parallel).

### ✅ M2.7 — Sparse-group variants

- ✅ `SparseGroupLasso` (convex) — Simon-Friedman-Hastie-Tibshirani
  two-step prox, with both single-weight (`with_weights`) and dual-weight
  (`with_coord_weights`) constructors. The dual-weight form holds
  per-group L2 weights AND per-position-in-group L1 weights, so it can
  serve as the LLA inner penalty for sparse-group MCP/SCAD.
- ✅ `surrogate_sparse_group_mcp` helper returning
  `(per_group_L2_weights, per_group-per-position L1 weights)` from the
  current iterate. Handles edge cases α = 0 (pure group MCP) and α = 1
  (pure scalar MCP) cleanly.
- ✅ `solve_block_path_lla` refactored to take a `Fn(β, &Groups, λ) →
  Box<dyn GroupPenalty>` closure. The user constructs the surrogate
  inner penalty from `surrogate_*` helpers; the path solver reads the
  per-group L2 weights via `inner.weights()` for the strong rule + KKT
  verifier.
- ✅ Within-group sparsity recovery test passes: on a problem with one
  active feature inside a group, sparse-group MCP via LLA zeros the
  inactive feature 1 while keeping feature 0 above 0.5 — what
  `SparseGroupLasso` (convex) can't do as cleanly because the L1
  surrogate weight stays at the base value rather than dropping toward
  zero for active coordinates.
- *Pending*: `SparseGroupSCAD` surrogate helper (analogous to MCP but
  with SCAD's piecewise-linear derivative); a `SparseGroupMcp` /
  `SparseGroupScad` type that implements `GroupPenalty` for objective
  reporting (currently the user computes objective via a manual
  surrogate sum).

### ✅ M2.8 — Outstanding integrations

- ✅ Block-level **gap-safe screening** for convex group lasso.
- ✅ **PyO3 surface** — 4 path functions
  (`solve_group_lasso_ls_path`, `solve_group_mcp_ls_path`,
  `solve_sparse_group_lasso_ls_path`, `solve_sparse_group_mcp_ls_path`)
  + 8 sklearn-style estimators (`Group{Lasso,MCP}{,Path}Regressor`,
  `SparseGroup{Lasso,MCP}{,Path}Regressor`) with sklearn-style
  label-vector group spec.
- ✅ **Group operator-norm cache** — `block_gap_safe_screen`,
  `block_cd_solve_subset`, and `block_cd_solve_subset_parallel` share a
  precomputed `group_lipschitz_cache` built once per fit. Public APIs
  unchanged; private `*_with_cache` variants are what the path solvers
  call.
- ✅ **SCAD surrogate helpers** — `surrogate_weights_group_scad` and
  `surrogate_sparse_group_scad` mirror the MCP variants.
- ✅ **Criterion benchmark scaffold** under
  `crates/skein-core/benches/` with `serial_vs_parallel` and
  `screening_modes` scenarios. Findings in `benches/README.md`:
  - Strong rule + gap-safe both deliver ~6× speedup over `Off` on a
    64-group path.
  - Jacobi parallel block-CD is *slower* than serial at small problem
    sizes — Rayon task overhead dominates. Larger-scale benchmarks
    (sparse X, n_groups in the thousands) need M4's `SparseCSC` first.

### Pending (post-M2 polish, not blocking M3)

- Larger-scale parallel benchmarks once M4's sparse design backend
  lands.
- Comparison benchmarks vs. `glmnet` / `ncvreg` / `grpreg` (M8).
- LLA-iteration-count benchmark for SCAD/MCP nonconvex paths.
- Cleanup pass for clippy 1.95 lints introduced by the toolchain bump
  (cosmetic; non-blocking).

---

## M3 — GLM datafits

`ncvreg` / `grpreg` cover Gaussian + binomial + Poisson + Cox. `skglm`
covers a wider GLM zoo. We need parity, then beat them on scale.

Decomposed into shippable sub-milestones; each closes one stack layer.

### ✅ M3.1 — Datafit trait refactor

Generalizes the M1/M2 solver call sites away from hardcoded
`X_jᵀr/n` formulas (correct only for unweighted LS) to dispatch through
the `Datafit` trait. This is the test of whether the trait surface
absorbs new datafits unchanged.

- ✅ Added `coord_grad(design, j, residual)` (required) and
  `full_grad(design, residual)` (default loops `coord_grad`; LS-shaped
  datafits override with one matvec) to the `Datafit` trait.
- ✅ `LeastSquares` honors `sample_weights` everywhere now (`value`,
  `coord_grad`, `full_grad`, `coord_lipschitz`); previously only `value`
  used them.
- ✅ Refactored ~10 call sites across `cd.rs`, `path.rs`, `block_cd.rs`,
  `block_path.rs`, `block_path_lla.rs` to dispatch gradient computations
  through the trait.

### ✅ M3.2 — Logistic regression (binomial logit)

- ✅ `BinomialLogit` type with `surrogate_at(β) → LeastSquares` (working
  response + per-sample weights) and `loss(β)` (numerically stable
  cross-entropy via `softplus(η)`).
- ✅ `prox_newton_solve` (single λ) and `prox_newton_solve_path` (λ-path
  with warm starts across both outer iters and λ steps). λ_max derived
  from the surrogate at β = 0.
- ✅ PyO3 surface: `solve_logistic_mcp_path`, `solve_logistic_scad_path`
  + 4 sklearn-style estimators (`LogisticMCP{,Path}Regressor`,
  `LogisticSCAD{,Path}Regressor`) with `predict` (class labels),
  `predict_proba` (P(y=1)), `decision_function` (linear scores).
- ✅ Intercept via internal X augmentation: append 1s column + extend
  per-feature penalty weights with `[…, 0.0]` (unpenalized).
- *Pending in M3.x*: gap-safe / strong-rule screening inside the
  prox-Newton inner CD (currently uses `cd_solve_warm`, no screening).

### ✅ M3.3 — Group/sparse-group + GLMs

Composes M2's group block-CD machinery with M3.2's prox-Newton outer
loop. Each outer iteration linearizes both the GLM loss (prox-Newton
quadratic surrogate) AND any non-convex group penalty (LLA surrogate),
yielding a convex weighted-LS-plus-weighted-group-lasso inner that
M2's solvers handle unchanged.

- ✅ **M3.3.1**: `prox_newton_block_solve_path` taking a
  `Fn(β, &Groups, λ) → Box<dyn GroupPenalty>` closure for the inner
  penalty (mirrors `solve_block_path_lla` shape). Drives all
  combinations: logistic + group lasso, logistic + group MCP via LLA,
  logistic + sparse-group lasso, logistic + sparse-group MCP via LLA.
- ✅ **M3.3.2**: PyO3 surface + 8 estimators —
  `solve_logistic_group_lasso_path`, `solve_logistic_group_mcp_path`,
  `solve_logistic_sparse_group_lasso_path`,
  `solve_logistic_sparse_group_mcp_path`; sklearn-style
  `LogisticGroupLasso{,Path}Regressor`,
  `LogisticGroupMCP{,Path}Regressor`,
  `LogisticSparseGroupLasso{,Path}Regressor`,
  `LogisticSparseGroupMCP{,Path}Regressor` (`predict`,
  `predict_proba`, `decision_function` inherited from the M3.2 logistic
  bases). Intercept handled by augmenting X with a 1s column AND adding
  a singleton intercept group at index `n_groups` with weight 0.

### ✅ M3.4 — Poisson regression

`PoissonLog` analogous to `BinomialLogit`: `μ_i = exp(η_i)`, weights
`w_i = μ_i`, working response `z_i = η_i + (y_i − μ_i)/μ_i`. Same
prox-Newton scaffold reused.

- ✅ **M3.4.1**: `GlmDatafit` trait (`surrogate_at`, `loss`) extracted;
  `BinomialLogit` and `PoissonLog` both impl it. `prox_newton_solve`,
  `prox_newton_solve_path`, `prox_newton_block_solve_path` all generic
  over `&dyn GlmDatafit`. `PoissonLog` clamps η to `[-30, 30]` before
  `exp()` and floors `w_i` at `1e-6` for numerical stability; per-sample
  weights honored. 11 new cargo tests (5 unit + 3 prox-Newton scalar +
  3 prox-Newton block group/LLA).
- ✅ **M3.4.2**: 6 PyO3 functions (`solve_poisson_{mcp,scad,group_lasso,
  group_mcp,sparse_group_lasso,sparse_group_mcp}_path`); 12 sklearn-
  style estimators with `decision_function` (η = log-rate),
  `predict` (μ = rate, matches sklearn `PoissonRegressor`); y ≥ 0
  validation. Two helpers shared with logistic via closure
  parameterization (`build_glm_path_outputs`, `build_glm_block_path_outputs`).
- *Pending in M3.x*: Poisson offsets (log-exposure for rate models)
  deferred to M3.7. Same `lambda_max-at-β=0` heuristic the logistic
  path uses; intercept warm-starting at `log ȳ` would tighten λ_max
  but isn't required for correctness.

### ✅ M3.5 — Cox proportional hazards

`CoxPH` with the Breslow tie-handling default. Different shape
(partial likelihood, no separate `y` per sample — `(time, event)`
instead). Reuses `GlmDatafit` trait from M3.4 — only `surrogate_at`
and `loss` semantics differ.

- ✅ **M3.5.1**: `CoxPH` datafit with time-sort permutation precomputed
  once; reverse-cumulative `S(t)` (risk-set sum) and forward-cumulative
  `CumH` / `CumH2` per outer iter (O(n)). Diagonal Hessian
  `w_i = exp(η_i)·CumH(t_i) − exp(2η_i)·CumH2(t_i)` floored at `1e-6`,
  working response `z_i = η_i − g_i/w_i`. η-clamp ±30. Breslow ties
  share `S(t)` within tie-blocks. 13 cargo tests (7 unit on hand-derived
  values + 3 prox-Newton scalar + 3 prox-Newton block group/LLA).
- ✅ **M3.5.2**: 6 PyO3 functions taking `(x, time, event, …)`
  (`solve_cox_{mcp,scad,group_lasso,group_mcp,sparse_group_lasso,sparse_group_mcp}_path`);
  12 sklearn-style estimators with 3-arg `fit(x, time, event)`.
  No `fit_intercept`, no `intercept_` (baseline hazard absorbs).
  `predict(x) = decision_function(x) = Xβ` (prognostic index, matches
  `glmnet::predict.cox`). Deferred to M3.7: per-sample weights
  (frequency vs. probability weighting), Efron ties, Breslow's
  cumulative-baseline-hazard estimator for absolute survival
  predictions.

### M3.6 — Multinomial / softmax

`MultinomialLogit` with the grouped-by-class parameterization so group
penalties penalize a feature's whole row of class coefficients. Reuses
the multi-task path from M7 once that lands.

### M3.7 — Opportunistic GLMs

Gaussian-with-offsets, negative binomial, Huber / quantile (smoothed) —
ship whichever has user demand.

---

## M4 — Design-matrix backends (the scale story)

This is where we leave R and `skglm` behind. Both assume the design
matrix fits in RAM as a dense or CSC numpy/Matrix object.

### ✅ M4.1 — `SparseCSC` Rust core

CSC layout matching scipy.sparse.csc_matrix (`data`, `indices`,
`indptr`); precomputed `col_sq_norms` keeps CD's per-coord Lipschitz
lookup O(1). `matvec` skips `β_j == 0` entries (warm-start friendly).
Constructor validates indptr invariants and row-index bounds.
`design.rs` reorganized into `design/` module with `dense.rs` +
`sparse_csc.rs`. 11 unit tests against a hand-built 4×3 reference
matrix plus 2 solver-equivalence tests proving sparse CD/path produces
the same β as dense within 1e-7 on the same data.

### M4.2 — Sparse PyO3 surface (in progress)

Each `solve_*_path` PyO3 function gets a `_sparse` sibling taking
`(data, indices, indptr, n_rows, n_cols, ...)`. Estimators sniff
`scipy.sparse.issparse(x)` and dispatch transparently. Sparse + intercept
uses column-augmentation (1s column with penalty weight 0), the same
scheme the GLM dense paths already use — different from the dense LS
centering trick, but mathematically equivalent at convergence.
`standardize_x=True` is rejected for sparse inputs with a clear error
message (centering would densify); a lazy `Standardized<D>` wrapper to
restore parity is M4.x.

- ✅ **M4.2a** (LS scalar): `solve_mcp_ls_path_sparse`,
  `solve_scad_ls_path_sparse`. `MCPRegressor` / `MCPPathRegressor` /
  `SCADRegressor` / `SCADPathRegressor` all dispatch on
  `scipy.sparse.issparse(x)`. `predict()` accepts sparse `x`.
  7 pytest tests prove dense ↔ sparse equivalence on a shared
  λ-grid (auto-grid differs slightly because dense centers `y`
  before computing λ_max while sparse computes it at β = 0 with no
  intercept warm-start).
- ✅ **M4.2b** (LS group): 4 sparse PyO3 wrappers
  (`solve_{group_lasso,group_mcp,sparse_group_lasso,sparse_group_mcp}_ls_path_sparse`)
  with column-augmented intercept matching the GLM scheme. Shared
  `_ls_group_dispatch_inputs` Python helper handles the dense-vs-sparse
  branch for all 8 group estimators in `python/skein/estimators.py`.
  6 pytest tests prove dense ↔ sparse equivalence on a shared λ-grid
  for group lasso, group MCP (γ=1e6 ≈ lasso), and sparse-group lasso,
  plus smoke tests for sparse-group MCP and the `standardize=True`
  rejection.
- ✅ **M4.2c** (GLMs): 18 sparse PyO3 wrappers covering logistic,
  Poisson, and Cox × scalar/group (6 each). Cox uses the no-intercept
  path; logistic/Poisson use column-augmentation. All 36 GLM
  estimators dispatch transparently on `scipy.sparse.issparse(x)`.
  Two shared dispatch helpers in `python/skein/estimators.py`
  (`_glm_dispatch_inputs`, `_cox_dispatch_inputs`) handle the
  dense/sparse branch + validation; each estimator's `fit()` becomes
  ~15 lines. `decision_function`, `predict`, and `predict_proba` on
  every base class accept sparse `x` without densifying.
  8 pytest tests prove dense ↔ sparse equivalence on shared λ-grids
  for logistic MCP path, logistic group lasso path, Poisson MCP path,
  and Cox MCP path; smoke tests cover Poisson group lasso, Cox group
  lasso, predict_proba/predict equivalence.

### ✅ M4.3 — Lazy `Standardized<D>` (scale-only, sparse LS)

`Standardized<D>` wraps any `DesignMatrix` with per-column scales.
`col_dot` divides by `s_j`, `col_sq_norm` by `s_j²`, `matvec` divides
its input β element-wise by `s`, `rmatvec` divides its output by `s`,
`columns()` densifies + scales — all O(nnz_j) per column, no extra
state. Generic over the base backend, so it composes cleanly with
`SparseCSC` and any future design.

The dense LS path centers + scales `X` (and centers `y`) and recovers
the intercept from `α = ȳ − x̄ᵀ(β/s)`. The sparse LS path can't center
without densifying, so it uses **scaling only** + the column-augmented
intercept already in place: append a 1s column with penalty weight 0,
wrap the augmented `SparseCSC` in `Standardized<SparseCSC>` with
`x_scale = [s, 1.0]` (intercept column unscaled), solve, and divide
the non-intercept β by `s` at the end. Mathematically equivalent to
the dense centering+scaling path at convergence (centering and
unpenalized intercept feature are dual parameterizations of the same
LS problem).

Per-column glmnet std `s_j = sqrt((‖X[:,j]‖² − n·x̄_j²)/n)` computed
in O(nnz) directly off the CSC arrays. Constant columns (s ≈ 0) clamp
to `1.0`. Per-feature penalty weights rescale by `w_j / s_j` (matches
the M1 `rescale_weights_for_standardize` convention); per-group
weights stay unchanged (group penalty applies in standardized space,
matching the dense LS group path).

- ✅ **Rust core**: 9 cargo tests against pre-scaled `DenseMatrix`
  reference (matvec/rmatvec/col_dot/col_sq_norm/columns); 1 solver-
  equivalence test proving CD path on `Standardized<SparseCSC>` matches
  pre-scaled `DenseMatrix` within 1e-7.
- ✅ **PyO3**: extended the 6 sparse LS PyO3 functions
  (`solve_{mcp,scad}_ls_path_sparse` + 4 group variants) with a new
  `standardize_x` kwarg.
- ✅ **Python estimators**: removed the `standardize=True` rejection
  for sparse LS. The 8 LS estimators (4 scalar + 4 group) now work
  with `standardize=True` on sparse input. 2 dense↔sparse equivalence
  pytest tests on inflated-scale problems prove the two
  parameterizations produce the same β at every λ in a shared grid;
  one smoke test covers sparse + group lasso + standardize.
- ✅ **GLMs + standardize** (dense + sparse): extended the 4 GLM/Cox
  helpers (`build_glm_path_outputs`, `build_glm_block_path_outputs`,
  and Cox counterparts) and their 4 sparse twins to thread
  `standardize_x` through every prox-Newton path. Compute glmnet
  scales before intercept augmentation, wrap the (possibly augmented)
  design in `Standardized<D>` with `x_scale = [s, 1.0]` (intercept
  unscaled; Cox has no intercept so the wrapper goes on the user
  matrix directly), divide non-intercept β by `s` at the end. Per-
  feature L1 weights rescale by `1/s_j`; per-group weights stay
  unchanged (matches the LS group standardize convention). Surfaced
  `standardize` on all 36 GLM estimators (logistic / Poisson / Cox ×
  scalar / group / sparse-group × `{,Path}Regressor`); threaded
  through `_glm_dispatch_inputs` / `_cox_dispatch_inputs`. Added a
  `compute_dense_glmnet_scales` helper mirroring the CSC version so
  dense GLMs use the same scale-only + augmentation recipe as sparse
  (unifies dense and sparse at convergence). 4 new cargo tests prove
  `Standardized<D>` ∘ prox-Newton matches pre-scaled `DenseMatrix`
  references for logistic / Poisson / Cox / logistic-group; 5 new
  pytest tests prove dense↔sparse equivalence on inflated-scale
  problems for logistic MCP path, logistic group lasso path, Poisson
  MCP path, Cox MCP path, plus an end-to-end signal-recovery test on
  a 50×-inflated column.

### M4.x — mmap, chunked, mixed precision, GPU
- **`MmapMatrix`**: memory-mapped dense `f32`/`f64` from disk. The
  trait already restricts the solver to `col_dot` / `columns`, so this
  drops in without algorithm changes.
- **`ChunkedMatrix`**: row-block streaming for out-of-core LS / GLM
  fits where columns are accessed via partial `Xᵀr` reductions per
  chunk. This is what makes `n` in the hundreds of millions tractable.
- **`Float32` / mixed precision**: parameterize the core over
  `T: Float`. Path solver in f32 for the bulk of work, refine at the
  active set in f64.
- **GPU backend** (stretch): `cubla`s / `wgpu` matvecs behind the same
  trait. Only worth it once dense `n × p` matvec is the bottleneck;
  measure first.

Bench target: a 10⁶ × 10⁵ sparse logistic group-lasso path in under
the time `glmnet` takes on a 10⁵ × 10³ dense one.

---

## M5 — Model selection & inference

Without these, we ship a solver, not a library people fit on real data.

### ✅ M5.1a — Cross-validation for LS scalar

`python/skein/cv.py` adds a `_PathCVMixin` that runs K-fold CV over a
shared λ-grid: a single full-data fit yields the auto-grid (and
doubles as the refit producing the final β); each fold then fits the
same path estimator on its train rows, predicts on the held-out rows,
and aggregates per-λ scores. Best λ minimizes the mean test MSE
(higher-is-better scorers are an opt-in via `_score_higher_better`).
Sequential folds; sparse input flows through unchanged via the
underlying estimator's dispatch.

- ✅ `MCPPathCV`, `SCADPathCV` — sklearn-compatible `cv_scores_`,
  `cv_mean_scores_`, `cv_std_scores_`, `lambdas_`, `lambda_best_`,
  `coef_`, `intercept_`, `n_features_in_`. `cv` accepts an int (KFold
  with shuffle) or any sklearn CV splitter; `random_state` seeds the
  default KFold shuffle.
- ✅ 7 pytest tests: shape + finiteness, sign recovery on a
  noiseless-ish problem, predict shape, explicit-λ-grid path,
  scipy.sparse input parity, SCAD smoke, sklearn `KFold` splitter.

### ✅ M5.1b — CV for LS group penalties

4 more `*PathCV` wrappers via the shared `_PathCVMixin`:
`GroupLassoPathCV`, `GroupMCPPathCV`, `SparseGroupLassoPathCV`,
`SparseGroupMCPPathCV`. Each exposes `cv` (int K or sklearn splitter)
+ `random_state` + the underlying path estimator's full constructor
surface (`groups`, `gamma`, `alpha`, `coord_weights`, `parallel`,
`max_outer`, `outer_tol`, etc.). 5 pytest tests: active-group recovery
under group lasso CV, smoke for group MCP / sparse-group lasso /
sparse-group MCP, and dense↔sparse predict parity for group lasso CV.

### ✅ M5.1c — CV for GLM families

3 GLM mixins added on top of `_PathCVMixin`:

- `_LogisticPathCVMixin` scores by binomial deviance (lower-is-better);
  the final estimator exposes `decision_function` (η), `predict_proba`
  (σ(η), 1D since CV picks a single λ), and class-label `predict` —
  inheriting from `ClassifierMixin`.
- `_PoissonPathCVMixin` scores by Poisson deviance with the y log y
  convention; the final estimator's `predict` returns the conditional
  mean `μ = exp(η)`, `decision_function` returns η.
- `_CoxPathCVMixin` is a separate mixin (different fit signature
  `fit(x, time, event)`, no intercept). Default `cv=int` uses
  `StratifiedKFold` by event indicator so heavy censoring doesn't
  produce event-empty train folds; folds with zero events are
  defensively skipped (NaN scores → `np.nanmean` aggregation).
  Scores by Harrell's concordance index (higher-is-better);
  `predict(x) = decision_function(x) = Xβ` (the Cox prognostic index).

18 wrappers added (6 per GLM family): `LogisticMCPPathCV`,
`LogisticSCADPathCV`, `LogisticGroupLassoPathCV`,
`LogisticGroupMCPPathCV`, `LogisticSparseGroupLassoPathCV`,
`LogisticSparseGroupMCPPathCV`, plus the parallel Poisson and Cox
families. 10 pytest tests cover: deviance shape and finiteness,
predict_proba / class-label semantics, group CV smoke for each GLM,
Cox c-index above 0.5 for a correctly-ordered model, no
`intercept_` on Cox CV, predict ≡ decision_function on Cox, and
sparse-input parity.

### ✅ M5.2 — Information criteria

`python/skein/ic.py` adds `select_by_ic(path_model, x, *outcomes,
criterion="bic", ebic_gamma=0.5)` — a single free function that picks
the best λ from any fitted `*PathRegressor` by AIC, BIC, or EBIC. No
per-estimator wrapper explosion: dispatch sniffs the path estimator's
class name to pick the right NLL helper (`_compute_nll_ls`,
`_compute_nll_logistic`, `_compute_nll_poisson`, `_compute_nll_cox`),
and the rest is one `coefs_`-shaped active-set count plus the
criterion arithmetic.

- LS NLL: `(n/2)·log(RSS/n)`. Logistic: `Σ softplus(η)−y·η`. Poisson:
  `Σ exp(η)−y·η`. Cox: `n · path_model.info_["final_losses"][k]` (the
  Breslow per-sample partial NLL the Rust core already computes).
- Effective df = `Σ |β_j| > 1e-12` per λ — the Zou-Hastie-Tibshirani
  unbiased estimator and the standard ncvreg/glmnet convention.
- AIC = `2k + 2·NLL`. BIC = `log(n)·k + 2·NLL`. EBIC = `BIC + 2γ·log
  C(p,k)` with `γ ∈ [0,1]` (default 0.5; matches `ncvreg::BIC`'s
  high-dim recommendation).
- 9 pytest tests covering: BIC sanity for LS, AIC-vs-BIC sensitivity
  (AIC keeps more features active for `n > e²`), EBIC stricter than
  BIC when `p > n`, per-GLM dispatch (logistic / Poisson / Cox), Cox
  rejecting a single-y outcome, criterion / `ebic_gamma`
  argument validation, and dense ↔ sparse score parity.

### M5.x — Other model selection (pending)

- **Stability selection** (Meinshausen–Bühlmann): bootstrap fits over
  a λ-path, return per-feature selection probabilities. Embarrassingly
  parallel; fits into the Rayon dispatch.
- **Adaptive weights**: one-shot `AdaptiveLasso` / `AdaptiveMCP`
  estimators that fit a coarse model first, derive `w_j ∝ 1/|β̂_j|^η`,
  then refit. This is one of the headline reasons the per-feature
  weight axis exists.
- **Debiased / desparsified lasso** for confidence intervals on the
  active set (Van de Geer–Bühlmann–Ritov). Optional, behind a feature
  flag — we are not a general inference library, but ignoring CIs
  cedes ground to R.
- **Rayon-parallel folds**: move CV's per-fold loop into Rust and
  dispatch across threads. Big speedup for fast solves.

---

## M6 — Penalty zoo expansion

Once the solver core is solid, penalties are cheap to add. Priority
ordered by user demand and by what differentiates us.

> Sparse-group lasso, sparse-group MCP, and the convex+LLA infrastructure
> for sparse-group SCAD are already done in **M2.7**.

- **Elastic net** (lasso + ridge), **group elastic net**.
- **`SparseGroupSCAD` end-to-end** — the M2.7 surrogate helper exists;
  what's left is wiring through to a `SparseGroupSCADRegressor` (Rust
  trait + PyO3) for users who want it directly.
- **Overlapping group lasso / latent group lasso** via duplication
  trick; surface a friendly group-construction API.
- **Fused lasso / generalized lasso**: 1D and graph-structured
  fusion. Solved via specialized prox (taut-string for 1D, ADMM for
  general). Lives behind `solver::fused`.
- **Adaptive group MCP / SCAD** with weights from a pilot fit.
- **Bridge penalty** (`|β|^q`, q < 1) — closes parity with `grpreg`.
- **Constrained variants**: nonneg lasso, box constraints. Implemented
  by post-prox projection.

---

## M7 — Multi-task / multi-response

`skglm` has multi-task lasso; R has it via custom packages. We wire it
through the existing trait surface.

- **Block-row coefficient matrix** `B ∈ ℝ^{p × K}`.
- **MultiTaskLasso / MultiTaskMCP / MultiTaskSCAD**: penalty acts on
  rows of `B` (so feature j is selected jointly across tasks).
- **Multi-response least squares + GLMs**.
- **Shared-support estimators** for the "same active features across
  related outcomes" use case that genomics + finance both want.

---

## M8 — Distribution & developer experience

Ship-grade polish. Without this, none of the above gets adopted.

- **Wheels**: `cibuildwheel` for Linux x86_64 / aarch64, macOS
  arm64+x86_64, Windows x86_64. ABI3 (already configured).
- **CI**: GitHub Actions running `cargo test`, `cargo clippy
  -- -D warnings`, `pytest`, `ruff`, `mypy`, plus a benchmark-regression
  job that fails PRs that slow the LS+MCP path by >5%.
- **Benchmarks**: criterion suites in `benches/`, asv suite in
  `benches/python/`, a published comparison page vs. `glmnet`,
  `ncvreg`, `grpreg`, `skglm`, `celer`.
- **Docs**: mkdocs site with a "porting from glmnet/ncvreg" cheat
  sheet, an "extending skein" guide that walks through implementing a
  custom `Penalty`, and worked examples for genomics, NLP, survival.
- **Numerical regression tests**: pin coefficient values from
  reference R fits so we never silently drift from `ncvreg`/`grpreg`
  numerics on canonical datasets.
- **Stable Rust API contract**: tag `skein-core` 0.x but document
  what's `pub` and intentional vs. `pub` and incidental. Downstream
  per-paper crates depend on this.

---

## Differentiators (the elevator pitch)

When someone asks "why not just `skglm` / `ncvreg`?":

1. **Three weight axes, first class** — per-sample, per-feature,
   per-group — wired through every solver. R packages support some,
   none support all. `skglm` partially.
2. **Nonconvex group penalties at scale** — group MCP, group SCAD,
   sparse-group MCP/SCAD via LLA + parallel block-CD. `grpreg` has the
   penalties but is single-threaded R; `skglm` has the parallelism but
   not the nonconvex group penalties.
3. **Design-matrix abstraction** — sparse, memory-mapped, chunked,
   standardized-on-the-fly, GPU later — all behind one trait.
   Algorithm code never sees the backend. R/Python competitors hard-code
   dense + CSC.
4. **Rust core, Python sklearn API, extension surface in both** —
   downstream researchers can prototype a custom penalty in Python
   against the same ABCs the Rust traits mirror, then port hot ones
   to Rust without re-architecting.
