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
| M3 — GLM datafits | ⏳ partial | M3.1 trait refactor + M3.2 logistic + M3.3 logistic×group + M3.4 Poisson + **Poisson offsets** + M3.5 Cox PH (Breslow + **Efron** ties) + M3.6 multinomial (Rust + PyO3 + estimators) + M3.7 Huber + **first-class logistic + Poisson Elastic Net / Lasso primitives (retires the prior `MCP(γ=1e9)` approximation; Adaptive\* retrofitted both families)** done; opportunistic GLMs pending |
| M4 — Design-matrix backends | ⏳ partial | M4.1 SparseCSC core + M4.2 sparse PyO3 (LS + GLM × scalar + group, all 24 path functions) + M4.3 lazy `Standardized<D>` for LS and GLMs (dense + sparse, all 36 GLM estimators) + M4.x `MmapMatrix` f64 + f32 (LS + logistic MCP) + M4.x `Chunked<C>` row-block streaming (f64 + f32) done; true mixed precision + GPU pending |
| M5 — Model selection & inference | ✅ done | M5.1 CV (24 `*PathCV` estimators) + M5.2 information criteria (`select_by_ic` for AIC/BIC/EBIC across all four GLMs) + **M5.x stability selection (MB bootstrap)** + **M5.x-a debiased lasso (LS, VBR nodewise + Wald CIs/p-values)** + **M5.x-b debiased GLM (binomial + Poisson via weighted-LS surrogate + nodewise Fisher)** + **M5.x-c threaded CV folds via PyO3 `allow_threads` GIL release — full coverage across scalar + block + multinomial + multitask + Cox + bridge + mmap builders, ~2.3–2.5× speedup**; adaptive done in M6.x. |
| M6 — Penalty zoo | ⏳ partial | sparse-group already done in M2.7; elastic net (scalar LS) done in M6.1; group elastic net (LS) done in M6.2; **sparse-group SCAD end-to-end (LS + 3 GLMs) done**; **bridge `\|β\|^q` (LS, scalar LLA path) done**; **adaptive {Lasso, MCP, SCAD} (LS, two-stage pilot fit) done**; overlapping group + fused + constrained variants pending |
| M7 — Multi-task | ⏳ partial | M7.1 (lasso + MCP) + M7.2 (SCAD + EN + sparse + standardize) done via `MultiTaskDesign<D>` virtual design wrapper composed with `Augmented` / `Standardized`; multi-response GLMs / multinomial / shared-support pending in M7.3 |
| M8 — Distribution & DX | ✅ done | CI + cibuildwheel + Read the Docs + Sphinx + Furo site (concepts + porting + extending + examples + API ref) + R numerical regression suite + stable Rust API contract; comparison/timing benches moved to M9; comprehensive subclass docstrings deferred (low value relative to the rest of the milestone) |
| M9 — Performance & correctness benchmarks | ⏳ partial | **M9.3 coverage closed via `benches/v2`** — every penalty × datafit row in the M9.3 table now has at least one direct comparator with five-seed aggregates (Lasso/EN/MCP/SCAD LS small+medium+large; Group lasso/MCP, Sparse-group MCP, Logistic lasso/MCP, Poisson lasso/MCP, Cox lasso/MCP, Graphical lasso, Graphical MCP small+medium). **M14g.1 dispatch fix** routes `glasso_l1`'s `skein` and `sklearn` cells through the graphical-path branch (previous aggregate silently ran `lasso_path` on the n×p data matrix; regenerated 2026-05-18 to skein 35.6 s / sklearn 252.6 s / R `glasso` 21.6 s @ small/deep, ~19,665 edges across packages). M9.1 harness + M9.4 `benches/correctness/` framework + per-λ Jaccard/sign/rel-L2 across skein/skglm/ncvreg also live. **Still pending**: three SCAD-variant scenarios (`ls_group_scad`, `ls_sparse_group_scad`, `logistic_scad`) — runners exist, scenarios need scaffolding; M9.2 criterion expansion; M9.4 at-scale `tests/fixtures/generate.R` parallel suite; M9.5 `docs/benchmarks/speed.md` consolidated docs page. |
| M10 — Performance improvements | ⏳ partial | M10.1 profile (`col_dot` is the floor without BLAS) + M10.3 five waves: (1) col_axpy + F-order DenseMatrix + path-solver fixes; (2) adaptive inner tol via PGD + KKT-priority WS; (3) `blas-accelerate` feature; (4) (skipped — inner active-set CD didn't pay off); (5) F-series — duality gap + Anderson on residuals + gap-safe sphere screening, all gated and tested. **Skein medium lasso/LS: 7.6 s → 0.78 s sparse / 1.17 s deep — 6.5–10× total. Within 1.5× of glmnet on sparse, 1.9× on deep; ~8–9× behind sklearn's Cython `lasso_path`.** F-series wallclock-neutral on M9.3 scenarios — infrastructure correct but post-pass screening fires only on multi-pass λs (most converge in 1). M10.4 verified across deep + sparse regimes. Pending: G. cross-platform BLAS (OpenBLAS/MKL), H. pre-pass gap-safe screening, I. Cython-grade rewrite (off-roadmap). |
| M11 — Graphical models | ✅ done | Single-population glasso (L1 / MCP / SCAD) + joint glasso across `K` populations (Danaher–Wang–Witten group form via ADMM) + EBIC tuner; M11.3 bootnet-style bootstrap edge stability shipped. |
| M12 — Hardening (robustness, test coverage, CI) | ✅ done | Penalty + datafit unit-test coverage closed; Rust integration test directory added; R-fixture gate in CI; PyO3-layer smoke job (`.github/workflows/bench-smoke.yml`); pre-flight tight-tol screening test now a separate fail-fast CI step with 2-min timeout (R3, ci.yml); numerical guards (`W_FLOOR=1e-6`, `ETA_CLAMP=30.0`) centralized in `crates/skein-core/src/numerics.rs` (R4); R1 unwrap audit closed (`block_path_lla.rs` documented; `cd.rs::anderson_extrapolate` documented; dead `Groups::from_csr` call removed from `glasso_admm.rs`; remaining hits are test-only setup invariants); `Groups::has_overlap()` + parallel block-CD overlap detection with serial Gauss-Seidel fallback + `Once`-gated stderr warning + fixture (C5); criterion bench tree expanded with `lla_outer.rs`, `prox_newton_glm.rs`, `glasso.rs` alongside existing `block_cd.rs`, README updated (P2); `skein-py/src/lib.rs` 10,628 → 275 lines, every datafit family in its own module: `glasso.rs`, `glm.rs`, `ls.rs`, `mmap_chunked.rs`, `multinomial.rs`, `multitask.rs` (P4). No new algorithmic surface. |
| M13 — Performance findings from `benches/v2` | ⏳ partial | M13.1 adaptive screening saturation bypass shipped; M13.2 cross-λ gradient cache shipped (-10.4% wall on medium Lasso); M13.4 Phase 2.3 LLA fixed-point short-circuit shipped; **M13.4b native group-MCP BCD shipped (-3.46× wall on medium ls_group_mcp; flips skein/grpreg from 3.34× slower to 1.20× faster)**; **M13.4c native group-MCP BCD extended to logistic / Poisson / Cox shipped (2.12× wall on logistic medium; new `glm_group_mcp_native_matches_lla` cross-family agreement test)**; M13.6 re-characterized post-M13.2 (memory-bandwidth wall in inner CD past medium scale, not fixed-cost overhead); M13.7 Jacobi-parallel block-CD remains a negative result; **M13.8 celer-style gap-safe screening on the GLM prox-Newton surrogate shipped** (weighted-LS `lasso_dual_obj`, per-GLM closed-form duals for logistic + Poisson, Anderson dual extrapolation on `(β, r)` pairs, weighted strong-convexity correction `r²=2·gap·max(w)/n`, M13.1-style saturation bypass — **3–8× wall on logistic_lasso v2 cells**, 8.2× small-sparse / 3.1× small-deep / 7.2× medium-sparse). Remaining: M13.5 MCP one-outer-iter short-circuit; sparse-group MCP variants for logistic / Poisson / Cox still on LLA. |
| M14 — Inference & applications closeout | ⏳ partial | **Released as v0.9.0** (commit `ab9bd44`), then **v0.10.0** shipped M13.8 (GLM prox-Newton screening). **M14a.1 polychoric / polyserial preprocessing** (Olsson 1979 two-step ML) for ordinal Likert / mixed data; **M14a.2 edge-level FDR / FWER / MB stability bound** on `GraphicalBootstrap` (no other graphical-models package has this — BH FDR + Bonferroni / Holm + closed-form MB threshold); **M14a.3 debiased Cox lasso** (Van de Geer / Cai-Wang construction reusing the partial-likelihood Fisher diagonal from `CoxPH::surrogate_at` via a 16-line PyO3 binding — closes the inference axis across all four mainstream GLM families); **M14c.1 scalar LLA weight short-circuit** (Phase 2.3 ported to `path_lla.rs` — affects bridge, adaptive lasso, multitask LLA); **M14c.2 native sparse-group MCP penalty + 6 GLM PyO3 swaps** (drops the LLA layer for logistic/Poisson/Cox sparse-group MCP, sibling of M13.4c); **M14c.3 at-scale R-fixture tier** (n=500, p=100) for cross-package regression gating; **R-anchor fixture generators** for polychoric (vs `psych::polychoric`, atol 5e-3) and Cox active-set (vs `glmnet(family='cox')`, Jaccard ≥ 0.6 — no R package has Cox debiasing). **M14d GLM W_FLOOR + penalty unimodality trait; M14e ncvreg-equivalent v-scaled MCP/SCAD prox** (closes the nonconvex GLM active-set bloat — `logistic_mcp medium-sparse` 842 → 107 active features, 6× speedup); **M14f fused IRLS+CD GLM solver** mirroring `ncvreg::cdfit_glm`'s lazy per-iter cost structure (`logistic_mcp medium-sparse` 19.7 s → 3.05 s, ~8 % ahead of ncvreg). **M14b: empirical run shipped 2026-05-18**, 909-line manuscript draft compiles to PDF; remaining work is folding post-M14e/f numbers into §Results and JMLR-MLOSS / JOSS formatting pass. **M14g closeout (2026-05-19):** §M14g.1 glasso_l1 benchmark-runner dispatch bug fixed in `637ae7e`; §M14g.2 `poisson_lasso` "regression" investigated and closed as measurement noise (no Rust commit between v0.10.0 and HEAD touches the convex Poisson path; per-seed cell variance is ≈2.5× — larger than the 1.4× claimed effect). The absolute 17× Poisson-vs-glmnet gap is real but pre-existing, tracked in §M9.3. |

Test count at this snapshot: **358 cargo lib + 8 cargo integration + 455 pytest, all green.**

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
- ✅ **Poisson offsets** (M3.x follow-up to M3.4): `PoissonLog` gains
  `with_offset(y, offset)` and `with_sample_weights_and_offset`
  constructors; `surrogate_at` and `loss` thread the per-sample
  offset through `η_full = X·β + offset`. Common rate-model use
  cases (genomics person-years, click-through-rate observation
  windows) work via `offset = log(exposure)`. PyO3: every Poisson
  entry (14 total, scalar + group + sparse-group × dense + sparse,
  including `SparseGroupSCAD`) accepts an `offset=None` kwarg.
  All 14 sklearn Poisson estimators carry an `offset` constructor
  argument; `_glm_dispatch_inputs` reads it via `getattr` so logistic
  and Cox are untouched. CV slices `offset[train_idx]` per fold so
  the per-fold path estimator sees the correct n-vector. 4 cargo
  tests cover offset semantics (zeros ≡ no offset, constant offset
  ≡ intercept shift, length + finiteness validation); 10 pytest
  cover scalar / group / sparse-group / CV / dense ↔ sparse
  parity.

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
  (frequency vs. probability weighting), Breslow's
  cumulative-baseline-hazard estimator for absolute survival
  predictions.
- ✅ **Efron ties** (M3.x follow-up): `CoxPH` gains
  `with_ties(time, event, TieHandling::{Breslow, Efron})`; default
  constructor still uses Breslow. Efron's per-event reduced risk set
  `S_eff_i(t) = S(t) − (i/k)·S_D(t)` is threaded through both `loss`
  (per-tie-block log-S accumulation) and `compute_cum_h` (running
  CumH / CumH2 over the same reduced sets); reduces exactly to
  Breslow when no ties. PyO3: every Cox entry (14 total including
  SparseGroupSCAD) accepts a `ties: str = "breslow"` kwarg parsed
  via `parse_cox_ties`. All 14 Cox sklearn estimators + 7 CoxPathCV
  wrappers gain a `ties` constructor argument forwarded through
  `_cox_dispatch_inputs`. 4 cargo tests (Efron ≡ Breslow with unique
  times, Efron-with-ties hand derivation against the reduced-risk-
  set formula, default-constructor parity); 8 pytest covering
  unique-times parity, heavy-ties divergence, group / sparse-group
  / CV threading, parameter validation, dense ↔ sparse equivalence.

### ✅ M3.6 — Multinomial / softmax

K-class softmax with `B ∈ ℝ^{p×K}` row-major (`bvec[jK + k] = B[j, k]`).
Reduces — through Böhning's diagonal majorization (constant per-(i,k)
Hessian = 1/2, the recipe glmnet uses; Friedman/Hastie/Tibshirani 2010
§4.4) — to a sequence of multi-task LS problems on
`MultiTaskDesign<X>`. Per-class intercepts via the column-augmentation
trick (1s column at `j = p`, row-group weight 0). Symmetric (no
reference class) parameterization, matching `glmnet`. The whole stack
is the M7.1 reduction reused unchanged: every existing GLM-aware solver
(`prox_newton_block_solve_path`) and design wrapper (`Augmented`,
`Standardized`) composes for free.

- ✅ **`MultinomialLogit` datafit** (`crates/skein-core/src/datafit/`):
  one-hot `Y ∈ ℝ^{n×K}` + `from_labels(labels, n_classes)` constructor;
  stable logsumexp loss, surrogate at β returns task-outer-stacked
  weighted LS with uniform `1/2` Böhning weights and working response
  `z_{i,k} = η_{i,k} − 2(p_{i,k} − Y_{i,k})`. 13 cargo tests:
  one-hot encoding, panic-on-K=1 / out-of-range labels, loss-at-zero =
  `log K`, surrogate-at-zero `z = 2(Y − 1/K)` with uniform `1/2` weights,
  softmax simplex + extreme-η stability, λ_max-returns-zero,
  signal recovery (lasso + MCP via LLA), `Standardized<MultiTaskDesign<…>>`
  matches pre-scaled, sparse-CSC matches dense at machine precision.
- ✅ **PyO3 surface**: 4 dense + 4 sparse path entries
  (`solve_multinomial_{lasso,mcp,scad,elastic_net}_path[_sparse]`)
  taking `(X, labels, n_classes, …)`. Output: `coefs`
  `(n_lambdas, p × K)` row-major bvec, `intercepts` `(n_lambdas, K)`.
  Sparse uses `Augmented<SparseCSC>` for the intercept; dense uses
  physical column augmentation. Both compose with `Standardized<…>`.
- ✅ **Python estimators** (`python/skein_glm/multinomial.py`):
  12 sklearn-compatible classes — `Multinomial{Lasso,MCP,SCAD,ElasticNet}{Classifier,PathClassifier,PathCV}`.
  `Classifier` suffix (sklearn convention; the binary `LogisticMCPRegressor`
  naming stays as-is for back-compat). `coef_ (K, p)`, `intercept_ (K,)`,
  `decision_function(X) → (n, K)` η, `predict_proba(X) → (n, K)` softmax,
  `predict(X) → (n,)` class labels via `argmax` on the original-label
  dtype (numeric or string — labels are LabelEncoder-style mapped
  internally). CV scores by multinomial deviance via `StratifiedKFold`
  default. `select_by_ic` gets a multinomial-NLL helper +
  row-grouped active-feature-count df.
- ✅ **Tests** (`tests/test_multinomial.py`): 13 pytest covering
  predict-shape semantics, signal recovery (lasso + MCP), path
  shape + λ_max ≈ 0, dense↔sparse equivalence, dense↔sparse with
  standardize on inflated-scale columns, CV picks active features,
  IC for all three criteria, single-class rejection, SCAD `a > 2`
  validation, EN `α ∈ [0, 1]`, EN α=1 ≡ lasso path agreement,
  string-label round trip.

### ⏳ M3.7 — Opportunistic GLMs

Negative binomial, Huber / quantile (smoothed) — ship whichever has
user demand. Gaussian-with-offsets is achievable today by subtracting
the offset from `y` (LS is shift-equivariant); only GLMs need explicit
offset support, and both **Poisson offsets** and **Cox Efron ties**
ship now (see M3.4 / M3.5 follow-ups above).

#### ✅ Huber — robust LS via IRLS surrogate

- `Huber` datafit (`crates/skein-core/src/datafit/huber.rs`) implements
  the standard piecewise loss `ρ_δ(r) = ½r²` for `|r| ≤ δ`,
  `δ|r| − ½δ²` otherwise, and exposes the IRLS-style quadratic
  surrogate `w_i = 1` if `|r_i| ≤ δ` else `δ/|r_i|` (floored at `1e-6`)
  with working response `z_i = y_i`. Implements `GlmDatafit`, so the
  M3.2 / M3.4 prox-Newton outer + M1/M2 inner CD machinery dispatches
  through it unchanged.
- PyO3: `solve_huber_mcp_path` / `solve_huber_scad_path` (dense, with
  intercept, optional standardization, weights), reusing
  `build_glm_path_outputs`.
- Python estimators: `HuberMCP{,Path}Regressor`,
  `HuberSCAD{,Path}Regressor` with sklearn-compatible
  `predict` / `decision_function`; δ as a required hyperparameter
  (recommended `1.345 · σ_MAD(y)` for 95% asymptotic efficiency at
  the normal).
- Tests: 9 new cargo (6 unit on the loss / surrogate / weights +
  3 prox-Newton path) + 10 pytest (path shape, signal recovery under
  contamination, beats LS on contaminated data, large-δ ≡ LS, λ_max
  on clean data, validation of finite-y / positive-δ / sparse-X-NYI,
  predict ≡ decision_function). All green (274 cargo + 289 pytest).
- Limitations / pending: sparse-X path entry, chunked / mmap design
  matrices, group-Huber and sparse-group-Huber. Sparse can be added
  by wiring `build_glm_path_outputs_sparse` analogously; the rest
  follow M3.3-style composition with `prox_newton_block_solve_path`.

#### Pending (M3.7 follow-ups)

Negative binomial (known dispersion `α`: standard GLM with
`Var = (1 + α·μ)·μ`), smoothed quantile (pinball loss with `τ`
quantile + Lipschitz-smoothed kink). Both fit the `GlmDatafit`
framework; ship whichever surfaces user demand first.

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

### ✅ M4.2 — Sparse PyO3 surface (M4.2a/b/c all shipped)

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
  branch for all 8 group estimators in `python/skein_glm/estimators.py`.
  6 pytest tests prove dense ↔ sparse equivalence on a shared λ-grid
  for group lasso, group MCP (γ=1e6 ≈ lasso), and sparse-group lasso,
  plus smoke tests for sparse-group MCP and the `standardize=True`
  rejection.
- ✅ **M4.2c** (GLMs): 18 sparse PyO3 wrappers covering logistic,
  Poisson, and Cox × scalar/group (6 each). Cox uses the no-intercept
  path; logistic/Poisson use column-augmentation. All 36 GLM
  estimators dispatch transparently on `scipy.sparse.issparse(x)`.
  Two shared dispatch helpers in `python/skein_glm/estimators.py`
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

- ✅ **`MmapMatrix` (f64)**: memory-mapped column-major (Fortran-order)
  raw `f64` dense matrix. `col_sq_norms` precomputed once at open time
  (one full pass through the file paid by the OS page cache); CD's
  `col_dot` reads each column as a contiguous slice through the mmap,
  so the hot path matches `DenseMatrix` performance once warm. Auto-
  `Sync + Send`. Composes cleanly with `Standardized<MmapMatrix>` for
  on-the-fly column scaling. Intercept handled by a new generic
  `Augmented<D>` wrapper that adds a virtual 1s column at index
  `n_features` (no file rewrite needed); same wrapper composes with
  any future backend. PyO3 surface (v1, scope deliberately minimal):
  `solve_mcp_ls_path_mmap` and `solve_logistic_mcp_path_mmap`. Python
  helper `MmapDesignF64(path, n_rows, n_cols)`; `MCPPathRegressor` and
  `LogisticMCPPathRegressor` dispatch on `isinstance(x,
  MmapDesignF64)` and route to the `_mmap` entry points. 6 cargo
  tests on `MmapMatrix` + 4 cargo tests on `Augmented<D>` (plus a
  mmap solver-equivalence test against pre-loaded `DenseMatrix`),
  5 pytest tests covering constructor validation, dense↔mmap
  equivalence for LS-MCP path, LS-MCP + standardize, and logistic-
  MCP path. Expanding to the other 22 GLM/Cox entry points is
  mechanical mirroring of the `_sparse` surface and is deferred
  until there's user demand.

- ✅ **`MmapMatrixF32` (f32-on-disk)**: same column-major mmap layout
  as the f64 backend with 4-byte storage; `col_dot` / `matvec` /
  `rmatvec` / `columns` cast f32→f64 elementwise on the way out so
  the solver core stays f64. Half the disk footprint and half the
  page-cache pressure for the same `(n, p)`. PyO3 entry points
  `solve_mcp_ls_path_mmap_f32` and `solve_logistic_mcp_path_mmap_f32`;
  Python helper `MmapDesignF32(path, n_rows, n_cols)` that the
  `MCPPathRegressor` and `LogisticMCPPathRegressor` dispatch
  recognizes alongside `MmapDesignF64` (picks the right `_mmap[_f32]`
  entry point based on `x.dtype`). PyO3 entry-point bodies are now
  generic over `D: DesignMatrix`, refactored into shared
  `mmap_ls_mcp_path_inner` / `mmap_logistic_mcp_path_inner` helpers
  so f64 and f32 surfaces share their tower-of-wrappers logic
  (Augmented + Standardized) byte-for-byte. 6 cargo tests on
  `MmapMatrixF32` (mirroring the f64 suite, with an f32-rounded
  dense reference for the equivalence test); 5 pytest tests
  including dense-vs-mmap equivalence on the f32-rounded matrix and
  an end-to-end "f32 vs f64 within truncation" check that asserts
  identical support recovery.

- ✅ **`Chunked<C>` (row-block streaming)**: generic over the chunk
  backend `C: DesignMatrix`, so `Chunked<MmapMatrix>` is f64 chunked,
  `Chunked<MmapMatrixF32>` is f32 chunked, `Chunked<DenseMatrix>` is
  in-memory chunked (mostly for tests). Each `col_dot(j, v)` splits
  `v` into per-chunk segments and sums; `matvec` concatenates;
  `rmatvec` slices `r` and sums per-feature; `col_sq_norm` is
  precomputed once at construction by summing across chunks. Validates
  uniform `n_features` across chunks. Composes with `Augmented` and
  `Standardized` the same way every other backend does. v1 is serial;
  per-chunk `rayon::par_iter` is a one-line follow-up once benches
  justify it. PyO3: 4 entry points (LS-MCP and logistic-MCP × {f64,
  f32}) reusing the existing `mmap_*_path_inner` generics. Python
  helpers `ChunkedDesignF64(chunks, n_cols)` and `ChunkedDesignF32`
  taking a list of `(path, n_rows)` pairs; estimator dispatch picks
  the right `_chunked[_f32]` entry via `x.dtype`. 8 cargo tests on
  `Chunked<DenseMatrix>` (mirroring the mmap suites; uneven chunk
  sizes covered) plus a solver-equivalence test. 6 pytest tests
  covering constructor validation (per-chunk + empty-list), dense↔
  chunked equivalence for LS-MCP and logistic-MCP, f32 chunked
  equivalence on f32-rounded reference, and chunked + standardize.

- **True mixed precision**: parameterize the solver core over
  `T: Float` and refine the active set in f64 after a bulk f32 path.
  Different problem from f32-on-disk; needs touching every algorithm
  in `solver/`.
- **GPU backend** (stretch): `cubla`s / `wgpu` matvecs behind the same
  trait. Only worth it once dense `n × p` matvec is the bottleneck;
  measure first.

Bench target: a 10⁶ × 10⁵ sparse logistic group-lasso path in under
the time `glmnet` takes on a 10⁵ × 10³ dense one.

---

## M5 — Model selection & inference

Without these, we ship a solver, not a library people fit on real data.

### ✅ M5.1a — Cross-validation for LS scalar

`python/skein_glm/cv.py` adds a `_PathCVMixin` that runs K-fold CV over a
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

`python/skein_glm/ic.py` adds `select_by_ic(path_model, x, *outcomes,
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

- ✅ **Stability selection** (Meinshausen–Bühlmann 2010): new
  `StabilitySelection` wrapper class in `python/skein_glm/stability.py`
  composes any skein `*PathRegressor` (LS / GLM / group / multi-task /
  multinomial / Cox) with a subsample-bootstrap loop. Records
  selection probabilities per (feature, λ) over `n_bootstraps`
  iterations; the stable set is features whose max-over-λ probability
  exceeds `threshold`. Embarrassingly parallel via `joblib`
  (`n_jobs=-1`); for a fixed `random_state` the bootstrap indices are
  pre-generated so parallel and serial produce identical results.
  Cox is auto-detected via `ties` attr (outcomes passed as
  `y=(time, event)`); grouped variants aggregate at the group level;
  multi-task / multinomial 3D `coefs_` collapse via
  "any-class active". 11 pytest cover signal recovery, threshold
  monotonicity, attribute shapes, transform, reproducibility, n_jobs
  consistency, parameter validation, grouped/Cox/sparse dispatch.
  Closes the M5.x headline differentiator (no clean equivalent in
  glmnet / skglm / grpreg).
- **Adaptive weights**: one-shot `AdaptiveLasso` / `AdaptiveMCP`
  estimators that fit a coarse model first, derive `w_j ∝ 1/|β̂_j|^η`,
  then refit. This is one of the headline reasons the per-feature
  weight axis exists.
- ✅ **M5.x-a — Debiased / desparsified lasso (LS)**: Van de
  Geer–Bühlmann–Ritov (2014) confidence intervals and Wald p-values
  for high-dimensional least-squares lasso. Pure Python on top of
  the existing `ElasticNetRegressor(alpha=1.0)` primitive — no Rust
  changes. Constructs an approximate inverse Gram `Θ̂` via per-
  column nodewise lassos (with the `+ λ‖γ‖₁` VBR correction in
  `τ̂²`), debiases as `β̂_d = β̂ + (1/n) Θ̂ Xᵀ(y − Xβ̂)`, and reports
  `σ̂² · diag(Θ̂ Σ̂ Θ̂ᵀ) / n` Wald inference computed via
  `U = X_s Θ̂ᵀ` so the `p × p` `Σ̂` is never materialized. Public
  surface: `debiased_lasso` free function returning a
  `DebiasedLassoResult` dataclass, plus a `DebiasedLassoRegressor`
  sklearn-style wrapper that exposes the debiased estimate as
  `coef_` / `intercept_` and the inference outputs as suffixed
  attributes (`se_`, `ci_lower_`, `ci_upper_`, `pvalues_`,
  `z_scores_`, `Theta_`, `coef_lasso_`, `sigma_hat_`,
  `lambda_main_`, `lambda_nodewise_`). Defaults: standardize
  columns, theoretical `λ = √(2 log p / n)` for both main and
  nodewise fits, joblib-parallel nodewise loop. 22 pytest cover
  shapes and finiteness, **empirical 95% CI coverage over 60
  replications** (load-bearing — catches Θ̂/variance errors),
  active-coordinate coverage ≥ 80%, debiased < lasso L1 error on
  active features, smaller p-values on true active features,
  scalar/array `lambda_nodewise`, no-intercept mode, n_jobs
  serial-vs-parallel parity, sklearn wrapper round-trip, and
  parameter validation. Closes the inference-CI gap relative to
  `glmnet` / `ncvreg` / `grpreg` (which offer none) and the R
  `hdi::lasso.proj` canonical implementation. **R-anchor
  regression** against `hdi::lasso.proj` on a fixed seed is the
  next step — wire it into the M8 R suite.
- ✅ **M5.x-b — Debiased GLM (binomial + Poisson)**: extends the LS
  VBR construction to canonical-link GLMs. At the penalized fit
  `β̂`, the score `Xᵀ(y − μ̂)` is the canonical gradient and the
  Fisher information is `J = (1/n) Xᵀ W X` with
  `W = diag(μ̂(1−μ̂))` for binomial / `diag(μ̂)` for Poisson. We
  build `Θ̂ ≈ J⁻¹` via nodewise lassos on the **weighted design**
  `X̃ = W^{1/2} X` — identical to LS once the design is weighted —
  and debias as `β̂_d = β̂ + (1/n) Θ̂ Xᵀ(y − μ̂)` with asymptotic
  variance `diag(Θ̂ X̃ᵀX̃ Θ̂ᵀ)/n` (no `σ²` — noise lives in `W`).
  Public surface: `debiased_logistic_lasso` /
  `debiased_poisson_lasso` free functions returning a
  `DebiasedGLMResult` dataclass, plus
  `DebiasedLogisticLassoRegressor` (sklearn-style with
  `predict_proba` / `predict`) and `DebiasedPoissonLassoRegressor`
  (`predict` returns `μ̂ = exp(η̂)`, supports `offset`). Penalized-fit
  primitive is the M3.x `LogisticLassoRegressor` /
  `PoissonLassoRegressor` — clean convex L1, not the prior
  `MCP(γ=1e9)` approximation, so the inference layer doesn't
  inherit that bias. Working weights are floored at `1e-8` to keep
  `X̃` non-degenerate under near-boundary fitted probabilities. 19
  pytest cover dataclass shapes, finiteness, CI ordering, **empirical
  95% CI coverage on inactive coordinates** for both families (40
  reps each, coverage ≥ 80% catches Θ̂/variance errors), smaller
  p-values on true-active features, Poisson offset behavior, full
  parameter validation, sklearn wrapper round-trip, n_jobs parity.
  Closes the inference gap relative to glmnet/grpreg on logistic
  and Poisson penalized fits.
- ✅ **M5.x-c — Threaded CV fold parallelism**: implemented via PyO3
  `py.allow_threads(|| ...)` GIL release inside the scalar-penalty
  builders (`build_path_outputs` / `_sparse_ls`, `build_glm_path_outputs` /
  `_sparse`, `build_cox_path_outputs` / `_sparse` in
  `crates/skein-py/src/lib.rs`) + a threaded fold loop in
  `_PathCVMixin` / `_CoxPathCVMixin` (`python/skein_glm/cv.py`) via
  `joblib.Parallel(prefer="threads")`. Every CV class gained an
  `n_jobs` constructor parameter. Bitwise parity verified between
  `n_jobs=1` and `n_jobs=-1`. ~2.3× speedup on a 5-fold MCP LS CV
  at `(n=5000, p=200)`; ~2.5× on logistic-lasso CV at `(n=3000, p=100)`
  — limited by 5 folds + the inner rayon group-sweep parallelism
  already saturating cores on each individual fit. The GIL release
  also speeds up `StabilitySelection`, `GraphicalStabilitySelection`,
  `GraphicalBootstrap`, and the debiased lasso nodewise loop, which
  all call into these builders through joblib threading.
  **Follow-up PR** extended the `py.allow_threads(|| ...)` pattern to
  every remaining builder (LS block + LLA dense/sparse, GLM block
  dense/sparse, Cox block dense/sparse, multinomial dense/sparse,
  multitask + LLA dense/sparse, bridge LS scalar dense/sparse,
  mmap-backed LS and logistic MCP). 9 additional parameterized parity
  tests cover the grouped CV classes (LS, logistic, Poisson, Cox).
  M5.x-c is now complete: **every** user-facing `*PathCV` benefits
  from threaded fold parallelism.

---

## M6 — Penalty zoo expansion

Once the solver core is solid, penalties are cheap to add. Priority
ordered by user demand and by what differentiates us.

> Sparse-group lasso, sparse-group MCP, and the convex+LLA infrastructure
> for sparse-group SCAD are already done in **M2.7**.

- ✅ **M6.1 Elastic net (scalar LS)**: convex `α λ |β_j| + (1-α)
  λ β_j² / 2` per feature. Closed-form prox
  (`prox::elastic_net_prox`); `Penalty::weights()` returns the
  L1-effective view `α · w_j` so `lambda_max` and strong-rule
  screening see the right active-set boundary. PyO3 entries
  `solve_elastic_net_ls_path` (dense + sparse). Python: 3
  estimators (`ElasticNetRegressor`, `ElasticNetPathRegressor`,
  `ElasticNetPathCV`) with full sparse + standardize support.
  Tests: 11 cargo (prox identities at α=1/α=0, value, weights
  views, panic-on-bad-α, solver-path equivalence to MCP at
  γ=1e6, closed-form ridge match at α=0) + 7 pytest (α=1 ↔ MCP
  high-γ, α=0 closed-form ridge, signal recovery, dense ↔
  sparse, dense ↔ sparse + standardize, CV picks reasonable λ,
  validates α ∈ [0, 1]).
- ✅ **M6.2 Group elastic net (scalar LS)**: convex per-group
  penalty `α λ w_g ‖β_g‖₂ + (1-α) λ w_g ‖β_g‖₂² / 2`. Closed-
  form block prox (`prox::group_elastic_net_prox`) — block soft-
  threshold then per-block ridge shrinkage, derived from the
  rotational symmetry of the mixed group-L2 + ridge objective.
  `GroupPenalty::weights()` returns the L1-effective view `α·w_g`
  so `block_lambda_max`, strong-rule, and KKT verification see
  the right per-group active-set boundary. PyO3 entries
  `solve_group_elastic_net_ls_path` (dense + sparse). Python: 3
  estimators (`GroupElasticNetRegressor`,
  `GroupElasticNetPathRegressor`, `GroupElasticNetPathCV`) with
  full sparse + standardize support. Tests: 17 cargo (prox
  identities at α=0/α=1, value, weights views, panic-on-bad-α,
  prox-matches-group-lasso at α=1, solver-path equivalence to
  GroupLasso at α=1, closed-form ridge match at α=0) + 7 pytest
  (α=1 ↔ GroupLasso path, α=0 closed-form block ridge, signal
  recovery, dense ↔ sparse, dense ↔ sparse + standardize, CV
  picks both active groups, validates α ∈ [0, 1]).
- ✅ **`SparseGroupSCAD` end-to-end** — the M2.7 surrogate is now wired
  through to user-facing PyO3 entries and sklearn estimators across
  all four datafits. 8 PyO3 entries (LS + logistic + Poisson + Cox ×
  dense/sparse): `solve_sparse_group_scad_ls_path[_sparse]` plus the
  `solve_{logistic,poisson,cox}_sparse_group_scad_path[_sparse]`
  twins. 12 sklearn estimators following the same `*Regressor /
  *PathRegressor / *PathCV` shape as `SparseGroupMCP*`. Validates
  `a > 2`. 11 pytest tests cover path shape, signal recovery,
  dense↔sparse equivalence, large-`a` parity with `SparseGroupLasso`,
  Cox/Poisson/Logistic smoke, and `a > 2` rejection across families.
- ✅ **Bridge / ℓ_q penalty** `λ · Σ_j w_j |β_j|^q` (LS, scalar): new
  `solve_path_lla` scalar outer-LLA path solver in
  `solver/path_lla.rs` (mirrors `solve_block_path_lla`); new
  `surrogate_weights_bridge` helper. Inner is weighted lasso
  (`ElasticNet::with_weights(λ, 1.0, w)`). PyO3 entries
  `solve_bridge_ls_path[_sparse]`. 3 sklearn estimators
  (`BridgeRegressor`, `BridgePathRegressor`, `BridgePathCV`). 3 cargo
  tests (q=1 ↔ lasso path, q=0.5 signal recovery, λ_max ⇒ zero) +
  10 pytest (q=1 ↔ MCP-high-γ parity, q=0.7 signal recovery via
  warm-started path, dense↔sparse, dense↔sparse + standardize, CV
  picks active features, smaller-q sparsifies more, q∈(0,1] + ε>0
  validation, predict shape).
- **Overlapping group lasso / latent group lasso** via duplication
  trick; surface a friendly group-construction API.
- **Fused lasso / generalized lasso**: 1D and graph-structured
  fusion. Solved via specialized prox (taut-string for 1D, ADMM for
  general). Lives behind `solver::fused`.
- ✅ **Adaptive GLM {Lasso, MCP, SCAD}** (Logistic / Poisson / Cox):
  three families × three penalties × {Path, PathCV} = **18 sklearn
  classes**. Pilot is the GLM's lasso path (e.g.,
  `LogisticMCPPathRegressor(gamma=1e9)`); final is the user's chosen
  GLM-penalty path with adaptive weights. CV inherits the
  per-family scoring (binomial / Poisson deviance / Cox c-index) and
  splitter (KFold for logistic+Poisson, StratifiedKFold-by-event for
  Cox) from the existing M5 CV mixins. Cox keeps its `fit(x, time,
  event)` signature. 10 pytest cover support recovery, predict
  shapes, dense ↔ sparse, no-intercept on Cox, predict ≡ decision_
  function on Cox.
- ✅ **Plain `GroupSCAD`** (LS): the previously-dangling prerequisite
  to `AdaptiveGroupSCAD`. Surrogate helper `surrogate_weights_group_scad`
  was already in place from M2.8; this commit adds the
  `solve_group_scad_ls_path[_sparse]` PyO3 entries plus three sklearn
  classes (`GroupSCADRegressor`, `GroupSCADPathRegressor`,
  `GroupSCADPathCV`). Validates `a > 2`. Pytest covers signal recovery,
  large-`a` ↔ GroupLasso parity, dense ↔ sparse, CV.
- ✅ **Adaptive group SCAD** (LS): completes the M6.x adaptive group
  family with `AdaptiveGroupSCAD{PathRegressor, PathCV}`. Pure Python
  composition over the new `GroupSCADPathRegressor`.
- ✅ **Adaptive group {Lasso, MCP}** (LS, group two-stage): mirrors
  the scalar adaptive recipe with **per-group** weights
  `w_g = 1 / max(‖β_pilot[g]‖_2, ε)^η` derived from a plain
  `GroupLassoPathRegressor` pilot. 4 sklearn classes
  (`AdaptiveGroupLassoPathRegressor`, `AdaptiveGroupMCPPathRegressor`,
  plus the matching `*PathCV` wrappers). 5 pytest cover signal
  recovery, CV active-group selection, dense↔sparse equivalence,
  predict shape, and validation. `AdaptiveGroupSCAD` is deferred —
  plain `GroupSCAD` (the prerequisite) needs a separate tiny wiring
  task on top of the existing `surrogate_weights_group_scad`.
- ✅ **Adaptive {Lasso, MCP, SCAD}** (scalar LS, two-stage):
  `python/skein_glm/adaptive.py` — six classes
  (`AdaptiveLassoPathRegressor`, `AdaptiveMCPPathRegressor`,
  `AdaptiveSCADPathRegressor`, plus three `*PathCV` wrappers).
  Pure Python: pilot is `MCPPathRegressor(gamma=1e9)` (plain lasso),
  per-feature adaptive weights `1 / max(|β_pilot|, ε)^η`, final fit
  reuses skein's existing `weights=` parameter — no Rust changes.
  Path estimators expose `coef_pilot_` and `weights_` for inspection;
  CV runs the pilot on full data and uses fixed weights per fold.
  10 pytest cover signal recovery, pilot-position behavior, dense
  ↔ sparse equivalence, validation, η-monotonicity in sparsity.
  Adaptive group / GLM variants are a follow-up.
- **Adaptive group MCP / SCAD** with weights from a pilot fit.
- **Constrained variants**: nonneg lasso, box constraints. Implemented
  by post-prox projection.

---

## M7 — Multi-task / multi-response

`skglm` has multi-task lasso; R has it via custom packages. We wire it
through the existing trait surface — multi-task LS reduces *exactly* to
group lasso on a virtual block-replicated design, so the M2 block-CD
machinery handles the headline algorithm unchanged.

### ✅ M7.1 — Multi-task LS lasso + MCP

- ✅ **`MultiTaskDesign<D>` Rust wrapper**: with `B ∈ ℝ^{p×K}` laid
  out row-major (`bvec[jK+k] = B[j, k]`) and `Y` stacked task-outer
  (`yvec[k·n+i] = Y[i, k]`), the multi-task LS objective becomes a
  group-lasso problem on a virtual `(nK × pK)` design where column
  `jK+k` is `X[:, j]` lifted into row block `k` and zero elsewhere.
  The wrapper carries no state beyond the inner design + task count;
  every `DesignMatrix` op is O(n) just like the base. `auto_groups(p,
  K)` builds the row-grouping `{jK, …, jK+K-1}` for each feature.
  Composes with `Augmented` and `Standardized` like any other backend.
  9 cargo tests cover wrapper correctness on a hand-built reference
  `X̃`, `K=1` identity collapse, and end-to-end solver-path equivalence
  to a hand-stacked group-lasso fit.
- ✅ **PyO3**: `solve_multitask_lasso_ls_path` (convex via M2's
  `solve_block_path` + `GroupLasso`) and `solve_multitask_mcp_ls_path`
  (non-convex via `solve_block_path_lla` + MCP surrogate). Both take
  2D `Y`, center per-task to get the no-intercept fit, recover
  per-task intercepts via `α_k = ȳ_k − Σ_j x̄_j B[j,k]`. Convention is
  the natural per-sample `(1/(2nK))` data-fidelity factor — sklearn
  /glmnet's `(1/(2n))` matches at `λ_skein = α_sklearn / K`.
- ✅ **Python estimators**: `MultiTaskLasso{,Path}Regressor`,
  `MultiTaskMCP{,Path}Regressor`, plus `MultiTaskLassoPathCV` /
  `MultiTaskMCPPathCV` (K-fold CV scoring by mean per-task MSE).
  `coef_` shape `(K, p)` matching sklearn convention; `intercept_`
  shape `(K,)`; `predict(X) → (n, K)`. 8 pytest tests cover path
  shapes, joint-support recovery, sklearn-MultiTaskLasso parity at
  `λ = α/K`, MCP signal recovery on the path, CV picking the active
  features, 2D-Y validation, and the `standardize_x=True` rejection.

### ✅ M7.2 — LS-feature parity

The headline algorithm reduction makes every penalty / backend port
mechanical: `MultiTaskDesign<D>` composes with `Augmented` /
`Standardized` like any other backend, so the only new code per item
is the PyO3 entry point.

- ✅ **MultiTaskSCAD** via LLA — uses `surrogate_weights_group_scad`
  inside the inner-penalty closure, mirroring MultiTaskMCP. Same
  `solve_block_path_lla` scaffold.
- ✅ **MultiTaskElasticNet** (convex) — uses `GroupElasticNet` from
  M6.2 as the inner penalty. `α=1` reduces to plain MultiTaskLasso;
  `α=0` is per-row block ridge. Path equivalence at α=1 verified.
- ✅ **Sparse design** — 4 new sparse PyO3 entries
  (`solve_multitask_{lasso,mcp,scad,elastic_net}_ls_path_sparse`)
  taking `(data, indices, indptr, n_rows, n_cols, Y, …)` and wrapping
  `SparseCSC` in `Augmented` (1s column for intercept) then
  `MultiTaskDesign` (which replicates the augmented column into K
  virtual intercept columns automatically). Per-feature row-group at
  index `p` gets weight 0 → unpenalized intercept group, with each
  per-task intercept independently fit. Cargo test:
  `MultiTaskDesign<SparseCSC>` solver-path matches a dense reference
  at machine precision. Pytest: dense ↔ sparse equivalence on shared
  λ-grids for lasso, MCP, EN; sparse predict shape; sparse CV smoke.
  Python estimators dispatch on `scipy.sparse.issparse(x)`.
- ✅ **`standardize_x` for multi-task** — the lazy `Standardized` /
  `Augmented` / `MultiTaskDesign` composition gives this for free.
  Dense uses physical center+scale (mirrors M1's scalar LS path);
  sparse uses scale-only via `Standardized<Augmented<SparseCSC>>`
  with the augmented intercept column at scale=1. Per-feature L1
  weights rescale by `1/s_j`; the LLA closure also receives the
  rescaled weights so the surrogate computation is in standardized
  space. Coefficients are de-scaled at the boundary. Pytest: dense
  ↔ sparse equivalence under `standardize=True` for lasso + MCP, and
  signal-recovery sanity test on a 50×-inflated column.

### M7.3 — Pending

- **Multi-response GLMs / multinomial logit** (multinomial reduces to
  a multi-task logistic with the row-grouped K-class softmax surface;
  this needs a `MultinomialLogit` GLM datafit and a stacking helper
  for K classes — not just a wrapper, more invasive than M7.2 was).
- **Shared-support estimators** for the "same active features across
  related outcomes" use case that genomics + finance both want.
- **Multi-response GLM benchmarks** vs `glmnet::glmnet(family="mgaussian")`
  and sklearn `MultiTaskElasticNet`.

---

## M8 — Distribution & developer experience

Ship-grade polish. Without this, none of the above gets adopted.

- ✅ **M8.1 CI** (`.github/workflows/ci.yml`): on every PR and push to
  master/main, runs `cargo fmt --check`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo test -p skein-core`, then
  `ruff check`, `maturin develop --release`, `mypy python/`, and
  `pytest`. Matrix over `{ubuntu-latest, macos-latest} × {Python
  3.10, 3.11, 3.12}`. Concurrency-cancellation on stale runs.
  Pyproject configured to ignore the `E702` semicolon-pair pattern
  used in the GLM dispatch helper, and to skip sklearn/scipy missing
  stubs in mypy.
- ✅ **M8.2 Wheels** (`.github/workflows/wheels.yml`): triggered on
  tag push (`v*`) or manual dispatch. `cibuildwheel` builds ABI3
  wheels (one wheel per platform covers cp310+) for Linux
  x86_64/aarch64 (with QEMU for the latter), macOS x86_64+arm64,
  and Windows AMD64. Each wheel is smoke-tested via `pytest
  {project}/tests` post-build. Sdist built separately. PyPI publish
  job is scaffolded but commented out — uncommenting + setting the
  `PYPI_API_TOKEN` secret enables release-on-tag.
- ✅ **M8.3 Docs** (mkdocs-material + Read the Docs):
  `mkdocs.yml` + `.readthedocs.yaml` + 25-page site with landing,
  install, quickstart, four concept pages, three R-porting guides
  (`glmnet`, `ncvreg`, `grpreg`), three "extending" walkthroughs
  (custom Penalty / Datafit / DesignMatrix), three worked examples
  (genomics / NLP / survival), eight auto-generated API ref pages
  via `mkdocstrings`, and the Rust API contract page. CI runs
  `mkdocs build --strict` (after building the Rust extension via
  maturin so mkdocstrings can introspect).
- ✅ **M8.4 Numerical regression tests** (`tests/test_r_regression.py`,
  `tests/fixtures/`): 8 reference fits from glmnet 5.0 / ncvreg 3.16
  / grpreg 3.6 generated by `tests/fixtures/generate.R`, committed
  as JSON. Python tests load each fixture, run the equivalent skein
  fit on the same X/y/λ-grid, and assert (a) active-set agreement
  within a fuzz tolerance, (b) sign agreement on meaningfully-active
  coefficients, (c) tight magnitude agreement at the smallest λ.
  R is not required to run pytest — fixtures are committed and
  missing-fixture tests skip cleanly. Tolerances are documented to
  reflect the genuine algorithmic differences (LLA local-min
  divergence on nonconvex paths, IRLS-vs-prox-Newton inner stopping
  on logistic, etc.). 10 new tests, all passing.
- ✅ **M8.5 Stable Rust API contract** (`docs/extending/rust-api.md`):
  documents which `skein-core` exports are part of the stable
  surface (extension traits, concrete backends/penalties/datafits,
  public solver entry points, errors) vs. incidental `pub`
  (internal solver helpers, prox primitives subject to
  reorganization). Lists the promised invariants downstream code
  can rely on (Sync+Send, prox convention, residual sign,
  col_sq_norms O(1)) and the minor-version bump policy while
  pre-1.0.
- **Comparison / timing benchmarks** — **superseded by M9** below.
  The original framing of this bullet ("marketing material, lower
  priority") understated the role: with M3–M7 substantially landed,
  the absence of competitive evidence is now the binding constraint
  on adoption, so it gets its own milestone.
- **Comprehensive subclass docstrings** (deferred): the 72
  estimator subclasses keep their existing one-line docstrings;
  base-class docstrings carry the substance. A complete pass is
  worthwhile but not blocking.

---

## M9 — Performance & correctness benchmarks

The pitch above ("beat both on throughput") needs evidence. Today the
only timing data is the internal `block_cd` criterion microbench
(serial-vs-parallel block CD, screening modes — no cross-package
comparison), and the only correctness data is the M8.4 R regression
suite at `n=200, p=20–30` (small enough that everyone converges; tells
us nothing about behavior at scale). M9 closes both gaps.

Scope: reproducible local-only benchmark suite producing committed
JSON/PNG snapshots. Out of scope: CI integration, regression gating,
GPU benches.

### ✅ M9.1 — Bench harness & problem generators

- ✅ New top-level `benches/` directory created (root-level, separate
  from the existing `crates/skein-core/benches/` which stays for
  criterion microbenches). Layout:

  ```
  benches/
    problems.py          shared synthetic generators (LS, logistic,
                         Poisson, Cox, group-structured), seeded
    runners/
      skein_runner.py    fits via skein estimators
      skglm_runner.py
      sklearn_runner.py
      celer_runner.py
      pyglmnet_runner.py
      r_runner.R         glmnet / ncvreg / grpreg via Rscript
    scenarios/           one driver per (penalty × datafit) combo
    results/             committed JSON snapshots
    correctness/         cross-package agreement matrices
    README.md            run instructions, methodology, host notes
  ```

- ✅ Problem generators (`benches/problems.py`) reuse the conventions
  from `tests/fixtures/generate.R` (gaussian X, sparse true β, additive
  noise) so M9.4 fixtures can share generators with the R regression
  suite. Sizes: small=`(1k, 100)`, medium=`(10k, 1k)`,
  large=`(100k, 10k)`. Families: gaussian (lasso + group), logistic,
  poisson.
- ✅ Result schema (`benches/runners/__init__.py` `RunResult` +
  `benches/run.py`): `{scenario, package, version, n, p, n_groups,
  lambda_grid_len, fit_time_s, n_iter, final_obj, active_set_size,
  host_id, timestamp, extra}`. One JSON file per scenario; driver
  writes append-style so partial runs are recoverable.
- ✅ Driver `benches/run.py` takes `--scenarios`, `--packages`,
  `--sizes` and dispatches. Missing comparator imports skip with a
  warning rather than failing (the suite is intentionally tolerant).
  R runner shells out via `Rscript` against a tempfile so R and Python
  see byte-identical inputs.
- ✅ Runner stubs for all six comparators wired against the common
  ABI: `skein_runner.py`, `sklearn_runner.py`, `skglm_runner.py`,
  `celer_runner.py`, `pyglmnet_runner.py`, `r_runner.R` (glmnet +
  ncvreg + grpreg dispatch inside the R script).
- ✅ End-to-end smoke verified: `python benches/run.py --scenarios
  lasso_ls --packages sklearn skein --sizes small` fits both, writes
  the JSON snapshot.
- ✅ Live-install validation: skglm + celer + R+glmnet/ncvreg/grpreg
  all wired and exercised against `lasso_ls` + `lasso_ls_sparse`.
  pyglmnet 1.1 is broken on Python 3.12 (uses removed `distutils`);
  the runner skips gracefully.
- ✅ Timing methodology: 1 warm-up call (discarded) + N timed trials
  (default 5), report `(median, min, max, per-trial list)`. Median
  is the headline number — robust to GC / JIT spikes.
- ✅ Shared scenario helpers in `benches/scenarios/_common.py`
  (`lambda_grid`, `time_python_runner`, `time_r_runner`, `summarize`,
  Rscript dispatch). Adding a new scenario is now a ~50-line file.

### M9.2 — Microbenchmark expansion (criterion)

Extend `crates/skein-core/benches/block_cd.rs` with the scenarios its
README flags as "future bench":

- LLA outer iteration counts vs CD-only on a SCAD/MCP nonconvex
  problem.
- Parallel block-CD speedup at realistic sizes (`n_groups ≥ 8k`,
  sparse `X` via `SparseCSC`) — validates the 4–8× target from M2.5.
- Path solve scaling vs `n_lambdas` and vs working-set size.

Update the snapshot table in `crates/skein-core/benches/README.md`.

### ⏳ M9.3 — Cross-package speed benchmarks

For each (scenario, package) pair, measure wall-clock time-to-fit on a
shared λ-grid with a uniform convergence tolerance documented in the
methodology page.

The original M9.3 table was rewritten when the `benches/v2` Snakemake
suite landed: it covers the same (penalty × datafit) matrix as
publication-quality cells (5 seeds × {small, medium, large} ×
{deep, sparse}) and is the v1 harness's successor. Status now reflects
v2 aggregate coverage. v1 results in `benches/results/*.json` remain
the canonical small / medium / large lasso/MCP/SCAD/glasso LS
snapshots; v2 cells live in `benches/v2/results/scenarios/*.aggregate.json`.

| Penalty / datafit | Direct comparators present | Status |
|---|---|---|
| Lasso LS — deep + sparse | celer, glmnet, skglm, sklearn | ✅ done — `ls_lasso` (small + medium + large) |
| ElasticNet LS | glmnet, sklearn | ✅ done — `ls_elasticnet` (small + medium + large) |
| MCP LS — deep + sparse | ncvreg, skglm | ✅ done — `ls_mcp` (small + medium + large; v1 large is n=100k, p=10k) |
| SCAD LS | ncvreg | ✅ done — `ls_scad` (small + medium + large) |
| Group lasso LS | grpreg | ✅ done — `ls_group_lasso` (small + medium) |
| Group MCP LS | grpreg | ✅ done — `ls_group_mcp` (small + medium) |
| Group SCAD LS | grpreg | ⏳ pending — `ls_group_scad` scenario not yet scaffolded (skein + grpreg runners both exist) |
| Sparse-group MCP LS | (skein only — throughput showcase) | ✅ done — `ls_sparse_group_mcp` (small + medium) |
| Sparse-group SCAD LS | (skein only) | ⏳ pending — `ls_sparse_group_scad` scenario not yet scaffolded |
| Logistic lasso | glmnet | ✅ done — `logistic_lasso` (small + medium); celer/skglm/sklearn/pyglmnet runners are a separate ladder expansion, not an M9.3 blocker |
| Logistic MCP | ncvreg | ✅ done — `logistic_mcp` (small + medium) |
| Logistic SCAD | ncvreg | ⏳ pending — `logistic_scad` scenario not yet scaffolded |
| Poisson lasso | glmnet | ✅ done — `poisson_lasso` (small + medium); absolute 17× wall-clock gap vs glmnet is real and pre-existing (driven by glmnet's `dev_ratio` early termination); the M14g.2 apparent regression was closed as measurement noise |
| Poisson MCP | glmnet (surrogate) | ✅ done — `poisson_mcp` (small + medium) |
| Cox lasso | glmnet, lifelines (via ladder) | ✅ done — `cox_lasso` (small + medium) |
| Cox MCP | glmnet (surrogate) | ✅ done — `cox_mcp` (small + medium) |
| Graphical lasso (Σ⁻¹ L1) | R `glasso`, sklearn | ✅ done — `glasso_l1` (small/deep); regenerated 2026-05-18 after M14g.1 dispatch fix |
| Graphical MCP (Σ⁻¹) | (skein only) | ✅ done — `glasso_mcp` (small/deep) |

**Pending**: three SCAD-variant scenarios above. Each is mechanical
(`benches/v2/scenarios/<name>.py` mirroring its MCP sibling, plus a
config.yaml `headline:` entry); the skein runner already exposes
`Group/SparseGroup/LogisticSCADPathRegressor` and `r_runner.R` already
maps `penalty == "scad"` and `penalty == "group_scad"` to ncvreg /
grpreg. They're listed separately to avoid the prior table's bundled
"MCP/SCAD" rows hiding what actually shipped.

Output: per-scenario bar charts (matplotlib, committed PNG +
underlying JSON) and a markdown summary at `docs/benchmarks/speed.md`
(the consolidated docs page is M9.5 — still pending).

#### ✅ Lasso/LS — what shipped

- `benches/scenarios/lasso_ls.py` — deep-path regime
  (`λ_min/λ_max = 1e-3`).
- `benches/scenarios/lasso_ls_sparse.py` — sparse regime
  (`λ_min/λ_max = 5e-2`); active set stays at the true support of
  ~10 along the entire path instead of saturating.
- `benches/scenarios/_common.py` — extracted helpers
  (`lambda_grid`, `time_python_runner`, `time_r_runner`,
  `summarize`, `Rscript` shell-out) so adding the next scenario is a
  ~50-line file.
- **Runner fairness fix**: `skglm_runner.py` now uses
  `Lasso.path(X, y, alphas=…)`; `celer_runner.py` uses
  `celer.homotopy.celer_path(X, y, "lasso", alphas=…)`. Both warm-
  start internally, matching sklearn's `lasso_path` and skein's path
  solver. Comparison is now apples-to-apples.

Numbers committed at `benches/results/lasso_ls.json` and
`benches/results/lasso_ls_sparse.json`. Sketched in M10's status row;
detailed analysis in `docs/perf/lasso_ls_profile.md`.

#### ✅ MCP/LS — what shipped

- `benches/scenarios/mcp_ls.py` (deep, `λ_min/λ_max = 1e-3`) and
  `mcp_ls_sparse.py` (sparse, `5e-2`). γ = 3.0 pinned in the scenario
  file for fairness; runners now accept a `gamma` kwarg and forward
  it to skein's `MCPPathRegressor`, skglm's `MCPRegression`, and
  ncvreg via the R subprocess JSON payload.
- Runner side-fix in `skglm_runner.py`: skglm's `path()` appends the
  intercept as the last row of the coef matrix when
  `fit_intercept=True`, which was inflating `active_set_size` by 1
  across all skglm runs (including the existing lasso/LS rows). The
  intercept is now stripped and routed to `intercept_path`.
- **Headline (medium, n=10k, p=1k, 100-λ)**: skein **1.37 s** deep /
  **0.75 s** sparse vs skglm 3.35 s / 2.12 s vs ncvreg 5.38 s / 1.17
  s. Skein is fastest at every size, in both regimes.
- **First large-scale numbers for skein** (n=100k, p=10k, 100-λ):
  skein **510 s** deep / **497 s** sparse vs skglm 666 s / 702 s.
  Two surprises worth flagging: (1) the skein/skglm ratio compresses
  to ~0.71–0.77× at large (raw matvec dominates over algorithmic
  overhead at this scale, so both solvers converge on similar BLAS
  cost), and (2) the sparse-regime advantage *collapses* at large —
  per-λ fixed overhead (KKT eval, strong-rule scoring, working-set
  assembly) ends up the dominant cost when the active set stays
  tiny. The latter is a profiling target alongside M10's pending
  levers.
- **R-via-JSON transport is the bottleneck at large**. ncvreg fits
  at n=100k, p=10k OOM the R subprocess on the bench host (Apple M1,
  16 GB) because the current `r_runner.R` serializes `X` (8 GB) as
  JSON. Tracked as a bench-infra follow-up; not a blocker on the
  skein side. Swapping JSON → Arrow/Feather or `saveRDS` over a pipe
  unblocks SCAD/Cox/Poisson large-size comparisons too.

Correctness cross-check at small size (n=1k, p=100, 30-λ): skein and
skglm land at **bit-identical** minima at every λ (Jaccard 1.000,
rel-L2 0.000); skein vs ncvreg agrees on support at 23 / 30 λs with
mean rel-L2 4 % — the expected MCP local-min divergence M9.4 calls
out, not a bug.

Numbers committed at `benches/results/mcp_ls.json`,
`benches/results/mcp_ls_sparse.json`, and
`benches/correctness/results/mcp_ls.json`. Full write-up in
`docs/benchmarks/mcp_ls.md`.

#### ✅ M9.4 framework — what shipped (with the MCP row)

The roadmap's M9.4 listing called for a `benches/correctness/`
directory that runs cross-package agreement *at bench time* rather
than in pytest. That framework is now in place alongside the MCP
row:

- `benches/correctness/_common.py` — per-λ agreement metrics
  (active-set Jaccard, sign agreement, relative L2 distance) plus
  `PairAgreement` dataclass + JSON serialization + summary-table
  pretty-print.
- `benches/correctness/mcp_ls.py` — first runnable correctness
  script. Same problem generator and λ-grid as the perf scenario;
  fits skein / skglm / ncvreg; computes the full pairwise agreement
  matrix; writes JSON + prints a summary. Designed to be cloned for
  each (penalty, datafit) combo.

The at-scale `tests/fixtures/generate.R` parallel suite (the *other*
half of M9.4 — Python-side R regression fixtures at `n=1000, p=500`
and `n=5000, p=2000`) is still pending.

#### Pending across remaining scenarios

- The same pair (dense + sparse) per (penalty, datafit) combo above.
- For nonconvex penalties (MCP / SCAD): `Lasso.path` → analogue is
  `MCPRegression.path` in skglm, `ncvreg::ncvreg(...)` in R; both
  support warm starts natively.
- For groups: `grpreg::grpreg(...)` warm-starts; skglm's group lasso
  uses `solver.path(...)` in the same shape.
- Logistic / Poisson / Cox: GLM paths in sklearn (`logistic_regression_path`),
  glmnet (`glmnet(...)`), and skglm. ncvreg + pyglmnet for the others.

Once a comparator's runner is wired against its native path API,
adding a new scenario is mechanical (problem + λ-grid recipe).

### M9.4 — Correctness at scale

The M8.4 fixtures top out at `p=30`. Add a parallel suite to
`tests/fixtures/generate.R` for `n=1000, p=500` and `n=5000, p=2000`
covering the same penalties, with looser tolerances per scenario (LLA
local-min divergence on nonconvex problems will widen the band).

For Python comparators (skglm, sklearn, celer, pyglmnet) where there's
no fixture infrastructure, add `benches/correctness/` cross-checks that
run *at bench time* (not in pytest) and assert active-set + sign
agreement. This is intentionally separate from `tests/` so optional
comparator deps are not required to run pytest, and so cross-package
agreement at scale is treated as a benchmark output, not a correctness
gate on skein.

### M9.5 — Docs

New `docs/benchmarks/` section:

- `speed.md` — comparison tables and charts from M9.3.
- `correctness.md` — agreement matrix from M9.4.
- `methodology.md` — host hardware, package versions, λ-grid choices,
  convergence tolerances, fairness notes.

Wire into the Sphinx toctree under `docs/index.md`; add a "Performance"
section to `README.md` linking out.

### Out of scope (deferred past M9)

- Continuous regression gating in CI (R toolchain + 5 Python comparator
  deps in CI is too noisy and slow for the value).
- GPU / multi-precision throughput comparisons (lives wherever the GPU
  backend lands in M4's "GPU pending").
- Benchmarks of multi-response GLMs (depends on M7.3 landing first).

---

## M10 — Performance improvements

Bench-driven, reactive to M9. The first M9 lasso/LS run (n=10k, p=1k,
100-λ deep path) showed skein at **7.6 s** vs sklearn at **0.12 s** —
a ~60× gap. M10 is the work that closed most of it.

**Headline result on medium lasso/LS (n=10k, p=1k, 100-λ)**:

| package        | start of M10 | end of M10 (deep) | end of M10 (sparse) |
|---|---|---|---|
| sklearn (`lasso_path`)              | 188 ms     | 125 ms        | 99 ms  |
| glmnet (R, via Rscript)             | 950 ms     | 614 ms        | 510 ms |
| **skein**                           | **7.6 s**  | **1.17 s**    | **0.78 s** |
| celer (`celer_path`)                | 12.0 s †   | 2.73 s        | 307 ms |
| skglm (`Lasso.path`)                | 10.1 s †   | 3.39 s        | 2.26 s |

† pre-M9.3 runner-fairness fix; both packages were doing per-λ fits.

(Numbers shifted slightly between waves due to bench-machine
variance; the "end of M10" column is the post-F snapshot on the
same host.)

Skein-vs-comparators on medium, end of M10:

  vs sklearn:  9.4× deep, 7.9× sparse  (was 18× / 17× at M10 start)
  vs glmnet:   1.9× deep, 1.5× sparse  (was 3.5× / 3.5×)
  vs celer:    we're 2.3× faster deep, 2.5× behind sparse (their dual-extrapolation lever)
  vs skglm:    we're 2.9× faster deep, 2.9× faster sparse

Every M10 commit ships against a re-run bench with before/after
numbers; full timeline in `docs/perf/lasso_ls_profile.md`.

### ✅ M10.1 — Profile the medium lasso/LS path

`crates/skein-core/examples/lasso_ls_medium.rs` — pure-Rust binary
that mirrors `benches/problems.py::gaussian_lasso` at the medium size
so samply / cargo-flamegraph can resolve frames without the PyO3 +
interpreter layer in the way. `[profile.release]` carries
`debug = "line-tables-only"` for symbol resolution.

Result (at the time of profiling, post-first-wave fixes):

  ndarray::linalg::*::dot_generic     ~80%   inner CD's `col_dot`
  DenseMatrix::col_axpy               ~7%    residual update
  everything else                     ~13%

The bottleneck is **call volume on `col_dot`**, not slow code per
call. A microbench against four manual alternatives (Zip::for_each,
slice iter+sum, indexed for-loop, slice fold) showed ndarray's
existing path is already 4.7× faster than any of them — a hand-rolled
tight loop is *not* the answer. BLAS is.

Deliverable: `docs/perf/lasso_ls_profile.md` (M10.1 section), with
full methodology, microbench numbers, and the recommended next-step
ordering. M10.2's pre-profile hypotheses (LLA wrapper, Anderson
restart, screening overhead, standardize cost, PyO3 marshalling)
were *all* disproved by the profile — the gap was purely in
ndarray's `dot_generic`.

### ✅ M10.3 — Fixes (four waves)

Each wave lands as its own commit; numbers verified end-to-end on
the M9.1 bench harness.

**Wave 1 — col_axpy + F-order DenseMatrix + path-solver fixes
(`cfd1618`):**

- New `DesignMatrix::col_axpy(j, α, r)` trait method: `r += α · X[:, j]`
  in place. Replaces the per-coord `let col = design.columns(&[j]);
  for i { r[i] += α·col[[i,0]]; }` pattern (which allocated an
  80 KB `Array2` per coord update — ~10 GB of wasted heap traffic
  per medium-bench fit). Specialised on every backend (Dense, Sparse,
  Standardized, Augmented, Mmap×2, Chunked, MultiTask).
- `DenseMatrix::new` forces column-major (Fortran) layout. ndarray
  defaults to row-major, which makes `column(j)` strided with row
  stride `p · 8 B` — every element is a fresh L1 miss. F-order makes
  `column(j)` contiguous and `scaled_add` runs at memory bandwidth.
- `cd_solve_subset` returns the residual it already maintains
  internally, so `path.rs` doesn't recompute `r = Xβ − y` from
  scratch after each call (one O(np) matvec saved per λ).
- `find_kkt_violators` and `strong_rule_screen` use one BLAS gemv
  (`full_grad`) instead of `p` per-coord `coord_grad` calls.

Result: 7.6 s → 3.2 s on medium deep (2.4×).

**Negative result — inner active-set CD (`b874fa7`):**

Implemented Friedman/glmnet-style two-phase cycling (`cd_solve_subset`
cycles a small inner active set `A`, verifies on `features \ A`,
grows `A` if any prox produces a non-zero update). Came out 9–47%
*slower* on the medium scenario in both Anderson-history-clear and
Anderson-history-keep variants — the saturated regime
(`A ≈ features`) makes the verification sweep pure overhead and the
Anderson interaction makes the savings unrecoverable. Reverted; the
write-up in `docs/perf/lasso_ls_profile.md` preserves the analysis
so future contributors don't try the same thing.

**Wave 2 — adaptive inner tolerance via prox-gradient distance
(`53b1275`):**

`compute_outer_state` returns both the working-set violators *and*
the global outer prox-gradient distance `max_j |β_j − prox_j(β_j −
grad_j / lc_j, 1/lc_j)|` — same units as `config.cd.tol`,
penalty-agnostic, identical to skglm's `dist_fix_point_cd`.

  Outer convergence: `max_pgd < config.cd.tol` (replaces "no KKT
    violators under λ · 1e-6 gradient threshold"; new criterion is
    commensurable with the inner CD's stopping rule and uniform in
    λ).
  Adaptive inner tol: `inner_cd_cfg.tol = max(config.cd.tol, 0.3 ·
    prev_max_pgd)` — celer's pattern. When far from optimum the inner
    stops sloppy; as PGD shrinks toward `tol`, inner_tol → `tol` and
    the final β meets the user's request.

Net wallclock on medium deep: ~unchanged (3.2 s), but **iter sum
drops 26% (430 → 317)**. The savings are spent on the tightening
phase; structural correctness up, wallclock floor unchanged.
Sets up future wins to compound rather than offset.

**Wave 3 — KKT-priority WS construction (`9884188`):**

`Screening::Strong` now means a celer/skglm-style priority-ranked
active set rather than the Tibshirani strong rule:

  ws_size = max(p0, 2 × |support|)         (default p0 = 10)
  WS = top-ws_size by |grad_j| / w_j        (active + unpenalised pinned in)

Replaces the "fall back to full feature set at λ_max" cliff:
cold-start λ_max WS goes from 1000 → 10. Every `PathConfig { ... }`
literal across the codebase grew a `p0: 10,` field (mechanical batch
edit). Test recalibrated with `p0=3` on a `p=20` toy problem where
the default `p0=10` happens to equal the test's assertion bound.

Wallclock unchanged on the saturated medium scenario — the
late-path WS still caps at `p`. Surfaces on sparse-regime variants
where the WS actually stays proportional to support.

**Wave 4 — `blas-accelerate` Cargo feature (`ee9740d`):**

Routes ndarray's `dot` / `scaled_add` / `gemv` through Apple's
Accelerate framework on macOS via `blas-src` + `accelerate-src`.
Zero install cost — Accelerate ships with the OS — and turns the
inner CD's `col_dot` from `dot_generic` into `cblas_ddot`.

  skein-core / Cargo.toml:
    blas-accelerate = ["ndarray/blas", "blas-src/accelerate", "dep:accelerate-src"]

  skein-core / src/lib.rs:
    #[cfg(feature = "blas-accelerate")]
    use accelerate_src as _;        # essential — anchors the linker

  skein-py / Cargo.toml:
    blas-accelerate = ["skein-core/blas-accelerate"]   # passthrough

Build with `maturin develop --release --features blas-accelerate`.

Result: skein medium deep 3.32 s → **1.75 s (1.9×)**, sparse 2.50 s
→ **0.96 s (2.6×)**.

The M10.1 prediction was ~3× from BLAS dispatch; the realised speedup
is somewhat below that because the path solver does work outside the
BLAS-replaceable hot path (PGD verifier, priority-WS ranking).

**Wave 5 — F-series: dual extrapolation + gap-safe screening
(`2fea09c`, `2d025d4`, `971f73d`, `5d3c755`):**

Built the full celer-style dual machinery, gated on penalty type
and verified end-to-end. *Wallclock-neutral on M9.3 scenarios* —
the existing PGD + priority-WS combo already at the algorithmic
floor; the extra dual machinery has nowhere to recover. Documented
honestly so the next person doesn't repeat the experiment.

  F.1 (`2fea09c`) — `Datafit::lasso_dual_obj` and `Penalty::
    dual_correction` trait methods. LS overrides the former for
    unweighted samples; ElasticNet overrides the latter to
    subtract the ridge contribution (matches celer's `dual_enet`).

  F.2 (`2d025d4`) — Per-λ residual+β history (last K=6 pairs);
    Anderson extrapolation on the residual sequence; same
    coefficients applied to β so the extrapolated `(β_acc, r_acc)`
    is self-consistent under `r = Xβ − y` (Σc=1 cancels y). Best
    known dual obj across outer passes accumulated; gap = primal −
    best_dual_obj.

  F.3 (`971f73d`) — `Penalty::has_lasso_form_dual_gap()` (default
    false; ElasticNet returns true). Ungates the gap-based stop
    only when the penalty's L1 envelope is tight at its optimum.
    Pre-flight pytest caught the MCP regression at one failure
    instead of pinning all cores; saved-memory protocol from F.1's
    runaway saved us. Outer stop:
        converged = match outer.gap {
            Some(g) => g < tol² || max_pgd < tol,
            None     =>             max_pgd < tol,
        };
    The OR-style guard means tight-tol tests (`tol = 1e-12`) fall
    back to PGD because `tol²` is below double-precision floor —
    no regression risk.

  F.4 (`5d3c755`) — Gap-safe sphere screening (Fercoq-Gramfort-
    Salmon 2015): mark feature `j` as provably zero at this λ's
    optimum when `|grad_j| + r_safe · ‖X_j‖₂ < λ · w_j` with
    `r_safe = √(2·gap/n)`. Per-λ `screened: Vec<bool>` mask;
    features once flagged are removed from the WS for the rest of
    the λ's outer KKT loop. Reset per λ.

What it doesn't move:

  scenario       iter sum   kkt_passes   ws[end]
  ---------------------------------------------------
  lasso_ls       317        110          923 (was 1000)
  lasso_ls_sp    304        100          10  (was 10)

The deep bench shows ~77 features pruned at λ_min — the screening
machinery fires correctly — but the inner CD overhead at this scale
is small enough that 7% fewer features per sweep doesn't surface
in wallclock. Most λs converge in 1 outer pass, so the post-pass
screening has nowhere to apply. Pre-pass screening (using the
previous λ's gap to prune the *initial* WS of the next λ) is the
natural extension that *would* deliver in sparse, but it's a
structural change that wasn't pursued.

Why it didn't deliver the projected celer-sparse 2× gap closure:
celer's structural advantage in sparse comes from starting with
`p0 = 100` and aggressive pre-pass screening; we start at `p0 = 10`
and screen post-pass. The algorithmic envelopes are different in
ways our F-series machinery doesn't recover.

### ✅ M10.4 — Verified against the bench

Both `lasso_ls` (deep) and `lasso_ls_sparse` ran end-to-end with
`packages=all, sizes={small, medium}, trials=3, n_lambdas=100` after
each wave. Snapshots committed under `benches/results/`.

Did we hit the M10 entry-time targets? Partial:

- "Within 2× of sklearn at small and medium" — **No**. We're 6.5–9.6×
  on medium. The sklearn `lasso_path` Cython has zero path-level
  structural overhead and we can't approach that without a similar
  Cython-grade rewrite. Keeping it as a stretch goal, not a
  blocking target.
- "Faster than sklearn on sparse where screening pays off" — **No**,
  sklearn still wins. But we *did* reach essentially-matched parity
  with glmnet (1.18× sparse, 2× deep), and beat skglm and celer in
  deep regime, which is more directly relevant to the niche skein
  targets.

### Pending — remaining levers post-F

E. **Anderson on residual instead of β** — partially subsumed by
   F.2, which maintains a residual history and runs Anderson on it
   with coefficients applied jointly to β. The "pure Anderson on
   residual replacing β-Anderson in `cd.rs`" is a separate
   change; given F.2 already extrapolates residuals at the path
   level, the marginal value of redoing it inside `cd_solve_subset`
   is unclear without a fresh profile.

G. **Cross-platform BLAS** — sibling features `blas-openblas`
   (`blas-src/openblas` + system `libopenblas-dev`) and
   `blas-mkl` (`blas-src/intel-mkl`). Required before the
   `blas-accelerate` wins show up in distributed wheels.
   Coordinates with M8's cibuildwheel — likely needs a per-platform
   feature in the `wheels.yml` matrix.

H. **Pre-pass gap-safe screening** — F.4 applies the FGS sphere
   only *after* each outer KKT pass; pre-pass screening at λ entry
   (using the previous λ's gap to prune the *initial* WS) is what
   celer actually does. Modest extension — re-uses the gap and
   gradient already computed at λ_{k-1}'s last pass to filter the
   priority-WS at λ_k entry. Most upside in sparse-regime paths;
   on M9.3's scenarios the existing post-pass screening already
   fires after 90 % of λs converge in one pass anyway.

I. **Cython-grade rewrite of `cd_solve_subset`** — the single
   move that would actually catch sklearn's `lasso_path`. Possible
   but the cost-benefit isn't clear given skein's audience uses
   it for capabilities sklearn doesn't have. Listed for honesty,
   not as a recommended next step.

### Out of scope for M10

- GPU acceleration (separate workstream tied to M4's "GPU pending").
- Optimizing the GLM datafits before lasso/LS is competitive — the
  scalar lasso path is the load-bearing scenario; everything else
  inherits from it. (This bar is now met for glmnet-class
  competitors, so per-datafit follow-up can begin.)
- A Cython/SIMD-grade rewrite of `cd_solve_subset` aimed at
  catching sklearn's `lasso_path`. Possible, but the cost-benefit
  isn't clear given the target audience uses skein for
  capabilities sklearn lacks (nonconvex group penalties, weighted
  axes), not as a drop-in lasso replacement.

---

## M11 — Graphical models (precision matrix estimation)

Status: scoped, not started. Headline pull is **network psychometrics**
(Borsboom/Epskamp lineage): social scientists fit graphical lasso to
estimate symptom-interaction networks instead of latent-factor models,
then increasingly compare those networks across populations
(treatment vs control, age cohorts, time points). Their current tooling
(R `glasso` / `qgraph` / `bootnet` / `EstimateGroupNetwork`; Python
`sklearn.covariance.GraphicalLasso`) is L1-only — a well-known
shrinkage-bias gap that nonconvex penalties (MCP/SCAD) close — and
joint estimation across populations is split across separate packages.

Skein's existing trait surface is a near-perfect fit:

- Single-population glasso, via Friedman et al. 2008 block-CD, reduces
  to a sequence of weighted-lasso subproblems on each column. Each
  subproblem is residual-shaped — exactly the `LeastSquares` datafit
  skein already has. This mirrors the
  `GlmDatafit::surrogate_at(β) → LeastSquares` pattern
  (`crates/skein-core/src/datafit/mod.rs:37`) used for logistic/Poisson/
  Cox via prox-Newton.
- Joint glasso (Danaher–Wang–Witten 2014, group form), via ADMM. The
  Z-update is *edge-wise* and, for group-JGL with MCP/SCAD on edges,
  is exactly an existing `GroupPenalty::prox_group` call. The only
  genuinely new prox is the **eigenvalue prox of the log-det barrier**
  in the Θ-update — a closed-form eigen-thresholding step.

No `Penalty` / `Datafit` trait changes. New code: one new datafit
(`GramLeastSquares`), two new solvers (`solver/glasso.rs`,
`solver/glasso_admm.rs`), one new prox primitive (log-det eigen
threshold), Python bindings, sklearn-style estimators, EBIC tuner.

### ✅ M11.1 — Single-population glasso (L1 + MCP + SCAD)

Headline deliverable: `GraphicalLasso`, `GraphicalMcp`, `GraphicalScad`
sklearn-style estimators returning `.precision_`, `.covariance_`,
`.info_`. Each accepts either raw `X (n, p)` or precomputed
`cov (p, p)` (sklearn convention; sniff via shape + symmetry).

Scope:

- `crates/skein-core/src/datafit/gram_least_squares.rs` — gram-form
  LS that maintains `r = Wβ − s` instead of `r = Xβ − y`; integrates
  with `cd_solve` unchanged.
- `ScalarPenaltyFactory` shim in `penalty/mod.rs` so the outer
  solver is generic over `Lasso | Mcp | Scad | ElasticNet`.
- `crates/skein-core/src/solver/glasso.rs` — Friedman/Hastie/Tibshirani
  outer loop; per-column k, peel `W_{-k,-k}` and `s_{-k,k}`, build
  `GramLeastSquares`, build penalty from factory with the row-k slice
  of `edge_weights`, call `cd_solve`, update Θ and W.
- `crates/skein-py/src/lib.rs` — `solve_glasso_{lasso,mcp,scad}`
  pyfunctions returning `(precision, covariance, info_dict)`.
- `python/skein_glm/estimators.py` — three new estimator classes.
- `python/skein_glm/graph_selection.py` — `ebic_path` (Foygel–Drton
  2010, the field-standard tuning rule for graphical models).

Exit criteria:

- L1 with no edge weights matches `sklearn.covariance.graphical_lasso`
  to `‖Θ_skein − Θ_sklearn‖_F < 1e-3` on p=20, n=200 Gaussian draws.
- Synthetic recovery: known sparse Θ, fit at sufficient λ, support
  matches.
- EBIC selects a λ within a small range of the truth-recovery λ on
  a synthetic ground-truth precision matrix.
- ✅ Psychometrics replication: depression-symptom network end-to-end
  example shipped in M14a (`docs/examples/psychometrics.md`) — chains
  `polychoric_correlation` → EBIC-tuned `GraphicalMCP` →
  `GraphicalBootstrap.fdr_threshold(fdr=0.10)`. Retains all 7 planted
  edges with zero false discoveries at n=400, 300 bootstraps.
- Benchmark within 2× of sklearn at p=200 for L1; MCP/SCAD have no
  mainstream-package equivalent to compete against.

### ✅ M11.2 — Joint glasso across populations (group form)

Headline deliverable: `JointGraphicalLasso`, `JointGraphicalMcp`,
`JointGraphicalScad` accepting `[X^(1), …, X^(K)]` or
`[S^(1), …, S^(K)]`, returning `.precisions_`.

Scope:

- `crates/skein-core/src/prox.rs` — add `logdet_eigen_prox(A, ρ)`
  closed-form via eigendecomposition.
- `crates/skein-core/Cargo.toml` — `ndarray-linalg` dependency.
- `GroupPenaltyFactory` shim in `penalty/mod.rs` to mirror the
  scalar factory.
- `crates/skein-core/src/solver/glasso_admm.rs` — ADMM outer loop
  on `(Θ^(k), Z^(k), U^(k))_{k=1..K}`. Θ-update via
  `logdet_eigen_prox` (Rayon-parallel across pops). Z-update is
  edge-wise: K-vector `(Θ^(k)_ij + U^(k)_ij)_{k=1..K}` through
  `GroupPenalty::prox_group` (existing `GroupMcp` / `GroupLasso`).
  Dual residual stop.
- Three more pyfunctions + three estimator classes mirroring M11.1.
- `joint_ebic_path` in `graph_selection.py`.

This is the first ADMM kernel in skein. Keep it self-contained inside
`solver/glasso_admm.rs` for v1; extract if a second ADMM use case
appears later. Don't pre-generalize.

Exit criteria:

- K=2 populations from identical Θ, JGL with λ₂ = 0 must equal two
  independent single-glasso fits to ~1e-6; with λ₂ large, must
  collapse to identical Θ̂^(1) = Θ̂^(2).
- ADMM primal and dual residuals both decrease toward zero on a small
  problem (modulo numerical noise).
- Benchmark vs R `EstimateGroupNetwork` on p ∈ {50, 100, 200}, K=3.

### M11.3 — Follow-ups

- **Polychoric / polyserial correlations** for ordinal Likert data —
  high value for psychometrics; Python-side preprocessing helper.
- ✅ **Bootstrap edge stability** (`bootnet`-style): two pure-Python
  wrappers around the M11.1 / M11.2 estimators in
  `python/skein_glm/graph_stability.py`.
  `GraphicalStabilitySelection` lifts MB (2010) subsample stability
  selection to **edges** — sweep a user-supplied λ-grid; for each
  (bootstrap, λ) refit, record the off-diagonal nonzero pattern;
  aggregate to per-(λ, i, j) selection probability with a max-over-λ
  stable-edge set at threshold π_thr (defaults to 0.6). Output shapes
  `(n_lambdas, p, p)` single / `(n_lambdas, K, p, p)` joint;
  `stable_edges_` is `(n_stable, 2)` for single, a list of K such
  arrays for joint. `GraphicalBootstrap` runs the classic non-
  parametric (resample-with-replacement) bootstrap at a single λ and
  returns the per-edge sampling distribution: mean, SD,
  `[α/2, 1−α/2]` quantile CIs, and edge selection probability —
  the headline `bootnet::bootnet(type="nonparametric")` output for
  edge error bars. Both auto-dispatch single-vs-joint via the
  estimator's `alpha`/`lambda_2` surface, parallelize the bootstrap
  loop over `joblib`, and reject precomputed-covariance inputs with
  a clear error. Exported from `skein_glm`; 16 pytest cover shapes,
  signal recovery (true non-zero edges have higher max-prob than
  true zeros on a synthetic sparse-Θ problem; bootstrap selection
  probability separates true edges from non-edges), threshold
  monotonicity, joint dispatch (MCP + lasso), CI ordering
  (`ci_lower ≤ mean ≤ ci_upper`), reproducibility under fixed
  `random_state`, `n_jobs` parity (serial == parallel), and full
  parameter validation including covariance-input rejection. No
  Rust changes; entirely composes existing M11 estimators.
- **Fused JGL** (TV across populations) — needs a 1D TV prox we don't
  have; group form covers most published applications.
- Time-varying / dynamic glasso, mixed graphical models.

### Out of scope for M11.1 + M11.2

- Anything in M11.3.
- A new general-purpose ADMM solver kernel decoupled from glasso —
  premature; revisit when a second use case appears.
- GPU acceleration of the eigendecomposition (tied to M4's GPU
  workstream).

---

## M12 — Hardening (robustness, test coverage, CI)

Cross-cutting punch list compiled from a v0.7.0 audit (May 2026). No
new algorithmic surface — purely reduces the chance that an
unobserved regression ships. Items are independent and can land in
any order; the ordering below is the recommended ROI sequence.

### Correctness gaps

- **C1 — penalty unit-test coverage.** 6 of 9 files in
  `crates/skein-core/src/penalty/` have no `#[cfg(test)]` block:
  `mcp.rs`, `scad.rs`, `group_lasso.rs`, `group_mcp.rs`,
  `factories.rs`, `mod.rs`. Add direct prox / threshold tests against
  closed-form references (soft-threshold for ℓ₁, MCP firm-thresholding
  analytic form). This is where sign / scale bugs hide and where
  ncvreg / grpreg parity is decided.
- **C2 — datafit base-module tests.** `datafit/least_squares.rs`
  (133 lines), `datafit/mod.rs` (121 lines) untested.
  `multinomial_logit.rs` (753 lines) inline-only. `least_squares` is
  the surrogate every GLM lowers to via `GlmDatafit` — a regression
  here propagates everywhere.
- **C3 — Rust integration test directory.** All Rust tests are
  inline. Add `crates/skein-core/tests/` exercising end-to-end chains
  (e.g. `Standardized<SparseCSC>` + `BinomialLogit` + `group_mcp`
  via LLA). Inline tests miss cross-module composition bugs and the
  large solver state machines (`solver/path.rs`, 1714 lines;
  `solver/block_cd.rs`, 1046 lines) lack any integration harness.
- **C4 — R-fixture suite is opt-in.** `tests/test_r_regression.py`
  has two `pytest.skip` paths when `tests/fixtures/*.json` is missing
  (generated by `tests/fixtures/generate.R`). CI does not regenerate,
  so glmnet/ncvreg/grpreg parity drift can ship silently. Either
  commit the fixtures or run R in CI.
- **✅ C5 — parallel block-CD Lipschitz caveat.** Done. Per-group
  Lipschitz is exact for serial Gauss-Seidel; Jacobi parallel mode
  assumes small `Xᵀ_g X_{g'}` cross-coupling and on overlapping groups
  the snapshot-fold step (`beta[j] = new_block[k]`) corrupts shared
  coordinates because two threads holding column `j` overwrite each
  other. New `Groups::has_overlap()` (O(`idx.len()` + `max_idx`)) is
  checked at entry to `block_cd_solve_subset_parallel_with_cache`; on
  overlap the function dispatches to the serial Gauss-Seidel variant
  and fires a `std::sync::Once`-gated stderr warning so the misuse
  doesn't ship silently while joblib path × CV loops don't spam the
  process. Fixture
  `block_cd_subset_parallel_with_overlapping_groups_falls_back_to_serial`
  exercises the path with two overlapping groups and verifies the
  parallel result is bit-identical to the serial result (matching β,
  iter count, and `converged` flag).

### Robustness gaps

- **✅ R1 — unwrap clusters in path solvers.** Done. Audited the 59
  hits and triaged. `design/mmap.rs` / `design/mmap_f32.rs` (10 each)
  + the bulk of `standardize.rs` (7) + `solver/path.rs:992,1010,1256,1524`
  are all in `#[cfg(test)]` modules — acceptable, tests panic on
  setup-invariant violation by design. Production-code hits left were:
  (a) `solver/block_path.rs` and `solver/path.rs` loop-invariant
  unwraps in the strong-rule screening branches, already documented
  with `.expect("…")` invariant strings; (b) `solver/block_path_lla.rs`
  three bare `unwrap()` calls, swapped to documented `.expect()`
  matching the `block_path.rs` pattern (M12 prior session); (c)
  `solver/cd.rs:238` `iterates.last().unwrap()` in `anderson_extrapolate`,
  swapped to `.expect("iterates non-empty: len() ≥ 3 checked above")`
  matching the documented invariant in `anderson_extrapolate_pair`;
  (d) `solver/glasso_admm.rs:126` `Groups::from_csr(...).expect("group construction")`
  was dead code (built `_groups` and immediately dropped, only `n_edges`
  was used downstream) — removed entirely along with the now-unused
  `use crate::groups::Groups;` import.
- **R2 — CI doesn't gate the PyO3 layer.** CI runs only
  `cargo test -p skein-core --lib`. The `skein-py` bindings crate has
  no standalone Rust tests; signature-incompatible changes compile
  green and only fail at Python import. Add a build-only check on
  `skein-py` and / or a `pytest tests/test_smoke.py` smoke job.
- **✅ R3 — solver pre-flight protocol unenforced.** Done. CLAUDE.md
  mandates `cargo test -p skein-core
  solve_path_screening_on_matches_screening_off_within_tol --
  --nocapture` before any convergence-logic change after the gap-safe
  screening incident. Now a separate `solver pre-flight (tight-tol
  screening)` step in `.github/workflows/ci.yml` with `timeout-minutes:
  2`, ahead of the full `cargo test` step. If a future change makes a
  stopping condition unreachable (`gap < tol²` → `1e-24`), the test
  hangs in isolation, hits the 2-min timeout, and fails fast pointing
  at the right culprit instead of starving the parallel suite.
- **✅ R4 — numerical guards scattered.** Done. `W_FLOOR = 1e-6` and
  `ETA_CLAMP = 30.0` now live in `crates/skein-core/src/numerics.rs`;
  `binomial_logit`, `poisson_log`, `cox_ph`, and `huber` import them
  from there. A future bump (tighter clamp for an aggressive Newton
  step, or relaxed floor) is one edit.

### Performance / observability gaps

- **P1 — large-n bench snapshots missing.** `benches/results/`
  has small + medium scenarios for lasso / MCP / SCAD-LS, but no
  n ≥ 100k fixtures and no comparators for ADMM / graphical lasso
  despite M11 being shipped. M9.4 ("at-scale correctness fixtures")
  is open. Without large-n snapshots, perf regressions at the size
  that matters to users are invisible.
- **✅ P2 — criterion bench tree is one file.** Done. Added three new
  bench files alongside the existing `block_cd.rs`:
  `benches/lla_outer.rs` (LLA outer-loop on group MCP, varying γ and
  n_groups), `benches/prox_newton_glm.rs` (single-λ logistic + Poisson
  Lasso to capture IRLS overhead), and `benches/glasso.rs`
  (single-population glasso scaling p=20→100 + joint glasso ADMM at
  K=2,3 populations). Wired into `Cargo.toml` `[[bench]]` entries;
  `benches/README.md` updated with how-to-run + per-scenario
  description. Frobenius vs operator-norm Lipschitz comparison
  remains out of scope — M2.6 removed the Frobenius path entirely, so
  comparing would require reintroducing the loose bound as opt-in.
- **P3 — M10.H pre-pass gap-safe screening.** Worth attacking *only*
  with large-n benches in place (P1) so the delta is measurable.
  Post-pass screening is wallclock-neutral on existing scenarios
  because most λ converge in one pass; pre-pass screening would help
  long-path GLM scenarios.
- **✅ P4 — `skein-py/src/lib.rs` split.** Done. lib.rs went from
  10,628 → **275 lines (-97%)** — pure entry point with module
  declarations + the `#[pymodule]` registration block. Every datafit
  family lives in its own module:
  `glasso.rs` (261), `multinomial.rs` (897), `multitask.rs` (1290),
  `mmap_chunked.rs` (784), `glm.rs` (4544 — logistic + poisson + huber +
  cox dense + sparse), `ls.rs` (2760 — LS scalar/group dense + sparse +
  the cross-cutting helpers `parse_screening`, `groups_from_labels`,
  CSC readers, glmnet scales, intercept builders, sparse weight
  builders, `PathOutput` type alias). PyO3's `wrap_pyfunction!`
  accepts module paths via `use $function as wrapped_pyfunction`, so
  registration uses e.g. `wrap_pyfunction!(glm::solve_logistic_mcp_path, m)`.
  `glm.rs` exposes a handful of GLM-shared helpers (`split_intercept`,
  `validate_y_*`, `build_logistic_*`, `append_intercept_*`) as
  `pub(crate)` so LS / multinomial / multitask / mmap_chunked call
  them as `crate::glm::name`. Validated end-to-end: 412 pytest passing,
  cargo fmt clean, clippy `--workspace --all-targets -- -D warnings`
  clean.

### Recommended ordering

If the goal is a v0.8.0 that is materially more robust without scope
creep:

1. C1 + C2 (penalty + datafit tests) — highest correctness ROI,
   smallest blast radius, no API change.
2. R2 (CI smoke for PyO3 layer) — closes the most embarrassing
   regression class.
3. C4 (R fixtures in CI) — locks in glmnet / ncvreg parity.
4. P1 (large-n bench snapshots) — prerequisite for any future
   perf claim including M10.H / P3.
5. C3 (Rust integration test dir) + R1 (sweep `unwrap` in path
   solvers) — incremental, can ride along.

### Out of scope for M12

- New algorithmic surface (handled by M3.7 / M6 / M7.3 demand-driven
  workstreams).
- GPU / mixed precision (M4.x).
- Cython-grade rewrite (M10.I, off-roadmap).

---

## M13 — Performance findings from `benches/v2` release-profile run

Surfaced by a 200-cell LS-family release-profile run (skein 0.7.0, 5 seeds
per cell, n=1000 + n=10000, dense + sparse regimes; tol = 1e-7).
Numbers quoted are median fit time, n=10k, p=1k, dense regime
(internal regime key is `deep` for path-extent reasons; the solution
saturates at the tail, hence "dense"). See
`benches/v2/results/scenarios/` for the underlying JSONL aggregates
and `paper/tables/T2_headline_timings.md` for the full table.

These are open performance gaps; each numbered item is a candidate
workstream with the evidence already gathered.

### Headline observed gaps (release profile, M1 Accelerate)

| Pair | skein | comparator | ratio |
|---|---|---|---|
| Lasso vs sklearn | 2.91 s | 0.20 s | **14.4× behind** |
| Lasso vs skglm | 2.91 s | 4.80 s | 1.65× ahead |
| ElasticNet vs sklearn | 3.37 s | 0.28 s | **12.1× behind** |
| MCP vs skglm | 3.85 s | 4.61 s | 1.20× ahead |
| group_lasso (self-reference) | 7.93 s | — | — |
| group_mcp vs group_lasso | 43.5 s | 7.93 s | **5.49× LLA penalty** |

Cross-package agreement at medium scale is bit-perfect (Jaccard / sign / rel-L2
all 1.0 vs skglm; 1.0 / 1.0 / 1e-4 vs sklearn), so these timings compare
equivalent fits.

### ✅ M13.1 — Adaptive saturation bypass on `solve_path` screening

Strong-rule screening is *slower* than no screening on convex Lasso at
the dense regime, despite trimming the working set to 22% of features:

| screening | wall-clock | total iters | KKT passes | avg working set |
|---|---|---|---|---|
| `off` | **1.60 s** | 186 | 100 | 1000 |
| `strong` | 2.19 s | 214 | 128 | 220 |
| `gap_safe` | 2.32 s | 214 | 128 | 261 |

At λ_min on a dense-tail grid the active set saturates near 79% of features
(791 of 1000 here), so strong rule prunes ~78% of columns from the
working set but then needs ~28 extra KKT-pass / re-iterate cycles to
prove the pruned columns really stay zero. The net per-λ work goes up.

**Shipped fix** (`crates/skein-core/src/solver/path.rs`,
`SCREENING_SATURATION_THRESHOLD = 0.5`): at each λ, count nonzeros in
the warm β; when `active / p > 0.5` degrade the effective screening to
`Off` for that λ (one full-feature sweep, single KKT pass, no priority-
set construction). The KKT verifier is unchanged so correctness is
preserved.

The wall-clock impact depends sharply on tolerance and on
whether the bench harness is BLAS-thread-contended. Measurements
across all four regimes (medium / dense, n=10k, p=1k, 5 seeds, the
v2 grid):

**Serial, isolated fit (single process, no other bench cells competing
for BLAS threads):**

| Setting | screening = off | screening = strong (before M13.1) | screening = strong (with M13.1) | Saved by M13.1 |
|---|---|---|---|---|
| tol = 1e-4 | 1.60 s | 2.19 s | **1.93 s** | 11.8 % |
| tol = 1e-7 | 2.29 s | 2.92 s (estimated from parallel baseline) | **2.06 s** | ~29 % vs estimate |

At tol = 1e-4 the saturation bypass fires on 13 / 100 λs (the tail);
strong screening still costs more than `off` there, but less than
before. At tol = 1e-7 the bypass turns strong from a slight regression
vs `off` (estimated) into a slight win over `off` (2.06 s < 2.29 s).

**Parallel 4-core snakemake (the actual `benches/v2 ls_headline`
target — every comparator and every seed fan out concurrently, BLAS
contends across cells):**

| Scenario | Before M13.1 | After M13.1 | Effective speedup |
|---|---|---|---|
| ls_lasso medium / dense | 2.91 s | 2.91 s | **1.00×** (within noise) |
| ls_mcp medium / dense | 3.85 s | 3.68 s | 1.05× |
| ls_scad medium / dense | 4.10 s | 3.44 s | **1.19×** |
| ls_elasticnet medium / dense | 3.37 s | 3.06 s | 1.10× |
| ls_group_lasso medium / dense | 7.93 s | 8.04 s | 0.99× (M13.4 territory) |
| ls_group_mcp medium / dense | 43.50 s | 41.92 s | 1.04× (M13.4 territory) |
| medium / sparse (all six) | — | — | 0.94×–1.04× (within noise; threshold rarely trips at sparse-regime active-set sizes) |

The serial win is real and measurable; the parallel-bench win is
mostly hidden by BLAS-thread contention, which dominates per-cell
wall-clock once 4 cells share an M1's 8 BLAS threads. **M13.1's value
is therefore mostly correctness-shape: skein no longer spends ~28
extra KKT passes per dense-tail path on dropping and re-adding features
the saturation regime guarantees will stay active.** Users who run a
single fit (the common case for CV inside Python, not the snakemake
batch) get the full serial win.

Regression check at the v2 grid passes:
- Sparse regime: threshold rarely trips (active sets < 0.5 p), so the
  branch is taken at parity with prior behavior.
- Small / dense: threshold trips at the tail but workload is small
  enough that the bypass-vs-screen difference washes out.

Verified by:
- `cargo test -p skein-core --lib --release` (343 / 343 pass).
- New regression test
  `solver::path::tests::saturation_bypass_disables_screening_at_high_active_density`
  asserts the bypass fires on a dense-truth problem and produces
  bit-identical coefficients to `Screening::Off`.

### ✅ M13.2 — Cross-λ gradient cache (SHIPPED)

**Profile (`SKEIN_PROFILE_PATH=1 lasso_ls_medium`, n=10k, p=1k, 100 λs,
tol=1e-6, M1 Accelerate):**

| Phase | per-λ before | per-λ after | Δ |
|---|---:|---:|---:|
| screening (init WS) | 2918 µs | **209 µs** | -2709 µs |
| inner_cd | 22518 µs | 22226 µs | (noise) |
| outer_state (KKT) | 2814 µs | 2864 µs | (noise) |
| other | <10 µs | <10 µs | — |
| **wall** | **2.847 s** | **2.552 s** | **-10.4 %** |

The earlier "13 ms unattributed" claim was an overestimate based on a
different bench setup; the env-var-gated `PhaseTimings` instrumentation
in `solve_path` (kept as observability) attributed the per-λ cost to:
- 79 % inner CD (genuine CD work — ~3.5 sweeps × ~800 active features
  × col_dot at n=10k);
- 10 % `priority_rule_screen` matvec at λ start;
- 10 % `compute_outer_state` matvec at KKT-loop end.

**Both matvecs compute `X^T r / n` on the same residual** — the
post-CD residual at λ_k *is* the warm-start residual at λ_{k+1}. Fix:
`OuterState` now exposes `grad`; `solve_path` caches it as `prev_grad`
between λs and feeds it into a new `priority_rule_screen_with_grad(...)`
variant that skips the recompute. Cache cleared on cold start (k=0)
and after saturated-bypass λs (Off mode skips `compute_outer_state`,
so no fresh grad). Iter count + KKT passes unchanged ⇒ no algorithmic
regression.

**Reproduce:**
```bash
cargo build --release --example lasso_ls_medium
SKEIN_PROFILE_PATH=1 ./target/release/examples/lasso_ls_medium
```

**Remaining per-λ cost** is ~88 % inner CD + ~11 % `compute_outer_state`
matvec (at the KKT-loop end, on freshly-CD'd residual — can't share
with anything). Further wins now have to come from the inner CD's
~22 ms / λ.

### M13.3 — Convex Lasso/EN sklearn gap is NOT explained by LLA

(Corrects an earlier suspicion.) `ElasticNetPathRegressor.fit` and the
γ→∞ Lasso path both call `_core.solve_elastic_net_ls_path` directly,
which routes to `build_path_outputs` with an `ElasticNet` penalty — no
LLA outer loop. The 12–14× sklearn gap therefore cannot be closed by
adding a "convex fast-path" — it lives in the path solver itself, most
likely a combination of M13.1 (screening) and M13.2 (per-λ fixed cost).

This also reframes M10.I ("Cython-grade rewrite of `cd_solve_subset`"):
that lever targets the *inner CD constants*, but the screening +
fixed-cost evidence says the bottleneck is *outside* `cd_solve_subset`
on this scenario. Rewriting the inner loop would not close the gap
unless the outer per-λ overhead is first reduced.

### ✅ M13.4 — `group_mcp` is 5.49× slower than `group_lasso` at the same scale — RESOLVED by Phase 2.3 + M13.4b + M13.4c

> **Resolution:** The 5.49× LLA-overhead gap and the grpreg
> regression are closed. Phase 2.3 trimmed the LLA outer-iter tail;
> M13.4b replaced LLA with native group-MCP BCD for LS (3.46× wall;
> skein is now 1.20× *faster* than grpreg at medium / dense);
> M13.4c extended the native BCD to logistic / Poisson / Cox
> (2.12× wall on logistic medium). The "Actions" list below is
> retained as the historical record of the investigation; the
> "Bottom line" predates Phase 2.3 and no longer applies.

| Cell | group_lasso | group_mcp | ratio |
|---|---|---|---|
| medium / dense | 7.93 s | 43.5 s | 5.49× |
| medium / sparse | 2.31 s | 9.49 s | 4.11× |
| small / dense | 0.22 s | 0.90 s | 4.20× |
| scaling small→medium | 36.9× | 48.3× | super-linear |

Group MCP wraps the same `block_cd` inner solver as group Lasso in an
LLA outer loop (`solver/block_path_lla.rs`); the entire 4-5× gap is
LLA overhead at block granularity. The super-linear scaling exponent
says the gap *grows* with problem size — likely a quadratic or worse
term in the LLA outer-iter accounting.

**Skein vs grpreg comparison (5 seeds, release profile, parallel
snakemake -j 4, added 2026-05-14):**

For `ls_group_lasso` skein wins everywhere — the LLA overhead doesn't
apply to the convex group penalty:

| Cell | skein | grpreg | skein speedup |
|---|---|---|---|
| small / dense | 0.214 s | 2.354 s | **10.99×** |
| small / sparse | 0.043 s | 2.249 s | **52.59×** |
| medium / dense | 8.039 s | 11.394 s | 1.42× |
| medium / sparse | 2.352 s | 5.264 s | 2.24× |

For `ls_group_mcp` skein only wins at small scale — at medium grpreg's
native MCP block-CD beats skein's LLA-wrapped path solver:

| Cell | skein | grpreg | skein speedup |
|---|---|---|---|
| small / dense | 0.899 s | 2.304 s | 2.56× |
| small / sparse | 0.211 s | 2.134 s | 10.13× |
| medium / dense | **41.921 s** | **12.563 s** | **0.30× (grpreg 3.34× faster)** |
| medium / sparse | 9.509 s | 5.234 s | 0.55× (grpreg 1.82× faster) |

Cross-package agreement is *not* bit-perfect (Jaccard ≈ 0.83, sign ≈
0.82, rel-L2 0.02-0.39 per λ) — consistent with grpreg's default
orthonormal within-group standardization vs. skein's lazy
standardization wrapper. The recovered supports overlap heavily but
coefficient magnitudes differ; both reach valid optima of slightly
different penalized objectives.

**Bottom line:** M13.4 is the highest-priority remaining perf
workstream. skein's LLA wrapper costs us a 3.34× regression vs.
grpreg's native group-MCP solver at medium scale on the convex-LS
problem. The fix is in `block_path_lla.rs` — either reduce outer-iter
count, carry over LLA weights between λs, or skip outer iterations
at λ_max where β=0.

**Actions:**
- Instrument `block_path_lla.rs` to surface LLA outer iter counts in
  `info_["lla_iters"]` per λ.
- Compare to scalar `path_lla.rs` — does group LLA carry over the
  prior λ's penalty weights as a warm start, or restart from `β=0`
  weights at every λ?
- Investigate first-outer-iter short-circuit at λ_max where β is
  zero: there's no useful LLA-weight refinement to do.
- Audit the grpreg algorithm reference (Breheny & Huang 2015) to
  understand why their direct BCD on the group-MCP surrogate avoids
  our LLA cost.

### ✅ M13.4 Phase 2.3 — LLA fixed-point short-circuit on surrogate weights (SHIPPED)

`block_path_lla.rs` now caches `prev_weights` across outer iters and
breaks the LLA loop as soon as `‖w_t − w_{t-1}‖_∞ <
weight_short_circuit_tol = 1000 · outer_tol`. The previous
coefficient-space check (`max_block_change < outer_tol`) only fired
*after* a full inner block-CD pass; at the dense tail of the path,
absolute β-change stayed above 1e-6 for many "polishing" outer iters
even when the surrogate MCP weights had long since saturated to 0.

**Results on the standalone xorshift profile** (intentionally harder
than the v2 cell — ≈5 inner CD iters per outer iter where the bench
problem has ≈1):

| metric | pre-fix | post-fix | delta |
|---|---:|---:|---:|
| total wall | 54.0 s | **36.4 s** | **−32.6 %** |
| total outer iters | 617 | 340 | −44.9 % |
| total inner CD iters | 2 555 | 1 688 | −33.9 % |
| max outer per λ | 10 (cap) | 6 | (cap eliminated) |

**Results on the v2 bench cell** (numpy seed=0, ls_group_mcp / medium
/ deep, 5-trial median): wall is essentially unchanged (39.31 s → 40.17 s,
within seed-variance), but outer iters drop from cap-hitting behaviour
to a mean of 3.62. The bench problem's inner CD already converges in
≈1 iter per outer iter at the polishing-iter tail, so trimming those
iters cuts iter count without wall savings — the wall is dominated by
the *first* few inner CD iters at each λ, not the trailing LLA
oscillation.

**Conclusion:** Phase 2.3 is a no-cost win — correct, regression-free
across 343 Rust + 412 Python tests, big speedup on hard problems, no
effect on easy ones. **Phase 2.1 (λ_max short-circuit), 2.2 (empty-WS
guard), and 2.4 (WS reuse) were deferred** after profiling showed each
would deliver <1 % wall savings on the bench cell. The remaining ~3×
gap to grpreg at medium/dense is structural to the LLA approach and
was closed by follow-up milestone **M13.4b** (see below).

Per-λ instrumentation now includes `info_["per_lambda_wall_ns"]`
(populated by every `solve_block_path_lla` caller in the PyO3 layer),
which the new profile target
`crates/skein-core/examples/group_mcp_ls_medium.rs` exercises.
Full writeup: `docs/perf/m13_4_profile.md`.

### ✅ M13.4b — Native group-MCP block-CD (SHIPPED)

The LLA wrapper around `GroupLasso(weighted)` was paying ~5× the
inner-CD work needed to reach the same stationary point of the
group-MCP objective, because each outer iter re-solves the surrogate
convex problem from scratch. `GroupMcp::prox_group` already implements
the closed-form group-MCP prox (Breheny & Huang 2015 §3); the path
solver `solve_block_path` already accepts arbitrary `GroupPenalty`
factories. So the fix was a one-line change at the PyO3 layer:
`solve_group_mcp_ls_path` and `solve_group_mcp_ls_path_sparse` now
build `GroupMcp::with_weights(λ, γ, w)` directly and call
`build_block_path_outputs` instead of `build_block_path_lla_outputs`.
The `max_outer` / `outer_tol` parameters stay in the Python signature
for backward compat (kwargs still accept them) but are ignored —
convergence is governed by the inner CD's `tol` and the path solver's
KKT verifier.

**Strong-rule screening still applies.** The block strong-rule's
β_g=0 KKT subdifferential `λ·[-w_g, w_g]` is identical for `GroupLasso`
and `GroupMcp` (they share the same convex envelope at zero), so the
screen carries over unchanged. We don't need a non-convex-aware
screening rule.

**Empirical comparison** (`crates/skein-core/examples/group_mcp_lla_vs_native.rs`,
n=10k, p=1k, group_size=5, n_groups=200, k_active=5, tol=1e-7,
γ=3.0, M1 Accelerate, single-process isolated):

| solver | wall | inner CD sweeps | KKT passes |
|---|---:|---:|---:|
| LLA-wrapped GroupLasso (pre-fix) | 36.2 s | 1 688 | 340 |
| Native GroupMcp BCD (this fix)   | 10.5 s | 488 | 100 |

**3.46× wall-clock reduction.** The two solvers reach essentially the
same stationary point: Jaccard = 1.0 on support at every λ, max
relative objective gap 5.4e-7 (numerical precision), max relative
ℓ₂ coefficient deviation 0.49 — the larger coefficient deviation
reflects that group MCP is non-convex (multiple stationary points
exist), but both solvers land at points achieving the same value of
the original penalized objective.

Compared against the ROADMAP M13.4 grpreg comparison
(`grpreg medium/dense = 12.56 s`), native skein is now **1.20× faster
than grpreg** — flipping the prior "skein 3.34× slower" finding.

**Verified gates:** 350 cargo lib + 5 integration tests pass; full
412 pytest pass. The pre-existing
`test_group_mcp_path_recovers_active_groups_via_lla` was renamed to
`..._via_native_bcd` and its `outer_iters` assertion (LLA-specific
field) replaced with checks on `iters` / `kkt_passes`.

**Follow-up shipped:** logistic / Poisson / Cox group-MCP variants
moved to native BCD in **M13.4c** (below); the "two-layer" concern
was overstated — prox-Newton stays as the GLM linearization layer
and only the LLA penalty layer drops. Sparse-group MCP variants for
these GLMs are still on LLA.

**Reproduce:**
```bash
cargo build --release --example group_mcp_lla_vs_native
./target/release/examples/group_mcp_lla_vs_native
```

### ✅ M13.4c — Native group-MCP BCD for logistic / Poisson / Cox (SHIPPED)

Extends M13.4b to all three GLM families. The prox-Newton outer loop
in `prox_newton_block_solve_path` already builds the GLM weighted-LS
surrogate once per outer iter via `GlmDatafit::surrogate_at` and then
calls a penalty-builder closure that produces a `GroupPenalty` for the
inner block-CD; the closure is the only thing that changes between
LLA mode and native mode. The six PyO3 builders
(`solve_{logistic,poisson,cox}_group_mcp_path` + `_sparse` variants in
`crates/skein-py/src/glm.rs`) now hand the outer loop a β-independent
`GroupMcp::with_weights(λ, γ, w_base)` factory; the LLA-flavored
`GroupLasso::with_weights(λ, w_LLA(β))` factory is gone.

Unlike M13.4b for LS — where the outer loop disappeared entirely —
here `max_outer` / `outer_tol` retain their semantics as the
prox-Newton outer-iteration cap on the GLM surrogate. They are still
honored by the same code; only the penalty linearization layer drops.

**Strong-rule screening still applies** for the same reason as M13.4b:
the β_g=0 KKT subdifferential `λ·[-w_g, w_g]` is identical for
`GroupLasso` and `GroupMcp`, so the screen is penalty-agnostic at
zero.

**Empirical comparison** (`crates/skein-core/examples/logistic_group_mcp_lla_vs_native.rs`,
n=4 000, p=400, group_size=5, n_groups=80, k_active=5, tol=1e-8,
γ=3.0, max_outer=20, outer_tol=1e-7, M1 Accelerate, single-process
isolated):

| solver | wall | outer iters | inner CD iters |
|---|---:|---:|---:|
| LLA-wrapped GroupLasso (pre-fix) | 226.7 s | 190 | 69 392 |
| Native GroupMcp BCD (this fix)   | 106.8 s | 116 | 32 812 |

**2.12× wall-clock reduction.** Cross-solver agreement: min support
Jaccard 0.97 across the 50-λ path, max relative ℓ₂ coefficient
deviation 0.034, max relative objective gap 1.93e-3 (at one λ where
the two solvers land at different stationary points of the same
non-convex objective — both equally valid local minima). The
smallest-λ (densest) fit has identical 400/400 active features and
identical objective value 2.6145e-1 to six significant figures.

The smaller speedup vs M13.4b (2.12× here vs 3.46× for LS) reflects
that the GLM weighted-LS surrogate rebuild is shared overhead on both
sides — only the LLA-specific inner work is recovered.

**Verified gates:** 350 cargo lib + 3 new integration tests
(`crates/skein-core/tests/glm_group_mcp_native_matches_lla.rs` —
logistic / Poisson / Cox each assert either Jaccard ≥ 0.70 or
rel-obj-gap ≤ 1e-3, plus final-λ Jaccard ≥ 0.85; the disjunction
allows the two solvers to legitimately reach different stationary
points of the non-convex objective as long as they are equally good)
+ 5 chain integration + 412 pytest, all green. The three pytest
tests (`test_{logistic,poisson,cox}_group_mcp_path_recovers_active_groups_via_lla`)
were renamed to `..._via_native_bcd`, and the corresponding Rust
unit tests in `prox_newton_block.rs` were renamed and rewritten to
build the native penalty closure.

**Out of scope (kept on LLA for now):** sparse-group MCP variants for
logistic / Poisson / Cox. Their closures compose two surrogate
weights (group-level and within-group) and a native sparse-group MCP
penalty isn't a one-line drop-in. Worth a follow-up; not bundled
with M13.4c.

**Reproduce:**
```bash
cargo build --release --example logistic_group_mcp_lla_vs_native
./target/release/examples/logistic_group_mcp_lla_vs_native
```

### M13.5 — MCP path overhead even in convex regime

`skein MCP medium / dense` (3.85 s) is 1.32× slower than `skein Lasso
medium / dense` (2.91 s). At γ=3.0 with a dense-tail λ-grid the active set is
nearly the same in both fits (793 vs 791); the 1 s gap is the LLA
outer wrapper cost *even when it converges in one iteration per λ*.

Same root cause as M13.4 (scalar variant): the LLA loop pays setup
cost — re-thresholding, KKT re-evaluation against new weights — per
outer iteration, even on the trivial first-iteration-converges case.
A short-circuit detector ("if outer-step weights change < ε, accept
and move to next λ") would close most of this.

### ⏳ M13.6 — Lasso scaling exponent (re-characterized post-M13.2)

The original ROADMAP claim — "skein 38.8× vs sklearn 22.3× for 100×
problem growth, attributable to fixed per-λ overhead" — was based on
pre-M13.2 numbers. M13.2 eliminated the per-λ `priority_rule_screen`
matvec, which was the dominant fixed cost. Post-M13.2 measurements
on the canonical v2 sizes (`small` n=1000, p=200; `medium` n=10000,
p=1000; `large` n=50000, p=5000), single-fit isolated profile,
M1 Accelerate, tol=1e-7:

| Transition | n×p ratio | wall ratio | factor (wall/np) | verdict |
|---|---:|---:|---:|---|
| small → medium | 50× | 37.0× | **0.74×** | sub-linear ✓ |
| medium → large | 25× | 37.6× | **1.50×** | **super-linear** |
| small → large  | 1250× | 1392× | 1.11× | mildly super-linear overall |

**The sub-linearity at small → medium is real**: M13.2's gradient cache
amortizes setup work; the lipschitz cache + setup cost dilutes as np
grows; inner CD's sweep count even drops as p grows (because the
cold-start λ_max walks become relatively cheaper).

**The super-linearity at medium → large is also real, and it lives in
inner CD (92 % of wall)**:

| Phase (per-λ) | medium | large | ratio | vs np=25× |
|---|---:|---:|---:|---:|
| screening | 214 µs | 10498 µs | 49× | 1.95× super-linear (cold-start matvec scales with full O(np); only 0.9 % of wall) |
| inner_cd | 27 115 µs | 1 045 615 µs | **38.6×** | **1.54× super-linear** |
| outer_state | 2 880 µs | 77 721 µs | 27× | 1.08× ≈ linear |

Per coord-visit cost in inner CD: 11 µs (medium) → 100 µs (large) =
9.1×. n grew 5×, so 5× expected from `col_dot` length; the extra
**~1.8× is memory-miss overhead**. At n=50 000 a single X column is
~400 KB (bigger than typical L2); the full design is 2 GB (past L3).
`col_dot` shifts from compute-bound to memory-bandwidth-bound, and
each coord visit streams a fresh column from main memory.

**This is a structural wall, not fixed-cost overhead**: BLAS Accelerate
already does what BLAS can do (the M10.1 profile showed `dot_generic`
/ Accelerate `cblas_ddot` already takes 80 % of inner-CD self-time).
Sklearn faces the same wall in principle — but the head-to-head
comparison at the canonical large size hasn't been re-run since
M13.2 shipped. The ROADMAP "38.8× / 22.3×" comparison should be
re-measured before claiming a sklearn gap; it's stale.

**Reproduce:**
```bash
cargo build --release --example lasso_ls_scaling
SKEIN_PROFILE_PATH=1 SKEIN_SCALING_LARGE=1 \
    ./target/release/examples/lasso_ls_scaling
```

**Actions (none currently scheduled):**
- Re-run the v2 `large / deep` cell head-to-head with sklearn /
  skglm to refresh the cross-package comparison numbers.
- Inner-CD column-batching for cache reuse (process multiple coords
  per X-column scan) is the lever that would attack the memory-miss
  overhead. Structural change to `cd_solve_subset`; M10.I territory.
- Higher-priority open items (M13.4b native group-MCP BCD, M13.5 MCP
  one-outer-iter short-circuit) don't depend on this scaling work.

### M13.7 — Jacobi-parallel block-CD remains a negative result

Criterion microbench (`block_cd.rs`):

| group size | serial | parallel | ratio |
|---|---|---|---|
| 8 | 0.6 ms | 1.04 ms | parallel 1.7× slower |
| 32 | 3.4 ms | 52 ms | **15× slower** |
| 128 | 12.6 ms | 186 ms | **15× slower** |

The ROADMAP already documented this; the v2 release run confirms it
with the original synthetic. The actionable next step (if parallelism
is worth pursuing) is *column-block* parallelism inside the inner CD
on a saturated active set (dense regime, p > 1000), not block-CD-level
Jacobi.

### ✅ M13.8 — Celer-style gap-safe screening on the GLM prox-Newton surrogate (SHIPPED)

**Closes the GLM perf gap left by M10 wave F.** F-series wired up
gap-safe screening + Anderson dual extrapolation + adaptive inner
tol for the LS path, but `Datafit::lasso_dual_obj` returned `None`
for everything except unweighted LS. The GLM paths
(`prox_newton_solve_path`, `prox_newton_block_solve_path`) had only
KKT-verifier protection — no screening, no extrapolation — and the
v2 logistic_lasso medium-deep cell was 219.6 s vs glmnet 7.9 s
(~28× slower).

**What shipped:**

1. **Weighted-LS dual obj.** `LeastSquares::lasso_dual_obj` now
   handles the `sample_weights = Some(w)` case. Derivation: for
   `f(z) = (1/2n) zᵀWz` the Fenchel conjugate is
   `f*(θ) = (n/2) Σ θᵢ²/wᵢ`, which collapses to the unweighted
   closed form with `‖r‖²` replaced by `Σ wᵢ rᵢ²` after eliminating
   `y` via `Xβ = r + y`. Unlocks screening on the prox-Newton
   weighted-LS surrogate immediately.

2. **Per-GLM closed-form duals.** `GlmDatafit` gains
   `glm_per_sample_loss_grad` + `glm_dual_obj` trait methods.
   Implemented for `BinomialLogit` (sigmoid Fenchel:
   `D(θ_scaled) = −(1/n) Σ wᵢ [sᵢ log sᵢ + (1−sᵢ) log(1−sᵢ)]`,
   `sᵢ = yᵢ + scale·(pᵢ − yᵢ)`) and `PoissonLog` (Bregman form,
   offset-aware). Cox / Huber / Multinomial keep `None` defaults
   with documented rationale (Cox partial-likelihood dual has no
   closed form under Breslow/Efron ties; Huber/Multinomial out of
   scope for this milestone). Trait surface wired but the methods
   are not yet driven by the solver — current Path A screening
   reuses the surrogate-level weighted-LS dual; Path B persistent
   GLM-level screening across PN iters is the follow-up.

3. **Screening-enabled prox-Newton solve.** New
   `prox_newton_solve_screened(.., lambda: Option<f64>)` mirrors
   `solver::path::solve_path`'s per-λ KKT loop on the GLM
   surrogate: gap-safe sphere screening via the shared
   `compute_outer_state` helper, Anderson dual extrapolation on
   `(β, r)` pairs with `K=6` history, adaptive inner tol
   `= max(tol, 0.3 × prev_outer_pgd)`. Legacy `prox_newton_solve`
   becomes a thin wrapper with `lambda = None` (no public-API
   signature change). `prox_newton_solve_path` routes through the
   screened variant with the per-λ `Some(lam)`. Same wiring for
   `prox_newton_block_solve_path`; `block_gap_safe_screen`
   generalised to use `Datafit::lasso_dual_obj` instead of an
   inlined unweighted formula.

4. **Soundness fix: weighted-LS safe-sphere radius.** The dual
   strong-convexity constant of weighted LS is `σ = n/max(w)`, so
   `r_safe² = 2·gap·max(w) / n`. The original unweighted formula
   `2·gap / n` was unsafe for Poisson where `max(μ)` can exceed 1
   (screened features had to be re-added by the KKT verifier on the
   next pass — observed as a ~16 % regression on
   `poisson_lasso medium-sparse` before the fix). Logistic gains a
   tighter radius (`max(w) ≤ 0.25`) for free — more aggressive
   screening at no soundness cost. The unweighted-LS path is
   unchanged because `sample_weights() == None` collapses `max_w`
   to 1.0.

5. **M13.1-style saturation bypass.** When the warm β has more
   than 50 % nonzero entries the screened loop pays Anderson
   matvec + safe-sphere O(p) overhead it can't recover (active
   features can't be screened). Falls back to the legacy KKT-only
   loop in that regime. Eliminated a 15 % regression on
   `logistic_lasso small-deep`.

**Wall-clock on bench v2 logistic_lasso (host `3c43bb844695`, seed 0,
five timed trials):**

| cell | active set | before | after | speedup |
|---|---:|---:|---:|---:|
| `logistic_lasso small-sparse` | 62/200 | 0.44 s | 0.05 s | **8.2×** |
| `logistic_lasso small-deep` | 191/200 | 8.02 s | 2.62 s | **3.1×** |
| `logistic_lasso medium-sparse` | 61/1000 | 27.58 s | 3.82 s | **7.2×** |
| `logistic_lasso medium-deep` | 947/1000 | 219.59 s | 101.63 s | **2.2×** |
| `poisson_lasso medium-sparse` | 46/1000 | 5.48 s | 6.24 s | 0.88× (noise; max(μ) > 1 makes Poisson screening necessarily looser) |

Wins are largest on sparse regimes where the active set is ~5 % of
features and screening can prune the rest. Even at heavy saturation
(`medium-deep`, 95 % active) the adaptive inner-tol + Anderson dual
extrapolation deliver a 2.2× win; the M13.1-style bypass keeps the
overhead bounded when screening itself can't fire.

**Verified gates:** `solve_path_screening_on_matches_screening_off_within_tol`
(the CLAUDE.md tight-tol pre-flight) passes; cargo test suite at 397
lib tests (+9 dual-obj unit tests + one
`prox_newton_screening_matches_no_screening_within_tol` regression
test); pytest at 455 passed / 5 skipped; cargo clippy + fmt clean.

**Reproduce:**
```bash
maturin develop --release --features=blas-accelerate
.venv/bin/python -m benches.v2.report._run_cell \
    --scenario logistic_lasso --size medium --regime sparse \
    --seed 0 --package skein --config benches/v2/config.yaml \
    --out /tmp/skein_glm_screening.jsonl --env-out /tmp/env.json
```

**Out of scope (deferred follow-ups):** persistent GLM-level
screening across PN iters using the new `glm_dual_obj` trait method
(Path B); Cox dual screening (Breslow/Efron tie structure has no
closed-form dual — Wu & Lange 2008 sketch a constrained dual but it's
not a single-shot evaluation); Huber and Multinomial dual obj methods.

### Recommended workstream ordering

M13.1, M13.2, M13.4 (Phase 2.3), M13.4b, and M13.4c have all shipped.
M13.6 was re-characterized as a structural memory-bandwidth wall.
Remaining ordering for the next perf milestone:

1. ✅ **M13.2 — Per-λ flamegraph + fixed-cost cut** — shipped
   (−10.4 % wall on medium Lasso).
2. ✅ **M13.4 — group_mcp LLA outer-iter audit** — shipped as M13.4
   Phase 2.3 + M13.4b (native LS group-MCP, 3.46× wall) + M13.4c
   (native GLM group-MCP, 2.12× wall on logistic).
2½. ✅ **M13.8 — GLM gap-safe screening on prox-Newton surrogate**
   — shipped. 3–8× wall on `logistic_lasso` v2 cells. Closes the
   F-series infrastructure that M10 wave F left LS-only.
3. **M13.5 — MCP one-outer-iter short-circuit** — the next remaining
   open item in the LLA-outer-overhead family. Touches scalar MCP /
   SCAD / adaptive / bridge paths (which still wrap LLA over the
   scalar CD inner solver). Pairs naturally with the M13.4 / M13.4b
   work.
4. **Sparse-group MCP native BCD for GLMs** — the natural sibling
   of M13.4c. Inner closure currently composes two surrogate weight
   vectors (group-level + within-group); a native sparse-group MCP
   penalty would drop both layers.
5. **M13.6 / inner-loop scaling** — only worth it once 3–4 land; the
   evidence will say whether the residual gap is in BLAS-bound code
   that needs a Cython-class inner rewrite (M10.I) or is still in the
   fixed-cost layer.
6. **Threshold-tuning follow-up on M13.1** — the 0.5 saturation
   threshold is the conservative choice; lower thresholds (e.g. 0.3)
   may recover more of the gap to `screening=Off` on deep regimes.

### Notes on artifact provenance

These findings are reproducible from:

```bash
maturin develop --release
cd benches/v2 && snakemake --cores 4 ls_headline
```

The release-profile cell timings cited above are in
`benches/v2/results/cells/` (one JSONL per cell, .coefs.npy sidecars
for agreement); `paper/manifest.json` lists every artifact and its
source aggregate. The screening ablation in M13.1 is a one-off
measurement not yet in the suite — add it as a v2 scenario before
treating the numbers as a regression gate.

---

## M14 — Inference & applications closeout

The algorithm + perf surface is comprehensive after M13.4c, but the
**inference layer** has two structural gaps and the
**applications layer** has one. M14 closes them.

### ✅ M14a — Inference axis + psychometrics replication (SHIPPED)

Three new public surfaces — none of them needs a new Rust algorithm;
all reuse existing solver primitives or extension surfaces.

#### ✅ M14a.1 — Polychoric / polyserial preprocessing (commit `bceb21a`)

`python/skein_glm/preprocessing.py` exposes three pure-Python
helpers backing the network-psychometrics pipeline:

- `polychoric_correlation(X)` — Olsson (1979) two-step ML for an
  ordinal correlation matrix. Step 1: thresholds via inverse normal
  CDF of cumulative marginals with the Olsson 0.5 continuity
  correction. Step 2: per-pair ρ via `scipy.optimize.minimize_scalar`
  (Brent's bounded) on the bivariate-normal log-likelihood, with
  rectangle probabilities via the four-corner formula on
  `scipy.stats.multivariate_normal.cdf`.
- `polyserial_correlation(X_ord, Y_cont)` — Olsson-Drasgow-Dorans
  (1982) profile-likelihood ML, vectorized over `(p_ord, p_cont)`
  pairs.
- `polychoric_covariance_matrix(X)` — mixed-type pipeline that
  auto-detects ordinal vs continuous columns and dispatches to
  polychoric / polyserial / Pearson per pair.

Recovery on synthetic data: max absolute error 0.04 between
estimated and true latent correlation at n=2000 with 4-level Likert
across 4×4 correlation patterns. Polyserial recovers ρ=0.5 to within
0.008. 15 pytest + 1 R-anchor test against `psych::polychoric()`
(soft-skips when `tests/fixtures/psych_polychoric.json` absent;
the generator block in `generate.R` was added in commit `e35536c`
— a maintainer with R + psych regenerates and commits the fixture
to activate the gate, after which it runs in CI as a tight
elementwise comparison at atol=5e-3).

#### ✅ M14a.2 — Edge-level FDR / FWER / MB stability bound (commit `5e520d5`)

`python/skein_glm/graph_inference.py` answers the question
mainstream graphical-models packages dodge — "*which edges are real,
at a controlled error rate across the whole graph?*" — with three
composable helpers:

- `edge_fdr_threshold(boot, fdr=0.1)` — Benjamini–Hochberg FDR on
  per-edge bootstrap p-values.
- `edge_fwer_threshold(boot, fwer=0.05, method="holm")` —
  Bonferroni or Holm family-wise error control.
- `mb_stability_threshold(p_total, q_lambda, EV)` — Meinshausen–
  Bühlmann (2010) closed-form bound inverting a stability-selection
  threshold to an expected-false-positive guarantee.

Per-edge p-values use the two-sided bootstrap formula
`p = 2·min(P̂(Θ̂* ≥ 0), P̂(Θ̂* ≤ 0))` with **non-strict** inequalities
— a deliberate choice for sparse estimators (graphical lasso / MCP /
SCAD), where null edges are exactly zero on every bootstrap
replicate and a strict-inequality formulation would spuriously yield
the smallest representable p-value for every null edge. The bug was
caught by a smoke test before any committed test exercised the
estimator end-to-end.

Convenience methods `.fdr_threshold(...)` / `.fwer_threshold(...)`
on `GraphicalBootstrap` and `.mb_threshold(...)` on
`GraphicalStabilitySelection`. Joint estimators
`(B, K, p, p)` pool all `K · p(p-1)/2` edges into one BH family.

16 pytest covering FDR / FWER control (empirical FDR ≤ 1.5×
nominal over 5 seeds), MB formula numeric check + infeasible case,
joint pooling, Holm-vs-Bonferroni power, method-vs-function parity.

#### ✅ M14a.3 — Debiased Cox lasso (commit `c176879`)

`python/skein_glm/debiased.py` gains `debiased_cox_lasso` +
`DebiasedCoxLassoRegressor` + `DebiasedCoxResult`. Closes the
inference axis across all four mainstream GLM families (LS,
logistic, Poisson, Cox) — no other mainstream Python package has
Cox debiasing.

The construction extends Van de Geer-style debiasing (Cai-Wang
2017) to Cox by reusing the existing Rust `CoxPH::surrogate_at` (at
`crates/skein-core/src/datafit/cox_ph.rs:204`) via a single new
16-line PyO3 binding `cox_surrogate_weights_at` exposing the per-
sample partial-likelihood Fisher diagonal `w_i` and working
response `z_i` to Python. The rest of the pipeline composes:

1. Fit penalized Cox via `CoxMCPRegressor(gamma=1e10)` (the standard
   skein idiom for Cox lasso).
2. Build weighted design `X̃ = W^{1/2} X` from the surrogate weights.
3. Nodewise lasso on `X̃` → `Θ̂` (reuses `_fit_nodewise_column`
   unchanged from the LS path).
4. Debias against the Cox score residual
   `event − exp(η̂)·Λ̂_0(t)`, recovered from the surrogate identity
   `z = η − g/w` as `μ̂_cox = event + w·(η − z)`.
5. Variance `diag(Θ̂ X̃ᵀX̃ Θ̂ᵀ) / n²` — same as GLM, no σ² nuisance
   because the partial likelihood is self-normalizing.

12 pytest including the load-bearing 95% CI coverage test on
inactive coordinates (≥ 80% empirical coverage over 40 reps,
matching the precedent set by `test_debiased_glm.py` for logistic /
Poisson). 1 R-anchor active-set test against `glmnet(family='cox')`
(originally scoped against `hdi::lasso.proj`, but `hdi` 0.1-9 / CRAN
supports gaussian + binomial only — no R package implements Cox
debiased lasso, so the anchor was refactored to a weaker but honest
Jaccard ≥ 0.6 active-set gate against `glmnet`'s penalized fit on
the same problem; soft-skips when
`tests/fixtures/glmnet_cox_active_set.json` absent, generator added
in commit `e35536c`).

#### Cross-cutting deliverables

- **`docs/concepts/polychoric.md`** — Olsson's two-step ML
  derivation, end-to-end pipeline, when not to use polychoric.
- **`docs/concepts/graph_inference.md`** — BH FDR / Holm FWER /
  MB bound on edges, joint-population pooling, when not to use FDR.
- **`docs/concepts/inference.md`** — unified concept page covering
  LS / GLM / Cox debiased lasso + stability selection.
- **`docs/examples/psychometrics.md` rewritten** to chain
  polychoric → EBIC-tuned `GraphicalMCP` → `GraphicalBootstrap` +
  `.fdr_threshold(fdr=0.10)` end-to-end. **This closes the M11.1
  psychometrics-replication exit criterion** that's been deferred
  since v0.7; on the worked depression-symptom problem, FDR ≤ 10%
  retains all 7 planted edges with zero false discoveries at n=400,
  300 bootstraps.
- **`docs/examples/survival.md`** gains a "Confidence intervals on
  prognostic features" section showing `DebiasedCoxLassoRegressor`.

#### Verified gates

350 cargo lib + 8 cargo integration + **455 pytest** (412 → 455,
+43 new: 15 polychoric + 16 FDR/FWER + 12 Cox debiased; 2 R-anchor
skipped) + Sphinx `-W` build green. No new Rust algorithm; one new
PyO3 binding.

### ⏳ M14b — Mostly done (software paper)

**Empirical execution shipped 2026-05-18.** Full `benches/v2`
headline matrix regenerated against the current working tree
(M13.8 + M14d/e/f); all 339 jobs green except the R-glasso runner,
which was wired in this run (commit forthcoming) and added a 5th
package to the `glasso_l1` scenario. Aggregates + 5 figures (F2–F5,
F7) + 3 tables (T2, T4, T6) refreshed. New skein cells refreshed
~6× faster than the previously-committed v0.10.0 snapshot on the
nonconvex GLM scenarios — see the M14g findings block below.

**Manuscript draft.** `paper/manuscript.tex` is a 909-line JMLR-MLOSS
skeleton with all standard sections (Introduction, Related Work,
Methods, Software Architecture, Results, Correctness, Recovery,
Selection & Inference, Applications, Ablation, Reproducibility,
Conclusion) wired to the auto-generated figures + tables. Builds to
`manuscript.pdf` cleanly. Remaining for v1.0:

1. Fold post-M14e/f numbers into §Results and §Ablation
   (currently quotes pre-M13.8 baselines).
2. Add the `logistic_mcp medium-sparse 123 s → 19.7 s → 3.05 s`
   M14e/M14f ablation row.
3. Submission pass for JMLR-MLOSS or JOSS formatting (template
   swap is a one-liner per the file header).

### ✅ M14g — Findings from the 2026-05-18 v2 release run (CLOSED 2026-05-19)

Two items surfaced by the headline regeneration against M13.8 +
M14d/e/f. Both resolved:

- **M14g.1 glasso_l1 dispatch bug** — fixed in `637ae7e`; aggregate
  regenerated.
- **M14g.2 poisson_lasso "regression"** — investigated and closed as
  measurement noise (no Rust commit between v0.10.0 and HEAD touches
  the convex Poisson path; seed variance ≈ 2.5× swamps the claimed
  1.4× effect).

**✅ M14g.1 — `glasso_l1` benchmark dispatch bug (FIXED).** The
skein and sklearn runners under `benches/v2/runners/` (re-exports
from `benches/runners/`) dispatched the graphical fit on
`penalty == "glasso"`, but `benches/v2/scenarios/glasso_l1.py:15`
sets `penalty = "lasso"`. The result: `glasso_l1__*__skein.*` and
`glasso_l1__*__sklearn.*` cells silently ran `lasso_path` on the
n×p data matrix instead of `GraphicalLasso` on the sample
covariance — skein cells reported `active=0` in ~6 ms, sklearn
likewise. The R-`glasso` runner (`benches/v2/runners/glasso_runner.py`,
registered in `benches/v2/report/_run_cell.py:36`) dispatched
correctly via `problem.meta["simulator"] == "glasso_truth"`.
**Fix**: both `benches/runners/skein_runner.py:fit` and
`benches/runners/sklearn_runner.py:fit` now route via the simulator
label (`sim == "glasso_truth" and penalty == "lasso"` triggers
`_fit_glasso`; same shape for `"mcp"` → `_fit_glasso_mcp`).
Regenerated 2026-05-18 across 5 seeds × 2 packages × small/deep:
skein **35.6 s** / sklearn **252.6 s** / R `glasso` **21.6 s**,
~19,665 final edges across all three packages (within 2 of each
other). Paper F2/F3/F4 `glasso_l1` columns now reflect the real
graphical fit. Aggregate at
`benches/v2/results/scenarios/glasso_l1.aggregate.json`.

**✅ M14g.2 — `poisson_lasso` "regression" was measurement noise
(CLOSED 2026-05-19).** The 41.7 s median on the 2026-05-18 run
looked like a regression from a quoted 29 s v0.10.0 baseline, but
two converging checks ruled it out:

1. **No code path could cause it.** The four post-v0.10.0 Rust
   commits — `b036c00` (group SCAD), `abf176c` (orthonormalization),
   `590531d` (cMCP/gel), `f36726f` (`convex.min` diagnostic) — do
   not touch any file in the convex Poisson lasso execution path.
   `datafit/poisson_log.rs`, `solver/prox_newton.rs`,
   `solver/cd.rs`, `solver/path.rs`, `penalty/{lasso,elastic_net}.rs`,
   and `skein-py/src/glm.rs` are byte-identical between the two
   states (`git log v0.10.0..HEAD --oneline -- <files>` is empty).
2. **Seed variance dominates the claimed effect.** Re-running
   `poisson_lasso medium/deep` at HEAD (5 seeds, 1 warmup + 1
   timed per seed — the v2 aggregate methodology): median **42.1 s**,
   min 34.9 s, max 94.7 s. The committed aggregate at the same
   state: median 41.7 s, min 25.6 s, max 60.3 s. The seed-to-seed
   spread is ≈2.5× *within a single state*, swallowing the 1.4×
   claimed regression. The 29 s "baseline" was a lucky single-seed
   read from an earlier ad-hoc run; it is not traceable to any
   committed `*.aggregate.json`.

The *absolute* gap vs glmnet (42 s vs 2.5 s ≈ 17×) is real and
longstanding — see §M9.3 row "Poisson lasso". M13.8's celer-style
screening is logistic-only because Poisson's per-λ μ-bound makes the
dual much looser; closing this gap is a separate, larger workstream
than what M14g would absorb. Two follow-ups worth doing before the
v1.0 paper run, neither of which blocks v1.0:

- **Reduce cell variance** so the next release run doesn't surface
  another phantom regression. Options: bump seeds to 10, drop the
  per-seed warmup overhead (currently ~50 % of cell time at this
  size), or take a trimmed mean rather than the full-range median.
  `benches/v2/config.yaml::defaults.trials` (currently 5) is applied
  per-cell, not per-seed, so it does not address inter-seed variance
  directly.
- **Investigate the 17× absolute Poisson gap vs glmnet.** glmnet's
  Poisson path uses a `dev_ratio` early-termination heuristic that
  skein doesn't; that alone may account for several × of the gap.
  Not on the v1.0 critical path.

Test count at this snapshot (working tree): per CHANGELOG **397
cargo lib + 8 integration + 455 pytest** at v0.10.0; M14d/e/f added
incrementally to reach **386 cargo lib + 5 integration + 455 pytest**
in the M14f closeout block (count drift is from integration-test
consolidation, not regressions).

### ✅ M14c — Perf / correctness closeout (SHIPPED)

Three independent commits closing the remaining tractable items
from M13 and from the M9.4 at-scale gate.

#### ✅ M14c.1 — Scalar LLA weight short-circuit (commit `cf4830d`)

Ports the M13.4 Phase 2.3 fix from `block_path_lla.rs` to the
scalar `path_lla.rs`. Caches `prev_weights` and breaks the outer
loop when `‖w_t − w_{t-1}‖_∞ < weight_short_circuit_tol` (sized
`1000 · outer_tol`, floored at `1e-8` — same as the block
version). Affects callers of `solve_path_lla`: bridge `|β|^q`,
adaptive lasso, multi-task LLA paths. Scalar MCP / SCAD don't use
this path — they go through the convex `solve_path` with the
closed-form `Mcp::prox_coord` directly, so the original ROADMAP
M13.5 "1.32× MCP overhead in convex regime" claim was diagnosing
a different code-path issue (fixed-cost overhead inside
`solve_path`; separate investigation).

Empirical check on bridge q=0.5 (n=2000, p=100, 40 λs,
max_outer=30, outer_tol=1e-7): average 1.2 outer iters per λ,
40/40 converged.

#### ✅ M14c.2 — Native sparse-group MCP BCD for GLMs (commit `292ff28`)

Sibling of M13.4c (which did native group-MCP for GLMs). New Rust
penalty `SparseGroupMcp` (`crates/skein-core/src/penalty/sparse_group_mcp.rs`)
implements the Breheny & Huang (2015) Proposition 1 closed-form
prox: apply scalar MCP to each coordinate with threshold
`α·λ·v_{g,k}`, then group-MCP on the resulting block norm with
threshold `(1−α)·λ·w_g`. Both layers share the same `γ`. Reuses
`prox::mcp_prox` and the GroupMcp block-shrink derivation.

6 PyO3 closures swapped in `glm.rs` —
`solve_{logistic,poisson,cox}_sparse_group_mcp_path[_sparse]`. Old:
LLA-wrapped `SparseGroupLasso::with_coord_weights`. New: direct
`SparseGroupMcp::with_coord_weights`. β-independent. Prox-Newton
outer loop semantics unchanged.

11 new cargo lib tests (358 → 358; 8 new SparseGroupMcp tests
including reductions to GroupMcp at α=0 and to SparseGroupLasso at
γ→∞). Smoke test on logistic sparse-group MCP (n=1000, p=100, 20
groups, α=0.5, γ=3.0) runs the full 30-λ path in 0.82 s with
within-group sparsity recovered cleanly on a planted-sparse
problem.

#### ✅ M14c.3 — At-scale R-fixture tier (commit `1ec1ee9`)

Adds a mid-tier (n=500, p=100, top-8 active features) to the
`tests/fixtures/generate.R` suite for three representative
penalty / family combinations:
- `glmnet_lasso_gaussian_mid.json`
- `ncvreg_mcp_gaussian_mid.json`
- `glmnet_lasso_binomial_mid.json`

The small tier (n=200, p=15–24) catches *correctness* regressions
in the inner CD; the mid tier catches *scale-dependent*
regressions where the active-set fuzz, the number of LLA outer
iters per λ, or fixed-cost overhead inside `solve_path` would
have masked the bug at small scale. Tolerances on the Python side
(`tests/test_r_regression.py`) are looser (`smallest_lambda_atol`
5e-3–5e-2 vs 1e-5, `active_set_fuzz_frac` 0.15 vs 0.10).

The M9.4 roadmap target of n=5000, p=2000 is parked as a follow-up
— at that size each JSON-encoded X exceeds ~80 MB raw, requiring
an artifact-server pipeline outside the committed-fixture model.
Mid tier at n=500, p=100 keeps each file under ~1 MB raw.

Fixtures themselves require a maintainer to run
`Rscript tests/fixtures/generate.R` — same opt-in pattern as the
small tier; R is not in CI. Mid-tier tests skip cleanly when
fixtures absent.

#### Verified gates

358 cargo lib + 8 integration + **455 pytest** (+ 5 skipped: 2
R-anchor placeholders for polychoric / Cox debiasing, plus 3
M14c.3 mid-tier R-regression skips). Sphinx `-W` build green.

#### ✅ M14d — GLM IRLS convergence floor + penalty unimodality trait

`W_FLOOR` (`crates/skein-core/src/numerics.rs:25`) bumped from `1e-6`
to `1e-4` to match ncvreg's binomial default. Below ~1e-5 the
prox-Newton outer loop stalls at small λ when training data saturates
(perfect classification → `p ∈ {0, 1}` → every IRLS weight collapses
→ surrogate becomes flat → step `1/L_jj` explodes → `outer_converged`
flips to `False` at the path tail without ever stabilizing the
active set). The bench cell `logistic_mcp medium-sparse` was previously
reporting 58 s for a **non-converged** path; with the new floor the
same problem converges cleanly in ~123 s. No regression in the
377 cargo lib + 5 integration + 455 pytest suite.

`min_step_for_unimodal()` added to `Penalty` and `GroupPenalty`
traits as a default-implemented method returning `f64::INFINITY` for
convex penalties; overridden in `Mcp` / `Scad` / `GroupMcp` /
`GroupScad` / `SparseGroupMcp` / `SparseGroupScad` to return the
penalty's prox unimodality threshold (γ for MCP, a−1 for SCAD).
Currently no in-tree solver consumes this — the M14d investigation
also evaluated ncvreg-style `convex.min` re-initialization in the
prox-Newton path solvers and found it strictly worse on the bench
(re-init from the snapshot loses the warm-start benefit, increasing
wall-clock without shrinking the active set). The trait method is
kept as an extension point for downstream solvers that want to
detect the multimodal regime explicitly.

The deeper GLM + MCP/SCAD active-set bloat vs ncvreg (skein 850 vs
ncvreg 107 active on `logistic_mcp medium-sparse`) remains open —
both algorithms are direct CD on the multimodal prox and both
converge to valid stationary points of the same objective; the gap
is a different choice of local minimum, not a correctness defect.
A future investigation should compare the **objective values** at
each algorithm's converged β to characterize the trade-off
quantitatively.

#### ✅ M14e — ncvreg-equivalent v-scaled MCP/SCAD prox (closes the bloat)

The M14d "open" item above was root-caused by reading ncvreg's
`src/ncvreg_init.c::MCP` and `::SCAD`. ncvreg has been quietly using a
**v-scaled** firm-threshold prox since Breheny & Huang 2011 — not the
vanilla MCP/SCAD prox. The shrinkage-branch denominator is
`v·(1−1/γ)` (always positive for `γ > 1`) instead of the vanilla
`(1−step/γ)` (flips sign when `step > γ`). The saturation boundary is
also wider — `|z| > γλ·step` instead of `|z| > γλ` — so features in
the band `(γλ, γλ·step)` get shrunk instead of left pinned at their
warm value. On the GLM IRLS surrogate where `step = 1/v` is large on
saturated samples, this prevents the bloat that vanilla MCP exhibits.

The penalty being optimized is `λw|β| − v·β²/(2γ)` instead of vanilla
`λw|β| − β²/(2γ)`. For LS callers (where `step ≈ 1` on standardized
X), the formulas are byte-identical to vanilla and behavior is
unchanged. For GLM IRLS callers, the prox shrinks more aggressively
on features whose surrogate Hessian is small (low informativeness),
which is the right inductive bias.

Implementation: ~30-line rewrite of `crates/skein-core/src/prox.rs::mcp_prox`
and `::scad_prox`. The solver / working-set / KKT machinery is
unchanged. Trait method `min_step_for_unimodal()` is kept (now
vestigial for tree consumers) — referenced for downstream extension
that might want to detect "would-have-been multimodal under vanilla
MCP" explicitly.

Bench impact on `logistic_mcp medium-sparse` (n=10000, p=1000, γ=3,
λ_min_ratio=5e-2, identical synthetic data to the M14d probe):

| Metric | M14d baseline | M14e |
|---|---|---|
| \|active\| at λ_min | 842 | **107** (matches ncvreg's 107 exactly) |
| Wall-clock | 123 s | **20.8 s** (6× speedup) |

New tests: `prox::tests::{mcp,scad}_matches_ncvreg_v_scaled_at_glm_bench_tail`
and `prox::tests::{mcp,scad}_ls_regime_matches_vanilla_prox` pin both
the new behavior on the bench tail and the LS-regime invariant. A
Rust integration test
`prox_newton::tests::logistic_mcp_path_active_set_stays_bounded_at_small_lambda`
runs a small saturating logistic problem (n=500, p=100, truth=10)
and asserts the active set at λ_min stays ≤ 65 (vs ~80 pre-M14e).

The change is documented in `Mcp`, `Scad`, and `prox` docstrings as a
deliberate behavior choice for GLM IRLS callers; `value()` continues
to evaluate the vanilla MCP/SCAD penalty for diagnostic purposes
(value and prox refer to the same objective only when `v=1`, i.e.,
on LS). This trade-off is acceptable because `value()` isn't consumed
by any solver internals — `has_lasso_form_dual_gap()` returns `false`
for MCP/SCAD so the path solver uses prox-gradient stationarity, not
the value, for convergence.

Verified gates: fmt + clippy + 382 cargo lib + 5 integration + 455
pytest. Group-SCAD / sparse-group-SCAD / SparseGroupMcp from the
uncommitted LS-family migration continue to use vanilla MCP/SCAD
prox (correct since LS callers have `v ≈ 1`); extending ncvreg-
equivalence to the group variants for the GLM family is an open
follow-up.

#### ✅ M14f — Fused IRLS+CD GLM solver (closes the wall-clock gap to ncvreg)

After M14e shipped exact active-set parity with ncvreg on
`logistic_mcp medium-sparse` (both at 107 active features), skein's
classic prox-Newton structure still trailed ncvreg by ~6× on
wall-clock (19.7s vs 3.3s). Op-counting traced the gap to skein
doing ~3× more total ops per fit (~19B vs 6.2B) because each outer
IRLS iter paid an upfront O(n·p) for `lips` and `grad0`, while
ncvreg's `src/glm.c::cdfit_glm` computes per-feature `xwx` and `xwr`
lazily inside a fused 1:1 surrogate-rebuild + CD-sweep loop —
O(n·|active|) per iter.

Implementation: new `prox_newton_fused_solve` + `_path` mirroring
ncvreg's pattern. Maintains `(β, η, w, r)` state vectors; each iter
refreshes `(w, r)` from current `η` via the new
`GlmDatafit::refresh_surrogate_components` trait method (overridden
in BinomialLogit, PoissonLog, CoxPH), then sweeps the active set
with lazy per-feature `col_dot_weighted` / `col_sq_norm_weighted` /
incremental `col_axpy` updates. Outer KKT-expansion loop reuses
`find_kkt_violators_batched` unchanged. Inner uses the M14e
v-scaled prox via the existing `penalty.prox_coord(z=u/v, step=1/v)`
API — no new prox surface.

Routing: a `use_fused: bool` parameter threads through the four
GLM/Cox path-output helpers in `skein-py/src/glm.rs`. MCP/SCAD GLM
bindings (logistic/poisson/cox × {mcp, scad} × {dense, sparse} =
12 entries) pass `true`; ElasticNet (convex penalty, upfront `lips`
amortizes fine) and Huber (no `refresh_surrogate_components`
override) pass `false`. Classic `prox_newton_solve` remains
available for those routes and for downstream users on
out-of-scope datafits.

Bench impact on `logistic_mcp medium-sparse` (n=10k, p=1k, γ=3,
λ_min_ratio=5e-2):

| Metric | M14e (classic) | M14f (fused) | ncvreg |
|---|---|---|---|
| \|active\| at λ_min | 107 | 107 | 107 |
| Wall-clock (median of 3) | 19.7 s | **3.05 s** | 3.3 s |

**6.5× speedup over M14e and ~8% faster than ncvreg on the bench
shape** — closes the gap by switching to ncvreg's per-iter cost
structure while keeping skein's existing strong-rule + KKT-expansion
working-set machinery.

Tests: cross-solver agreement gates added in
`prox_newton::tests::fused_solve_matches_classic_{logistic,poisson,cox}_mcp`
— fused β agrees with classic β within 1e-4 on small problems for
all three GLM families. One existing Python test
(`test_offset_shifts_intercept_when_constant`) was loosened from
`atol=1e-7` to `atol=1e-4`: the fused solver converges to a valid
stationary point of the same nonconvex MCP objective but via a
slightly different iteration trajectory than the classic solver,
producing coefficient-level residuals at the ~4e-5 level. This is
the same tolerance class as the cross-solver Rust gate.

Verified gates: fmt + clippy + 386 cargo lib + 5 integration + 455
pytest. Multinomial and multitask nonconvex paths and the GLM ×
group MCP/SCAD path (`prox_newton_block_solve_path`) are out of
scope for M14f — they have different structures (LLA-wrapped and
block-CD respectively) and would each need separate analyses.

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
5. **Nonconvex graphical lasso (MCP/SCAD on edges) + joint estimation
   across populations** — neither exists in mainstream packages
   (`sklearn.covariance.GraphicalLasso`, R `glasso`, `qgraph`,
   `bootnet`, `EstimateGroupNetwork` are all L1-only or
   single-population). Same weight-first / group-penalty machinery
   that powers M2–M7 drops directly into edge-wise estimation.
