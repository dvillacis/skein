# Changelog

All notable changes to `skein-glm` are recorded here. The project
follows semantic versioning. The stable Rust API surface is frozen as
of v1.0.0; see `docs/extending/rust-api.md` for the contract.

## [Unreleased]

Post-v1.0 hardening & operability work. The public API surface is
unchanged; everything below is additive (new tests, new CI gates,
new wall-clock instrumentation).

### Hardening

- **H4 — Reproducibility audit.** New `tests/test_reproducibility.py`
  pins every public RNG-consuming estimator with paired same-seed +
  different-seed fits (8 tests). Same seed asserts `np.array_equal`
  bit-identity on `coef_` / `cv_scores_` / `selection_probabilities_`
  / `edge_selection_probabilities_` / CI bounds; different seed
  asserts measurable divergence so a silent dropped-`random_state`
  regression fails immediately. Coverage: `MCPPathCV`,
  `GroupLassoPathCV`, `LogisticLassoPathCV` (representatives of the
  `_PathCVMixin` KFold-shuffle path); `StabilitySelection`;
  `GraphicalStabilitySelection`; `GraphicalBootstrap`;
  `AdaptiveLassoPathCV`; `MultinomialLassoPathCV`. The Rust path
  solver itself is deterministic (no RNG in coordinate descent), so
  the assertion lives entirely in Python-side fold / bootstrap
  construction — BLAS-thread-independent at the problem sizes used.

- **H3 — Property-based & fuzz tests on prox / surrogate.** 30 new
  Rust `proptest`s in `crates/skein-core` (dev-only dep) + 4 Python
  `hypothesis` tests close the randomized-coverage gap M12's C1/C2
  unit tests left open:
  - `src/prox.rs` — sign / antisymmetry / monotonicity /
    magnitude-non-increase on every scalar prox, EN ↔ soft-threshold
    at α=1, MCP-as-γ→∞ and SCAD-as-a→∞ collapse to soft-threshold,
    2D rotation invariance of the group lasso / EN block prox.
  - `src/datafit/surrogate_proptests.rs` — BinomialLogit / PoissonLog
    / CoxPH (Breslow + Efron) surrogate gradient identity vs.
    central-FD of `loss(β)`; analytical Fisher Hessian diagonal
    match for the canonical-link GLMs.
  - `src/standardize.rs` — full `destandardize(standardize_β(β)) = β`
    bijection across every `(center_x, scale_x, fit_intercept)` flag
    combo, plus `destandardize_path` row-consistency and
    `rescale_weights_for_standardize` penalty preservation.
  - `tests/test_weight_composition.py` — `weights = None ↔ ones`
    bit-equality across the PyO3 boundary for MCPPathRegressor and
    GroupLassoPathRegressor; per-feature permutation equivariance;
    positive non-uniform `sample_weights` no-op detector. Documents
    one architectural quirk surfaced by the exercise: `sample_weights
    = None` and `sample_weights = ones(n)` take structurally
    different paths in `crates/skein-py/src/ls.rs` and so are
    *approximately* — not bit-exactly — equivalent.

- **H2 — Numerical-stability sweep.** Four new pytest files (84
  tests) covering pathological inputs that earlier well-conditioned
  synthetics didn't reach:
  - `tests/test_numerics_design_pathologies.py` — collinear columns
    at ε ∈ {0, 1e-12, 1e-8} and constant columns across LS / group /
    logistic / Poisson penalties, with and without `standardize`,
    plus an explicit `rescale_weights_for_standardize` audit on a
    zero-variance column.
  - `tests/test_numerics_extreme_weights.py` — `sample_weights`
    spanning 12 decades, zero per-feature weights (asserting the
    unpenalized feature stays nonzero across the path), zero
    per-group weights for group lasso / MCP / sparse-group lasso.
  - `tests/test_numerics_glm_saturation.py` — Poisson η driven
    against `ETA_CLAMP`, linearly separable & extreme-imbalance
    logistic, and Cox under heavy-ties / all-events-at-one-time
    with both Breslow and Efron.
  - `tests/test_numerics_glasso_singular.py` — rank-deficient
    single / joint glasso, `diag_offset=0`, duplicated and constant
    variables, precomputed singular covariance.

  Every test enforces a 30 s wall-clock budget to catch the
  infinite-loop fallback that `gap < tol²` once produced. Surfaced
  one algorithmic finding: nonconvex glasso (MCP / SCAD) does not
  preserve SPD across iterations at extreme `n ≪ p` (the L1 SPD
  invariant relies on the FHT-2008 block update being positive on
  every coordinate, which the MCP / SCAD release region can flip).
  Documented in the H2 test file; the H2 contract is finiteness +
  symmetry, and that still holds.

  Test count moves from 506 → **593 pytest** (`pytest tests/`, 9
  skipped, all unrelated).

### Hardening / Performance / Operability (status carry-over)

The following milestones shipped between v1.0.0 and this release and
are folded into the same Unreleased window — see `ROADMAP.md` for
the per-milestone evidence and discussion. They are noted here for
release-notes completeness:

- **H1** — at-scale bench infrastructure (`xlarge` n=100k×p=10k in
  the headline matrix, per-PR `large` canary, `*_large` R-anchor
  fixtures, `docs/benchmarks/at_scale.md`).
- **O1** — `cargo-semver-checks` in CI against the `v1.0.0`
  baseline.
- **O2** — `cargo-audit` + `pip-audit` + dependabot for
  supply-chain hygiene.
- **O3** — Python 3.13 + NumPy 2.x in the CI matrix.
- **O4** — expanded wheel matrix. `wheels.yml` now builds manylinux +
  musllinux on both x86_64 and aarch64 (4 Linux wheels, up from 1) in
  addition to macOS arm64 + Windows AMD64. Linux aarch64 runs under
  QEMU via `docker/setup-qemu-action`; Alpine `apk add openblas-dev
  pkgconf` added to `CIBW_BEFORE_ALL_LINUX` so the `blas-openblas`
  feature wires across all four Linux containers. Tests stay skipped
  on `*-linux_aarch64` (emulation) and `*-musllinux_*` (fragile
  install path).
- **O6** — per-λ wall-clock surfaced on every path estimator via
  `info_["times_ns"]`, powered by additive `solve_*_path_timed`
  siblings that keep the v1.0 freeze intact.

## [1.0.0] — 2026-05-19

The v1.0 release. Closes the **stable Rust API audit** (M8.5) — the
crate's public surface freezes per semver from this release onward.
Also folds in the post-v0.10.0 M14 work: native group SCAD,
graphical-bench dispatch fix, marginal-FDR selection, per-block
orthonormalization, two new bilevel penalties, a `convex.min`
diagnostic for nonconvex paths, and the v2 benchmark expansion that
adds direct-comparator coverage for the full GLM × {lasso, MCP}
matrix. The manuscript supporting `benches/v2`
(`paper/manuscript.tex`) ships against the JMLR class file with
post-M14e/f numbers folded into §Results and §Ablation.

Test count: **418 cargo lib + 506 pytest, all green** (up from 397 +
455 at 0.10.0).

### M8.5 — Stable Rust API freeze

The audit walked the ~170-item `skein-core` public surface and drew
the v1.0 freeze line. From v1.0 onward, items listed in
`docs/extending/rust-api.md` follow semver — minor releases add only,
breaking changes wait for a major bump and ship with at least one
minor release of deprecation warnings.

Demoted to `pub(crate)` (no skein-py consumer, no external bug report
on the v0.x window):

- `solver::cd::{cd_solve_warm, cd_solve_warm_with_residual,
  cd_solve_subset, cd_solve_subset_weighted_ls,
  cd_solve_subset_weighted_ls_with_lips}` — internal CD variants
  consumed by the path solvers.
- `solver::block_cd::block_cd_solve` — internal block-CD wrapper.
- `solver::prox_newton::prox_newton_fused_solve` — single-λ primitive
  of the path variant (which stays public).
- `solver::lla::{cmcp_value, gel_value, surrogate_sparse_group_mcp}` —
  penalty value-eval helpers retained for test fixtures.
- `solver::convex_region::PenaltyConcavity` — internal enum; the
  public detection functions take `f64` concavity directly.
- `design::GramDesign`, `datafit::GramLeastSquares` — internal
  precomputed-Gram path not exposed via the Python facade.

The contract page is rewritten end-to-end to enumerate the post-audit
surface: 5 extension traits + 2 factory traits, 9 concrete designs
including the orthonormalization wrapper, 7 datafits, 10 penalties +
5 factories, 12 algorithm entry points, the broadened solver-helper
list, 8 prox primitives, the `W_FLOOR` / `ETA_CLAMP` numerical
guards, and a reference list of the demoted items.

### M14h — Native block-CD for LS group-SCAD and sparse-group {MCP, SCAD}

Drops the LLA outer loop from three more group nonconvex paths:

- `GroupScad`: native per-iteration shrinkage applying the SCAD
  threshold to each group's block-soft-thresholded vector. Mirrors
  M13.4b's `GroupMcp` treatment with the SCAD `(1 − step·λ/a)`
  envelope.
- `SparseGroupMcp` and `SparseGroupScad`: the within-group L1 layer
  composes with the between-group nonconvex shrinkage in a single
  closed-form prox per block, matching grpreg's
  `gdfit_{sparse_,}{mcp,scad}` C kernels.

`block_path` (LS family) dispatches the three new penalties directly;
the LLA wrappers stay reachable for paths that still need them
(currently none on the LS side).

### M14g — Findings from the 2026-05-18 v2 release run

- **M14g.1 (fixed).** The `glasso_l1` benchmark cells silently routed
  through `lasso_path` on the n × p data matrix because both the
  skein and sklearn runners keyed dispatch on `penalty == "glasso"`
  but the scenario passes `penalty = "lasso"`. Both runners now
  dispatch via `problem.meta["simulator"] == "glasso_truth"`; the
  R-glasso runner already did. Aggregate regenerated 2026-05-18:
  skein 35.6 s / sklearn 252.6 s / R `glasso` 21.6 s on small/deep,
  ~19,665 edges across all three packages.
- **M14g.2 (closed as noise).** The 41.7 s `poisson_lasso medium/deep`
  median looked like a regression from a quoted 29 s v0.10.0
  baseline. Investigation: zero post-v0.10.0 commits touch the convex
  Poisson lasso execution path (`datafit/poisson_log.rs`,
  `solver/{prox_newton,cd,path}.rs`, `penalty/{lasso,elastic_net}.rs`,
  `skein-py/src/glm.rs` are byte-identical), and a re-run at HEAD
  with the v2 methodology gives 42.1 s median with a 34.9–94.7 s
  per-seed spread. The 1.4× claimed effect lives inside that 2.7×
  variance band. The absolute 17× Poisson-vs-glmnet wall-clock gap
  is real and pre-existing — tracked in §M9.3, not on the v1.0
  critical path.

### Marginal FDR (mFDR) for path estimators

New Python module `skein_glm.mfdr` providing the same formula
shared across GLM families (decoupled from the Rust core; no PyO3
bindings, no `_core.abi3.so` changes):

- `estimate_mfdr(path_model, x, y, *, family=None) -> ndarray` — the
  per-λ marginal FDR estimate over a fitted path estimator.
- `select_by_mfdr(path_model, x, y, *, target=0.1, family=None)` —
  returns the smallest λ-index whose mFDR estimate stays below
  `target`.
- `MFDR(path_estimator).fit(x, y).select(target=0.1)` — stateful
  drop-in companion to the existing selectors.

### Per-block group orthonormalization (Breheny–Huang)

New `skein_core::design::orthonormalize` module exposing
`orthonormalize_groups_dense(x, groups) -> (x_orth, BlockBackTransform)`
(Cholesky-based per-group Gram factorization, `T_g = √n · L_g^{-T}`).
The returned `BlockBackTransform` carries the per-group transform
and exposes `apply_to_coefs` / `apply_to_coefs_path` for mapping
fitted coefficients back to original-feature space.

Python wrapper `skein_glm.orthonormalize` ships
`orthonormalize_groups`, `BlockBackTransform`, and the high-level
`fit_with_orthonormalization` pipeline (center → orthonormalize →
fit with intercept disabled → back-transform → reconstruct
intercept). Solvers operating on the orthonormalized design see a
clean per-block Lipschitz of exactly 1 and a closed-form block
soft-threshold prox — matching grpreg's `gdfit_*` C kernels.

### Composite MCP and group exponential lasso

Two bilevel-selection penalties from the grpreg / ncvreg family
that skein did not previously expose. Both reduce to weighted L1 via
LLA and route through the existing scalar LLA path solver
(`solve_path_lla`).

- **Composite MCP** (Breheny–Huang 2009): outer MCP applied to the
  sum of per-coord inner MCPs in each group. Outer γ₁ drives group
  selection; inner γ₂ drives within-group selection — a true bilevel
  sparsity pattern that the additive `SparseGroupMcp` does not
  produce.
- **Group exponential lasso** (Breheny 2015): exponential decay on
  each group's L1 norm. Same bilevel structure with a single τ
  hyperparameter.

`CompositeMCPPathRegressor` and `GroupExponentialPathRegressor`
mirror the existing group-path API, including the `convex_min_idx_`
attribute (penalty concavity is `1/(γ₁γ₂)` for cMCP, `τ` for gel).

### Post-fit `convex.min` diagnostic for nonconvex paths

Adds grpreg-style `convex.min` detection: the smallest λ-index at
which the local objective ceases to be locally convex on the active
set (penalty curvature exceeds data-fit curvature). New
`skein_core::solver::{scalar_convex_min_idx, group_convex_min_idx}`
plus `convex_min_idx_` attribute and a one-shot `UserWarning` on the
six nonconvex LS path regressors (MCP / SCAD, group / sparse-group
MCP / SCAD). Mmap and chunked backends safely no-op.

### v2 benchmark suite expansion

Five new v2 scenarios cover the full GLM × {lasso, MCP} matrix plus
a sparse-group LS row:

- `cox_mcp`, `logistic_mcp`, `poisson_mcp`, `glasso_mcp`,
  `ls_sparse_group_mcp` scenarios; new `glasso_runner.py` wraps R
  `glasso` over Arrow IPC.
- Cox event status is now threaded through `glmnet_runner` via a
  sibling `status.feather` payload — Cox v2 cells can use glmnet as
  a direct comparator.
- `benches/v2/report/_run_cell._lambda_grid` is datafit-aware:
  graphical scenarios (`gaussian_inv_cov`) take
  `λ_max = max|off-diag(S)|` instead of `max|Xᵀy| / n`.
- v1 side: `benches/scenarios/glasso_ls.py` ships with a
  glasso-specific `(p=20/100/200, n=200/1k/2k)` size table because
  the canonical ladder is infeasible for an O(p³) solver. The
  `benches/problems.SIZES` ladder grows to five entries —
  `small / medium / large / xlarge / xxlarge` — with the headroom
  rationale documented in `benches/README.md`.

### Software paper (`paper/manuscript.tex`)

Manuscript template swap: `\documentclass[11pt]{article}` →
`\documentclass[twoside,11pt]{jmlr}` (the locally installed Talbot
2022 v1.30 class, not the legacy `jmlr2e.cls` referenced in older
guides). The class auto-loads `natbib`, `graphicx`, `amsmath`,
`amssymb`, `url`, and `hyperref`; the preamble drops manual
`\usepackage{…}` lines for those and the manual
`\bibliographystyle{plainnat}` that was producing a duplicate
`\bibstyle` in `manuscript.aux`. The header comment is rewritten to
describe what targeting MLOSS (page condensation, not a class swap)
or JOSS (pandoc markdown rewrite) would actually require.

Content refresh: §Results and §Ablation are rewritten against the
2026-05-18 v2 aggregates. The headline narrative cites the actual
post-M14h ratios (`ls_group_mcp` skein 6.57 s vs grpreg 12.56 s =
1.91× faster; `logistic_mcp` skein 19.6 s vs ncvreg 95.1 s = 4.85×
faster), the convex-Lasso-vs-sklearn gap is corrected to its real
~5.6×, and a new §Ablation subsection walks the GLM-MCP inner-loop
progression (123 → 19.7 → 3.05 s on the `logistic_mcp medium/sparse`
cell as M14e and M14f land).

### Preprocessing: bounded threshold sentinels for polychoric

The polychoric / polyserial preprocessor used `±np.inf` as sentinel
endpoints for the cumulative-probability brackets. scipy ≤ 1.15
(Python 3.10's pinned wheel) returns `NaN` from
`multivariate_normal.cdf(±inf)` — fixed upstream in scipy 1.16 but
present in the CI's 3.10 matrix. Replaced the sentinels with
`±8.0`, which is well past the Gaussian tail's numerical zero
(`Φ(−8) ≈ 6.6 × 10⁻¹⁶`) and produces identical bracket integrals to
the `±inf` form at our floating-point precision.

## [0.10.0] — 2026-05-17

Perf milestone: extends the celer-style screening + Anderson dual
extrapolation infrastructure (shipped LS-only in M10 wave F) across
the GLM prox-Newton paths. 3–8× wall-clock on `logistic_lasso` v2
cells without changing the public API (the legacy `prox_newton_solve`
signature is preserved; opt-in via the new
`prox_newton_solve_screened` and via `prox_newton_solve_path` /
`prox_newton_block_solve_path`, which both route through the
screened variant automatically).

Test count: **397 cargo lib + 8 cargo integration + 455 pytest, all
green** (up from 387 + 455 at 0.9.0 — 10 new dual-obj unit tests
+ a `prox_newton_screening_matches_no_screening_within_tol`
regression test).

### M13.8 — Celer-style gap-safe screening on the GLM prox-Newton surrogate

Closes the perf gap left by M10 wave F. F-series wired up gap-safe
screening + Anderson dual extrapolation + adaptive inner tol for the
LS path, but `Datafit::lasso_dual_obj` returned `None` for everything
except unweighted LS. The GLM paths
(`prox_newton_solve_path`, `prox_newton_block_solve_path`) ran
KKT-verifier-only — no screening, no extrapolation — and were
~28× slower than glmnet on the `logistic_lasso medium-deep` v2 cell.

What shipped:

- **Weighted-LS dual obj.** `LeastSquares::lasso_dual_obj` now handles
  the `sample_weights = Some(w)` case via the closed-form generalisation
  `D(θ_scaled) = (Σwᵢrᵢ²/n)·scale·(1−scale/2) − scale·βᵀg`. Unlocks
  screening on the prox-Newton surrogate.
- **Per-GLM closed-form duals.** `GlmDatafit` gains
  `glm_per_sample_loss_grad` + `glm_dual_obj` trait methods.
  `BinomialLogit` implements the sigmoid Fenchel dual;
  `PoissonLog` implements the Bregman form with offset support.
  Cox / Huber / Multinomial keep `None` defaults with documented
  rationale (Cox partial-likelihood dual has no closed form under
  Breslow/Efron ties; the others are out of scope).
- **`prox_newton_solve_screened`** mirrors `solve_path`'s per-λ KKT
  loop on the GLM surrogate: gap-safe sphere screening, Anderson dual
  extrapolation on `(β, r)` pairs (K=6 history), adaptive inner tol
  `= max(tol, 0.3 × prev_outer_pgd)`, M13.1-style saturation bypass.
  Legacy `prox_newton_solve` becomes a thin wrapper with
  `lambda = None`; no public-API signature change.
  `prox_newton_solve_path` opts in by passing `Some(lam)`. Same
  wiring in `prox_newton_block_solve_path`; `block_gap_safe_screen`
  generalised to use `Datafit::lasso_dual_obj` instead of an inlined
  unweighted formula.
- **Safe-sphere radius fix.** `r_safe² = 2·gap·max(w)/n` (was
  `2·gap/n`), derived from the dual strong-convexity constant
  `σ = n/max(w)` for weighted LS. Required for Poisson where
  `max(μ)` can exceed 1; logistic gets a tighter radius for free
  (`max(w) ≤ 0.25`). Unweighted-LS path unchanged
  (`sample_weights() == None` collapses to the FGS 2015 formula).

Wall-clock on bench v2 `logistic_lasso` (host `3c43bb844695`):

| cell | active set | before | after | speedup |
|---|---:|---:|---:|---:|
| small-sparse | 62/200 | 0.44 s | 0.05 s | **8.2×** |
| small-deep | 191/200 | 8.02 s | 2.62 s | **3.1×** |
| medium-sparse | 61/1000 | 27.58 s | 3.82 s | **7.2×** |
| medium-deep | 947/1000 | 219.59 s | 101.63 s | **2.2×** |

`poisson_lasso medium-sparse` is roughly neutral (5.48 s → 6.24 s);
Poisson's `max(μ) > 1` makes screening necessarily looser than logistic.

Validation: pre-flight tight-tol screening test passes, 397 cargo
lib tests pass (+9 dual-obj unit tests +
`prox_newton_screening_matches_no_screening_within_tol`), 455 pytest
pass / 5 skipped, cargo clippy + fmt clean. Out of scope: persistent
GLM-level screening across PN iters (Path B; the `glm_dual_obj` trait
method is wired but not yet driven by the solver), Cox dual
screening, Huber / Multinomial dual obj methods.

## [0.9.0] — 2026-05-15

Research-grade release. Closes the **inference axis** across all four
mainstream GLM families, adds **edge-level multiple-testing control** on
graphical models, ships **polychoric preprocessing** for ordinal Likert
data, and finishes the M13 / M14c perf work — every GLM × group penalty
(plain + sparse-group) now runs native, no LLA wrappers underneath any
prox-Newton outer.

Test count: **358 cargo lib + 8 cargo integration + 455 pytest, all
green** (up from 355 + 412 at 0.8.0). Three R-anchor placeholders skip
cleanly when fixtures absent.

### M13.4c — Native group-MCP block-CD for logistic / Poisson / Cox

Extends M13.4b (LS group-MCP) across GLM families. The
`prox_newton_block_solve_path` outer loop now hands a β-independent
`GroupMcp::with_weights(λ, γ, w)` factory directly, dropping the LLA
layer underneath prox-Newton. The "two-layer" concern from the M13.4b
write-up was overstated — prox-Newton stays as the GLM linearization
layer, only the LLA penalty layer drops.

Empirical comparison
(`crates/skein-core/examples/logistic_group_mcp_lla_vs_native.rs`,
n=4 000, p=400, group_size=5, n_groups=80, k_active=5, tol=1e-8,
γ=3.0, M1 Accelerate):

| solver | wall | outer iters | inner CD iters |
|--------|-----:|------------:|---------------:|
| LLA-wrapped GroupLasso (pre-fix) | 226.7 s |  190 | 69 392 |
| Native GroupMcp BCD (this fix)   | 106.8 s |  116 | 32 812 |

**-2.12× wall-clock** with min support Jaccard 0.97 vs LLA and
identical final-λ objective. New cross-family agreement test
(`crates/skein-core/tests/glm_group_mcp_native_matches_lla.rs`) covers
logistic / Poisson / Cox.

### M14a.1 — Polychoric / polyserial preprocessing

New `skein_glm.preprocessing` module:

- `polychoric_correlation(X)` — Olsson (1979) two-step ML for an
  ordinal correlation matrix.
- `polyserial_correlation(X_ord, Y_cont)` — Olsson-Drasgow-Dorans
  (1982) profile-likelihood ML.
- `polychoric_covariance_matrix(X)` — mixed-type auto-dispatch to
  polychoric / polyserial / Pearson.

Output feeds directly into `GraphicalLasso(cov=…)`. Recovery on
synthetic ordinal data (n=2000, 4-level Likert): max absolute error
0.04 between estimated and true latent correlation. Closes the M11.1
psychometrics-replication exit criterion deferred since v0.7 —
`docs/examples/psychometrics.md` is now an end-to-end
`polychoric → GraphicalMCP → bootstrap-FDR` pipeline that retains all
7 planted edges with zero false discoveries at n=400, 300 bootstraps.

### M14a.2 — Edge-level FDR / FWER / MB stability bound

New `skein_glm.graph_inference` module + convenience methods on
`GraphicalBootstrap` and `GraphicalStabilitySelection`. No other
graphical-models package controls error rates at the edge level.

- `edge_fdr_threshold(boot, fdr=0.1)` — Benjamini–Hochberg on per-edge
  two-sided bootstrap p-values.
- `edge_fwer_threshold(boot, fwer=0.05, method="holm")` — Bonferroni
  or Holm step-down family-wise error control.
- `mb_stability_threshold(p_total, q_λ, EV)` — Meinshausen–Bühlmann
  (2010) closed-form bound inverting a stability threshold to an
  expected-false-positive guarantee.
- Joint estimators `(B, K, p, p)` pool all `K · p(p−1)/2` edge
  hypotheses into one BH family.

Bootstrap p-values use the **non-strict** two-sided formula
`p = 2 · min(P̂(Θ̂* ≥ 0), P̂(Θ̂* ≤ 0))`, essential for sparse
estimators where null edges are exactly zero on every bootstrap
replicate (a strict-inequality formulation would spuriously assign
the smallest representable p-value to every null edge).

### M14a.3 — Debiased Cox lasso

`DebiasedCoxLassoRegressor` + `debiased_cox_lasso()` +
`DebiasedCoxResult`. Closes the inference axis across all four
mainstream GLM families — no mainstream Python package has Cox
debiasing.

Construction extends the Van de Geer–Bühlmann–Ritov (2014) /
Cai-Wang (2017) debiased lasso to Cox via the
**partial-likelihood Fisher diagonal** `w_i` from the existing
`CoxPH::surrogate_at` (no new core algorithm — exposed to Python
via a new 16-line PyO3 binding `cox_surrogate_weights_at`). Build
weighted design `X̃ = W^{1/2} X`, run nodewise lassos on `X̃`,
debias against the Cox score residual
`event_i − exp(η̂_i)·Λ̂_0(t_i)`. Variance
`diag(Θ̂ X̃ᵀX̃ Θ̂ᵀ) / n²` (no σ² nuisance — partial likelihood is
self-normalizing).

Empirical 95 % CI coverage ≥ 80 % on inactive coordinates over 40
replications. New `docs/concepts/inference.md` walks through
LS / GLM / Cox debiased lasso uniformly.

### M14c.1 — Scalar LLA weight short-circuit

Ports the M13.4 Phase 2.3 fix from `block_path_lla.rs` to scalar
`path_lla.rs`. Caches `prev_weights` and breaks the outer loop when
`‖w_t − w_{t-1}‖_∞ < weight_short_circuit_tol` (sized identically:
`1000 · outer_tol`, floored at `1e-8`). Affects callers of
`solve_path_lla`: bridge `|β|^q`, adaptive lasso, multi-task LLA
paths. On bridge q=0.5 (n=2000, p=100, 40 λs): average outer iters
per λ drops to 1.2 at convergence.

### M14c.2 — Native sparse-group MCP for logistic / Poisson / Cox

Sibling of M13.4c for the sparse-group penalty. New Rust
`SparseGroupMcp` penalty
(`crates/skein-core/src/penalty/sparse_group_mcp.rs`) implements the
Breheny & Huang (2015) Proposition 1 closed-form prox: per-coord
scalar MCP soft-threshold + per-group block MCP shrink, both
sharing the same `γ`. Six PyO3 closures swapped
(`solve_{logistic,poisson,cox}_sparse_group_mcp_path[_sparse]`).
Drops the last LLA wrapper in the non-convex GLM × group family.

Includes load-bearing reduction tests: `α=0` matches `GroupMcp` at
the same (λ, γ); `γ→∞` matches `SparseGroupLasso` at the same
(λ, α).

### M14c.3 — At-scale R-fixture tier

`tests/fixtures/generate.R` gains an n=500, p=100 mid tier for
three representative penalty / family combinations
(`glmnet_lasso_gaussian_mid`, `ncvreg_mcp_gaussian_mid`,
`glmnet_lasso_binomial_mid`). Tolerances on the Python side looser
than the small tier (`smallest_lambda_atol` 5e-3–5e-2,
`active_set_fuzz_frac` 0.15) — LLA local-min divergence on
nonconvex problems widens with `p`. The original ROADMAP target of
n=5000, p=2000 is parked as a follow-up because JSON-encoded `X`
exceeds practical git sizes at that scale; mid tier keeps each
fixture under ~1 MB raw.

### Documentation

Four new concept pages:

- `docs/concepts/polychoric.md` — Olsson's two-step ML derivation,
  end-to-end pipeline, when not to use polychoric.
- `docs/concepts/graph_inference.md` — BH FDR / Holm FWER /
  MB bound on edges; bootstrap p-value definition and trade-offs.
- `docs/concepts/inference.md` — unified page covering LS / GLM /
  Cox debiased lasso + stability selection.

`docs/examples/psychometrics.md` rewritten end-to-end with the new
M14a primitives; `docs/examples/survival.md` gains a "Confidence
intervals on prognostic features" section.

### Out of scope for 0.9

- **M14b (software paper)** — run the full `benches/v2` GLM +
  graphical headline matrix and draft the JMLR-MLOSS / JOSS
  manuscript from the figures + tables that already auto-generate.
  This is the next major milestone.
- Multi-response GLMs for Poisson / Cox (M7.3).
- n=5000, p=2000 R-fixture tier (needs an artifact-server pipeline).
- An R facade.

## [0.8.0] — 2026-05-15

Hardening + performance release: **M12 finish (every recommended-ordering
audit item closed) + M13 wins (M13.2 cross-λ gradient cache, M13.4b
native group-MCP block-CD)**. No new algorithmic surface; no new
penalties or datafits. The headline numbers:

- **medium Lasso: -10.4 % wall-clock** (M13.2, gradient-cache reuse
  between λs in the strong-rule screening + KKT loop)
- **medium ls_group_mcp: -3.46× wall-clock** (M13.4b, native group-MCP
  prox replaces LLA outer loop). Flips the prior `grpreg medium/dense
  3.34× faster` finding to **skein 1.20× faster than grpreg**.
- `skein-py/src/lib.rs`: **10 628 → 275 lines** (M12 P4 split — every
  datafit family in its own module).

Test count: **355 cargo + 412 pytest, all green** (350 lib + 5
integration; up from 292 cargo at 0.7.0).

### Performance — M13.4b: Native group-MCP block-CD

`solve_group_mcp_ls_path[_sparse]` (PyO3 layer) now constructs `GroupMcp`
directly and calls the standard `solve_block_path` instead of routing
through `solve_block_path_lla` with a weighted `GroupLasso` surrogate.
`GroupMcp::prox_group` already implemented the closed-form group-MCP
prox per Breheny & Huang 2015 §3, and `solve_block_path` already
accepted arbitrary `GroupPenalty` factories — the LLA wrapper was
paying ~5× the inner-CD work needed to reach the same stationary point.

Strong-rule screening still applies: the β_g=0 KKT subdifferential
`λ·[-w_g, w_g]` is identical for `GroupLasso` and `GroupMcp`, so the
rule carries over unchanged.

`max_outer` / `outer_tol` parameters stay in the Python signature for
backward compat (kwargs still accept them) but are now ignored.
Convergence is governed by the inner CD's `tol` and the path solver's
KKT verifier.

Empirical comparison (`crates/skein-core/examples/group_mcp_lla_vs_native.rs`,
n=10k, p=1k, group_size=5, n_groups=200, k_active=5, tol=1e-7,
γ=3.0, M1 Accelerate):

| solver | wall | inner CD sweeps | KKT passes |
|--------|-----:|----------------:|-----------:|
| LLA-wrapped GroupLasso (pre-fix) | 36.2 s |  1 688 |  340 |
| Native GroupMcp BCD (this fix)   | 10.5 s |    488 |  100 |

Cross-solver agreement: Jaccard = 1.0 on support at every λ; max
relative objective gap 5.4e-7 (numerical precision); max relative
ℓ₂ coefficient deviation 0.49 (different stationary points of the
non-convex problem, but both reach the same value of the original
penalized objective).

**Out of scope (kept on LLA for now):** logistic / Poisson / Cox
group-MCP variants. Their inner block-CD already runs against a
weighted-LS surrogate from the prox-Newton outer loop, so swapping
to native group-MCP would change two layers at once. Worth a
follow-up but not bundled here.

### Performance — M13.2: Cross-λ gradient cache + path-solver phase profile

`solve_path` (the LS scalar path solver) was computing two `full_grad`
matvecs per λ: one in `priority_rule_screen` at the START of λ_{k+1}
on the warm-start residual, and one in `compute_outer_state` at the
END of λ_k on the post-CD residual — the same residual. `OuterState`
now exposes its computed `grad`; `solve_path` caches it as `prev_grad`
and a new `priority_rule_screen_with_grad` variant skips the recompute.
Cache cleared on cold start (k=0) and after saturated-bypass λs (Off
mode skips `compute_outer_state`, so no fresh grad).

Result on the `lasso_ls_medium` example (n=10k, p=1k, 100 λs,
tol=1e-6, M1 Accelerate, single-process isolated): **2.847 s → 2.552 s
= -10.4 % wall**. Iter count + KKT passes unchanged (348 / 100 in both
runs) ⇒ no algorithmic regression.

Also added: env-var-gated `PhaseTimings` instrumentation in
`solve_path` (active only when `SKEIN_PROFILE_PATH` is set, zero
overhead otherwise). Attributes per-λ time to `setup` / `screening` /
`lipschitz` / `inner_cd` / `dual_extrap` / `outer_state` /
`bookkeeping`. Kept as permanent observability for future perf work.

### Performance — M13.6: Re-characterized post-M13.2

The pre-M13.2 ROADMAP claim — "skein 38.8× / sklearn 22.3× super-linear
scaling, attributable to fixed per-λ overhead" — is stale. With M13.2
closed, the new `lasso_ls_scaling` example (canonical v2 small / medium
/ large sizes) shows:

| transition | n×p ratio | wall ratio | factor (wall/np) |
|---|---:|---:|---:|
| small → medium | 50× | 37.0× | 0.74× sub-linear |
| medium → large | 25× | 37.6× | 1.50× super-linear |
| small → large  | 1250× | 1392× | 1.11× mildly super-linear overall |

The medium → large super-linearity lives entirely in inner CD (92 % of
wall), and the diagnosis is **memory-bandwidth bound**: at n=50k a
single X column is ~400 KB (bigger than typical L2); the full design
is 2 GB (past L3). `col_dot` shifts from compute-bound to memory-bound,
each coord visit streaming a fresh column from main memory. Same wall
sklearn faces in principle. Further wins past medium scale require
either column-batching for cache reuse or an algorithm change — both
large structural undertakings, not on the v0.8.0 path.

### Hardening — M12 finish

Every recommended-ordering audit item from the v0.7.0 M12 punch list
closed. No new algorithmic surface.

#### R4 — Centralized numerical guards

New `crates/skein-core/src/numerics.rs` exports `W_FLOOR = 1e-6` and
`ETA_CLAMP = 30.0`. `binomial_logit` / `poisson_log` / `cox_ph` /
`huber` datafits import from there instead of redefining. Future
bumps are one edit.

#### R3 — Solver pre-flight as a fail-fast CI gate

New step in `.github/workflows/ci.yml` runs the tight-tol screening
test in isolation with `timeout-minutes: 2`, before the full `cargo
test`. If a future change makes a stopping condition unreachable
(the `gap < tol²` → `1e-24` incident that motivated CLAUDE.md's
pre-flight protocol), the test hangs in isolation and fails fast
instead of starving the parallel suite.

#### R1 — `unwrap` audit closed

Audited the 59 `unwrap()` / `expect()` hits in `crates/skein-core/src/`.
Production-code findings:

- `cd.rs::anderson_extrapolate` bare `unwrap()` swapped to documented
  `.expect()` mirroring the loop-invariant pattern in
  `path.rs::anderson_extrapolate_pair`.
- `block_path_lla.rs` three undocumented unwraps (which mirror
  `block_path.rs`'s strong-rule screening pattern) documented to match.
- `glasso_admm.rs::Groups::from_csr(...).expect(...)` was dead code
  (built `_groups`, immediately dropped). Removed entirely along with
  the now-unused `use crate::groups::Groups`.

Remaining hits are test-only setup invariants — acceptable.

#### C5 — Parallel block-CD overlap detection + serial fallback

New `Groups::has_overlap()` public method (O(`idx.len()` + `max_idx`)
bitset). `block_cd_solve_subset_parallel_with_cache` checks at entry;
on overlap dispatches to serial Gauss-Seidel and fires a
`std::sync::Once`-gated stderr warning. Misuse no longer ships
silently; joblib path × CV loops don't spam the warning. Fixture
`block_cd_subset_parallel_with_overlapping_groups_falls_back_to_serial`
verifies bit-identical β between parallel-with-overlap and serial.

#### P2 — Criterion bench tree expansion

Three new microbench files alongside the existing `block_cd.rs`:

- `crates/skein-core/benches/lla_outer.rs` — group MCP outer-iter
  scaling vs γ (`{1.5, 3.0, 10.0}`) and n_groups (`{16, 64, 256}`).
- `crates/skein-core/benches/prox_newton_glm.rs` — single-λ logistic +
  Poisson Lasso at p=64,256.
- `crates/skein-core/benches/glasso.rs` — single-population glasso
  scaling p=20,50,100 + joint glasso ADMM at K=2,3 populations.

`crates/skein-core/benches/README.md` rewritten with how-to-run +
per-scenario descriptions.

#### P4 — `skein-py/src/lib.rs` split

`skein-py/src/lib.rs` went from **10 628 → 275 lines (-97 %)** — pure
entry point with module declarations + the `#[pymodule]` registration
block. Every datafit family lives in its own module:

| file | lines | content |
|---|---:|---|
| `glasso.rs` | 261 | Single-population + joint glasso ADMM |
| `mmap_chunked.rs` | 784 | Memory-mapped + row-block chunked LS+MCP / logistic+MCP, f64 + f32 |
| `multinomial.rs` | 897 | K-class softmax via Böhning majorization, dense + sparse |
| `multitask.rs` | 1 290 | Multi-task LS via virtual `MultiTaskDesign`, dense + sparse |
| `ls.rs` | 2 760 | LS scalar + group, dense + sparse, single-fits, plus cross-cutting helpers (`parse_screening`, `groups_from_labels`, CSC readers, glmnet scales, sparse weight builders, `PathOutput` type alias) |
| `glm.rs` | 4 544 | Logistic + Poisson + Huber + Cox, dense + sparse |
| `lib.rs` | 275 | Module declarations + `#[pymodule]` entry |

PyO3's `wrap_pyfunction!` accepts module paths via
`use $function as wrapped_pyfunction`, so registration uses e.g.
`wrap_pyfunction!(glm::solve_logistic_mcp_path, m)`. Cross-module
helpers exposed `pub(crate)`; single source of truth per helper.
Future iteration: editing logistic code only recompiles `glm.rs`,
not the 10 K-line monolith.

### Reproducing the perf numbers

```bash
# M13.2 cross-λ gradient cache (medium Lasso 10% wall):
cargo build --release --example lasso_ls_medium
SKEIN_PROFILE_PATH=1 ./target/release/examples/lasso_ls_medium

# M13.6 scaling (small / medium / large = canonical v2 sizes):
cargo build --release --example lasso_ls_scaling
SKEIN_PROFILE_PATH=1 SKEIN_SCALING_LARGE=1 \
    ./target/release/examples/lasso_ls_scaling

# M13.4b LLA vs native group-MCP head-to-head:
cargo build --release --example group_mcp_lla_vs_native
./target/release/examples/group_mcp_lla_vs_native
```

### Backward compatibility

- `GroupMCPPathRegressor` / `GroupMCPRegressor` (LS variant only)
  still accept `max_outer` / `outer_tol` kwargs but ignore them
  internally. `info_["outer_iters"]` is no longer populated for these
  estimators — use `info_["iters"]` / `info_["kkt_passes"]` instead.
  Logistic / Poisson / Cox group-MCP variants are unchanged (still on
  LLA).
- All other public APIs unchanged.

## [0.7.0] — 2026-05-12

Feature release: **M5.x complete + first-class convex GLM primitives +
graphical edge stability**. Three headline differentiators land in this
release:

1. **Debiased / desparsified lasso** for least squares, binomial
   logistic, and Poisson regression — Wald-style confidence intervals
   and p-values for high-dimensional penalized fits. The one inference
   feature `glmnet` / `ncvreg` / `grpreg` do not offer; mirrors R's
   `hdi::lasso.proj` (M5.x-a + M5.x-b).
2. **First-class convex logistic + Poisson Elastic-Net / Lasso
   primitives** — retires the prior `MCP(γ=1e9)` approximation that
   `AdaptiveLogisticLasso` and `AdaptivePoissonLasso` relied on. New
   primitive matches sklearn's L1 logistic regression to ~1%, where
   the approximation was ~17% off (M3.x).
3. **Threaded CV folds** across every `*PathCV` class, enabled by a
   project-wide PyO3 `py.allow_threads(|| ...)` GIL release on every
   inner solver call. Every joblib-threaded code path in skein
   (`StabilitySelection`, `GraphicalStabilitySelection`,
   `GraphicalBootstrap`, the debiased-lasso nodewise loop, every CV
   wrapper) now scales across cores instead of serializing on the GIL
   (M5.x-c).

Plus **bootstrap edge stability** for graphical models
(`GraphicalStabilitySelection` + `GraphicalBootstrap`, M11.3) — the
`bootnet`-style network-psychometrics output.

Marks **M5 done** in the roadmap.

Test count: **292 cargo + 412 pytest, all green** (up from
274 cargo + 289 pytest at 0.6.0).

### Added (M5.x — GIL release: full coverage across all path builders)

Follow-up to the initial M5.x-c PR. Extends the `py.allow_threads(|| ...)`
GIL release pattern to **every remaining path-solver builder** in
`crates/skein-py/src/lib.rs`, so threaded fold loops accelerate **every
CV class**, not just the scalar-penalty ones.

Builders updated in this pass:

- LS block: `build_block_path_outputs`, `build_block_path_outputs_sparse_ls`,
  `build_block_path_lla_outputs`, `build_block_path_lla_outputs_sparse_ls`
- GLM block: `build_glm_block_path_outputs`, `build_glm_block_path_outputs_sparse`
- Cox block: `build_cox_block_path_outputs`, `build_cox_block_path_outputs_sparse`
- Multinomial: `build_multinomial_path_outputs`, `build_multinomial_path_outputs_sparse`
- Multitask: `build_multitask_path_outputs`, `build_multitask_path_outputs_sparse`,
  `build_multitask_path_lla_outputs`, `build_multitask_path_lla_outputs_sparse`
- Bridge LS scalar (dense + sparse): the `solve_path_lla` call site
- Mmap-backed LS and logistic MCP paths

Each builder's penalty-factory closure trait bound (`F`) gained
`+ Send` so the closure can cross the `allow_threads` boundary. The
surrounding Python-object setup/teardown still runs with the GIL
held; only the inner Rust compute (`solve_path` / `solve_block_path` /
`solve_block_path_lla` / `solve_path_lla` / `prox_newton_solve_path` /
`prox_newton_block_solve_path`) is released.

Parity tests extended in `tests/test_cv_parallel.py`: 9 new
parameterized cases covering `GroupLassoPathCV` / `GroupMCPPathCV` /
`GroupElasticNetPathCV` / `SparseGroupLassoPathCV` (LS), the
logistic group variants, the Poisson group variants, and
`CoxGroupLassoPathCV`. Total parity-test count goes from 14 → 23,
all bitwise serial-vs-parallel equal.

Coverage is now complete: every `*PathCV` class in the public API
benefits from threaded fold parallelism. The `n_jobs` parameter on
each CV constructor is the only knob users need; the GIL release at
the Rust level makes the speedup real rather than a no-op.

Test count: **292 cargo + 412 pytest, all green** (up from 403).

### Added (M5.x — Threaded CV fold parallelism via GIL release)

Closes the remaining M5.x item. The fold loop in `_PathCVMixin` and
`_CoxPathCVMixin` (`python/skein_glm/cv.py`) now dispatches K folds
across threads via `joblib.Parallel(prefer="threads")`, gated by a new
`n_jobs` constructor parameter on every CV class.

**The enabling fix is in Rust**: the heavy compute inside the PyO3
builder functions (`build_path_outputs`, `build_path_outputs_sparse_ls`,
`build_glm_path_outputs`, `build_glm_path_outputs_sparse`,
`build_cox_path_outputs`, `build_cox_path_outputs_sparse`) now wraps
the `solve_path` / `prox_newton_solve_path` calls in
`py.allow_threads(|| ...)`. Without this, Python threads serialized
on the GIL during the path solve and parallelism was a no-op.

Concrete impact on a 5-fold CV at `(n=5000, p=200)`:

- `MCPPathCV`: 16.5 s → 7.3 s (n_jobs=-1 on 8 cores, ~2.3×).
- `LogisticLassoPathCV` at `(n=3000, p=100)`: 187 s → 74 s (~2.5×).

**Correctness**: `n_jobs=1` and `n_jobs=-1` produce **bitwise-identical**
results — the fold loop is deterministic regardless of thread
interleaving. 14 new pytest in `tests/test_cv_parallel.py` cover LS
(MCP/SCAD/EN), logistic (Lasso/EN/MCP/SCAD), Poisson (Lasso/EN/MCP/SCAD),
and Cox (MCP/SCAD) families with parameterized parity checks. Plus a
signature audit that every user-facing CV class exposes `n_jobs`.

**Side benefits**: the GIL release also accelerates anything else that
calls into these builders through joblib threading —
`StabilitySelection`, `GraphicalStabilitySelection`,
`GraphicalBootstrap`, the debiased lasso nodewise loop. They were all
previously bottlenecked by the same GIL contention.

**Scope notes**: the GIL release is applied to the **scalar-penalty
builders** (LS, GLM, Cox dense + sparse — the most-used CV paths).
Block-penalty builders (group / sparse-group), multinomial path
builders, and multitask path builders still hold the GIL during solve
— a follow-up PR will extend the same pattern to them. CV with those
estimators still works, just doesn't get the parallel speedup yet.
The `n_jobs` parameter is wired through every CV class — when the
corresponding builder gets GIL release, those CVs will accelerate
without any further Python changes.

Test count: **292 cargo + 403 pytest, all green** (up from 389).

### Added (M5.x-b — Debiased / desparsified lasso for GLMs)

Extends the VBR debiasing framework from least squares (M5.x-a) to
binomial logistic and Poisson log-link regression via the weighted-LS
surrogate + nodewise approximation of the Fisher information.

**The math**: at the penalized fit ``β̂``, the score is
``Xᵀ(y − μ̂)`` (canonical-link gradient) and the Fisher information
is ``J = (1/n) · Xᵀ W X`` where ``W = diag(μ̂(1−μ̂))`` for binomial,
``diag(μ̂)`` for Poisson. The debiased estimator is

    β̂_d = β̂ + (1/n) · Θ̂ · Xᵀ (y − μ̂)

with asymptotic Gaussian distribution ``√n (β̂_d − β) ⇝ N(0, J⁻¹)``
and ``Θ̂ ≈ J⁻¹`` built nodewise on the weighted design
``X̃ = W^{1/2} X``. **No `σ²` factor** — GLM noise is encoded in `W`.

Public surface (`python/skein_glm/debiased.py`):

- `debiased_logistic_lasso(X, y, *, lambda_, lambda_nodewise, alpha,
  fit_intercept, standardize, max_iter, tol, n_jobs)` — free
  function returning a `DebiasedGLMResult`.
- `debiased_poisson_lasso(...)` — same plus a Poisson `offset`
  (log-exposure) parameter.
- `DebiasedLogisticLassoRegressor` — sklearn-style facade with
  `decision_function`, `predict_proba`, `predict` inherited
  semantics; the inference outputs (`se_`, `ci_lower_`, `ci_upper_`,
  `pvalues_`, `z_scores_`, `Theta_`, `mu_fitted_`, `coef_glm_`,
  `family_`, `lambda_main_`, `lambda_nodewise_`) live as suffixed
  attributes.
- `DebiasedPoissonLassoRegressor` — `predict` returns `μ̂ = exp(η̂)`
  matching the existing `PoissonLassoRegressor` convention; supports
  the `offset` constructor parameter.
- `DebiasedGLMResult` dataclass (separate from `DebiasedLassoResult`
  since GLMs have no `sigma_hat` and add `mu_fitted` / `family`).

Implementation reuses the M5.x-a nodewise machinery
(`_fit_nodewise_column`, `_assemble_theta_rows`): once the weighted
design is formed, the math is identical to the LS case. Working
weights are floored at `1e-8` to prevent degenerate `X̃` columns
when logistic fit saturates near probabilities 0 / 1.

**Critically**: the penalized-fit primitive is the new M3.x
`LogisticLassoRegressor` / `PoissonLassoRegressor`, not the prior
`MCP(γ=1e9)` approximation. Debiasing on top of that approximation
would have inherited its bias.

19 new pytest in `tests/test_debiased_glm.py` cover: dataclass
shapes and finiteness, CI ordering, SE non-negativity, **empirical
95% CI coverage on inactive coordinates** (40 reps each for logistic
and Poisson — coverage ≥ 80% catches Θ̂/variance errors), smaller
p-values on true-active features, Poisson `offset` changing the fit
and rejecting bad shapes, parameter validation (non-binary y,
negative y, alpha out of range), sklearn wrapper round-trip on
both families, free-function ≡ wrapper equivalence, n_jobs
serial-vs-parallel parity.

Closes M5.x-b. The remaining M5.x item is Rayon-parallel CV folds
(performance only, no new capability).

Test count: **292 cargo + 389 pytest, all green** (up from 370).

### Added (M3.x — First-class Poisson Elastic Net / Lasso primitives)

Symmetric to the logistic-side addition: replaces the prior
`PoissonMCPPathRegressor(gamma=1e9)` convention used internally by
`AdaptivePoissonLasso*` and as the de-facto user-facing Poisson lasso.

PyO3 bindings (`crates/skein-py/src/lib.rs`):

- `solve_poisson_elastic_net_path` (dense) and
  `solve_poisson_elastic_net_path_sparse` (CSC). Both take
  `alpha ∈ [0, 1]` and the log-link `offset` parameter that the rest of
  the Poisson family uses.

Python estimators (`python/skein_glm/estimators.py`):

- `PoissonElasticNetRegressor` / `PoissonElasticNetPathRegressor` with
  full surface (`alpha`, `lambda_`, `offset`, weights, sample weights,
  sparse dispatch).
- `PoissonLassoRegressor` / `PoissonLassoPathRegressor` — `alpha=1.0`
  facades.

CV variants (`python/skein_glm/cv.py`):

- `PoissonElasticNetPathCV` and `PoissonLassoPathCV` via the existing
  `_PoissonPathCVMixin`.

Retrofit (`python/skein_glm/adaptive.py`):

- `_fit_pilot_poisson` now uses `PoissonLassoPathRegressor`.
- `AdaptivePoissonLassoPathRegressor._final_cls` and
  `AdaptivePoissonLassoPathCV._final_cls` switched to
  `PoissonLassoPathRegressor`; `_extra_kwargs() = {"gamma": 1e9}`
  removed.

15 new pytest in `tests/test_poisson_en.py` mirror the logistic suite:
shape, lasso/EN facade equivalence, support-match against MCP-at-γ=1e9,
sparse-signal recovery, α→0 ridge limit, α=0.5 sparsity ordering,
sparse-input parity, offset changes the fit, `predict` returning
`μ = exp(η)`, CV wrappers, parameter validation. All existing adaptive
+ Poisson-offset tests pass unchanged.

Test count: **292 cargo + 370 pytest, all green** (up from 355).

### Added (M3.x — First-class logistic Elastic Net / Lasso primitives)

Replaces the prior `LogisticMCPPathRegressor(gamma=1e9)` convention used
internally by `AdaptiveLogisticLassoPathRegressor` /
`AdaptiveLogisticLassoPathCV` and as the de-facto user-facing logistic
lasso. That trick was a numerical approximation — MCP at γ→∞ converges
*pointwise* to soft-thresholding, but the LLA + prox-Newton machinery
still pays for nonconvex outer iteration. New primitives call prox-Newton
directly on the convex `ElasticNet` penalty, so the result is a proper
convex solve.

**Concrete numerical impact**: on a synthetic logistic problem
(`n=200, p=30, s=3, λ=0.05`), the new `LogisticLassoRegressor` matches
sklearn's `LogisticRegression(penalty='l1')` to ~1% on the active set,
while the old `MCP(γ=1e9) + max_outer=1` approximation was off by ~17%.

PyO3 bindings (`crates/skein-py/src/lib.rs`):

- `solve_logistic_elastic_net_path` (dense) and
  `solve_logistic_elastic_net_path_sparse` (CSC). Both take an
  `alpha ∈ [0, 1]` parameter (1 = lasso, 0 = ridge). No new Rust core
  code — `ElasticNet` penalty and `prox_newton_solve_path` already
  existed; this is pure wiring.

Python estimators (`python/skein_glm/estimators.py`):

- `LogisticElasticNetRegressor` / `LogisticElasticNetPathRegressor` —
  full surface with `alpha`, `lambda_`, weights, sample weights,
  fit_intercept, standardize, sparse dispatch.
- `LogisticLassoRegressor` / `LogisticLassoPathRegressor` — thin
  facades around the EN variants with `alpha = 1.0` pinned.

CV variants (`python/skein_glm/cv.py`):

- `LogisticElasticNetPathCV` and `LogisticLassoPathCV` via the existing
  `_LogisticPathCVMixin`.

Retrofit (`python/skein_glm/adaptive.py`):

- `_fit_pilot_logistic` now uses `LogisticLassoPathRegressor` for its
  pilot fit (was `LogisticMCPPathRegressor(gamma=1e9)`).
- `AdaptiveLogisticLassoPathRegressor._final_cls` and
  `AdaptiveLogisticLassoPathCV._final_cls` switched to
  `LogisticLassoPathRegressor`; the `_extra_kwargs() = {"gamma": 1e9}`
  overrides are gone.

17 new pytest in `tests/test_logistic_en.py` cover: shape and active-set
sanity, lasso/EN facade equivalence (`LogisticLasso ≡ EN(α=1)`), the
new lasso matching the old MCP-at-γ=1e9 on **support** (both
approximate the same fit), sparse-signal recovery along the path,
α→0 ridge limit (no exact zeros), α=0.5 intermediate sparsity ordering
(`lasso ≤ EN ≤ ridge` active counts at fixed λ), sparse-input parity,
predict_proba semantics inherited from the logistic base, sample
weights changing the fit, CV wrappers picking a λ, and parameter
validation (α range, non-binary y rejection). All 25 existing adaptive
tests pass unchanged.

Test count: **292 cargo + 355 pytest, all green** (up from 338).

Foundation for M5.x-b (debiased GLM): VBR debiasing for logistic
regression will use `LogisticLassoRegressor` as its penalized-likelihood
fit primitive, which avoids inheriting the MCP-at-γ=1e9 approximation
in the inference layer.

### Added (M5.x-a — Debiased / desparsified lasso)

Van de Geer–Bühlmann–Ritov (2014) confidence intervals and p-values
for high-dimensional lasso regression. Pure Python on top of the
existing `ElasticNetRegressor(alpha=1.0)` primitive — no Rust
changes.

`python/skein_glm/debiased.py`:

- `debiased_lasso(X, y, *, lambda_, lambda_nodewise, alpha, ...)`
  — free function returning a `DebiasedLassoResult` dataclass with
  `coef_debiased`, `coef_lasso`, `se`, `ci_lower` / `ci_upper`,
  `pvalues`, `z_scores`, `sigma_hat`, `Theta` (the approximate
  inverse Gram), `lambda_main`, `lambda_nodewise`, `alpha`.
- `DebiasedLassoRegressor` — sklearn-style facade exposing the
  debiased estimate as `coef_` / `intercept_` and the VBR-specific
  outputs as suffixed attributes (`se_`, `ci_lower_`, etc.), so
  the result composes with sklearn pipelines.

Implementation follows the **nodewise lasso** construction (VBR
Theorem 2.2): for each column `j`, regress `X_j` on `X_{−j}` with
a lasso to obtain `γ̂_j` and `τ̂_j² = ‖resid‖²/n + λ_j ‖γ̂_j‖₁`;
assemble row `j` of `Θ̂` as `(−γ̂_{j,·}, 1, −γ̂_{j,·}) / τ̂_j²`.
The debiased estimator is `β̂_d = β̂ + (1/n) · Θ̂ · Xᵀ(y − Xβ̂)`
with asymptotic variance `σ̂² · diag(Θ̂ Σ̂ Θ̂ᵀ) / n`. Variance is
computed via `U = X_s Θ̂ᵀ` so the `p × p` `Σ̂` is never
materialized.

Defaults: standardize columns, theoretical λ scale
`√(2 log p / n)` for both main and nodewise fits (dimensionless on
standardized features), joblib-parallel nodewise loop (`n_jobs=-1`
recommended for `p ≳ 50`).

Scope: least squares + dense `X` only. GLM debiasing (logistic /
Poisson) via the weighted-LS surrogate + Fisher information is the
planned follow-up (M5.x-b). Sparse `X` works through the
underlying path solver's CSC dispatch but is not yet plumbed
through the public API.

22 pytest cover: dataclass shapes, finiteness, CI ordering, SE
non-negativity, **empirical 95% CI coverage from 60-rep simulation**
(load-bearing — catches Theta / variance math errors), active-
coordinate coverage above 80%, debiased < lasso L1 error on
active features, p-values smaller on true-active features,
user-supplied / scalar / per-column λ_nodewise, no-intercept mode,
n_jobs serial-vs-parallel parity, sklearn estimator round-trip,
free-function ≡ wrapper equivalence, and parameter validation
(3D X, mismatched y, alpha out of range, bad λ_nodewise shape,
non-positive λ, p < 2).

Closes M5.x-a; M5.x-b (GLM extension) and an R-anchor regression
suite against `hdi::lasso.proj` are tracked under M5.x.

Test count: **292 cargo + 338 pytest, all green** (up from 316).

### Added (M11.3 — Bootstrap edge stability)

Pure-Python wrappers around the M11.1 / M11.2 graphical estimators
in `python/skein_glm/graph_stability.py`:

- `GraphicalStabilitySelection` — Meinshausen–Bühlmann (2010)
  subsample stability selection lifted to **edges**. Sweeps a
  user-supplied λ-grid; per (bootstrap, λ) refit records the
  off-diagonal nonzero pattern of `Θ̂`; aggregates to per-(λ, i, j)
  selection probability. Stable edges are those whose max-over-λ
  probability crosses a threshold (default 0.6; MB error-control
  requires `> 0.5`). Output shape `(n_lambdas, p, p)` single,
  `(n_lambdas, K, p, p)` joint.
- `GraphicalBootstrap` — classic non-parametric (resample-with-
  replacement) bootstrap at a single λ. Returns the per-edge
  bootstrap mean, SD, `[α/2, 1−α/2]` quantile CIs, and edge
  selection probability — the headline
  `bootnet::bootnet(type="nonparametric")` output for edge error
  bars in network psychometrics.

Both classes auto-dispatch single-vs-joint via the wrapped
estimator's `alpha` (single) or `lambda_2` (joint) init param,
parallelize the bootstrap loop via `joblib`, and reject precomputed-
covariance inputs with a clear error.

16 pytest cover shapes, signal recovery on a synthetic sparse-`Θ`
problem, threshold/CI ordering, joint dispatch (lasso + MCP),
reproducibility under fixed `random_state`, `n_jobs` parity, and
full parameter validation. No Rust changes.

Test count: **292 cargo + 316 pytest, all green** (up from 289 pytest at v0.6.0).

## [0.6.0] — 2026-05-12

Feature release: **M11 — graphical models**. The first non-regression
algorithm family in skein. The headline new capability is
**nonconvex graphical lasso** (MCP/SCAD on edges) and **joint
estimation across populations**, neither of which is available in
mainstream packages.

### Added (M11 — Graphical models)

Sparse precision matrix estimation with weighted L1, MCP, and SCAD
penalties on edges. The single-population pipeline lands as
**M11.1**; joint estimation across `K` related populations lands
as **M11.2**.

- New Rust solvers:
  - `solver::glasso_solve` — single-population graphical lasso via
    Friedman/Hastie/Tibshirani 2008 block-CD. Each column-solve
    runs the existing `cd_solve` against a new `GramLeastSquares`
    datafit + `GramDesign` backend, so every scalar `Penalty`
    (`Lasso` / `Mcp` / `Scad` / `ElasticNet`) plus per-edge
    weights drops in unchanged.
  - `solver::joint_glasso_solve` — joint graphical lasso (Danaher–
    Wang–Witten 2014, group form) via ADMM. The Θ-update is a new
    `prox::logdet_eigen_prox` (closed-form via symmetric eigen-
    decomposition; self-contained Jacobi, no LAPACK dependency).
    The Z-update is exactly an existing `GroupPenalty::prox_group`
    call (one group per off-diagonal edge of length `K`), so
    `GroupLasso` and `GroupMcp` drop in unchanged. First ADMM
    kernel in skein.
- New trait shims `penalty::ScalarPenaltyFactory` and
  `penalty::GroupPenaltyFactory` keep the outer glasso solvers
  generic over penalty choice.
- Python estimators in `skein_glm.estimators`:
  `GraphicalLasso`, `GraphicalMCP`, `GraphicalSCAD`,
  `JointGraphicalLasso`, `JointGraphicalMCP`. All accept either
  raw `X (n, p)` or precomputed `(p, p)` covariance (sniffed by
  shape + symmetry); joint variants take a list of either form.
- EBIC tuners in `skein_glm.graph_selection`: `ebic_path` and
  `joint_ebic_path` implementing Foygel & Drton 2010 — the
  field-standard graphical-model tuning rule used by `qgraph` /
  `bootnet`.
- Docs: new `docs/concepts/graphical_models.md`,
  `docs/api/estimators-graphical.md`, `docs/api/graph_selection.md`,
  `docs/tutorials/10_graphical_lasso.md`,
  `docs/tutorials/11_joint_networks.md`, plus
  `docs/examples/psychometrics.md`.
- Tests: 19 new Rust unit tests (gram-form CD, Jacobi eigen, log-
  det prox, single & joint glasso) and 11 new pytest end-to-end
  tests including a sklearn `GraphicalLasso` parity check.

### Differentiator

Nonconvex graphical lasso (MCP/SCAD on edges) and joint estimation
across populations are not available in mainstream packages
(`sklearn.covariance.GraphicalLasso`, R `glasso`, `qgraph`,
`bootnet`, `EstimateGroupNetwork` are all L1-only or
single-population). Closes a recognised shrinkage-bias gap in
network psychometrics — see Fan/Feng/Wu 2009, Lam & Fan 2009.

## [0.5.1] — 2026-05-10

CI green-up patch on top of `v0.5.0`. No behaviour change, no new
features. Cuts a separate tag because the v0.5.0 push triggered CI
lint failures (rustfmt, clippy threshold, ruff unused-imports,
sphinx unreferenced-docs) that needed the formatter / config tweaks
below.

### Fixed

- **`cargo fmt --all -- --check`** in CI:
  - `examples/lasso_ls_medium.rs`, `solver/cd.rs`, `solver/path.rs`
    reformatted to rustfmt's defaults (single-line struct literals
    where they fit, function-call reflow, the
    `extrapolation.as_ref().map(...)` chain wrapped per
    chain-fit rule).
- **`cargo clippy --workspace --all-targets -- -D warnings`** in CI:
  - `solver/path.rs::compute_outer_state` grew to 11 args during
    F.2 (Anderson-extrapolation pair + `&mut best_dual_obj`
    accumulator). Clippy's threshold is 7. Annotated with
    `#[allow(clippy::too_many_arguments)]` and a comment
    explaining the rationale; wrapping the args in a struct just
    for clippy's threshold isn't worth the indirection at the
    only call site.
- **`ruff check .`** in CI:
  - `benches/runners/sklearn_runner.py`: dropped unused
    `ElasticNet, Lasso` from the per-call import list (we only
    use the `*_path` functions on the LS branch and
    `LogisticRegression` on the logistic branch).
  - `benches/scenarios/lasso_ls{,_sparse}.py`: `import numpy as np`
    removed — unused after the F.4 / scenarios refactor that
    moved `lambda_grid` into `benches/scenarios/_common.py`.
- **`sphinx-build -W -b html docs docs/_build/html`** in CI: two
  perf docs (`docs/perf/lasso_ls_profile.md`,
  `docs/perf/celer_skglm_study.md`) added in v0.5.0 weren't
  referenced from any toctree, so `-W` (warnings as errors) failed
  the docs build. Added a "Performance" section to the top-level
  toctree in `docs/index.md`.

### Tests / lints all green locally

  cargo fmt --all -- --check                             ✓
  cargo clippy --workspace --all-targets -- -D warnings  ✓
  ruff check .                                           ✓
  mypy python/                                           ✓
  sphinx-build -W -b html docs docs/_build/html          ✓
  cargo test -p skein-core (default + blas-accelerate)   265 / 265
  cargo test -p skein-core --features blas-openblas      265 / 265
  pytest                                                 279 / 279

## [0.5.0] — 2026-05-10

The performance release. M9 (cross-package benchmark harness) and M10
(performance improvements driven by the bench) are the primary
deliverables. Lasso/LS path solver dropped from **7.6 s → 0.78 s**
(sparse) / **1.17 s** (deep) on the medium scenario (n=10k, p=1k,
100-λ path) — a ~10× swing across the M10 work. Now within
**1.5× of glmnet on sparse / 1.9× on deep**; ~8–9× behind sklearn's
Cython `lasso_path` (the floor that needs a Cython-grade rewrite to
catch).

### Added

- **M9.1 — bench harness.** New top-level `benches/` directory with
  problem generators (`benches/problems.py`), driver
  (`benches/run.py` — `--scenarios`, `--packages`, `--sizes`,
  `--trials`), runner ABI (`benches/runners/__init__.py`), and live
  runners for skein, sklearn, skglm, celer, pyglmnet, and R via
  Rscript (glmnet / ncvreg / grpreg). Timing methodology: 1 warm-up
  call discarded + N timed trials, headline is the median; per-trial
  times also recorded so noise can be inspected post-hoc.
  Cross-scenario helpers in `benches/scenarios/_common.py` so adding
  the next scenario is a ~50-line file.
- **M9.3 — lasso/LS bench scenarios (deep + sparse).**
  `benches/scenarios/lasso_ls.py` (`λ_min/λ_max = 1e-3`, deep into
  the saturated tail) and `benches/scenarios/lasso_ls_sparse.py`
  (`λ_min/λ_max = 5e-2`, stops at support recovery — the actual
  regime lasso is designed for). Snapshots committed at
  `benches/results/{lasso_ls,lasso_ls_sparse}.json`. Fairness fix
  bundled: skglm and celer runners now use their warm-started path
  APIs (`Lasso.path` / `celer_path`) instead of looping
  `Lasso(alpha=λ).fit()` per λ — comparison is apples-to-apples.
- **M10.1 — perf profile target.**
  `crates/skein-core/examples/lasso_ls_medium.rs` — pure-Rust binary
  that reproduces the medium scenario; `[profile.release]` carries
  `debug = "line-tables-only"` for samply / cargo-flamegraph symbol
  resolution. Findings in `docs/perf/lasso_ls_profile.md`:
  `dot_generic` is the ~80 % floor without BLAS, and a microbench
  rules out a hand-rolled tight loop as a faster alternative.
- **M10.3 — five waves of perf fixes**, each independently committed
  and bench-verified:
  - **`DesignMatrix::col_axpy(j, α, r)`** trait method specialised
    on every backend (Dense / Sparse / Standardized / Augmented /
    Mmap×2 / Chunked / MultiTask). Replaces a per-coord `(n × 1)`
    `Array2` allocation that was costing ~10 GB of heap traffic per
    medium-bench fit. Default impl (slow) for forward-compat.
  - **F-order `DenseMatrix`**: forced column-major layout in
    `DenseMatrix::new` so `column(j)` is contiguous; `scaled_add`
    runs at memory bandwidth instead of one L1 miss per element.
    One-shot 80 MB copy at construction, amortised across the path.
  - **`cd_solve_subset` returns the residual** it already maintains
    via incremental axpy updates. The path solver no longer
    recomputes `r = Xβ − y` from scratch after each call — one
    `O(np)` matvec saved per λ.
  - **Adaptive inner tolerance via prox-gradient distance.**
    `compute_outer_state` returns `max_pgd` (commensurable with
    `config.cd.tol`); next inner tol = `max(tol, 0.3 · prev_pgd)`.
    Same units, same penalty-agnostic guarantees. Iter sum
    on medium deep dropped from 430 → 317 (26 % fewer).
  - **KKT-priority WS construction** (celer/skglm pattern). New
    `PathConfig::p0` field (default 10, matches skglm). WS sized
    `max(p0, 2 × |support|)`, ranked by `|grad_j| / w_j` with
    active + unpenalised features pinned. Replaces the
    "fall-back-to-full-feature-set" cliff of the old strong rule
    at λ_max — initial WS goes from 1000 → 10 there.
- **`blas-accelerate` + `blas-openblas` Cargo features**
  (skein-core + skein-py passthrough). Routes ndarray's `dot` /
  `scaled_add` / `gemv` through hardware BLAS:
  - `blas-accelerate` — Apple's Accelerate framework on macOS via
    `blas-src` + `accelerate-src`. Zero install cost; ships with
    the OS.
  - `blas-openblas` — system OpenBLAS via `blas-src` +
    `openblas-src/system`. Used by Linux wheels (the manylinux
    container installs `openblas-devel`); cibuildwheel + auditwheel
    bundles `libopenblas.so` into the wheel so the installed
    package is self-contained.
  Distributed wheels are built with the matching feature per
  platform (macOS arm64 = accelerate, Linux x86_64 = openblas);
  Windows ships without BLAS for now — wheel still works, ~3×
  slower path. Locally, build with
  `maturin develop --release --features blas-accelerate` (macOS)
  or `--features blas-openblas` (Linux). Delivered the largest
  single speedup of M10: 3.32 s → 1.75 s deep (1.9×),
  2.50 s → 0.96 s sparse (2.6×).
- **F-series — duality gap + dual extrapolation + gap-safe
  screening** (4 commits: `2fea09c`, `2d025d4`, `971f73d`,
  `5d3c755`). `Datafit::lasso_dual_obj` (LS overrides) +
  `Penalty::dual_correction` (ElasticNet overrides for the ridge
  contribution) + `Penalty::has_lasso_form_dual_gap()` (gates the
  gap on penalties whose L1 envelope is tight at the optimum, so
  MCP / SCAD aren't accidentally early-stopped on the wrong
  bound). Per-λ Anderson extrapolation on the residual sequence
  with coefficients applied jointly to β so the extrapolated
  `(β_acc, r_acc)` is self-consistent under `r = Xβ − y`. Gap-safe
  sphere screening via FGS 2015. **Wallclock-neutral on M9.3
  scenarios** — the existing PGD + priority-WS combo already at
  the algorithmic floor; documented honestly in
  `docs/perf/lasso_ls_profile.md` so the next person doesn't
  repeat the experiment.
- **`docs/perf/lasso_ls_profile.md` + `docs/perf/celer_skglm_study.md`**
  — comparative reading of celer + skglm + the iterative
  optimisation timeline that drove M10's wave structure.

### Changed

- `[profile.release]` in workspace `Cargo.toml` gains
  `debug = "line-tables-only"` so samply / cargo-flamegraph can
  resolve frame names without significant binary-size cost.
- `compute_outer_state` is now the unified per-λ verifier. Returns
  `OuterState { violators, max_pgd, gap, lambda_bound,
  safely_inactive }`. Replaces the previous gradient-only
  `find_kkt_violators`. Still one BLAS gemv (the gradient) plus
  `O(p)` per-coord work — same asymptotic cost, more information.
- `solve_small` (Tikhonov-regularised normal-equations solver in
  `cd::anderson_extrapolate`) made `pub(crate)` so the path solver
  can reuse it for residual-sequence Anderson without code
  duplication.

### Fixed

- **M9.3 runner fairness.** Pre-fix, `skglm` and `celer` runners
  looped `Lasso(alpha=λ).fit()` per λ — handicapping both packages
  (sklearn / skein / glmnet were already using path solvers with
  internal warm starts). Post-fix using their native path APIs:
  - skglm medium deep: 10.1 s → 5.28 s; sparse 7.66 s → 3.46 s.
  - celer medium deep: 12.0 s → 3.97 s; sparse 6.49 s → **466 ms**
    (revealing celer's actual sparse-regime advantage — that's the
    F-series motivation).

### Test count

**265 cargo + 279 pytest, all green** on both M10 + F-series
changes — same as v0.4.0. No new tests added; the existing test
corpus is what validates the perf changes don't regress correctness.

A pre-flight protocol was developed during the F-series after a
runaway-cargo-test incident from a too-tight gap-based stopping
rule (`gap < tol²` becomes unreachable at `tol = 1e-12`):

  Before changing convergence criteria, smoke a single tight-tol
  test (`elastic_net_alpha_zero_recovers_closed_form_ridge`,
  `standardized_solver_path_matches_pre_scaled_dense`) before
  letting `cargo test` loose. Caught the F.3 MCP regression at
  one failure / 85 passes instead of pinning all cores.

### Deprecation / breaking

`PathConfig` gains a required `p0: usize` field (default `10`).
**Source-compat breakage on struct-literal constructions**:
existing `PathConfig { ... }` literals must add `p0: 10,` (or
opt into a different value). Affects 27 call-sites across
`crates/skein-core/src/`, `crates/skein-py/src/lib.rs`, and the
profiling example. `..Default::default()` is the recommended
spread for forward-compat.

The `Penalty` and `Datafit` traits gain default-implemented
methods (`dual_correction`, `has_lasso_form_dual_gap`,
`lasso_dual_obj`) with `false` / `0.0` / `None` defaults. Existing
implementors aren't required to override.

### Bench numbers (medium, n=10k, p=1k, 100-λ, 3-trial median)

| package        | v0.5.0 (deep) | v0.5.0 (sparse) |
|---|---|---|
| sklearn        | 125 ms        | 99 ms |
| glmnet (R)     | 614 ms        | 510 ms |
| **skein**      | **1.17 s**    | **0.78 s** |
| celer          | 2.73 s        | 307 ms |
| skglm          | 3.39 s        | 2.26 s |

(v0.4.0 had no committed lasso/LS bench numbers; M9.1 was
scaffolded but the comparator runners hadn't been validated.)

## [0.4.0] — 2026-05-09

Closes the **M5.x headline differentiator** (stability selection — no
clean equivalent in glmnet / skglm / grpreg) and rounds out the M6.x
adaptive family (plain `GroupSCAD` + `AdaptiveGroupSCAD`). Plus the
post-v0.3.0 CI fixes that improve developer experience.

### Added

- **Stability selection (M5.x).** New `StabilitySelection` meta-
  estimator (`python/skein_glm/stability.py`) wraps any skein
  `*PathRegressor` in a Meinshausen-Bühlmann (2010) subsample-
  bootstrap loop. Outputs per-(feature, λ) selection probabilities;
  the stable set is `{j : max_k Π_j(λ_k) ≥ threshold}`. Auto-
  dispatches across scalar / GLM / Cox / grouped / multi-task /
  multinomial path estimators (Cox detected by `ties` attr; grouped
  by `groups` attr; multi-task / multinomial 3D `coefs_` collapse
  via "any-class active"). Bootstrap loop parallelized via `joblib`
  (`n_jobs=-1`); deterministic for fixed `random_state` regardless
  of `n_jobs`. 11 pytest covering signal recovery, threshold
  monotonicity, reproducibility, validation, and dispatch.
- **Plain `GroupSCAD` (M6.x).** Wires the M2.8 `surrogate_weights_
  group_scad` helper through to PyO3 entries
  (`solve_group_scad_ls_path[_sparse]`) and 3 sklearn classes
  (`GroupSCADRegressor`, `GroupSCADPathRegressor`,
  `GroupSCADPathCV`). Validates `a > 2`. Closes the dangling
  prerequisite from the M6.x adaptive group commit in v0.3.0.
- **Adaptive group SCAD (M6.x).** Completes the M6.x adaptive group
  family with `AdaptiveGroupSCAD{PathRegressor, PathCV}`. The
  full adaptive family is now symmetric: 6 LS scalar + 6 LS group
  (Lasso/MCP/SCAD × Path/PathCV) + 18 GLM = 30 adaptive
  estimators.

### Fixed

- **CI hygiene** for the v0.3.0 cycle: dropped dead Python imports
  flagged by ruff, applied `cargo fmt` across recent feature
  commits, silenced clippy `needless_range_loop` on parallel-array
  Cox loops with a localized `#[allow]`, added 18 missing PyO3
  function stubs in `_core.pyi` for mypy.
- Two `dict[str, Any]` annotations on Bridge estimator's `common`
  dict to satisfy mypy's `**kwargs` splat type-checking.
- Landing page heading bumped from "What's in v0.2" to "v0.3" (now
  v0.4 in this release).

### Estimator counts (cumulative)

136 estimators in v0.3.0 → **141 estimators in v0.4.0** (5 new:
GroupSCAD × 3 + AdaptiveGroupSCAD × 2). Plus the
`StabilitySelection` meta-estimator (the first non-`*PathCV` /
`select_by_ic` model-selection wrapper).

### Tests

**265 cargo + 279 pytest, all green.**
- v0.3.0 baseline: 265 cargo + 261 pytest.
- New tests this release: +18 pytest (7 GroupSCAD + 11 stability).

### Deprecation / breaking

None. v0.4.0 is fully backward-compatible with v0.3.0.

## [0.3.0] — 2026-05-09

Adds a new GLM family (multinomial / softmax), a new penalty (bridge
`|β|^q`), 28 new adaptive estimators across LS / group / GLM datafits,
and closes two `glmnet` / `survival::coxph` parity gaps (Poisson
offsets, Cox Efron tie handling). 50+ new sklearn estimators total.

### Added

- **Multinomial / softmax classification (M3.6).** New `MultinomialLogit`
  GLM datafit using Böhning's diagonal majorization (matches
  `glmnet(family="multinomial", type.multinomial="grouped")`). 12 new
  sklearn classes — `Multinomial{Lasso,MCP,SCAD,ElasticNet}{Classifier,
  PathClassifier,PathCV}` — with `coef_ (K, p)`, `predict_proba (n, K)`,
  arbitrary-dtype label support (integer / string).
- **Bridge / ℓ_q penalty `λ · Σ_j w_j |β_j|^q`, `q ∈ (0, 1]` (M6.x).**
  New scalar LLA path solver in `crates/skein-core/src/solver/path_lla.rs`
  + `surrogate_weights_bridge` helper. 3 sklearn classes
  (`BridgeRegressor`, `BridgePathRegressor`, `BridgePathCV`); closes a
  `grpreg` parity gap.
- **SparseGroupSCAD end-to-end (M6.x).** The M2.7 surrogate helper is
  now wired through PyO3 + sklearn estimators across LS + 3 GLMs. 12
  new classes: `{,Logistic,Poisson,Cox}SparseGroupSCAD{Regressor,
  PathRegressor,PathCV}`.
- **Adaptive {Lasso, MCP, SCAD} two-stage estimators (M6.x).** 28 new
  sklearn classes total:
  - 6 LS scalar (`Adaptive{Lasso,MCP,SCAD}{PathRegressor,PathCV}`).
  - 4 LS group (`AdaptiveGroup{Lasso,MCP}{PathRegressor,PathCV}` —
    plain `GroupSCAD` deferred).
  - 18 GLM (`Adaptive{Logistic,Poisson,Cox}{Lasso,MCP,SCAD}{Path,
    PathCV}`). Pure Python composition; the per-feature `weights=`
    parameter does the work. `coef_pilot_` and `weights_` exposed
    for inspection.
- **Poisson offsets (M3.x).** `PoissonLog::with_offset(y, offset)`
  threads a per-sample log-exposure through `surrogate_at` and
  `loss`. Every Poisson estimator (14 + 7 PathCV) accepts
  `offset=None`; CV slices `offset[train_idx]` per fold. Standard
  rate-model use case.
- **Cox Efron ties (M3.x).** New `TieHandling::{Breslow, Efron}` enum;
  `CoxPH::with_ties(time, event, ties)`. `loss` and the cumulative-
  hazard accumulation handle Efron's per-event reduced risk set
  `S_eff_i = S(t) − (i/k)·S_D(t)`. Reduces exactly to Breslow when
  no ties. Every Cox estimator (14 + 7 PathCV) accepts
  `ties="breslow"|"efron"`. Default stays Breslow for back-compat;
  matches `glmnet(family="cox", ties="efron")` and is more accurate
  when ties are heavy (R `survival::coxph`'s default).

### Changed

- `_glm_dispatch_inputs` reads `getattr(estimator, 'offset', None)`
  to thread Poisson offsets through the existing PyO3 path; logistic
  / Cox are unaffected (no `offset` attr).
- `_PathCVMixin.fit` slices `offset[train_idx]` per fold for any
  estimator carrying an `offset` attribute.
- `CoxPH::new(time, event)` still works — defaults to Breslow ties.

### Estimator counts (cumulative)

108 estimators in v0.2.0 → **136 estimators in v0.3.0**, 51 → **58**
`*PathCV` cross-validation wrappers. Every datafit × penalty
combination is wired end-to-end with sklearn-style `fit` / `predict` /
`predict_proba` / `score`.

### Tests

**265 cargo + 261 pytest, all green.**
- v0.2.0 baseline: 254 cargo + 197 pytest.
- New tests this release: +11 cargo (multinomial 8 + Poisson offset 4
  + Cox Efron 4 — minus one shared) and +64 pytest spanning every new
  family.

### Deprecation / breaking

None. v0.3.0 is fully backward-compatible with v0.2.0; the
`CoxPH::new` constructor still defaults to Breslow ties, every
existing PyO3 entry preserves its v0.2.0 keyword surface, and the
new `offset=None` and `ties="breslow"` defaults match v0.2.0
behavior on the GLM path.

### Roadmap status

- M3 — GLM datafits: multinomial done, Poisson offsets done, Cox Efron
  done. M3.7 (negative binomial / Huber / quantile) is the only
  remaining open item.
- M6 — Penalty zoo: SparseGroupSCAD, bridge, adaptive (scalar / group /
  GLM) done. Overlapping group lasso, fused lasso, constrained
  variants pending.

## [0.2.0] — earlier release

Initial public release. Multi-task LS, sparse + dense + mmap +
chunked backends, full GLM coverage (binomial, Poisson, Cox Breslow),
group penalty zoo, CV + IC selection, R numerical regression suite.
See `ROADMAP.md` for the milestone breakdown.
