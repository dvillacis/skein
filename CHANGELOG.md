# Changelog

All notable changes to `skein-glm` are recorded here. The project follows
semantic versioning, with the pre-1.0 minor-bump-on-feature policy
documented in `docs/extending/rust-api.md`.

## [Unreleased]

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
