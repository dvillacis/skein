# skein Roadmap (post-v1.0)

v1.0.0 (2026-05-20) froze the public API surface. The library is
feature-complete relative to its original niche: nonconvex
structured-sparse models with first-class weight axes across LS,
GLM, multi-task, multinomial, Cox, and graphical-lasso families.

This roadmap is the **forward plan for v1.x**. The throughline is
hardening and performance — making what already exists faster, more
robust, more reproducible, and easier to operate. New algorithmic
surface (penalties, datafits, design backends, inference layers) is
explicitly out of scope; the trait surface that downstream projects
extend is what the v1.0 stability promise is protecting.

The v0.x history (M0–M14, with all the per-milestone evidence and
benchmark snapshots) lives in `ROADMAP_old.md`. Cross-references
below point into it when a v1.x milestone is the closeout of an
open v0.x lever.

## Status snapshot

| Milestone | Theme | Status | Notes |
|-----------|-------|--------|-------|
| H1 — At-scale bench + fixture tier (n ≥ 100k) | Hardening | ✅ infra | infrastructure shipped 2026-05-20: `xlarge` (100k × 10k) in headline matrix for ls_lasso / ls_mcp / logistic_lasso / ls_group_lasso, comparator gap captured in `paper/manifest.json` under `at_scale_comparator_gap`, per-PR `large` canary in bench-smoke, `*_large` R-anchor fixtures (n=5k/p=500 default, env-tunable to 50k/p=2k), `docs/benchmarks/at_scale.md`. Rendered `xlarge` snapshots are maintainer-overnight, not part of this closeout. |
| H2 — Numerical-stability sweep | Hardening | ✅ done | 84 new pytests across four files (`tests/test_numerics_design_pathologies.py`, `test_numerics_extreme_weights.py`, `test_numerics_glm_saturation.py`, `test_numerics_glasso_singular.py`) covering collinear / zero-variance designs, 12-decade sample-weight spreads, zero per-feature and per-group weights, Poisson η near `ETA_CLAMP`, separable & class-imbalanced logistic, heavy-ties Cox under both Breslow and Efron, and rank-deficient single / joint glasso. Surfaced one finding: nonconvex glasso (MCP / SCAD) does not preserve SPD in extreme `n ≪ p` (n=5, p=40 produced min eigvalue −8.8e-3) — documented as an algorithmic property; finiteness + symmetry remain the asserted contract. All tests run under a 30 s wall-clock budget to catch infinite-loop fallback. |
| H3 — Property-based & fuzz tests | Hardening | ✅ done | 30 Rust `proptest`s + 4 Python `hypothesis` tests covering randomized invariants the hand-picked unit fixtures don't reach: sign / antisymmetry / monotonicity / magnitude-non-increase on `soft_threshold` / `elastic_net_prox` / `mcp_prox` / `scad_prox`, large-γ / large-a limit collapse to soft-threshold, group-prox rotation invariance for ℓ₂ group lasso / EN, BinomialLogit / PoissonLog / CoxPH surrogate gradient match against FD-of-loss (with both tie-handlers for Cox), Binomial / Poisson Hessian-diagonal match against the analytical Fisher diagonal, full `destandardize(standardize_β(β)) = β` bijection across every flag combo (center × scale × intercept) including `destandardize_path` and `rescale_weights_for_standardize` penalty preservation, and Python-side `weights = None ↔ ones` bit-equality through the PyO3 boundary plus per-feature permutation equivariance and a positive `sample_weights` no-op detector. Surfaced one architectural quirk (documented in `tests/test_weight_composition.py`): `sample_weights=None` and `sample_weights=ones(n)` take structurally different code paths in `crates/skein-py/src/ls.rs` (centered destandardize vs. augmented intercept column) and converge to the same optimum but along different iterate trajectories, so the identity holds approximately, not bit-exactly. |
| H4 — Reproducibility audit | Hardening | ✅ done | `tests/test_reproducibility.py` pins every public RNG-consuming estimator with paired same-seed + different-seed fits: MCPPathCV / GroupLassoPathCV / LogisticLassoPathCV / AdaptiveLassoPathCV / MultinomialLassoPathCV (KFold-shuffle path), StabilitySelection (bootstrap subsampling), GraphicalStabilitySelection (graph stability), GraphicalBootstrap (CI bounds). Same seed → `np.array_equal` on `coef_` / `cv_scores_` / `selection_probabilities_` / `edge_selection_probabilities_` / `ci_lower_` / `ci_upper_`; different seed → measurable divergence (catches a silent dropped-RNG regression). BLAS-thread caveat documented inline: the Rust path solver itself is deterministic — the reproducibility we assert lives entirely in Python-side fold construction + bootstrap resampling, which is BLAS-thread-independent at the small problem sizes used. |
| P1 — Native sparse-group SCAD block-CD for GLMs | Performance | ✅ done | dropped the LLA layer for logistic / Poisson / Cox sparse-group SCAD (dense + sparse, six PyO3 builders) by routing each closure through `SparseGroupScad::with_coord_weights` instead of `surrogate_sparse_group_scad` + `SparseGroupLasso`. Closes the last LLA-wrapped non-convex group family in the GLM PyO3 surface (sparse-group MCP was already native per M14c.2). All 11 `tests/test_sparse_group_scad.py` cases pass and the broader 605-pytest / 448-cargo-lib suites stay green |
| P2 — Scalar MCP/SCAD path overhead investigation | Performance | ✅ closed (no structural lever) | profiled with `SKEIN_PROFILE_PATH` and a microbench + per-coord-visit attribution harness (`crates/skein-core/examples/{mcp_ls_medium,mcp_vs_lasso_micro,mcp_cd_attribution}.rs`); the medium/deep MCP-vs-Lasso gap localises entirely to `inner_cd` (penalty `prox_coord` is actually *faster* than Lasso's; `value(beta)` adds <2 µs/λ) and the driver is **MCP's bias-correction property creating ~31% more total coord work**: ~20% more inner CD iters per λ (firm-threshold settles slower) and a ~6% larger per-λ working set (warm-started support carries the bias-corrected coefficients forward). Per-coord-visit BLAS cost is identical (axpy 1.18×, grad 1.21× — exactly proportional to visit count). This is the same property that makes MCP useful versus Lasso; the right next attack is P6's uniform inner-CD coord-visit speedup. |
| P3 — Cross-platform BLAS in distributed wheels | Performance | ⏳ in progress | `skein_glm.__build_features__` shipped 2026-05-21 (P3 acceptance criterion — users can introspect what BLAS the wheel was built with); Windows BLAS audit + MKL question still open |
| P4 — Pre-pass gap-safe screening | Performance | ✅ closed (subsumed by existing screening cascade) | Implemented and A/B-measured 2026-05-21; pre-pass screen at λ_k entry reuses the M13.2 `prev_grad` cache to invoke `gap_safe_screen_with_grad` before the first inner-CD pass. At large (n=50k, p=5k) the change registered as a **+11–15 % regression** in both deep and sparse regimes despite identical iter / KKT-pass counts; the cascade of M13.1 saturation bypass + M13.2 priority rule + post-pass `compute_outer_state.safely_inactive` already prunes everything the pre-pass would catch, so the only net effect is the screen's O(p) overhead repeated per λ. Implementation + tests reverted; closeout below records what was tried and why it didn't pay. |
| P5 — M13.1 saturation-threshold tuning | Performance | ✅ closed (0.5 already optimal) | 10-replicate ablation across {0.3, 0.4, 0.5, 0.6, 0.7} × {deep, sparse} on medium (n=10k, p=1k): **0.5 strictly dominates** in both regimes (p25 wall, locale-immune Python parse). Closest competitor is 0.7 sparse at +1.8 % (effectively tied); every other (threshold, regime) is 8–24 % slower than 0.5. M13.1's original calibration validated, not displaceable. The `saturation_threshold()` helper + `SKEIN_SATURATION_THRESHOLD` env-var hook are kept in tree as a permanent ablation surface (sibling to `SKEIN_PROFILE_PATH`), so future maintainers can re-run the sweep if the cascade ever changes shape. |
| P6 — Inner-CD column batching at large n | Performance | ✅ closed (no measurable lever) | M13.6's 1.5× super-linearity does not reproduce on current HEAD — full-path large (n=50k, p=5k) scales 15.79× wall for 25× n×p growth (0.63× factor, sub-linear), and per-coord-visit BLAS cost is *lowest* at large (0.37 ns/elem vs 0.82 at medium — Accelerate amortises call overhead over longer vectors). Both the bandwidth-wall premise and the path-level super-linearity premise are falsified; the M14.x perf work between M13.6 and v1.0 appears to have already closed the gap. |
| O1 — `cargo-semver-checks` in CI | Operability | ✅ done | v1.0 stability promise machine-checked on every PR via `--baseline-rev v1.0.0`; 222 checks vs the freeze surface |
| O2 — `cargo-audit` + `pip-audit` + dependabot | Operability | ✅ done | supply-chain hygiene baseline; weekly cron + per-PR on dep-manifest changes; 1 documented advisory ignore (RUSTSEC-2025-0020, pyo3 unreachable API) |
| O3 — Python 3.13 + NumPy 2.x in CI matrix | Operability | ✅ done | 3.13 added to the `python` job matrix; local pytest on 3.13 + NumPy 2.4.6 green (506 passed). NumPy 2.x was already the resolver default at v1.0 — `numpy>=1.24` floor stays; no API-removal hazards in the Python codebase |
| O4 — Expanded wheel matrix (musllinux + Linux aarch64) | Operability | ⏳ planned | currently `CIBW_SKIP: "*-musllinux_*"`; aarch64 dropped from v0.1.x matrix |
| O5 — `docs/benchmarks/speed.md` consolidation | Operability | ✅ done | One-page headline summary at `docs/benchmarks/speed.md` with full provenance block (host_id, BLAS, skein version, snapshot date, git rev) and per-scenario speedup tables sourced from `paper/tables/T2_headline_timings.md`. Honest about which cells skein loses on (logistic_lasso / poisson_lasso / cox_lasso pre-M13.8 snapshot) and which scenarios have no comparator. Linked into the docs index toctree + `docs/benchmarks/index.md`. |
| O6 — Structured timing / iteration surface | Operability | ✅ done | per-λ wall time surfaced via `info_["times_ns"]` on every path estimator; powered by additive `solve_path_timed` / `solve_block_path_timed` / `prox_newton_*_solve_path_timed` siblings that keep the v1.0 freeze intact |

Test count at v1.0.0: **358 cargo lib + 8 cargo integration + 455
pytest, all green.** Each milestone below either keeps this number
flat (perf work) or grows it (hardening). Current HEAD: **448 cargo
lib + 606 pytest** (post-O5; O5 is docs-only, counts unchanged from P3a).

---

## Hardening

### ✅ H1 — At-scale bench + fixture tier (n ≥ 100k)

**Shipped 2026-05-20 (infrastructure).** The headline matrix, the
per-PR canary, the R-anchor scaffolding, and the documentation
landed together. Rendered `xlarge` snapshots are maintainer-
overnight and do not gate this milestone — H1's contract was
*making it possible to measure at n ≥ 100k*, not generating one
specific snapshot run.

What shipped:

- **`xlarge` (n=100k, p=10k) in `benches/v2/config.yaml` headline**
  for `ls_lasso`, `ls_mcp`, `logistic_lasso`, `ls_group_lasso` —
  five seeds × two regimes per scenario. Cross-package comparators
  kept where they fit (celer, skglm for LS Lasso/MCP); R packages
  + `sklearn.coordinate_descent` dropped at this tier with the
  asymmetry captured in `paper/manifest.json` under
  `at_scale_comparator_gap` so paper figures flag the gap rather
  than silently dropping the comparator.
- **`bench-smoke-at-scale` job** running one `large/sparse` skein-
  only cell under release maturin + OpenBLAS, `--trials 1`, every
  PR. Target ≤15 min wall-clock; emits a workflow warning at >10
  min so budget creep is visible. Existing dev-profile `small`
  canary unchanged.
- **`--trials` CLI override** on `benches/v2/report/_run_cell.py`
  so the smoke job can short-circuit the 5-trial default without
  shipping a separate config.
- **`*_large` R-anchor fixtures** in `tests/fixtures/generate.R` for
  LS + logistic Lasso/MCP (four new optional tests in
  `tests/test_r_regression.py`). Default size n=5k × p=500;
  `SKEIN_FIXTURE_LARGE_N` / `SKEIN_FIXTURE_LARGE_P` env vars let
  maintainers regenerate at the roadmap's aspirational n=50k × p=2k
  on a machine with adequate RAM. Never committed (same pattern as
  M14c.3 mid-tier); CI silently skips when fixtures are absent.
- **`docs/benchmarks/at_scale.md`** as the durable home for the
  tier definitions, comparator asymmetry, reproduction recipe, and
  per-PR canary semantics.

**Not in this closeout** (deferred to a maintainer run):

- Actually generating `xlarge` aggregates and committing them under
  `benches/v2/results/scenarios/`. The matrix is ~10–12 hours of
  wall-clock on one laptop; that's a maintainer-overnight job, not
  a v1.x infra task.
- `xlarge` extension to the rest of the headline scenarios
  (logistic_mcp, poisson_*, cox_*, ls_group_mcp,
  ls_sparse_group_mcp). The four scenarios picked here are the
  H1 list per the original deliverable; extending the rest is a
  natural follow-up but not in scope.

**Carries forward**: M12 P1, M9.4.

`benches/results/` (v1 harness) and `benches/v2/results/` both stop
at `medium` (n=10k, p=1k) for headline scenarios; M13.6 used a
one-off `lasso_ls_scaling` example to characterize the n=50k,
p=5k memory-bandwidth wall, but those numbers are not in the suite
and not under regression watch. Without large-n snapshots, perf
regressions at the size that matters to users are invisible.

Deliverable:

- `large` cells (n=100k, p=10k) added to `benches/v2/config.yaml`
  for at least Lasso/LS, MCP/LS, Logistic Lasso, Group Lasso. Five
  seeds per cell, BLAS build only.
- Cross-package comparators kept where they fit in memory; for cells
  where comparators OOM, snapshot skein alone and note the
  asymmetry in `paper/manifest.json`.
- `tests/fixtures/generate.R` extended with at-scale R-anchor cells
  (n=5000, p=500 is the current upper bound; bump to n=50k, p=2k for
  the LS + logistic Lasso/MCP families). Gating: parity must hold at
  the at-scale tier or the build fails.
- One short `docs/benchmarks/at_scale.md` page so the cells have a
  durable home.

Acceptance: bench-smoke runs one `large` cell per PR; the rest are
maintainer-driven overnight.

Risks: bench-smoke wall-clock budget. Cap the per-PR cell so a green
PR still completes in under 15 minutes.

### ✅ H2 — Numerical-stability sweep

**Shipped 2026-05-20.** Four pytest files (84 tests, ~9 s combined)
cover the regimes that bit us historically (M12 R4, M14d W_FLOOR,
M14e v-scaled prox) and that earlier well-conditioned synthetics
missed.

Each test asserts (a) all coefficients along the path are finite,
(b) a linear prediction on the training matrix is finite, and (c)
the fit completes inside a 30 s wall-clock budget — the budget
exists specifically to catch the infinite KKT loop that
`gap < tol²` once produced. Coverage:

- **`tests/test_numerics_design_pathologies.py`** (41 tests).
  Collinear columns at ε ∈ {0, 1e-12, 1e-8} for LS lasso / MCP /
  SCAD / EN, group lasso / group MCP / sparse-group lasso, and
  logistic / Poisson lasso. Constant columns (both value=1.0 and
  value=0.0) with and without `standardize=True`. Explicit
  zero-variance × per-feature-weight rescale audit and a zero
  per-feature-weight regression test.
- **`tests/test_numerics_extreme_weights.py`** (20 tests).
  `sample_weights` spanning 12 decades across scalar LS / logistic /
  Poisson path estimators (groups don't accept `sample_weights`,
  noted). Zero `sample_weights` rows. Zero per-feature weights
  (asserts the unpenalized feature stays nonzero across the path,
  with and without `standardize`). Zero per-group weights for group
  lasso, sparse-group lasso, and group MCP.
- **`tests/test_numerics_glm_saturation.py`** (11 tests). Poisson η
  driven against `ETA_CLAMP` (paired with `y ~ Poisson(μ_clamped)`
  to avoid unfeasible targets) for both lasso and MCP, plus a
  large-counts variant. Logistic on linearly separable data, a
  95%-separable variant, and 1-positive-vs-999-negatives class
  imbalance. Cox heavy ties (`n_unique_times ∈ {2, 3}`) under both
  Breslow and Efron, plus the pathological "all events at one time".
- **`tests/test_numerics_glasso_singular.py`** (12 tests). `n=20,
  p=50` and `n=5, p=40` rank-deficit. `diag_offset=0` removing the
  safety ridge. Duplicated / near-duplicated / constant variables.
  Precomputed rank-deficient covariance. Joint glasso (group form
  + MCP) with per-population rank deficit.

**Surfaced finding.** Nonconvex glasso (MCP / SCAD) does not
preserve SPD across iterations the way L1 glasso does — at extreme
rank deficit (`n=5, p=40`) the released-shrinkage region pushed the
smallest eigenvalue to −8.8e-3. The block-CD inner solver does not
project iterates back to the SPD cone; the L1 piece + `diag_offset`
do that for L1 but the MCP / SCAD tail can flip the gradient sign.
We document this as an algorithmic property of nonconvex glasso
rather than a regression — the H2 contract is finiteness +
symmetry, and that still holds. (The L1 SPD check is now a
separate, stricter test on `GraphicalLasso`.)

Roll-up: test count moves from **506 pytest pre-H2 to 593
(`pytest tests/`, 9 skipped, all unrelated)**.

### ✅ H3 — Property-based & fuzz tests on prox / surrogate

Closure of the C1/C2 randomized-coverage gap M12 left open. `proptest`
in Rust covers the in-tree numerical contracts; `hypothesis` in Python
covers what crosses the PyO3 boundary.

**Rust (`proptest`, in `crates/skein-core`, dev-only dep):**

- `src/prox.rs` — 22 properties on `soft_threshold`, `elastic_net_prox`,
  `mcp_prox`, `scad_prox`, `group_soft_threshold`, `group_elastic_net_prox`:
  sign preservation, antisymmetry, zero fixed-point, monotonicity in
  `z`, magnitude non-increase; large-γ and large-a limit collapse to
  `soft_threshold`; 2D rotation invariance of the group prox.
- `src/datafit/surrogate_proptests.rs` — 5 properties on the GLM
  surrogates. BinomialLogit / PoissonLog / CoxPH (both tie-handlers):
  surrogate's `coord_grad` at β matches central-FD of `loss(β)`.
  BinomialLogit / PoissonLog (with optional sample-weights / offset):
  surrogate's `coord_lipschitz` matches the analytical Fisher Hessian
  diagonal. Cox's diagonal-IRLS is approximate by construction so the
  Lipschitz identity is not asserted.
- `src/standardize.rs` — 3 properties: `destandardize(β · s) = β` for
  every (center_x × scale_x × fit_intercept) flag combo plus the
  documented intercept formula, `destandardize_path` agrees with
  per-row `destandardize`, and `rescale_weights_for_standardize`
  preserves the L1 penalty value under the standardized-space lift.

**Python (`hypothesis`, dev-only dep):**

- `tests/test_weight_composition.py` — 4 properties through the public
  estimators: `weights=None ≡ weights=ones(p)` for MCPPathRegressor
  (bit-equal — both reach the same internal `Array1::ones(p)`),
  per-group `weights=None ≡ weights=ones(n_groups)` for
  GroupLassoPathRegressor, per-feature column-permutation equivariance
  for MCPPathRegressor at tight tol, and a positive non-uniform
  `sample_weights` no-op detector. Inputs are derived from a
  hypothesis-drawn RNG seed (X / y aren't fuzzed element-wise — the
  invariances are bit-equality assertions strengthened by repeated
  runs, not by pathological draws).

One architectural finding documented inline in the Python module
docstring: `sample_weights=None` and `sample_weights=ones(n)` take
structurally different code paths in `crates/skein-py/src/ls.rs` — the
no-weights path centers via `standardize`/`destandardize_path`; the
explicit path uses an augmented intercept column. Both formulations
target the same penalised LS objective and converge to the same
optimum, but their iterate trajectories and λ-grids differ, so this is
*not* a bit-equality invariance and we don't test it as one.

### ✅ H4 — Reproducibility audit

Every public estimator that consumes an RNG is pinned in
`tests/test_reproducibility.py` with paired same-seed + different-seed
fits — same seed asserts `np.array_equal` bit-identity on the natural
state-vector (CV coefs + scores, stability probabilities, bootstrap CI
bounds); different seed asserts a measurable divergence so a silent
"`random_state` parsed but never reaches the RNG consumer" regression
fails immediately.

Coverage by RNG-consumer family (8 tests):

- **CV KFold-shuffle path** — `MCPPathCV`, `GroupLassoPathCV`,
  `LogisticLassoPathCV` (parametrized representatives of the
  `_PathCVMixin` family, which all dispatch through the same
  `KFold(shuffle=True, random_state=…)` call site).
- **Stability selection** — `StabilitySelection` with `MCPPathRegressor`
  base, exercising the bootstrap-subsampling RNG.
- **Graphical stability + bootstrap** — `GraphicalStabilitySelection`
  and `GraphicalBootstrap` with `GraphicalLasso` base, covering the
  graph-side analogues.
- **Nested CV** — `AdaptiveLassoPathCV` (pilot + refit each consume
  the same `random_state`).
- **Multinomial CV** — `MultinomialLassoPathCV`'s separate
  `_MultinomialPathCVBase` code path.

BLAS-thread caveat documented inline in the test docstring: the Rust
path solver itself has no RNG (coordinate descent is deterministic
from `β=0`), so all reproducibility-relevant randomness is in
Python-side fold / bootstrap construction, which is unaffected by
hardware-BLAS thread scheduling. At the small problem sizes used
(n=40, p=8) BLAS stays single-threaded anyway; if future tests scale
beyond that regime they should gate with `OMP_NUM_THREADS=1` /
`OPENBLAS_NUM_THREADS=1`.

---

## Performance

### ✅ P1 — Native sparse-group SCAD block-CD for GLMs

**Shipped 2026-05-20.** Closes out the last LLA-wrapped non-convex
group family on the GLM PyO3 surface. Sibling of **M13.4c** (native
group-MCP BCD for logistic / Poisson / Cox) and **M14c.2** (native
sparse-group MCP — the MCP side of this work was already done in
M14c.2; the original P1 entry mislabelled what was left). The native
`SparseGroupScad` penalty itself shipped in M14h alongside the four
LS PyO3 swaps; the six GLM swaps are what this milestone landed.

What shipped:

- `crates/skein-py/src/glm.rs` — all six sparse-group SCAD builders
  (`solve_{logistic,poisson,cox}_sparse_group_scad_path` and their
  `_sparse` counterparts) now build `SparseGroupScad::with_coord_weights`
  directly inside the prox-Newton `make_inner` closure, mirroring the
  M14c.2 pattern for sparse-group MCP. The closures are β-independent
  (the LLA β-iterate is no longer needed).
- `surrogate_sparse_group_scad` is dropped from the `glm.rs` import
  list; the function itself remains in `skein-core` (v1.0 stable
  surface) for downstream users who still want the LLA surrogate.
- `SparseGroupScad` added to the `penalty::` import list.

Validation: 11 / 11 `tests/test_sparse_group_scad.py` cases pass —
covers LS shape / recovery / dense-sparse equivalence / `a < 2`
rejection / `a → ∞` limit-to-sparse-group-lasso / path-CV, plus
logistic predict-proba smoke + dense-sparse equivalence, Poisson
smoke, Cox smoke, and GLM `a < 2` rejection. Full suite stays at 448
cargo lib + 605 pytest, all green.

### ✅ P2 — Scalar MCP/SCAD path overhead investigation

**Closed 2026-05-20 with a "no structural lever" outcome.** The
gap localises to inner-CD coord-visit count, which is a direct
consequence of MCP's bias-correction property — the same property
that motivates choosing MCP over Lasso in the first place. P6's
column-batching is the right next attack.

Background: the initial framing (carried from v0.x M13.5) was that
scalar MCP/SCAD pays "LLA outer wrapper cost". That turned out to
be wrong — scalar MCP/SCAD call `solve_path` directly with the
`Mcp`/`Scad` closed-form prox; M14c.1's short-circuit already
covered the LLA-side surface (bridge / adaptive / multitask) and
nothing was deferred.

The `medium/deep` snapshot still shows a real gap:

| cell          | Lasso | MCP   | MCP/Lasso | EN/Lasso |
|---------------|------:|------:|----------:|---------:|
| medium/deep   | 1.13s | 1.70s | **1.50×** | 1.24×    |
| medium/sparse | 0.37s | 0.46s | 1.24×     | 1.01×    |

EN-vs-Lasso accounts for the generic non-trivial-prox cost; the
MCP-specific excess on top is ≈1.21× on deep, ≈1.24× on sparse.

**Investigation artifacts** (kept as reusable profiling tools for
P6 and future perf work):

- `crates/skein-core/examples/mcp_ls_medium.rs` — MCP sibling of
  `lasso_ls_medium.rs`. Runs four cells (lasso/deep, mcp/deep,
  lasso/sparse, mcp/sparse), emits the `SKEIN_PROFILE_PATH` phase
  breakdown per cell, prints per-λ working-set distribution + a
  Σ ws × iter coord-work proxy.
- `crates/skein-core/examples/mcp_vs_lasso_micro.rs` — isolates
  `prox_coord` and `value(beta)` so we can attribute the inner-CD
  gap to penalty-side virtual calls vs design-side BLAS.
- `crates/skein-core/examples/mcp_cd_attribution.rs` — re-implements
  the `cd_solve_subset` loop in-line so we can count nonzero updates
  per sweep and time `coord_grad` / `col_axpy` / `value(r)` separately.

**Findings** (host = macOS arm64 + Accelerate):

Phase log (medium/deep, 100 λs, 1 warm-up + 1 measured fit):

```
              Lasso        MCP         Δ
  setup       0.15 ms      0.11 ms     ~0
  screening   5.2 ms       5.2 ms      ~0
  lipschitz   0.36 ms      0.39 ms     ~0
  inner_cd    697 ms       920 ms      +223 ms   ← gap is here
  dual_extrap 0.00 ms      0.00 ms     ~0
  outer_state 143 ms       143 ms      ~0
  bookkeeping 0.00 ms      0.00 ms     ~0
```

Microbench (80k prox calls + 400 value calls):

```
  prox_coord  Lasso   6.5 ns/call   MCP   3.1 ns/call   (MCP faster — Lasso routes through ElasticNet with α=1)
  value(beta) Lasso  1.9 µs/call    MCP   2.4 µs/call   (MCP slower, but <2 µs/λ total)
```

Per-λ projected penalty cost: ~25 µs for Lasso, ~17 µs for MCP.
Observed inner_cd gap: **2230 µs / λ**. So `prox_coord` and
`value(beta)` together account for <2 % of the gap.

Single-λ inner-CD attribution at λ_max × 1e-3, cold-start, full
1000-feature sweep:

```
              iters   visits   nz_updates    grad      axpy
  Lasso       10      10000    8701 (87.0%)  25.2 ms   17.9 ms
  MCP         12      12000   10323 (86.0%)  30.6 ms   21.1 ms
              ratio   1.20×   1.20×          1.21×     1.18×
```

Per-coord-visit BLAS cost is essentially identical; MCP just needs
~20 % more sweeps to converge from cold start.

Path-level `working_set_sizes` × `iters` coord-work proxy:

```
  Lasso deep:   91,967 coord-visits   (1.395 s wall)
  MCP   deep:  120,648 coord-visits   (1.155 s wall)   1.31× more work
  Lasso sparse:  2,636
  MCP   sparse:  4,866                                  1.85× more work (tiny absolute)
```

The 1.31× coord-work ratio matches the inner_cd wall ratio (1.32×)
almost exactly. So the per-iter wall difference is entirely
explained by **more total coord work**, not higher per-coord cost.

**Why this is structural.** MCP's firm-threshold de-biases
moderate-|β| coordinates that Lasso shrinks aggressively. Two
downstream consequences:

1. **Larger warm-started support.** Once a coord activates on the
   MCP path it tends to stay active (firm-threshold caps shrinkage
   at γλ then flattens). The priority rule sizes the WS as
   `(n_support × 2).max(p0).min(p)`, so MCP's larger warm-started
   support yields a ~6 % larger average WS per λ.
2. **More inner CD sweeps to settle.** Soft-threshold reaches a
   coord's optimum in one shot when far from the boundary; firm-
   threshold's stair-step rescaling needs more passes to converge
   to coefficient-space tol.

Both factors compound multiplicatively. Combined: ~20 % more iters
× ~6 % bigger WS ≈ 27–31 % more total coord work, which is exactly
what we measure.

**Levers considered and rejected:**

- *Tightening the WS-growth factor* (replace `n_support × 2` with
  a smaller multiplier for MCP). Cuts WS but pushes work onto more
  outer KKT passes; the existing `outer.violators` mechanism then
  re-adds features one-at-a-time. Net effect: not a free win, and
  the same factor governs the strong-rule WS sizing for every
  separable penalty — regressing Lasso/EN to win on MCP is a bad
  trade.
- *MCP-aware coord ordering* (visit large-|β| coords first since
  they're more likely to be at firm-saturation and need no update).
  Speculative, and re-ordering breaks the inner-CD cyclic-convergence
  argument; not worth a load-bearing implementation.

**Why P6 is the right next step.** The remaining cost lives in
`cd_solve_subset`'s `O(|ws| · n)` per-sweep BLAS work. P6's
inner-CD column batching (process multiple coords per X-column
scan) speeds up the coord-visit path uniformly, regardless of
penalty. MCP has more visits per fit, so it benefits proportionally
more from a P6 fix — and P6's structural change is what M10.I
already flagged in the v0.x roadmap.

### P3 — Cross-platform BLAS in distributed wheels (in progress)

**Carries M10.G forward.** Two-part milestone: (a) introspection
surface so users can tell what BLAS their wheel ships with, and
(b) actually wire BLAS on Windows.

**(a) `__build_features__` — shipped 2026-05-21.** New
`pub fn build_features() -> Vec<&'static str>` in
`crates/skein-py/src/lib.rs` returns the active BLAS feature flags
determined at compile time via `cfg!(feature = "blas-*")`. Wired
through to Python as `skein_glm.__build_features__: tuple[str, ...]`
in `python/skein_glm/__init__.py`. Empty tuple ⇒ no hardware BLAS
(ndarray's pure-Rust `matrixmultiply` fallback, ~3× slower on the
inner-CD hot path). Backed by `tests/test_smoke.py::test_build_features_attribute_shape`
pinning the type contract; the `_core.pyi` stub teaches mypy about
the new function. P3 acceptance criterion as written
(`skein_glm.__build_features__` exposed) is now satisfied.

**(b) Windows BLAS wiring — still open.** Current state:

- macOS arm64 wheels: `blas-accelerate` (✓).
- Linux x86_64 wheels: `blas-openblas` via manylinux2014 OpenBLAS
  package (✓ — wired in `.github/workflows/wheels.yml`).
- **Windows wheels: no BLAS feature enabled.** The Cython-grade
  matvec gap is therefore largest on Windows. After (a),
  `skein_glm.__build_features__ == ()` on a Windows wheel makes
  the gap user-visible.
- **MKL feature unwired.** Listed in the v0.x roadmap as an option;
  not built, not exposed.

Remaining deliverable:

- Audit Windows wheel for whether `blas-openblas` (`vcpkg openblas`)
  is reachable in cibuildwheel; if so, wire it. Otherwise document
  the gap. Three plausible paths to try in order: (1) `vcpkg install
  openblas:x64-windows` + `OPENBLAS_DIR` env hint for `openblas-src/system`;
  (2) `openblas-src/static` (builds OpenBLAS from source — slow CI but
  reliable); (3) `intel-mkl-src` prebuilt-binary download as a `blas-mkl`
  opt-in feature. (3) doubles as the MKL question — only worth wiring
  if (1) and (2) both fail.

### ✅ P4 — Pre-pass gap-safe screening

**Closed 2026-05-21 with a "subsumed by existing screening cascade"
outcome.** Third milestone this week to close as a no-lever after
implementation + measurement (P2, P6, P4); same pattern of carrying
forward a roadmap premise that intervening v0.x work had already
addressed via a different mechanism.

Background framing (now stale): F-series shipped post-pass
gap-safe screening (`compute_outer_state.safely_inactive`, fires
inside the KKT loop after each inner CD pass). celer's documented
pattern is pre-pass — at λ_k entry, use the cached gradient from
λ_{k-1}'s converged state to prune the priority working set
*before* the first inner-CD pass instead of waiting one CD pass.
The proposed lever was to add this pre-pass step to
`Screening::Strong`, intersecting the priority WS with the
gap-safe inactive set, using the M13.2 `prev_grad` cache so the
screen costs only an O(p) dual-feasibility sweep rather than an
extra O(np) matvec.

**What was implemented and reverted.** A two-stage refactor of
`crates/skein-core/src/solver/path.rs`: (1) `gap_safe_screen` split
into `gap_safe_screen_with_grad` (takes precomputed gradient) plus
a thin wrapper that recomputes the gradient for the existing
`Screening::GapSafe` callers; (2) a pre-pass block in the
`Strong` branch that — when `prev_grad.is_some()`, the penalty has
a lasso-form dual gap (`pen.has_lasso_form_dual_gap()`), and the
datafit is unweighted — calls the helper at λ_k entry and seeds
the per-λ `screened` mask with the resulting inactive set. Three
new tests pinned soundness: Lasso ↔ `Screening::Off` equivalence
at `tol=1e-12`, ElasticNet (α=0.5) ↔ `Off` equivalence, and a
cold-start no-op assertion (pre-pass is gated out when
`prev_grad.is_none()` at k=0). `SKEIN_DISABLE_PRE_PASS=1` was
added as an A/B kill switch.

**Measurements** (medians of 3-5 runs per condition, A/B from the
same release binary, host = macOS arm64 + Accelerate):

| cell                 | P4 on    | P4 off   | delta            |
|----------------------|---------:|---------:|:-----------------|
| small (1k × 200) / deep   |    44.2 ms |    44.5 ms | −0.7 % (noise)         |
| small (1k × 200) / sparse |    10.3 ms |    11.3 ms | −8.9 %                 |
| medium (10k × 1k) / deep   |    1.57 s  |    1.52 s  | +3.0 % (noise band)    |
| medium (10k × 1k) / sparse |     616 ms |     618 ms | −0.3 % (noise)         |
| **large (50k × 5k) / deep**   | **27.55 s**  | **24.77 s**  | **+11.2 % regression** |
| **large (50k × 5k) / sparse** | **17.46 s**  | **15.19 s**  | **+14.9 % regression** |

Iter counts and KKT-pass counts are **identical** between P4 on
and P4 off across every cell. The pre-pass screen is changing
which features land in the first-pass WS, but not the
convergence trajectory — same total work, just rearranged.

Phase breakdown on large/sparse with `SKEIN_PROFILE_PATH=1`
isolates where the regression lives:

| phase             | P4 on    | P4 off   | delta   |
|-------------------|---------:|---------:|:--------|
| screening (init WS) | 156 ms   | 131 ms   | +19 %   (expected — pre-pass O(p) cost) |
| inner_cd          | 10626 ms | 9544 ms  | **+11 %** (~+1 s)                          |
| outer_state (KKT) | 4188 ms  | 3788 ms  | **+11 %** (~+0.4 s)                        |

Both `inner_cd` and `outer_state` grow proportionally by ~11 %.
Pre-pass overhead alone is ~25 ms; nowhere near the +1.5 s
tracked delta. Same KKT-pass count, same total work counted in
iterations, yet wall is consistently higher — most likely
memory-layout / allocation interactions from carrying the larger
`screened` mask through the loop, but not localised by the
existing phase counters.

**Why the cascade was already complete.** Three earlier
milestones, taken together, leave essentially no work for a
pre-pass screen to do:

1. **M13.1 saturation bypass** (50 % active-density threshold)
   routes the entire dense tail through `Screening::Off`, so the
   second half of the deep-regime path never sees screening at
   all — pre-pass included.
2. **M13.2 prev_grad cache** feeds the priority rule with the
   exact KKT-violation gradient at the warm-start residual, so
   the rule's top-`max(p0, 2·|support|)` features are already a
   gradient-driven candidate set, not a generic correlation
   sample.
3. **Post-pass `safely_inactive`** in `compute_outer_state` runs
   gap-safe sphere screening on the WS at the FIRST KKT
   verification — one pass *after* what P4 would do, but with a
   tighter gap (the inner CD has run; the gap evaluated at the
   converged inner-iterate is much smaller than the gap evaluated
   at the warm-start residual). The tighter gap prunes more
   aggressively than P4's pre-pass ever could.

P4's pre-pass at λ_k entry uses the *loose* gap from the
warm-start residual; the post-pass at the same λ uses the
*tight* gap from the just-converged inner iterate. The latter
strictly dominates, and the cost of running both is the
overhead we measured.

**Investigation artifacts.** The `lasso_ls_scaling.rs` example
already in tree was the regression detector; the
`SKEIN_DISABLE_PRE_PASS=1` kill switch was reverted with the rest
of the implementation. No new examples or test files survive the
revert. Closeout itself is in this entry and `git log` (the
implementation branch lived locally, never landed on `main`).

**What this implies for the v1.x perf queue.** Same conclusion
the P6 closeout reached from a different angle: skein's screening
machinery is mature enough that further pre-pass / mid-pass
tweaks face diminishing returns. The remaining open perf
candidates — P3 (Windows BLAS) and P5 (saturation-threshold
tuning) — both target different surfaces (wheel build, scalar
hyperparameter) and aren't subject to the same cascade
saturation. P5 in particular is a cheap ablation that adjusts
how often the M13.1 bypass fires, which is closer to a
threshold-tuning experiment than a structural change.

### ✅ P5 — M13.1 saturation-threshold tuning

**Closed 2026-05-21 with a "0.5 already optimal" outcome.** Different
flavor of "no lever" from P2 / P6 / P4: those closed because the
proposed change couldn't deliver a measurable win; P5 closes
because the *current* parameter value is the optimum of the
ablation grid — moving it in either direction strictly regresses
medium-cell wall in both regimes.

Background framing (now resolved): M13.1 shipped at
`SCREENING_SATURATION_THRESHOLD = 0.5` — chosen "conservatively"
because the original M13.1 measurement only checked threshold = 0.5
vs the alternatives of full Strong (no bypass) or full Off (no
screening). The roadmap entry conjectured that lower thresholds
(0.3) might recover more of the off-vs-strong gap on deep
regimes; higher thresholds might be safer on borderline-saturated
cells. P5 actually measured the trade-off across a 5-point grid.

**What was implemented and kept.** A small refactor of
`crates/skein-core/src/solver/path.rs` that replaces the
compile-time `const SCREENING_SATURATION_THRESHOLD: f64 = 0.5`
with a `DEFAULT_SATURATION_THRESHOLD` const plus a
`pub(crate) fn saturation_threshold()` helper that reads
`SKEIN_SATURATION_THRESHOLD` (validated as a float in `[0, 1]`)
and falls back to the default on absent / invalid input. The two
GLM sibling constants (`PN_SCREENING_SATURATION_THRESHOLD` in
`prox_newton.rs` and `BLOCK_PN_SCREENING_SATURATION_THRESHOLD` in
`prox_newton_block.rs`) are removed in favour of importing the
shared helper, so all three solver entrypoints (LS path, GLM
prox-Newton, group prox-Newton) respect the same env-var override.
This is kept in tree as a permanent ablation hook — same pattern
as `SKEIN_PROFILE_PATH` — so future maintainers can re-run the
sweep if the screening cascade ever changes shape.

The `lasso_ls_scaling.rs` example also picked up a
`SKEIN_REGIME=sparse` env var (5e-2 λ_min/λ_max) so the sweep can
hit both the dense-tail and support-recovery regimes from the
same binary.

**Measurement.** 10 replicates per (threshold × regime) at medium
(n=10k, p=1k) on the same release binary, locale-immune Python
parse of the example's `fit in <Duration>` lines (the prior
locale=es_MX bash/awk pipeline truncated decimal-point values to
integer seconds — corrected before drawing conclusions). p25
medium wall (lower quartile, robust to occasional load spikes):

| threshold | deep medium    | sparse medium   |
|-----------|---------------:|----------------:|
| 0.3       | 1709 ms (+19.7 %) | 732 ms (+22.2 %) |
| 0.4       | 1755 ms (+23.0 %) | 647 ms ( +8.0 %) |
| **0.5**   | **1427 ms (best)**   | **599 ms (best)**   |
| 0.6       | 1624 ms (+13.8 %) | 690 ms (+15.3 %) |
| 0.7       | 1763 ms (+23.6 %) | 610 ms ( +1.8 %) |

0.5 is the **strict optimum** in both regimes; every other point
on the grid is slower by p25 wall. The closest competitor is 0.7
on sparse (+1.8 %, effectively tied — well within the per-run
variance), but it still doesn't beat the default; in deep regime
0.7 is +23.6 % worse, which kills it as a global pick.

**Why 0.5 happens to be optimal.** The threshold trades inner-CD
work against KKT-verifier work, and the two costs cross at
roughly equal active-density. Lower threshold → bypass fires
earlier → more λs do full-feature sweeps (cheaper per-λ inner CD
in absolute terms once the active set is dense, but more total
work over the path). Higher threshold → bypass fires later →
more λs run Strong screening (fewer features per CD pass, but
more KKT-verifier rounds and more screen-then-re-add overhead).
The crossover at 0.5 reflects skein's specific per-coord BLAS
cost / KKT-verifier cost ratio under Accelerate; a different
BLAS or a structural change to either phase would shift the
optimum, which is why the env-var hook stays.

**What was NOT measured** (deferred to a maintainer cell if ever
needed):

- Large cells (n=50k, p=5k) at the full grid. ~25 s per fit ×
  50 conditions = ~20 min wall; expected to track medium given
  the same scaling story, but not confirmed.
- GLM prox-Newton paths (logistic / Poisson / Cox). The helper
  refactor wires their saturation bypasses to the same env var,
  but the threshold was only measured at the LS surface. If a
  future investigation finds the GLM optimum differs, the
  natural fix is a per-solver default (not a single shared
  constant).

### ✅ P6 — Inner-CD column batching at large n

**Closed 2026-05-21 with a "no measurable lever" outcome.** Same
disposition as P2: the original framing carried an M13.6
measurement forward without re-validating against current HEAD;
once the gates (H1 + P1 + P2) cleared and we re-measured, the
super-linearity the milestone was meant to attack had already been
closed by the M14.x perf work that landed between M13.6 and v1.0.
There is no structural lever to ship.

Background framing (now stale): M13.6 reportedly saw a 37.6× wall
for the medium → large (n=50k, p=5k) transition vs a 25× n×p
growth — i.e. 1.5× super-linear — and attributed the excess to a
column-streaming bandwidth wall as per-column vectors exceed L2.
The proposed lever was to process multiple coords per X-column
scan, structurally reworking `cd_solve_subset` (the M10.I
"Cython-grade rewrite" parked back in v0.x).

**Investigation artifacts** (kept as reusable scaling-attribution
tools for future perf work):

- `crates/skein-core/examples/lasso_cd_attribution_scaling.rs` —
  re-implements the `cd_solve_subset` inner loop with per-call
  `Instant` timers and runs the canonical small / medium / large
  cells. Reports per-element ns for `col_dot` (gradient) and
  `col_axpy` separately so the bandwidth premise — whether per-call
  cost spikes as the column outgrows cache — is directly testable.
- `crates/skein-core/examples/lasso_ls_scaling.rs` already shipped
  with M13.6; re-run with `SKEIN_PROFILE_PATH=1 SKEIN_SCALING_LARGE=1`
  produces the path-level phase breakdown reported below.

**Findings** (host = macOS arm64 + Accelerate, 2026-05-21):

Single-sweep cold-start at λ_max × 1e-3, full feature set:

```
              wall      visits   grad ns/elem   axpy ns/elem
  small         4.8 ms    3600    0.583          0.466
  medium      178   ms   11000    0.818          0.771
  large      1797   ms   50000    0.365          0.392
```

Per-element column-streaming cost is *lowest* at large, not
highest — BLAS amortises call overhead over the longer vectors,
and the column read is at peak memory throughput regardless of
whether it fits in L2. No headroom to recover.

Full-path scaling (100 λs, warm-start, `SKEIN_PROFILE_PATH=1`):

```
                  total wall   wall ratio   n×p ratio   factor
  small            44.85 ms       —           —          —
  medium            1.43 s     31.92×       50×        0.64×   sub-linear
  large            22.61 s     15.79×       25×        0.63×   sub-linear
```

Both transitions are strongly sub-linear in n×p. M13.6's 37.6× ÷
25× = 1.504× super-linear reading is replaced by today's 15.79× ÷
25× = 0.63× sub-linear. The reduction came in over the v0.x M14.x
window without an attributable single commit (the perf work was
spread across several milestones — native penalty BCD swaps,
`weighted_col_sq_norms` batching, BLAS hot-path tightening).

At large, the per-phase share also shifts: `inner_cd` drops from
94.7 % (small) → 89.1 % (medium) → 79.2 % (large) of tracked time,
with `outer_state (KKT)` taking the relative weight (5.3 % → 10.4 %
→ 20.2 %). KKT per-pass cost scales 30.3× medium→large (vs 25×
n×p, mildly super-linear at 1.21×) but on the smaller phase the
net path total stays sub-linear.

**Levers considered and rejected:**

- *Fused `col_dot + col_axpy` in a single column pass.* Premise
  was that the second column read on each nonzero update compounds
  the bandwidth tax. Falsified by the per-element data: the
  per-update cost is already at peak BLAS throughput at large, so
  there's nothing to fuse against.
- *Sample-block (row-tiled) CD.* Premise was that processing the
  inner sweep in row blocks improves L2 hit rate. Falsified by
  the same data — column streaming is not the bottleneck.
- *Gram-cache CD for LS* (covariance-update rule, glmnet-style).
  Would help at n ≫ p but is LS-only (GLM weighted-LS surrogate
  has changing W per outer iter) and duplicates the existing
  `GramLeastSquares` opt-in path. Out of P6 scope; a separate
  "when to use Gram path" docs / dispatch question.

**What this implies for the remaining v1.x perf queue.** The
outer KKT phase is now the only sub-component growing in relative
weight at large, which is exactly the surface P4 (pre-pass
gap-safe screening) attacks. That milestone's premise — "most
upside is on sparse-regime paths where post-pass screening fires
after the (single) pass has already converged" — is reinforced
by the 20.2 % share `outer_state` carries at large.

---

## Operability

### ✅ O1 — `cargo-semver-checks` in CI

**Shipped 2026-05-20.** The v1.0 stability promise was a written
policy (`docs/extending/rust-api.md` + the M8.5 audit in commit
`226b88e`); now machine-enforced. New `semver` job in
`.github/workflows/ci.yml` runs `cargo semver-checks check-release -p
skein-core --default-features --baseline-rev v1.0.0` on every PR.
222 checks against the freeze surface; breaking changes (removed
item, renamed export, signature change, new required trait method)
fail the job. The only path to a breaking change is a 2.0 release:
bump the baseline tag, list breakage in `CHANGELOG.md`, ship the new
major.

Notes on the implementation:

- `--default-features` only. `skein-core`'s default is empty (no
  BLAS) and the BLAS feature flags are mutually exclusive
  implementation switches that don't alter the public surface;
  `--all-features` would fail to build because `blas-accelerate` +
  `blas-openblas` both alias `blas-src as raw`.
- Ubuntu-only — the check is platform-independent.
- Binary install via `taiki-e/install-action@v2` (prebuilt, seconds)
  rather than `cargo install` (~6 min cold).
- Skein-py is intentionally not checked: the PyO3 macro-generated
  symbols are not the contract, the Python API is. A Python
  equivalent would need a different tool (e.g. `griffe`).

### ✅ O2 — Supply-chain hygiene

**Shipped 2026-05-20.** Three coupled pieces landed together:

- **`cargo-audit`** in a new `.github/workflows/security.yml` job.
  Runs against the RustSec advisory DB. Triggers: PRs that touch
  `Cargo.lock` / workspace `Cargo.toml` / crate manifests / the
  audit allowlist / the workflow itself; push to main on the same
  paths; weekly Monday 06:00 UTC cron so a new advisory against an
  unchanged tree still fails CI. `--deny warnings` promotes
  non-vulnerability advisories (unmaintained / unsound / notice) to
  hard failures.
- **`pip-audit`** as the sibling job in the same workflow. Audits
  the same dep tree users get from `pip install skein-glm[dev]` —
  builds with `MATURIN_PEP517_ARGS=--profile dev` so the maturin
  step matches the regular python job's ~3-4 min budget. `--strict`
  fails the gate if any package gets skipped by the resolver. The
  `[bench]` extra is intentionally excluded (per CLAUDE.md its
  resolution fails on the project's `requires-python` floor; bench
  is a maintainer tool, not a user-facing surface).
- **`.github/dependabot.yml`** with weekly Monday updates across the
  three ecosystems that ship from this repo: `cargo` (workspace
  `Cargo.lock`), `pip` (`pyproject.toml` runtime + extras), and
  `github-actions` (the `@vN` pins in `.github/workflows/*.yml`).
  Open-PR cap of 5 per ecosystem so a quiet week doesn't flood the
  PR queue.

Advisory allowlist landed at `.cargo/audit.toml`. One entry:

- **RUSTSEC-2025-0020** — pyo3 0.22.6 buffer-overflow risk in
  `PyString::from_object`. Verified unreachable from skein's binding
  surface (`grep -rn "PyString" crates/ python/` → zero hits). The
  0.22 → 0.24 upgrade is a deliberate breaking refactor (Bound<'py,
  T> default API + matching numpy crate bump) that earns its own
  milestone, not a security-driven emergency. The ignore is paired
  with the rationale inline so a future reviewer can re-evaluate.

Local pre-flight verification before shipping:

- `cargo audit` against current `Cargo.lock`: 177 deps scanned, 1
  vulnerability found (RUSTSEC-2025-0020, allowlisted), 0 others.
- `pip-audit --strict` against a fresh `[dev]` install: 0 known
  vulnerabilities once `idna` resolved to ≥3.15 (CVE-2026-45409 fix
  version); fresh CI installs already resolve to the patched
  version, so no manifest pin needed.

Notes on the implementation:

- The audit jobs went in a separate workflow rather than appended to
  `ci.yml` because `schedule:` triggers apply per-workflow; pinning
  it to `security.yml` keeps the weekly cron from re-running the
  full rust + python matrix.
- Ubuntu-only — advisory checks are platform-independent, and
  doubling the matrix would just slow the cron without changing
  what it catches.
- No CHANGELOG entry: this is CI tooling, not a user-visible v1.x
  behavior change (matches the O1/O3 precedent).

Follow-up tracked outside O2: schedule the pyo3 0.22 → ≥0.24 bump as
its own milestone so RUSTSEC-2025-0020 can come off the allowlist.

### ✅ O3 — Python 3.13 + NumPy 2.x in CI matrix

**Shipped 2026-05-20.** Python 3.13 added to the `python` job
matrix (`ci.yml`); matrix is now `["3.10", "3.11", "3.12", "3.13"]`
on both ubuntu-latest and macos-latest. `fail-fast: false` was
already set on the python job, so a flake on any single row doesn't
gate the others.

Findings during the audit:

- **NumPy 2.x was already the resolver default at v1.0.** The
  `numpy>=1.24` / `scipy>=1.10` floors in `pyproject.toml` had no
  upper cap, and modern pip on every Python in the matrix already
  picks NumPy 2.x. Local `.venv/` (Python 3.12) was on NumPy 2.4.4
  before this milestone landed; the 3.13 install picks NumPy 2.4.6.
  The matrix bump is what makes "we support 3.13" a tested
  guarantee instead of a hopeful one.
- **No Python-code hazards.** Grepped `python/` and `tests/` for
  removed NumPy-2 APIs (`np.float_`, `np.cast`, `np.NaN`, `np.product`,
  `np.alltrue`, `np.in1d`, `np.trapz`, `numpy.core`, etc.) — zero
  hits. Nothing to migrate.
- **Single abi3 wheel covers all four Python versions.** The
  maturin build emits `cp310-abi3` (per `crates/skein-py/Cargo.toml`'s
  `abi3-py310`), so 3.11/3.12/3.13 all consume the same artifact.
  The matrix exercises Python-level dispatch + each interpreter's
  stdlib + NumPy resolution, not separate Rust builds.
- **NumPy 1.x compatibility lane not added.** The `numpy>=1.24`
  floor stays as a written promise but is no longer tested in CI;
  modern resolvers always pick 2.x. If a user-reported regression
  shows up, the cheapest fix is to bump the floor to `numpy>=2.0`
  in a future minor and drop the unenforced 1.x claim.

Acceptance: 506 tests passed on Python 3.13 + NumPy 2.4.6 +
SciPy 1.17.1 + sklearn 1.8.0 locally (`/tmp/skein_py313`,
`SKEIN_REQUIRE_FIXTURES=1`, ~7m38s wall). Import smoke passed.

### O4 — Expanded wheel matrix

`.github/workflows/wheels.yml` currently:

- `CIBW_SKIP: "*-musllinux_*"` — Alpine / distroless users have no
  prebuilt path.
- Linux aarch64 dropped from the v0.1.x matrix.

Both decisions made sense at v0.1; v1.0 is a different audience.
Re-evaluate both.

### ✅ O5 — `docs/benchmarks/speed.md` consolidation

**Shipped 2026-05-21.** `docs/benchmarks/speed.md` is now the
canonical landing page for skein's wall-clock headline claims. It
sources its numbers directly from `paper/tables/T2_headline_timings.md`
(the v2 benchmark suite's machine-generated headline table) and surfaces
them as scenario-by-scenario speedup ratios against glmnet / ncvreg /
celer / skglm / grpreg, with explicit provenance for the snapshot
(host_id `3c43bb844695`, Apple M1, Accelerate, skein 0.10.0,
2026-05-19, git rev `474c68a1` per-cell / `08b1d378` bundle assembly).

What landed:

- **Provenance-first.** The "Provenance" section is the page's first
  table — host, CPU, BLAS, OS, Python, every comparator's pinned
  version, snapshot date, and the git rev. No headline number on the
  page is unmoored from this block.
- **Three speedup tables.** LS family (skein wins everywhere it has a
  peer — 1.45× over glmnet on `ls_lasso`, 4.59× over ncvreg on
  `ls_mcp`, 2.71× over celer on `ls_lasso`, etc.); group/structured
  family (~2× over grpreg on both `ls_group_lasso` and `ls_group_mcp`);
  GLM family (4.85× over ncvreg on `logistic_mcp`; losing to glmnet on
  `logistic_lasso` / `poisson_lasso` / `cox_lasso`).
- **Honest about the losses.** The page calls out explicitly that
  the snapshot is from skein 0.10.0, predating the M13.8 celer-style
  screening cascade that the README's "2.2–8.2× wall-clock on
  `logistic_lasso`" claim was measured against. v1.0+ users running a
  fresh v2 bundle should see different numbers on those three cells;
  re-snapshot is maintainer-overnight.
- **Caveats spelled out.** Single-threaded inner CD only (CV / stability
  parallelism is a separate axis). Deep-tail (`λ_min/λ_max = 1e-3`)
  cells only — sparse cells track the same direction qualitatively
  but finish 3–10× faster.
- **Reproduction recipe.** The exact `pip install -e '.[bench]'` +
  `maturin develop --release --features=blas-accelerate` +
  `snakemake --profile profiles/m1-headline` sequence that
  regenerates the snapshot.
- **Linked from the doctree.** Added to `docs/index.md`'s benchmark
  toctree and surfaced as section "0. Headline summary" at the top of
  `docs/benchmarks/index.md`.

Closes the M9.5 carryover. Future maintainers re-running the v2
bundle should regenerate `speed.md`'s headline table from the new
T2 — the script for that is a few lines of Python (currently inlined
in the commit message that landed this milestone; promoting it to a
`benches/v2/report/render_speed_md.py` driver is a follow-up if
re-snapshots become frequent).

### ✅ O6 — Structured timing / iteration surface

**Shipped 2026-05-20.** The roadmap framing was partly stale at
write-time: the Python `info_` dict already carried per-λ
`working_set_sizes` / `kkt_passes` / `iters` / `converged` /
`final_objs` (CD path) and `outer_iters` / `outer_converged` /
`inner_iters` / `final_losses` (prox-Newton path). What was actually
missing was wall-clock — the only datum that needs solver-internal
instrumentation. O6 ships exactly that, plus documentation of the
existing schema:

- New `skein_core::solver::{solve_path_timed, solve_block_path_timed,
  prox_newton_solve_path_timed, prox_newton_fused_solve_path_timed,
  prox_newton_block_solve_path_timed}` — sibling functions returning
  `(betas, report, Vec<u64>)` where the trailing vec is per-λ
  wall-clock nanoseconds. The existing 2-tuple variants delegate to
  these and discard the timing, so the v1.0 frozen API surface is
  untouched (`cargo semver-checks check-release` against
  `--baseline-rev v1.0.0` continues to pass — 222/222 checks).
- The PyO3 layer (`crates/skein-py/src/{ls,glm,multinomial,multitask,mmap_chunked}.rs`)
  routes every path builder through the `_timed` variant and adds a
  `times_ns: List[int]` key to the returned info dict.
- `python/skein_glm/estimators.py` module docstring now documents the
  full `info_` dict schema (which keys appear for CD-path vs
  prox-Newton-path estimators).
- `tests/test_path_report.py` pins the schema (3 tests, +3 to the
  pytest total).

The `path_report_` attribute name floated in the original framing
was dropped — `info_` is already the documented attribute on every
estimator, and adding an alias would have created two redundant
paths users have to choose between.

**Verification considered.** Adding a field to the existing
`PathReport` struct in `skein-core` was the first attempt and was
correctly flagged by `cargo-semver-checks` as
`constructible_struct_adds_field` (a 2.0-requiring break, since
downstream code can construct `PathReport { ... }` directly). The
shipped solution avoids that entirely.

---

## Out of scope for v1.x

- **New penalties, datafits, design backends.** The trait surface
  remains the extension surface; downstream researchers can still
  subclass the Python ABCs or implement the Rust traits in their
  own crate. We won't merge new variants upstream during v1.x.
- **GPU acceleration.** Carried over as out-of-scope from M4; the
  cost-benefit hasn't improved.
- **Inference layer additions.** Debiased Cox (M14a.3) closed the
  inference axis across the four main GLM families. No further
  inference machinery during v1.x.
- **Cython-grade inner rewrite (M10.I).** Re-evaluated as P6 with
  explicit gates; outside those gates, still parked.
- **Application-specific helpers** (psychometrics, finance,
  bioinformatics shortcuts). Build downstream.

A 2.0 release exists only when (a) we accumulate enough breaking
changes that an API-frozen 1.x can't accommodate them, or (b) a GPU
or precision-flexible compute backend lands. Neither is in v1.x scope.
