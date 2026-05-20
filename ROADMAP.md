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
| P2 — Scalar MCP/SCAD path overhead investigation | Performance | ⏳ planned | M13.5 carryover, re-scoped — scalar MCP/SCAD route through `solve_path` directly (closed-form prox), not `path_lla`; the M14c.1 short-circuit already covers the LLA-side users (bridge/adaptive/multitask). Current `medium/deep` gap is 1.50× MCP vs Lasso (was 1.32× pre-M14e); EN-vs-Lasso is 1.24× so most of MCP's gap is generic non-trivial-prox cost, ~1.21× excess is genuinely MCP-specific. Profile-then-fix; smaller upper bound than originally claimed. |
| P3 — Cross-platform BLAS in distributed wheels | Performance | ⏳ planned | M10.G carryover — Linux/manylinux2014 already wires OpenBLAS; Windows wheels still ship without BLAS; MKL feature unwired |
| P4 — Pre-pass gap-safe screening | Performance | ⏳ planned | M10.H carryover — requires H1 to measure |
| P5 — M13.1 saturation-threshold tuning | Performance | ⏳ planned | conservative 0.5 may leave headroom on deep regime; cheap ablation, gated on H1 |
| P6 — Inner-CD column batching at large n | Performance | ⏳ planned | M13.6 follow-up — memory-bandwidth wall confirmed at n=50k, p=5k; structural change to `cd_solve_subset` |
| O1 — `cargo-semver-checks` in CI | Operability | ✅ done | v1.0 stability promise machine-checked on every PR via `--baseline-rev v1.0.0`; 222 checks vs the freeze surface |
| O2 — `cargo-audit` + `pip-audit` + dependabot | Operability | ✅ done | supply-chain hygiene baseline; weekly cron + per-PR on dep-manifest changes; 1 documented advisory ignore (RUSTSEC-2025-0020, pyo3 unreachable API) |
| O3 — Python 3.13 + NumPy 2.x in CI matrix | Operability | ✅ done | 3.13 added to the `python` job matrix; local pytest on 3.13 + NumPy 2.4.6 green (506 passed). NumPy 2.x was already the resolver default at v1.0 — `numpy>=1.24` floor stays; no API-removal hazards in the Python codebase |
| O4 — Expanded wheel matrix (musllinux + Linux aarch64) | Operability | ⏳ planned | currently `CIBW_SKIP: "*-musllinux_*"`; aarch64 dropped from v0.1.x matrix |
| O5 — `docs/benchmarks/speed.md` consolidation | Operability | ⏳ planned | M9.5 carryover — single landing page for all perf claims with provenance |
| O6 — Structured timing / iteration surface | Operability | ✅ done | per-λ wall time surfaced via `info_["times_ns"]` on every path estimator; powered by additive `solve_path_timed` / `solve_block_path_timed` / `prox_newton_*_solve_path_timed` siblings that keep the v1.0 freeze intact |

Test count at v1.0.0: **358 cargo lib + 8 cargo integration + 455
pytest, all green.** Each milestone below either keeps this number
flat (perf work) or grows it (hardening). Current HEAD: **448 cargo
lib + 605 pytest** (post-P1; P1 is a perf-swap, counts unchanged from H4).

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

### P2 — Scalar MCP/SCAD path overhead investigation

**Carries M13.5 forward, but re-scoped.** Initial framing (taken
from the v0.x roadmap) was that scalar MCP/SCAD pays "LLA outer
wrapper cost" and that porting Phase 2.3's short-circuit would
close it. Investigation post-v1.0 shows that's wrong: scalar
MCP/SCAD don't go through `solve_path_lla` at all — they call
`solve_path` directly with the `Mcp`/`Scad` penalty's closed-form
prox. M14c.1 already shipped the short-circuit for `path_lla.rs`,
and its scope (bridge / adaptive / multitask) really was the whole
LLA-side surface; nothing was deferred.

Current `medium/deep` snapshot (committed in
`benches/v2/results/scenarios/`):

| cell          | Lasso | MCP   | MCP/Lasso | EN/Lasso |
|---------------|------:|------:|----------:|---------:|
| medium/deep   | 1.13s | 1.70s | **1.50×** | 1.24×    |
| medium/sparse | 0.37s | 0.46s | 1.24×     | 1.01×    |

So the MCP gap exists, but EN-vs-Lasso shows the same 1.24× on deep
even with a near-soft-threshold prox. The MCP-specific excess on top
of that is ≈1.21× on deep, ≈1.24× on sparse — meaningful but smaller
than M13.5's pre-M14e 1.32× claim. Upper bound on the medium/deep
fix is ~0.3 s.

The real work is therefore:

1. Profile (`SKEIN_PROFILE_PATH=1` + the
   `crates/skein-core/examples/lasso_ls_medium.rs` pattern adapted
   to MCP) to identify where the genuine MCP-specific overhead lives
   — prox call cost, KKT-pass re-evaluation, weight-vector construction
   per λ, or something else.
2. Decide if the bottleneck is amenable to a focused fix or is
   structural ("MCP firm-threshold is just more work per coord
   update than soft-threshold, full stop").
3. Either ship the fix or close P2 with a "no structural lever
   available" note.

Lower priority than originally assigned given the smaller upper
bound and the fact that the fix shape isn't pre-known. Reasonable
to defer until another investigation surfaces a similar pattern.

### P3 — Cross-platform BLAS in distributed wheels

**Carries M10.G forward.** Current state:

- macOS arm64 wheels: `blas-accelerate` (✓).
- Linux x86_64 wheels: `blas-openblas` via manylinux2014 OpenBLAS
  package (✓ — wired in `.github/workflows/wheels.yml`).
- **Windows wheels: no BLAS feature enabled.** The Cython-grade
  matvec gap is therefore largest on Windows.
- **MKL feature unwired.** Listed in the v0.x roadmap as an option;
  not built, not exposed.

Deliverable:

- Audit Windows wheel for whether `blas-openblas` (`vcpkg openblas`)
  is reachable in cibuildwheel; if so, wire it. Otherwise document
  the gap.
- Decide whether to add `blas-mkl` as a documented opt-in build
  feature. Not in the default wheel matrix.

Acceptance: a published wheel that lacks BLAS prints a one-line
runtime notice on first import (or in `skein_glm.__version__` /
`__build_features__`) so users know what they have.

### P4 — Pre-pass gap-safe screening

**Carries M10.H forward.** F-series shipped post-pass screening (run
after each outer KKT pass). celer's actual pattern is pre-pass — use
λ\_{k-1}'s last gap + gradient to prune the priority working set at
λ\_k entry. Modest extension; most upside is on sparse-regime GLM
paths where post-pass screening fires after the (single) pass has
already converged.

Gated on H1 — without large-n / long-path scenarios the win is below
measurement noise.

### P5 — M13.1 saturation-threshold tuning

M13.1 shipped at `SCREENING_SATURATION_THRESHOLD = 0.5` — the
conservative choice. Lower thresholds (e.g. 0.3) likely recover
more of the gap to `screening = Off` on deep regimes; higher
thresholds are safer on borderline-saturated cells. Ablation cell in
`benches/v2/` to measure at every regime × scenario; gate on H1
landing first so the ablation has scale to measure against.

### P6 — Inner-CD column batching at large n

**Carries M13.6 forward.** The medium → large transition is 1.5×
super-linear in `cd_solve_subset` (37.6× wall for a 25× problem
growth, with the entire excess in inner-CD coord-visit cost). At
n=50k the per-column X-vector exceeds L2; `col_dot` shifts from
compute-bound to memory-bandwidth-bound, and each coord visit
streams a fresh column from main memory.

The lever is processing multiple coords per X-column scan
(reorder the inner CD loop so a column is touched once for a batch
of pending updates). Structural change to `cd_solve_subset`;
explicitly the M10.I "Cython-grade rewrite" lever the v0.x roadmap
called out and parked.

Gated on:

1. H1 landing so we have a regression gate.
2. P1 + P2 landing so the LLA-side overhead is out of the way and
   the residual gap really does live in inner CD.

If after (1)+(2) the medium→large super-linearity is still >1.3×,
this milestone proceeds; otherwise it stays parked.

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

### O5 — `docs/benchmarks/speed.md` consolidation

**M9.5 carryover.** Headline numbers live in five places: `README.md`,
`ROADMAP_old.md`, `paper/tables/T2_headline_timings.md`, individual
`docs/perf/*.md` profile notes, and `paper/BUNDLE.md`. Consolidate
into a single `docs/benchmarks/speed.md` landing page with explicit
provenance (host_id, BLAS feature, commit SHA, snapshot date) so
users can tell at a glance what the "1.9× ahead of glmnet" claim
covers.

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
