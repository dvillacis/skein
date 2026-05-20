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
| H1 — At-scale bench + fixture tier (n ≥ 100k) | Hardening | ⏳ planned | M12 P1 + M9.4 carryover; prerequisite for any future perf claim at the size users care about |
| H2 — Numerical-stability sweep | Hardening | ⏳ planned | collinear / zero-variance / extreme-weight / near-singular design fixtures across every solver |
| H3 — Property-based & fuzz tests | Hardening | ⏳ planned | `proptest` on prox / threshold / surrogate identities; closes the long tail C1/C2 left |
| H4 — Reproducibility audit | Hardening | ⏳ planned | RNG-seed coverage across stability selection, CV, bootstrap, multinomial init |
| P1 — Native sparse-group MCP block-CD for GLMs | Performance | ⏳ planned | drops the LLA layer for logistic / Poisson / Cox sparse-group MCP (sibling of M13.4c) |
| P2 — Scalar LLA one-outer-iter short-circuit | Performance | ⏳ planned | M13.5 carryover — bridge / adaptive / scalar MCP / SCAD still pay full LLA setup per λ in the convex regime |
| P3 — Cross-platform BLAS in distributed wheels | Performance | ⏳ planned | M10.G carryover — Linux/manylinux2014 already wires OpenBLAS; Windows wheels still ship without BLAS; MKL feature unwired |
| P4 — Pre-pass gap-safe screening | Performance | ⏳ planned | M10.H carryover — requires H1 to measure |
| P5 — M13.1 saturation-threshold tuning | Performance | ⏳ planned | conservative 0.5 may leave headroom on deep regime; cheap ablation, gated on H1 |
| P6 — Inner-CD column batching at large n | Performance | ⏳ planned | M13.6 follow-up — memory-bandwidth wall confirmed at n=50k, p=5k; structural change to `cd_solve_subset` |
| O1 — `cargo-semver-checks` in CI | Operability | ✅ done | v1.0 stability promise machine-checked on every PR via `--baseline-rev v1.0.0`; 222 checks vs the freeze surface |
| O2 — `cargo-audit` + `pip-audit` + dependabot | Operability | ⏳ planned | supply-chain hygiene baseline |
| O3 — Python 3.13 + NumPy 2.x in CI matrix | Operability | ⏳ planned | 3.13 GA was 2024-10; NumPy 2.x has been stable for 12+ months |
| O4 — Expanded wheel matrix (musllinux + Linux aarch64) | Operability | ⏳ planned | currently `CIBW_SKIP: "*-musllinux_*"`; aarch64 dropped from v0.1.x matrix |
| O5 — `docs/benchmarks/speed.md` consolidation | Operability | ⏳ planned | M9.5 carryover — single landing page for all perf claims with provenance |
| O6 — Structured timing / iteration surface | Operability | ⏳ planned | optional per-λ breakdown returned from path solvers; enables user-driven profiling without rebuilding |

Test count at v1.0.0: **358 cargo lib + 8 cargo integration + 455
pytest, all green.** Each milestone below either keeps this number
flat (perf work) or grows it (hardening).

---

## Hardening

### H1 — At-scale bench + fixture tier (n ≥ 100k)

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

### H2 — Numerical-stability sweep

Existing fixtures exercise well-conditioned synthetics. The classes
that have bitten us historically (M12 R4, M14d W_FLOOR, M14e v-scaled
prox) all came from real datasets in degenerate regimes that the
synthetics didn't cover.

In scope:

- **Collinear designs** — `X[:, j] = X[:, k] + ε` for ε ∈ {0, 1e-8,
  1e-12} across every penalty × datafit. Verify path solver and CV
  produce finite β, no NaN, no infinite KKT loops.
- **Zero-variance columns** — `X[:, j] = c` (constant). The
  `Standardized<D>` wrapper handles this lazily but the per-feature
  weight rescaling path through `rescale_weights_for_standardize`
  has not been audited under zero-variance.
- **Extreme weights** — `sample_weight` spanning 12+ orders of
  magnitude, per-feature weights with zeros (effective inactive
  feature), per-group weights with one zero in a sparse-group
  setting.
- **GLM tail saturation** — Poisson with `μ` near `ETA_CLAMP`,
  binomial with predicted probabilities pinned at `W_FLOOR`, Cox
  with ties heavier than Efron's exact-tie formula assumes.
- **Graphical lasso** near-singular sample covariance (n < p with
  effective rank deficit).

Each scenario lands as a `crates/skein-core/tests/numerics_*.rs`
integration test or a `tests/test_numerics_*.py` pytest. The test
asserts finiteness, monotone objective on the path, and reasonable
runtime (no infinite-loop fallback).

### H3 — Property-based & fuzz tests on prox / surrogate

`proptest` (Rust) and `hypothesis` (Python) over:

- **Prox identities**: `prox(prox(x)) = prox(x)`; soft-threshold
  monotonicity; group-prox rotation invariance for ℓ₂ group lasso;
  MCP/SCAD agreement with closed-form references at random points.
- **Surrogate identities**: `GlmDatafit::surrogate_at(β)` returns a
  `LeastSquares` whose quadratic approximation matches the GLM
  gradient + Hessian at β to machine precision (we have this for
  logistic + Poisson + Cox in test fixtures; property-based version
  catches Lipschitz-bound regressions).
- **Standardize bijection**: `destandardize(standardize(X, β)) ≈ β`
  for arbitrary X / weights.
- **Penalty / weight composition**: per-feature × per-sample ×
  per-group weight combos that don't currently have a unit test but
  are documented as valid through the public API.

This is the closure of C1/C2 from M12 — those milestones added
direct unit tests but didn't introduce randomized coverage.

### H4 — Reproducibility audit

Stability selection, CV fold construction, multinomial init, and
bootstrap edge stability all consume `rng` somewhere. Audit:

- Every `rng` consumer accepts a `random_state` parameter and
  documents the exact semantics.
- Two fits with the same `random_state` produce bit-identical β
  (modulo BLAS-thread nondeterminism, which we document and gate
  with `OMP_NUM_THREADS=1` in the reproducibility test).
- The `joblib`/Rayon path doesn't reorder work in a way that
  depends on scheduling — current `allow_threads` + Rayon pattern is
  deterministic per fold but verify.

Deliverable: one `tests/test_reproducibility.py` pinning every
randomized estimator with two seeds and asserting equality / fold
consistency.

---

## Performance

### P1 — Native sparse-group MCP block-CD for GLMs

**Sibling of M13.4c** (native group-MCP BCD for logistic / Poisson /
Cox) and **M14c.2** (native sparse-group MCP penalty + 6 GLM PyO3
swaps were already done for LS; the GLM swap is what's left). The
GLM sparse-group MCP path still routes through LLA, paying the full
outer-iter cost on what is, in the convex regime, a one-iteration
problem.

Expected impact: same order as M13.4c (2–3× wall on the
medium-deep cell). Touches `solver/prox_newton_block.rs` and the
six PyO3 builders that dispatch sparse-group MCP for GLMs.

### P2 — Scalar LLA one-outer-iter short-circuit

**Carries M13.5 forward.** Phase 2.3 (M14c.1) ported the scalar-LLA
short-circuit to `path_lla.rs` for bridge, adaptive lasso, and
multitask LLA, but the scalar MCP / SCAD paths in the convex regime
(γ large enough that LLA converges in one iter per λ) still pay full
setup cost. Same fix shape: at outer iter 1, if the surrogate-weight
delta is below ε, accept the current β as the solution and move to
the next λ without a second pass.

Expected impact: closes the ~25–30 % gap from `skein Lasso medium /
dense` to `skein MCP medium / dense` that M13.5 measured.

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

### O2 — Supply-chain hygiene

- `cargo-audit` job in `.github/workflows/ci.yml` (one cron + on
  every PR touching `Cargo.lock`).
- `pip-audit` over the resolved `requirements-dev.lock` (current
  resolution issue with the `bench` extra notwithstanding — audit
  the resolved dev set).
- Dependabot config for `Cargo.toml`, `pyproject.toml`,
  `.github/workflows/*.yml`. Weekly cadence.

### O3 — Python 3.13 + NumPy 2.x in CI matrix

CI matrix currently pins `["3.10", "3.11", "3.12"]` (`ci.yml`). 3.13
is in scope; NumPy 2.x is in scope. Pin both as additive matrix
entries first (fail-fast = false), then promote once green.

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

### O6 — Structured timing / iteration surface

The Rust path solvers maintain a `PathReport` (per-λ working-set
sizes, KKT pass counts, screening mode) — the Python facade does not
expose this. Wire it through as an optional return field (e.g.
`MCPRegressor(verbose_report=True).fit(X, y).path_report_`) so
downstream users can profile their fit without rebuilding skein
with `SKEIN_PROFILE_PATH=1`.

Strictly additive — does not extend the v1.0 frozen API surface
(adds a new field, does not remove or rename anything).

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
