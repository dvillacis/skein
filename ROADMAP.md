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
| H2 — Numerical-stability sweep | Hardening | ⏳ planned | collinear / zero-variance / extreme-weight / near-singular design fixtures across every solver |
| H3 — Property-based & fuzz tests | Hardening | ⏳ planned | `proptest` on prox / threshold / surrogate identities; closes the long tail C1/C2 left |
| H4 — Reproducibility audit | Hardening | ⏳ planned | RNG-seed coverage across stability selection, CV, bootstrap, multinomial init |
| P1 — Native sparse-group MCP block-CD for GLMs | Performance | ⏳ planned | drops the LLA layer for logistic / Poisson / Cox sparse-group MCP (sibling of M13.4c) |
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
flat (perf work) or grows it (hardening).

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
