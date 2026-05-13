# M10.1 — Profile of medium lasso/LS path

**Scenario**: scalar lasso, Gaussian LS, `n=10_000`, `p=1_000`,
`k_active=10`, SNR=5, 100-λ path, `tol=1e-6`, strong rule + Anderson(5).

**Profiling target**: `crates/skein-core/examples/lasso_ls_medium.rs` —
pure-Rust binary that mirrors `benches/problems.py::gaussian_lasso` at
the medium size, so samply / `cargo flamegraph` can resolve frames
without the PyO3 + interpreter layer in the way.

**Run**:

```bash
cargo build --release --example lasso_ls_medium
samply record --save-only -o /tmp/skein_lasso.profile.json \
    ./target/release/examples/lasso_ls_medium
samply load /tmp/skein_lasso.profile.json   # opens browser flamegraph
```

The `[profile.release]` in the workspace `Cargo.toml` carries
`debug = "line-tables-only"` so symbol names resolve in the profile
without a meaningful binary-size cost.

## Headline result (post-M10.3 first wave)

| measurement | value |
|---|---|
| total fit | 3.21 s |
| inner CD iterations summed across λ | 430 |
| KKT verification passes | 100 (one per λ — strong rule + screening agree first try) |
| working-set size at smallest λ | 827 / 1000 |
| samply samples | 6,715 |

## Where the time goes

After resolving samply offsets to the example binary's symbols (`nm
-n target/release/examples/lasso_ls_medium`):

| function | self time | role |
|---|---|---|
| `ndarray::linalg::impl_linalg::*::dot_generic` | **~80%** | inner `col_dot(j, r) = X[:, j] · r` for the per-coord gradient |
| `<DenseMatrix as DesignMatrix>::col_axpy` | ~7% | residual update `r += δ · X[:, j]` after a non-zero coord change |
| everything else | ~13% | prox, obj eval, screening, Anderson, allocations |

(Self time is the leaf frame at sample time. Total time / inclusive
time per function is dominated by `solve_path → cd_solve_subset →
LeastSquares::coord_grad → DenseMatrix::col_dot → dot_generic` at the
top-95% of samples.)

## Read-out

The bottleneck is **call volume on `col_dot`**, not slow code per
call. Each coordinate visit issues one `col_dot` (always — for the
gradient) and conditionally one `col_axpy` (only when the prox shrinks
to a non-zero delta). Once CD nears convergence, most coords have
`δ=0`, so `col_axpy` fires for ~10% of visits while `col_dot` fires
for 100% — that's the 11× ratio in the profile.

## Microbench: `col_dot` on a 10 k-long contiguous F-order column

```
ndarray .dot()                  323 ms / 100k calls
Zip::from(col).and(v).for_each  1.52 s / 100k calls
slice iter().zip().map().sum()  1.53 s / 100k calls
indexed for-loop                1.52 s / 100k calls
slice iter().zip().fold(...)    1.53 s / 100k calls
```

ndarray's `dot()` (dispatching to `dot_generic`, which under the hood
goes through `matrixmultiply`'s tuned f64 microkernel) is **4.7×
faster** than every manual variant I tried. So the path ahead **is
not** "write a tighter pure-Rust loop"; that's already what's
happening, and we'd lose ~5× by replacing it.

## What this means for the next M10.3 wave

Three categories of follow-up, in order of expected impact ÷ cost.

### 1. Enable ndarray's `blas` feature

Per call cost on `col_dot` would drop from ~3.2 µs (`dot_generic`,
generic `matrixmultiply`-backed loop) to ~1.0 µs (BLAS `ddot`,
hand-tuned vector assembly with prefetch). With ~800 k `col_dot` calls
on the medium bench, the projected fit time drops from 3.2 s to
roughly 1.0–1.2 s. That would put skein at ~6× sklearn (still slow but
closing the gap meaningfully) and ~30% faster than glmnet.

Cost: a build-time dependency. On macOS, `accelerate-src` ships zero
install — Apple's Accelerate framework is part of the OS. On Linux,
`openblas-system` requires `apt install libopenblas-dev` (or the
distro equivalent); on Windows, it's most easily handled via
`vcpkg`. Wheels can either statically link via `openblas-static` or
declare a runtime requirement.

CI cost: install one BLAS dev package per OS in the existing matrix
(low). Wheel-size cost: 100s of KB if statically linked, zero if
dynamically.

### 2. Skip `col_dot` when we can prove the gradient is below threshold

Active-set CD ideas (Tseng-Yun, Friedman et al. for glmnet) maintain
an inner active set inside each λ that shrinks across iterations. If
feature `j` had `|grad_j| ≪ λ w_j` last iter and `‖Δr‖` is small
enough that the gradient can't have moved by more than the safe
margin, skip the recompute. Ranges from "track ‖Δr‖² incrementally
and bound by Cauchy–Schwarz" to a full per-iter screening. Real wins
without new dependencies, but real coding work.

Expected impact: 2–4× depending on how aggressively we shrink the
inner working set. Compatible with #1.

### 3. Restructured CD with a cached gradient vector

Compute `g = Xᵀ r` once at the start of each outer iter (one gemv —
much faster than `p` separate `col_dot`s if BLAS is on, since gemm/gemv
microkernels amortise the inner reduction across columns). Read
`g[j]` per coord; on a non-zero update, refresh `g` incrementally via
the column dot of `X` against the changed column — but that's still
`O(np)` per single coord update, so this only pays off if multiple
coords update before a refresh. The clean form requires a Gram-matrix
precompute (`G = XᵀX`, `p × p`, 8 MB at p = 1000, 800 GB at
p = 10⁵), so it's a regime-dependent optimization rather than a
general win.

Expected impact: large at small `p`, irrelevant at the scales this
library is meant to handle. Probably skip.

## Recommendation

Pursue #1 (BLAS) next. It's the single change with the largest
expected impact and the most predictable cost. If it lands skein in
glmnet's neighbourhood, the M9 elevator-pitch claim is at least
partially defensible while #2's harder algorithmic work proceeds in
parallel.

## Postscript — inner active-set CD (#2) tried, did not pay off

Implemented `cd_solve_subset` with Friedman-style two-phase cycling:

- **Phase 1**: cycle on an inner active set `A ⊆ features` (the
  currently-non-zero coordinates) until `max_delta < tol`.
- **Phase 2**: one verification sweep over `features \ A`. Any
  coordinate whose prox produces a non-zero update joins `A` and we
  re-enter Phase 1. Convergence is declared when a Phase 2 sweep
  finds no growth.
- Cold start (β = 0) does one full sweep first to populate `A`, then
  enters the loop.

Used a length-`p` boolean mask for `O(1)` membership during Phase 2.
All 265 cargo tests passed.

**Result on the medium lasso/LS bench**:

| variant | medium fit |
|---|---|
| baseline (post-M10.3, no inner active-set) | 3.2 s |
| active-set, history-clear on Phase 2 grow | 3.5 s (+9%) |
| active-set, no history-clear | 4.7 s (+47%) |

Why it didn't help:

- The medium scenario reaches `787 / 1000` features active at the
  smallest λ. Late-path λs are saturated — `A ≈ features`, so
  Phase 1's cycle is the same size as plain cycling, and the per-λ
  Phase 2 verification sweep is pure overhead.
- At the cold-start λ where active-set CD *could* help most (`A` of
  size 1–3 from 1000), the absolute cost is small enough that the
  savings don't compensate later regressions.
- Anderson interaction: clearing history on Phase 2 grow was needed
  to prevent stale snapshots from generating bad extrapolations
  (the obj-decrease safeguard caught them, but the rejected attempts
  cost an extra `init_residual` matvec each — that's where the
  +47% goes when we *don't* clear). With clearing, Anderson never
  accumulates the 6 iterates it needs to fire on this saturated
  problem, losing speedup it had under plain cycling.

Reverted in commit *(see git log)*. The negative result narrows the
remaining options: BLAS (#1) is the only path that doesn't require a
deeper algorithmic redesign (sklearn's tight-loop micro-optimisations,
celer's dual extrapolation, or a Gram-cached CD). On a sparser-regime
benchmark (`k_active ≈ √p`, `lambda_min_ratio = 0.05`, well short of
saturation) inner active-set CD might still pay off; revisit when
that scenario lands in M9.3.

## Postscript — sparse-regime bench (`lasso_ls_sparse`)

Added `benches/scenarios/lasso_ls_sparse.py` with `λ_min/λ_max =
5e-2` (vs the dense-path `1e-3`). Same problem generator (`k_active=10,
n=10k, p=1k, snr=5`); the only thing that changes is how far the path
runs before λ stops shrinking. With the sparse grid the active set at
the smallest λ stays at the true support size (10) instead of
saturating to 791.

This is the regime priority-WS + adaptive inner tol were designed for,
so it's the right test of whether the M10.3 structural changes are
delivering wallclock value the saturated regime hides.

**Results (medium, n=10k, p=1k)**:

| comparator | dense regime | sparse regime |
|---|---|---|
| sklearn (`lasso_path`)        | 18.3× slower | **17.3× slower** |
| glmnet (R, via Rscript)       |  3.6× slower | **3.1× slower** |
| skglm (per-λ runner — unfair) |  0.3× (we're faster) | 0.3× |
| celer (per-λ runner — unfair) |  0.3× | 0.4× |

skein medium drops from 3.29 s (dense) to **2.53 s (sparse)** — the
priority WS keeps the working set at 10–20 features along the entire
path (vs growing to 1000 in the dense regime), and the adaptive inner
tol's iteration savings finally compound:

  ws across path:  10–20 features   (was 10 → 1000)
  iter sum:        304               (was 317)
  kkt_passes sum:  100               (was 110 — every λ converges in 1 outer pass)

**Honest takeaway**: the gap to glmnet narrows by ~14% and the gap to
sklearn by ~5%. Real improvements but not the order-of-magnitude
move. The remaining ~17× to sklearn lives in `dot_generic` per the
M10.1 profile; sparser regimes don't change that floor.

skglm and celer numbers in this table are not informative — both
runners do per-λ `Lasso(alpha=λ).fit()` fresh, while sklearn /
skein / glmnet use native warm-started path solvers. The comparison
becomes meaningful once the M9.3 runner fairness fix lands.

## Postscript — runner fairness fix landed

Updated `benches/runners/skglm_runner.py` to use
`skglm.estimators.Lasso.path(X, y, alphas=…)` and
`benches/runners/celer_runner.py` to use
`celer.homotopy.celer_path(X, y, "lasso", alphas=…)` — the same
warm-started path APIs each package's own benchmarks use. Both
packages now get the same warm-start advantage skein and sklearn have.

**Apples-to-apples results (medium, n=10k, p=1k, 100-λ path)**:

| package | dense regime | sparse regime |
|---|---|---|
| sklearn (`lasso_path`)              | 188 ms | 147 ms |
| glmnet (R, via Rscript)             | 950 ms | 707 ms |
| skein                               | 3.32 s | 2.50 s |
| celer (`celer_path`)                | 3.97 s | **466 ms** |
| skglm (`Lasso.path`)                | 5.24 s | 3.47 s |

Two real findings:

1. **skein beats both celer and skglm in the dense regime**. When the
   path runs into the saturated tail, our M10.3 structural
   changes + strong-rule-equivalent priority WS combine to beat
   celer's dual-extrapolation overhead and skglm's per-feature numba
   dispatch. We stay 17× behind sklearn (BLAS gap) and 3.5× behind
   glmnet (also Fortran BLAS) but ahead of the Python comparators.

2. **celer dominates in the sparse regime — 5× faster than glmnet,
   5.4× faster than skein**. Their dual-extrapolation builds a
   tighter screening sphere as the path progresses, so the WS shrinks
   aggressively when the true support is small. skein's PGD-based
   screening only compresses fit time 1.3× from dense to sparse;
   celer compresses 8.5×.

The first is encouraging, but the second is the bigger lesson:
**celer has an algorithmic advantage on sparse paths that priority
WS + adaptive inner tol don't capture**. That's the next non-BLAS
lever after Option C — but it's a real algorithmic addition (dual
extrapolation, tracked dual feasibility), not a simple
restructuring.

Order of remaining levers, post-runner-fairness:

  C. ndarray `blas` feature: ~3× on dense, would close most of the
     gap to sklearn / glmnet. The single largest move.
  E. Anderson on residual instead of β (smaller code, ~1.05–1.2×).
  F. Dual extrapolation (celer's lever): would close the celer-sparse
     gap. Significant new code.

skglm at 5.24s dense / 3.47s sparse — note that skglm's *AndersonCD*
is what drives our `subdiff_distance` adoption philosophy in the
adaptive PGD work. skglm uses numba @njit for the inner CD; we use
ndarray's pure-Rust dot path. Numba's BLAS dispatch on
`X[:, j].dot(Xw)` apparently isn't meaningfully faster than
`dot_generic` here — both packages cluster in the 3–5 s range.

## Postscript — `blas-accelerate` feature landed

Added the `blas-accelerate` Cargo feature on `skein-core` and a
`blas-accelerate` passthrough on `skein-py`:

  ndarray = { workspace = true, features = ["blas"] }   (gated)
  blas-src   = "0.11"  (default-features = false, optional)
  accelerate-src = "0.3"  (optional, anchors the linker directive)

Build the wheel with `maturin develop --release --features
blas-accelerate` to route ndarray's `dot` / `scaled_add` /
`gemv` through Apple's Accelerate framework on macOS — zero
install cost, the framework ships with the OS. The hot path is
`col_dot` inside the inner CD, which is now BLAS `ddot` instead
of `matrixmultiply`'s generic loop.

(For Linux / Windows, add a sibling feature like
`blas-openblas = ["ndarray/blas", "blas-src/openblas"]` and
ensure `libopenblas-dev` is present at build time. That work is
deferred until cibuildwheel / CI integration; the local-dev
macOS path is what enabled the M10.1 prediction to be tested.)

**Results (medium lasso/LS, n=10k, p=1k, 100-λ path)**:

| package        | dense before | dense with BLAS | sparse before | sparse with BLAS |
|---|---|---|---|---|
| sklearn        | 188 ms      | 181 ms         | 147 ms        | 147 ms |
| glmnet (R)     | 950 ms      | 883 ms         | 707 ms        | 810 ms |
| **skein**      | **3.32 s**  | **1.75 s** ← 1.9× | **2.50 s**    | **0.96 s** ← 2.6× |
| celer          | 3.97 s      | 3.97 s         | 466 ms        | 466 ms |
| skglm          | 5.24 s      | 5.28 s         | 3.47 s        | 3.46 s |

skein-relative-to-comparators (medium, with BLAS):

- vs sklearn:  **9.6× dense, 6.5× sparse** (was 18× / 17×)
- vs glmnet:   **2.0× dense, 1.18× sparse** (was 3.5× / 3.5× — sparse is essentially matched!)
- vs celer:    0.44× dense (we're 2.3× faster), 2.1× sparse (was 0.84× / 5.4×)
- vs skglm:    0.33× dense (we're 3× faster), 0.28× sparse (3.6× faster)

**Read-out**:

The M10.1 prediction was ~3× from BLAS dispatch on `col_dot`. The
realised speedup (1.9× dense / 2.6× sparse) is somewhat below that but
qualitatively correct — the BLAS replaces the per-call `dot_generic`
floor with `ddot`, and the extra 0.4× to 1.4× went to other parts of
the path solver (the prox-grad-distance verifier, the priority-WS
ranking) that haven't been BLAS-accelerated.

The biggest practical change: **skein is now within striking distance
of glmnet** (essentially matched in sparse, 2× off in dense). We still
sit ~10× behind sklearn's `lasso_path` Cython — that's a tight inner
loop with no path-level structural overhead, hard to match without a
similar Cython-grade rewrite. But for the niche this library targets
(nonconvex group penalties, weighted formulations, Rust-native
extension surface), being a stone's throw from glmnet is more than
defensible.

Order of remaining levers, post-BLAS:

  E. Anderson on residual instead of β       — small change, ~1.05–1.2×.
  F. Dual extrapolation (celer's lever)      — significant new code; would
                                                close the celer-sparse gap
                                                where their priority screen
                                                + dual coastlines beat us 2×.
  G. Tighter inner loop (sklearn-level Cython-equivalent in Rust) — unclear
                                                if there's room beyond
                                                BLAS dispatch given LLVM is
                                                already vectorising what we
                                                have.
