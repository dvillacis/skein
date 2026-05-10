# What celer & skglm do that skein doesn't

Comparative reading of:

- `celer/celer/lasso_fast.pyx` (the load-bearing inner solver)
- `celer/celer/cython_utils.pyx` (BLAS wrappers, `set_prios`, dual point construction)
- `skglm/skglm/solvers/anderson_cd.py` (the lasso/MCP/SCAD outer/inner driver)
- `skglm/skglm/solvers/common.py` (gradient construction, fix-point distance)
- `skglm/skglm/penalties/separable.py` (per-penalty `subdiff_distance` for KKT scoring)

These are the two Python sparse-GLM solvers our M9 bench compares against.
celer 0.7 beats us 1.7× on the medium scenario (when we let it use a
warm-started path); skglm 0.5 is in our neighbourhood but uses a
similar architecture. Reading their code closely reveals that **the
algorithmic shape they share is significantly different from ours, and
many of those design choices line up with our M10.1 profile findings.**

## Shared architecture (celer + skglm + glmnet)

Both use a **two-level loop with a growing working set**:

```
outer t = 0..max_iter:           # ~50–100 outer iters
    1. Build WS_t from KKT priorities
    2. Solve subproblem on WS_t (inner loop on |WS_t| features only)
    3. Stop when outer KKT/gap criterion is satisfied
```

```
inner epoch = 0..max_epochs:      # up to 50_000
    one CD sweep over WS_t
    every K=10 epochs:
        recompute KKT/gap on WS_t
        if inner stop crit < 0.3 × outer crit: break  # adaptive inner tol
```

**This differs from skein in seven places.**

### 1. Working-set growth from `p0`, not strong-rule full set

celer (`lasso_fast.pyx:188-203`): `ws_size = p0` at first iter (default
`p0=100`), then `ws_size = 2 * nnz` (or `2 * ws_size`) per outer iter.
skglm same idea (`anderson_cd.py:131-133`):
`ws_size = max(p0 + n_unpen, 2 * |support| - n_unpen)`.

Skein currently uses the strong-rule output as the WS. At λ_max (cold
start, no previous gradient), strong rule degrades to the full feature
set — so we sweep all p columns once before any screening helps. celer
sweeps only 100 features at the same point.

### 2. WS selection by KKT priority, not just strong rule

celer's `set_prios` (`cython_utils.pyx`) computes per-feature priority
from the **dual point** (after projecting `θ = -r/n` into the
dual-feasible set). Lower priority = more deserving of inclusion.
Active features always get priority −1 (always included).

skglm's `subdiff_distance` (`penalties/separable.py:36-56` for L1) is
the ℓ₂ distance from `−grad_j` to the subdifferential of `λ |·|` at
`w_j`:

```python
if w_j == 0:        d_j = max(0, |grad_j| - λ)
else:               d_j = |grad_j + λ * sign(w_j)|
```

`d_j = 0` ⇔ feature `j` is at a KKT-stationary point. WS = features
with the largest `d_j`. Active features always pinned via `d_j = ∞`.

Skein's `strong_rule_screen` is correct but coarser — features pass a
threshold, but we don't rank them. That makes our WS **conservative**
(may include features the priority-rank would exclude).

### 3. Outer stopping by duality gap (celer) or max KKT distance (skglm)

celer: `gap = primal − dual` per outer iter; stop when `gap ≤ tol`.
skglm: `stop_crit = max_j subdiff_distance(j)`; stop when `≤ tol`.

Both are penalty-aware and tighter than skein's coefficient-space
`max_j |β_j_new − β_j_old| < tol`. **Concretely, skein currently
declares convergence on a max_delta criterion that doesn't reflect
sub-optimality for the primal problem — only the iterate's stationarity
under the prox map.** Either of celer/skglm's criteria would stop
sooner without losing solution quality.

### 4. Periodic gap/KKT check at `gap_freq=10`, not per-coord

celer (`lasso_fast.pyx:231`): `if epoch != 0 and epoch % gap_freq == 0:
... recompute gap`. Inner CD just runs full epochs; convergence is
checked every 10.

skglm (`anderson_cd.py:183-209`): `if epoch % 10 == 0:` recompute
`stop_crit_in` on the WS. Same pattern.

Skein checks `max_delta` **per coordinate update** in the inner sweep,
which is the simplest convergence signal but means a max-comparison
on every coord visit. Cheap individually; not a bottleneck on its own,
but it ties our convergence definition to coefficient distance rather
than KKT.

### 5. Adaptive inner tolerance: `inner_tol = 0.3 × outer`

celer (`lasso_fast.pyx:220`): `tol_in = 0.3 * gap`.
skglm (`anderson_cd.py:206`): `if stop_crit_in < 0.3 * stop_crit:
break`.

The inner subproblem doesn't try to converge below the current outer
KKT distance. This **prevents over-converging** the inner problem
when the outer is far from done — common at early λs in a path.

Skein's inner CD always converges to `tol`, not a relaxed inner
tolerance. On a path solve this means inner work happens at
unnecessary precision early in the path.

### 6. Anderson on `Xw` (the residual), not on `β`

celer (`lasso_fast.pyx:75-79, 251-278`): `last_K_Xw[k][:]` snapshots
of the residual; extrapolate `thetacc`; accept on **dual** objective
improvement.

skglm (`solvers/anderson_cd.py:169-181`): `extrapolate(w[ws], Xw)` —
extrapolates **both** on the WS-restricted parameter and the residual;
accepts on primal objective decrease.

Skein currently extrapolates `β` on the full feature set. For sparse
problems with many zero features, that's a lot of zero-stride dimensions
in the Anderson matrix; the extrapolation has nothing to model on
inactive features. Restricting to WS or moving to residual (length n)
is the more standard approach.

### 7. Per-coord update uses BLAS `daxpy` / `ddot`

celer (`lasso_fast.pyx:316`):
```cython
w[j] += fdot(&n_samples, &X[0, j], &inc, &Xw[0], &inc) / norms_X_col[j]**2
...
faxpy(&n_samples, &tmp, &X[0, j], &inc, &Xw[0], &inc)
```

`fdot` and `faxpy` are direct BLAS calls (`ddot` / `daxpy`).

skglm uses numpy + numba `@njit`. numba's BLAS dispatch on
`X[:, j].dot(Xw)` similarly hits `ddot`.

Skein, with ndarray's `blas` feature off, uses `dot_generic` — pure
Rust loops via `matrixmultiply`. Per the M10.1 microbench, that's
already ~5× faster than any manual Rust loop, but ~3× slower than
true BLAS `ddot` on n=10k vectors.

## Cost taxonomy of the gap

What of the remaining ~17× we owe sklearn (and 5× we owe celer's
warm-started path) comes from each design choice?

| Difference | Estimated impact |
|---|---|
| BLAS `ddot` vs `dot_generic` on per-coord gradient | ~3× — the M10.1 profile floor |
| Adaptive inner tol + periodic gap check (fewer epochs) | ~1.3–1.5× on path solves |
| WS growth from `p0` (smaller initial WS at λ_max) | ~1.1× on the first few λs |
| KKT-priority WS selection (tighter WS at all λ) | ~1.1× |
| Outer stop on duality gap / KKT (instead of `max_delta`) | ~1.0–1.1× |
| Anderson-on-residual vs on-β | unclear, possibly 1.05–1.2× |

Compounded: the non-BLAS items product to roughly **1.5–2×**. BLAS is
the single ~3× lever. Together they put us in glmnet/celer territory
on dense lasso/LS at this size.

## What's worth porting

In rough order of impact-per-LOC:

### High-value, low-friction

**A. Periodic KKT/gap check + adaptive inner tolerance** — restructure
the path-solver inner CD to: run full epochs without per-coord
`max_delta`, compute KKT distance every 10 epochs, break when below
`0.3 × outer_KKT`. This is a structural change to `cd_solve_subset`
+ `path.rs` but doesn't add new dependencies.

**B. WS growth from `p0`** — replace the strong-rule full-set WS at
the first λ with a `p0`-sized WS, grow `2 × nnz` thereafter. Affects
only the early-path performance significantly (cold start), but
those iters do cost in absolute terms.

### High-value, high-friction

**C. ndarray's `blas` feature** with Apple Accelerate (macOS) /
OpenBLAS (Linux/Windows). The single largest lever. CI / wheel
complexity is real but bounded.

### Medium-value structural

**D. KKT priority via `subdiff_distance` per penalty** — add a
`Penalty::subdiff_distance` trait method, replace `strong_rule_screen`
with priority-rank WS selection. Affects all penalties, not just
lasso. Touches every `Penalty` impl.

**E. Anderson on residual instead of β** — switch the Anderson
history from β snapshots (length p) to residual snapshots (length n).
Wins on the n < p regime; less clear at our medium where n > p.
Smaller code change than D.

### Probably skip

**F. Gram-cached CD** — celer doesn't use it; only `gram_cd.py` in
skglm does, marked as a niche solver for `p ≪ n` problems.

## Recommendation

Land **A** next. It restructures `cd_solve_subset` to celer/skglm's
outer/inner shape — adaptive inner tol, periodic gap check, no per-coord
max_delta tracking. Expected ~1.3-1.5× speedup on the medium bench
without new dependencies. **Then land C** for the remaining ~3×.

D, E, B are smaller wins to chase after A and C.
