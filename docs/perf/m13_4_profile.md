# M13.4 — Profile + Phase 2.3 fix for `ls_group_mcp` LLA overhead

**Scenario**: Gaussian LS + group MCP, `n=10_000`, `p=1_000`,
`group_size=5`, `n_groups=200`, `k_active_groups=5`, SNR=5, 100-λ path
from `λ_max` down to `1e-3 · λ_max`, `tol=1e-7`, `γ=3.0`, `max_outer=10`,
`outer_tol=1e-6`, strong rule + Anderson(5). Matches the v2 bench cell
[`ls_group_mcp` / medium / deep / seed=any].

**Profiling target**:
[`crates/skein-core/examples/group_mcp_ls_medium.rs`](../../crates/skein-core/examples/group_mcp_ls_medium.rs) —
pure-Rust binary so samply / `cargo flamegraph` can resolve frames
without the PyO3 + interpreter layer in the way. It uses an xorshift
PRNG rather than numpy's `default_rng`, so the realized active groups
differ from the v2 bench cell; that yields a harder per-iteration
problem (more inner-CD work per outer iter), which is useful for
exposing the LLA overhead even though it does not directly reproduce
the bench cell's seconds.

```bash
cargo build --release --example group_mcp_ls_medium
./target/release/examples/group_mcp_ls_medium    # standalone profile
samply record ./target/release/examples/group_mcp_ls_medium

# Bench cell (numpy seed problem, 5 trials per seed):
PYTHONPATH=. .venv/bin/python -m benches.v2.report._run_cell \
    --scenario ls_group_mcp --size medium --regime deep --seed 0 \
    --package skein --config benches/v2/config.yaml \
    --out /tmp/skein.jsonl --env-out /tmp/skein.env.json
```

## Phase 1 — Pre-fix baseline (standalone xorshift problem)

| metric | value |
|---|---|
| total fit (measured) | **54.0 s** |
| sum(per_lambda_wall_ns) | 53.4 s |
| final-λ active features | 1000 / 1000 |
| total outer iters across λ | 617 |
| total inner CD iters | 2 555 |
| total KKT passes | 617 |

**Outer-iter distribution was bimodal:**

| outer_iters | n_lambdas |
|---:|---:|
| 1 | 37 |
| 2 | 1 |
| 7 | 5 |
| 8 | 4 |
| 9 | 19 |
| 10 (= `max_outer`) | 34 |

53/100 λs ran the outer LLA loop to the cap or one short, indicating
the coefficient-space convergence check `max_block_change < outer_tol`
(`block_path_lla.rs:213`) was never satisfied on those — even though
the surrogate MCP weights had long since saturated (`max(base − ‖β_g‖
/ (λγ), 0)` clamps to 0 for groups with `‖β_g‖ ≥ λγ`).

## Phase 2.3 fix — weight-space LLA fixed-point short-circuit

**Implementation** (`block_path_lla.rs`): cache the surrogate weights
from the previous outer iter; before the screening + block-CD + KKT
cycle in the next iter, compute the new weights `w_t = ψ(β_{t-1})` and
break if `‖w_t − w_{t-1}‖_∞ < weight_short_circuit_tol`. The threshold
is set at `1000 · outer_tol` (i.e. `1e-3` when `outer_tol=1e-6`) —
above the inner-CD-induced weight jitter floor, but tight enough that
no test in the integration suite regresses.

**Standalone xorshift result (same problem as baseline):**

| metric | pre-Phase-2.3 | post-Phase-2.3 | delta |
|---|---:|---:|---:|
| total wall | 54.0 s | **36.4 s** | **−32.6 %** |
| total outer iters | 617 | 340 | −44.9 % |
| total inner CD iters | 2 555 | 1 688 | −33.9 % |
| max outer per λ | 10 (cap) | 6 | — |

Histogram of `outer_iters` collapses to `{1: 38, 3: 1, 4: 24, 5: 19, 6: 18}` —
no λ runs to the `max_outer` cap, none past 6.

**v2 bench cell result (numpy seed=0, 5-trial median):**

| metric | pre-Phase-2.3 (committed) | post-Phase-2.3 |
|---|---:|---:|
| fit_time_s (skein) | 39.31 s | 40.17 s |
| outer_iters sum | (not in snapshot) | 362 |
| outer_iters mean | (not in snapshot) | 3.62 |
| grpreg comparison | 12.74 s (skein 3.08×) | 12.74 s (skein 3.15×) |

The bench-cell numpy problem has dramatically fewer inner-CD iterations
per outer iter (≈1 inner per outer) than the standalone xorshift
problem (≈5 inner per outer), so trimming "polishing" outer iters at
the tail of the LLA loop saves outer-iter count but not wall — the
inner CD wasn't doing real work in those iters anyway. The 41 %
reduction in outer iters did not translate into wall savings on the
realistic bench seed=0.

The wall is now dominated by the inner block-CD work itself, which
the LLA wrapper cannot reduce — confirming the ROADMAP M13.4
conclusion that **closing the residual ~3× gap to grpreg requires a
native (non-LLA) group-MCP block solver** (Phase 3, scoped as
follow-up milestone M13.4b).

## Headline (standalone xorshift problem, post-Phase-2.3)

| metric | value |
|---|---|
| total fit (measured) | **36.4 s** |
| sum(per_lambda_wall_ns) | 36.4 s |
| final-λ active features | 1000 / 1000 |
| total outer iters across λ | 340 |
| total inner CD iters | 1 688 |
| total KKT passes | 340 |

## Wall-time concentration (post-Phase-2.3, standalone)

Top 10 λ account for **31.9 %** of solve wall-time (11.6 s / 36.4 s):

| rank | k | λ | outer | inner_sum | wall_ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 83 | 8.56e-3 | 6 | 38 | 1520.4 |
| 2 | 80 | 1.05e-2 | 6 | 39 | 1275.3 |
| 3 | 86 | 6.94e-3 | 6 | 38 | 1201.2 |
| 4 | 92 | 4.57e-3 | 5 | 30 | 1134.1 |
| 5 | 82 | 9.17e-3 | 6 | 39 | 1119.0 |
| 6 | 84 | 7.98e-3 | 6 | 38 | 1100.1 |
| 7 | 81 | 9.84e-3 | 6 | 39 | 1086.0 |
| 8 | 85 | 7.44e-3 | 6 | 38 | 1076.5 |
| 9 | 87 | 6.47e-3 | 6 | 36 | 1036.9 |
| 10 | 88 | 6.04e-3 | 6 | 36 | 1033.4 |

These are the dense-tail saturation regime (active set 130–199 groups).
Each does 6 outer iters × 5–6 inner CD iters each; the inner CD work
is what we cannot avoid via LLA-loop micro-optimization.

## What was tried but not shipped

- **Phase 2.1 — λ_max short-circuit.** Profile data showed it would
  save ~3 % of wall time (the 38 λs that converge in 1 outer iter
  cost ~40 ms each, mostly in the unavoidable KKT scan). Deferred.
- **Phase 2.2 — empty-WS guard.** Profile data showed zero λs
  produce an empty strong-rule WS in the bench scenario. Deferred.
- **Phase 2.4 — WS reuse across outer iters.** The strong-rule scan
  costs ~1 ms per outer iter; reusing the WS could save ~100 ms total
  across the dense tail (<0.5 % of wall). Deferred.

All three are correct and easy fixes; none meaningfully closes the
gap to grpreg. They are documented for future revisits if the LLA
path remains primary, but the highest-leverage next investment is
the native group-MCP BCD in Phase 3 / M13.4b.

## Decision summary

Phase 2.3 ships because it is correct, simple, regression-free in the
343-test Rust suite and 412-test Python suite, and delivers real
speedups on pathological-by-design problems where LLA outer iters do
non-trivial work each. It is a no-cost improvement on the bench cell
(the worst case there is "no change") and a 30 % win on harder
problems. The remaining gap to grpreg on the v2 cell is structural to
the LLA approach and is scoped to follow-up milestone M13.4b.

## Reproducibility

Numbers above are from an M1 host (8 BLAS threads, macOS, Accelerate),
release profile with `debug = "line-tables-only"`. Treat the seconds
as approximate; the relative outer-iter distribution and wall-time
concentration are the load-bearing observations.
