# skein-core microbenchmarks

Run all benches:

```bash
cargo bench -p skein-core
```

Or one at a time:

```bash
cargo bench -p skein-core --bench block_cd
cargo bench -p skein-core --bench lla_outer
cargo bench -p skein-core --bench prox_newton_glm
cargo bench -p skein-core --bench glasso
```

Quick smoke (1-2s per scenario):

```bash
cargo bench -p skein-core --bench <name> -- --quick --noplot
```

## Scenarios

### `block_cd`

#### `serial_vs_parallel/{serial,parallel}/{8,32,128}`

Compare `block_cd_solve_subset` (serial Gauss-Seidel) against
`block_cd_solve_subset_parallel` (Jacobi via `rayon::par_iter`) on
contiguous-block group lasso, group size 4, n=200, λ=0.01.

#### `screening_modes/{off,strong,gap_safe}`

Full path solve (n_lambdas=20) on n_groups=64 group lasso under each
screening strategy.

### `lla_outer` (M12 P2)

Outer-loop convergence on the LLA wrapper for non-convex group MCP.
Captures wall-clock so future short-circuit optimisations (e.g.
M13.4 Phase 2.3 surrogate-fixed-point check) show up here.

#### `lla_outer_gamma/gamma/{1.5, 3.0, 10.0}`

Single-λ group MCP solve at fixed (n=200, p=256, n_groups=64),
varying the MCP shape parameter γ. Smaller γ = more aggressive
nonconvexity = more outer iters expected; γ → ∞ recovers convex
group lasso.

#### `lla_outer_n_groups/n_groups/{16, 64, 256}`

Group MCP solve at fixed γ=3.0, varying group count to expose how
outer-loop work scales with problem size.

### `prox_newton_glm` (M12 P2)

GLM datafits (logistic, Poisson, Cox) reach the M1 separable CD via a
prox-Newton outer loop that re-linearises the loss at every iterate.
This bench measures the IRLS-on-CD overhead independently from the
outer path solver.

#### `prox_newton_glm_logistic/p/{64, 256}`

Single-λ logistic Lasso on synthetic Bernoulli data, n=200.

#### `prox_newton_glm_poisson/p/{64, 256}`

Single-λ Poisson Lasso with `μ_i = exp(x_iᵀ β_true)`, n=200.

### `glasso` (M12 P2)

Graphical lasso family — single-population block-coordinate solver
(M11.1) and joint ADMM across populations (M11.2).

#### `glasso_single/p/{20, 50, 100}`

L1 single-population glasso on a synthetic SPD covariance built from
n=200 random samples. Wall-clock should grow ~p³.

#### `glasso_joint/K/{2, 3}_p=20`

Joint glasso ADMM at fixed p=20, varying population count K.
Group penalty couples the same edge across populations.

## Snapshot results (apple silicon, dev profile, 2026-05-06)

```
serial_vs_parallel/serial/128       ≈  12.6 ms
serial_vs_parallel/parallel/128     ≈ 186   ms      [15× SLOWER than serial]

screening_modes/off                 ≈  45.4 ms
screening_modes/strong              ≈   7.6 ms      [5.9× faster than off]
screening_modes/gap_safe            ≈   8.0 ms      [5.7× faster than off]
```

### What `block_cd` says

**Strong rule + gap-safe deliver real wins** — both screening modes drop
the path solve to ~⅙ of the no-screening cost on this 64-group example.
Strong rule is slightly faster than gap-safe (gap-safe pays for one
matvec to compute the global gradient + a per-group dual feasibility
loop; on convex problems both produce the same β so the user picks
whichever's faster).

**Parallel block-CD is microbenchmark-dominated by Rayon overhead.**
For 128 groups of 4 features each, Jacobi parallel CD is *15× slower*
than serial Gauss-Seidel — the per-group prox + matvec is so cheap
(microseconds) that Rayon's task-spawn cost dwarfs the work. The M2.5
parallel mode pays off when:

- groups are larger (more work per task amortizes overhead)
- the problem has many more groups (8k+) and is correlation-light
- the user is fitting many λ values where each LLA outer iteration's
  inner can dispatch in parallel

A future bench should validate the speedup target (4–8× on
`n_groups ≫ n_threads` with uncorrelated groups) by scaling up to
realistic problem sizes (n=10k+, p=10k+, sparse X via M4's `SparseCSC`).

### Out of scope today

- Frobenius vs operator-norm Lipschitz iter-count comparison: we no
  longer have a Frobenius path; M2.6 replaced it everywhere with the
  power-iteration operator norm. The comparison would require
  reintroducing the loose bound as an opt-in for benchmarking.
- Comparison vs `glmnet` / `ncvreg` / `grpreg`: lives in `benches/`
  (cross-package v1) and `benches/v2/` (publication suite), not this
  microbench tree.
- Large-n / sparse-X scaling: that's the `benches/v2/` suite's job.
  This tree intentionally stays at sizes that fit in a few seconds
  per scenario so it runs on every dev box.
