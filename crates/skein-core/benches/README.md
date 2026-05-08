# skein-core microbenchmarks

Run with:

```bash
cargo bench -p skein-core --bench block_cd
```

For a quick smoke run:

```bash
cargo bench -p skein-core --bench block_cd -- --quick
```

## Current scenarios

### `serial_vs_parallel/{serial,parallel}/{8,32,128}`

Compare `block_cd_solve_subset` (serial Gauss-Seidel) against
`block_cd_solve_subset_parallel` (Jacobi via `rayon::par_iter`) on
contiguous-block group lasso, group size 4, n=200, lambda=0.01.

### `screening_modes/{off,strong,gap_safe}`

Full path solve (n_lambdas=20) on n_groups=64 group lasso under each
screening strategy.

## Snapshot results (apple silicon, dev profile, 2026-05-06)

```
serial_vs_parallel/serial/128       ≈  12.6 ms
serial_vs_parallel/parallel/128     ≈ 186   ms      [15× SLOWER than serial]

screening_modes/off                 ≈  45.4 ms
screening_modes/strong              ≈   7.6 ms      [5.9× faster than off]
screening_modes/gap_safe            ≈   8.0 ms      [5.7× faster than off]
```

### What these say

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
- LLA outer iteration counts vs CD-only: needs a SCAD/MCP nonconvex
  benchmark scenario.
- Comparison vs `glmnet` / `ncvreg` / `grpreg`: lives in M8 alongside
  the published comparison page.
