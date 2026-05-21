# Benchmarks

The skein-glm benchmark surface has three layers:

## 0. Headline summary

[`speed.md`](speed.md) is the one-page consolidation of skein's
wall-clock headline numbers with explicit provenance (host_id, BLAS,
skein version, snapshot date, git rev). Start here if you want to
know "how fast is skein vs glmnet / ncvreg / skglm / celer / grpreg on
the canonical medium cell".

## 1. Per-scenario notes (this directory)

Hand-curated v1 benchmark pages from `benches/`, each with detailed
headline numbers and correctness summaries for one (datafit, penalty)
combination. These are stable and won't be rewritten:

- [`mcp_ls`](mcp_ls.md) — MCP / Gaussian-LS vs skglm + ncvreg
- [`scad_ls`](scad_ls.md) — SCAD / Gaussian-LS vs ncvreg
- [`lasso_ls_correctness`](lasso_ls_correctness.md) — Lasso / LS cross-package agreement
- [`at_scale`](at_scale.md) — at-scale tier (n ≥ 100k) infrastructure and reproduction recipe (H1 closeout)

## 2. Publication-quality bundle (`benches/v2/`)

The full benchmark suite backing the software paper lives in
`benches/v2/` (see `benches/v2/README.md` in the repo). It produces
ten figures (F1..F10) and five tables (T1..T6) committed to `paper/`
(see `paper/README.md`) — every artifact regenerable from a clean
checkout via Snakemake:

```bash
pip install -e '.[bench]'
maturin develop --release
cd benches/v2 && snakemake --profile profiles/m1-headline
```

The bundle covers:
- **Coverage matrix** (F1, T1) — every public estimator vs every comparator
- **Headline timings + scaling curves** (F2, F3, T2)
- **Cross-package agreement** (F4) — per-λ Jaccard / sign / rel-L2
- **Recovery on synthetic truth** (F5, T4) — support F1, β-RMSE
- **Real-dataset case studies** (F6, T5) — Riboflavin, Leukemia, PBC, Birthwt
- **IC selection** (F7) and **stability selection** (F8)
- **Screening + parallelism ablation** (F9) — from criterion microbenches
- **CV threading speedup** (F10) — validates the M5.x-c 2.3-2.5× claim

See `benches/v2/README.md` for the design rationale (why this suite
exists, what's in it, how to run, reproducibility contract) and
`paper/BUNDLE.md` for the artifact provenance manifest after a run.

## CI

A lightweight regression canary (`.github/workflows/bench-smoke.yml`)
runs two cells per PR to catch pipeline breakage. The full headline
matrix is a maintainer-driven overnight job on a release build, not a
per-PR check.
