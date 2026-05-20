# At-scale benchmarks (H1)

Headline cells in `benches/v2/` historically topped out at
`medium` (n=10k, p=1k), with `large` (n=50k, p=5k) populated only
for the Lasso/LS family and `xlarge` (n=100k, p=10k) marked
"opt-in only." H1 (post-v1.0 roadmap) closes that gap by promoting
`xlarge` into the headline matrix and adding a per-PR canary so
regressions at the size that actually matters to users — n on the
order of 10⁵ — surface before they ship.

## What runs where

| Tier                  | n × p              | Headline scenarios                                                  | Comparators                       | Cadence                |
|-----------------------|--------------------|---------------------------------------------------------------------|-----------------------------------|------------------------|
| `small`               | 1 000 × 200        | every scenario                                                      | full set (Python + R)             | per release            |
| `medium`              | 10 000 × 1 000     | every scenario                                                      | full set (Python + R)             | per release            |
| `large`               | 50 000 × 5 000     | `ls_lasso`, `ls_mcp`, `ls_scad`, `ls_elasticnet`                    | full set (Python + R)             | per release            |
| **`xlarge`** *(H1)*   | 100 000 × 10 000   | `ls_lasso`, `ls_mcp`, `logistic_lasso`, `ls_group_lasso`            | reduced — see below               | maintainer overnight   |
| `large` (per-PR canary) | 50 000 × 5 000   | `ls_lasso / sparse / seed 0 / skein`                                | skein only, `--trials 1`          | every PR (bench-smoke) |

The per-PR canary uses the existing `large` tier (not `xlarge`)
because a single `xlarge` cell at n=100 000 × p=10 000 with one
warmup + one timed trial still exceeds the 15-minute CI budget
once the release maturin build is included. The `large` canary
catches the same class of regression (memory-bandwidth wall,
strong-rule misfire at scale) at a budget that fits on the free
ubuntu-latest runner.

## Comparator asymmetry at `xlarge`

R comparators and `sklearn.linear_model.coordinate_descent` are
dropped at `xlarge` because they hit memory or wall-clock ceilings
before n=100 000 × p=10 000 dense. The exclusions are mechanical,
not philosophical, and are captured in `paper/manifest.json` under
`at_scale_comparator_gap` so downstream paper figures can flag the
gap rather than silently dropping the comparator. Summary:

| Scenario          | Included at `xlarge`          | Excluded — and why                                                                |
|-------------------|-------------------------------|-----------------------------------------------------------------------------------|
| `ls_lasso`        | skein, celer, skglm           | `sklearn` (CD OOM on dense 8 GB X); `glmnet` (32-bit `nlam × nvar` index space)   |
| `ls_mcp`          | skein, skglm                  | `ncvreg` (`p × p` intermediate ≈ 800 MB at p=10k)                                 |
| `logistic_lasso`  | skein                         | `glmnet` (binomial path exceeds per-cell wall ≈ 1 h)                              |
| `ls_group_lasso`  | skein                         | `grpreg` (Fortran core copies `X`, peak RSS ≈ 3× X size)                          |

When a future paper figure consumes the `xlarge` aggregates,
**flag** the missing comparators rather than drop them silently —
a "no R bar shown at xlarge" is information about the comparator,
not about skein.

## Reproducing the `xlarge` matrix

Maintainer-overnight, on a machine with ≥32 GB RAM:

```bash
# Release build with BLAS — `xlarge` under dev profile or
# BLAS-less release falls back to ndarray's pure-Rust matvec and
# costs ~3× wall-clock.
pip install -e '.[bench]'
maturin develop --release --features=blas-accelerate    # macOS
maturin develop --release --features=blas-openblas      # Linux

# Drive only the `xlarge` cells:
cd benches/v2
snakemake --profile profiles/m1-headline \
  $(python -c "
import yaml
cfg = yaml.safe_load(open('config.yaml'))
ids = sorted({s['id'] for s in cfg['headline']
              if 'xlarge' in s['sizes']})
for sid in ids:
    for regime in ('deep', 'sparse'):
        for seed in range(5):
            print(f'results/cells/{sid}__xlarge__{regime}__seed{seed}__skein.jsonl')
")
```

Wall-clock budget: the M13.6 lasso_ls_scaling characterization put
the per-fit time at ~500 s for skein on an Apple M1 + Accelerate at
n=100k × p=10k dense regime. With 1 warmup + 5 timed trials × 2
regimes × 5 seeds × 4 scenarios × ~3 packages on average, the
`xlarge` matrix is roughly **10–12 hours of wall-clock** on a
single laptop. Linux + OpenBLAS should land in the same order of
magnitude.

## R-anchor fixtures at scale

`tests/fixtures/generate.R` ships `*_large` problems for
LS + logistic Lasso/MCP. The default size is **n=5 000, p=500** —
a 10× extension over the M14c.3 mid-tier (n=500, p=100) that's
large enough to catch scale-dependent regressions (Phase 2.3
strong-rule misfire, LLA local-min drift at high p) while keeping
each JSON ~25 MB raw.

The roadmap aspirational size is n=50 000, p=2 000, which produces
~800 MB JSON per fixture. Override via environment variable when
regenerating on a machine with adequate RAM:

```bash
SKEIN_FIXTURE_LARGE_N=50000 SKEIN_FIXTURE_LARGE_P=2000 \
  Rscript tests/fixtures/generate.R
```

Like the mid-tier, `*_large.json` files are **never committed**.
The Python tests use `_skipped_if_missing_optional` so CI silently
skips them; the maintainer regenerates locally when gating a
scale-dependent regression. See
`tests/test_r_regression.py::test_*_large_*` for the four
`*_large` test functions.

## Per-PR canary

`.github/workflows/bench-smoke.yml` has two parallel jobs:

- **`smoke`** — two `small/sparse` cells under dev maturin profile,
  ~1 minute total. Catches pipeline breakage (renamed runner,
  broken Snakemake rule, missing module export).
- **`smoke-at-scale`** — one `large/sparse` cell under release
  maturin + OpenBLAS, `--trials 1`. Target ≤15 minutes; emits a
  workflow warning if the cell itself exceeds 10 minutes wall-clock.
