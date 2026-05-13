# Paper artifact bundle

Generated at 2026-05-13 22:17:14 UTC.
Git rev: `d7bd7270c6b69e99245fd23ad938dd04bc292ad3` (dirty working tree)

Each artifact under `figures/` and `tables/` is the build output of a deterministic driver — never edit them by hand. Re-run the relevant Snakemake rule or driver script to regenerate.

## §Ablation

| Artifact | Driver | Source |
|---|---|---|
| `figures/F10_cv_parallel_speedup.pdf` | `report/run_cv_parallel.py → figures.py cv_parallel_speedup` | results/cv_parallel/cv_parallel.jsonl |
| `figures/F9_screening_parallel.pdf` | `cargo bench → report/ingest_criterion.py → figures.py screening_parallel` | results/criterion.jsonl |

## §Applications

| Artifact | Driver | Source |
|---|---|---|
| `figures/F6_realdata_boxplots.pdf` | `report/run_realdata.py → figures.py realdata_boxplots` | results/realdata/*.jsonl |
| `tables/T5_realdata.tex` | `report/tables.py realdata` | results/realdata/*.jsonl |

## §Correctness

| Artifact | Driver | Source |
|---|---|---|
| `figures/F4_agreement.pdf` | `report/figures.py agreement` | results/scenarios/*.aggregate.json (cells with agreement_vs_skein_mean) |

## §Introduction

| Artifact | Driver | Source |
|---|---|---|
| `figures/F1_coverage_matrix.pdf` | `report/coverage_matrix.py` | benches/v2/runners/registry.py (LADDER) |
| `tables/T1_estimator_surface.tex` | `report/tables.py estimator_surface` | skein_glm.* (introspection) |

## §Recovery

| Artifact | Driver | Source |
|---|---|---|
| `figures/F5_recovery_curves.pdf` | `report/figures.py recovery_curves` | results/scenarios/*.aggregate.json (recovery_per_lambda_mean) |
| `tables/T4_recovery.tex` | `report/tables.py recovery` | results/scenarios/*.aggregate.json (selection_mean, regime=deep) |

## §Reproducibility

| Artifact | Driver | Source |
|---|---|---|
| `tables/T6_environment.tex` | `report/tables.py environment` | results/cells/*.env.json (host_id, BLAS, package versions) |

## §Results

| Artifact | Driver | Source |
|---|---|---|
| `figures/F2_headline_timing.pdf` | `report/figures.py headline_timing` | results/scenarios/*.aggregate.json (size=medium, regime=deep) |
| `figures/F3_scaling_curves.pdf` | `report/figures.py scaling_curves` | results/scenarios/*.aggregate.json (all sizes, regime=deep) |
| `tables/T2_headline_timings.tex` | `report/tables.py headline_timings` | results/scenarios/*.aggregate.json (size=medium, regime=deep) |

## §Selection

| Artifact | Driver | Source |
|---|---|---|
| `figures/F7_ic_selection_accuracy.pdf` | `report/figures.py ic_selection` | results/scenarios/*.aggregate.json (selection_mean) |
| `figures/F8_stability_fdr_power.pdf` | `report/run_stability.py → figures.py stability_fdr_power` | results/stability/stability.jsonl |
