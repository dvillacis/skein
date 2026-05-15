# Paper artifacts

This directory holds (a) the **build output** of the publication
benchmark suite at `../benches/v2/` and (b) the human-authored
software-paper manuscript that wraps it. Auto-regenerated artifacts
(`figures/`, `tables/`) must not be edited by hand; the manuscript
(`manuscript.tex`, `references.bib`) is the hand-authored narrative.

## Layout

```
paper/
  manuscript.tex   # software paper draft (JMLR-MLOSS target)
  references.bib   # bibliography
  figures/         # *.pdf, vector, locked style (see report/figures.py)
  tables/          # *.tex (booktabs) + *.md mirror
```

## Build the manuscript

```bash
cd paper && pdflatex manuscript.tex && bibtex manuscript && \
    pdflatex manuscript.tex && pdflatex manuscript.tex
```

The skeleton is portable — compiles against any LaTeX `article`
class. For JMLR-MLOSS submission, swap the documentclass to the
JMLR-MLOSS template (see comment block at the top of
`manuscript.tex`) before final compilation.

## How to regenerate from a clean checkout

```bash
# install bench extras (snakemake, pyarrow, comparators)
pip install -e '.[bench]'
maturin develop --release

# headline figures (~12h on Apple Silicon)
cd benches/v2 && snakemake --profile profiles/m1-headline

# one figure only (re-uses cached aggregates)
snakemake paper/figures/F2_headline_timing.pdf
```

## What goes where in the paper

| Artifact                              | Section       | Source                         |
|---------------------------------------|---------------|--------------------------------|
| `figures/F1_coverage_matrix.pdf`      | Intro         | `benches/v2/runners/registry.py` |
| `figures/F2_headline_timing.pdf`      | §Results      | aggregate of medium/deep cells |
| `figures/F3_scaling_curves.pdf`       | §Results      | all sizes, deep regime         |
| `figures/F4_agreement.pdf`            | §Correctness  | direct-comparator cells        |
| `figures/F5_recovery.pdf`             | §Recovery     | synthetic-truth cells          |
| `figures/F6_realdata_boxplots.pdf`    | §Applications | real-dataset CV folds          |
| `figures/F7_ic_selection.pdf`         | §Selection    | AIC/BIC/EBIC accuracy          |
| `figures/F8_stability.pdf`            | §Selection    | StabilitySelection vs nominal  |
| `figures/F9_screening_parallel.pdf`   | §Ablation     | criterion microbench JSON      |
| `figures/F10_cv_parallel.pdf`         | §Ablation     | CV threaded vs serial          |
| `tables/T1_estimator_surface.tex`     | Intro         | hand-curated + introspection   |
| `tables/T2_headline_timings.tex`      | §Results      | aggregate medium/deep          |
| `tables/T3_agreement.tex`             | §Correctness  | direct-comparator cells        |
| `tables/T4_recovery.tex`              | §Recovery     | synthetic-truth cells          |
| `tables/T5_realdata.tex`              | §Applications | real-dataset CV results        |
| `tables/T6_environment.tex`           | §Reproducibility | env.json sidecars           |

F4–F10 and T3–T5 land here as Phases C–G fill in.
