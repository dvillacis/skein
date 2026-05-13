# `benches/v2` — Publication-Quality Benchmark Suite for `skein-glm`

This directory contains the benchmark, accuracy, and reproducibility
machinery used to back the claims in the `skein-glm` software paper
(JMLR-MLOSS / JOSS target). Everything is driven by a single
[Snakemake](https://snakemake.readthedocs.io/) DAG that turns the
declarative `config.yaml` into JSON-Lines snapshots and then into the
publication artifacts under `../paper/`.

The original benchmark harness at `benches/` is **not** removed —
v2 sits alongside it. The old harness keeps its committed snapshots
so the historical numbers in `docs/benchmarks/*.md` remain
reproducible during the migration.

---

## 1. Why this suite exists

A software paper about `skein-glm` has to convincingly argue three
distinct claims, and the benchmark suite has to produce evidence for
each one:

| Claim          | What the suite produces                                          |
|----------------|------------------------------------------------------------------|
| **Correctness** | Per-λ cross-package agreement (Jaccard / sign / rel-L2) against glmnet, ncvreg, grpreg, sklearn, skglm, celer wherever a direct comparator exists. |
| **Speed**       | Median wall-clock timings with 5-seed confidence intervals **as scaling curves** in (n, p), not just three discrete sizes; plus the headline bar chart for the medium size. |
| **Reach**       | Statistical recovery (support F1, β-RMSE) and real-dataset prediction error for the ~60% of skein's estimator surface that **has no external comparator** — group MCP/SCAD on GLMs, sparse-group nonconvex, Cox MCP/SCAD, joint graphical, debiased GLM, stability selection. |

Existing benches measure only speed on six LS scenarios, at two sizes,
with hand-curated markdown tables. This suite generalizes that to the
entire public estimator surface and adds the statistical-quality
evidence a software paper needs.

---

## 2. What's in it

### Directory layout

```
benches/v2/
  Snakefile                 # DAG: scenarios → cells → aggregates → figures/tables
  config.yaml               # the matrix: scenarios × sizes × seeds × packages
  envs/                     # lockfiles + auto-captured system.json
  profiles/                 # snakemake profiles (m1-headline, appendix, smoke)
  scenarios/                # one .py per (datafit, penalty) family
  simulators/               # known-truth data generators (β_true exposed)
  datasets/                 # cached real-dataset loaders
  runners/                  # one adapter per comparator package
  metrics/                  # agreement, recovery, deviance, selection
  report/                   # JSONL → PDF figures and LaTeX tables
  results/                  # JSONL snapshots (one file per scenario)
```

### Scenario taxonomy

A scenario is a tuple `(datafit, penalty, design, regime)`:

| Axis     | Values                                                                   |
|----------|--------------------------------------------------------------------------|
| Datafit  | LS, Logistic, Poisson, Cox, Multinomial (K=3, K=10), Graphical            |
| Penalty  | Lasso, MCP, SCAD, EN, grLasso, grMCP, sgLasso, sgMCP, sgSCAD              |
| Design   | dense, sparse (CSC, 5% density)                                          |
| Regime   | `deep` aka **dense** (λ_min/λ_max=1e-3; saturated tail), `sparse` (5e-2; stops near support recovery), `screening` (1e-4), `CV` (10-fold). Internal key is `deep` for backward compat; user-facing prose says "dense" because the regime name describes the *solution* density at the tail, not the path geometry. |
| Size     | small (n=1k, p=200), medium (n=10k, p=1k), large (n=50k, p=5k), xlarge (n=100k, p=10k; opt-in) |
| Seeds    | 5 for headline, 1 for appendix                                           |

The **headline** matrix (run on every release) and the **appendix**
matrix (one cell per estimator to fill the coverage table) are both
listed declaratively in `config.yaml`.

---

## 3. How to run

The suite expects the bench extras to be installed:

```bash
pip install -e '.[bench]'    # snakemake, pyarrow, seaborn, jupytext, comparators
```

R comparators (glmnet, ncvreg, grpreg, glasso) are invoked via
`Rscript` + the `arrow` R package (feather IPC transport, replaces
the legacy 8 GB JSON round-trip). Install them with:

```bash
bash benches/v2/envs/install_r_deps.sh    # arrow + glmnet + ncvreg + grpreg + glasso
```

`arrow` takes ~5–10 min to compile on first install (no prebuilt
macOS binary on CRAN); the others are fast. See `envs/r-bench.lock`
for the pinned versions.

### Profiles

```bash
# Headline suite (~12h on an M1/M2 laptop): everything for the main paper body.
snakemake --profile profiles/m1-headline

# Appendix suite: every public estimator at one cell, fills the coverage table.
snakemake --profile profiles/appendix

# Smoke: two cells, used by CI as a regression canary.
snakemake --profile profiles/smoke
```

### Single cells (debugging)

```bash
# Run one explicit cell:
snakemake results/cells/ls_mcp__medium__seed1__skein.jsonl

# Re-generate one figure from cached aggregates:
snakemake paper/figures/F2_headline_timing.pdf
```

All wall-clock numbers are reported as median over 5 timed trials
after one warm-up. The trial array is preserved in the JSONL row so
the figure builder can show min/max bars without rerunning.

---

## 4. How a cell becomes a figure (worked example)

The DAG for `F2_headline_timing.pdf` looks like this:

```
config.yaml + scenarios/ls_mcp.py
            │
            ▼
  rule run_cell  →  results/cells/ls_mcp__medium__seed{0..4}__{skein,skglm,ncvreg}.jsonl
            │            (one Snakemake job per cell — parallelizable)
            ▼
  rule aggregate →  results/scenarios/ls_mcp.aggregate.json
            │            (medians + CIs across seeds + comparators)
            ▼
  rule make_figure → paper/figures/F2_headline_timing.pdf
            │            (consumes aggregates from every headline family)
            ▼
  rule paper_bundle → paper/  (symlinks/copies for the final paper artifact)
```

Every cell is independent — Snakemake can fan out across cores. Each
`run_cell` invocation:
1. Captures the environment (`envs/system.json` + `pip freeze` + `R sessionInfo`)
2. Builds the problem via the simulator (`simulators/<family>_truth.py`)
3. Calls the runner (`runners/<package>_runner.py`) `1 + n_trials` times
4. Computes metrics (`metrics/{agreement,recovery,deviance,selection}.py`)
5. Writes one JSONL row to `results/cells/<cell_id>.jsonl`

---

## 5. Comparator ladder + caveats

For each cell, the suite picks comparators by walking this fallback
ladder:

1. **Direct external comparator** — same penalty, same datafit, same
   convergence semantics. Example: `ncvreg` for scalar Gaussian MCP,
   `grpreg` for group MCP/LS. These cells produce agreement panels in
   the paper.

2. **Surrogate comparator** — when no nonconvex external implementation
   exists. Example: glmnet's Lasso vs. skein's Logistic-MCP. Surrogates
   are **explicitly labeled** in plot legends ("convex surrogate, not
   equivalent"). Useful only for *speed* comparison, never agreement.

3. **Internal-only** — no comparator at all (joint graphical, debiased
   GLM, sparse-group nonconvex GLMs, multinomial nonconvex). For these,
   the paper reports statistical recovery vs. ground truth instead of
   cross-package agreement.

The `runners/registry.py` module maps each (datafit, penalty) tuple to
the ladder, with an explicit `level: direct | surrogate | none` tag
that propagates into the JSONL output and the figure legends.

---

## 6. Synthetic-truth simulator parameterization

The existing `benches/problems.py` already exposes `beta_true` in the
`Problem` dataclass — v2 reuses it directly. New v2 simulators add
controllable knobs that are essential for recovery curves:

| Knob               | Values                                          | Used in                  |
|--------------------|-------------------------------------------------|--------------------------|
| Correlation        | Toeplitz ρ ∈ {0, 0.5, 0.9}, equicorrelation, block | All families             |
| SNR (LS)           | {0.5, 1, 3, 10}                                 | F5 recovery curves       |
| Sparsity           | s = k · √p, k ∈ {0.5, 1, 2}                     | All families             |
| Censoring (Cox)    | {0.3, 0.6, 0.9}                                 | Cox simulator only       |
| Group block corr   | ρ_w ∈ {0.5, 0.9}, ρ_b ∈ {0, 0.3}                | Group simulators         |
| Graph topology     | banded, hub, random sparse Ω                    | Graphical simulators     |

A 3×3×3 grid (correlation × SNR × sparsity) at the small size, with
5 seeds, generates the recovery curves with shaded error bands.

---

## 7. Real datasets — provenance, license, preprocessing

| Dataset           | n     | p    | Datafit  | Source / License                                | Preprocessing                            |
|-------------------|-------|------|----------|-------------------------------------------------|------------------------------------------|
| Riboflavin        | 71    | 4088 | LS       | `hdi` R package (GPL-3)                         | None — column-standardize at fit time    |
| Leukemia (Golub)  | 72    | 7129 | Logistic | `golubEsets` Bioconductor (Artistic-2.0)        | log2 + quantile normalize                |
| TCGA-BRCA subset  | ~500  | 2000 | Cox      | TCGA portal (open access)                       | Hallmark gene sets → groups; right-censor |
| Bardet / birthwt  | 189   | 24   | LS group | `grpreg` package (GPL-2)                        | Indicator-coded factors → groups         |
| Co-expression     | 100   | 200  | Graphical | Synthetic from GTEx subset (open access)        | log-CPM + winsorize                      |

Loaders cache to `~/.cache/skein-bench/<dataset>/` with SHA-256
checksums committed to the loader source. Re-fetching is gated on
checksum mismatch.

---

## 8. Reproducibility contract

- **Seeds**: every cell's `seed` is part of the cell ID; rerunning the
  same cell ID is bit-for-bit reproducible on coefficients (timing
  within 10% modulo CPU thermal variance).
- **Env capture**: every cell writes `env.json` with `platform.uname()`,
  CPU model (`sysctl -a` on macOS), BLAS link (`otool -L`),
  `pip freeze`, `R -e 'sessionInfo()'`, `git rev-parse HEAD`, dirty-tree
  flag. The paper's T6 table is auto-rendered from these files.
- **Host tagging**: every JSONL row carries `host_id` (hash of CPU
  model + BLAS + cores). Aggregators refuse to mix host_ids in a
  single figure — that prevents "I ran half on the M1 and half on the
  Linux box" foot-guns.
- **Lockfiles**: `envs/skein-bench.lock` (uv-style pinned Python),
  `envs/r-bench.lock` (renv).
- **Determinism**: figures are generated from JSONL only; rebuilding
  figures does not rerun benchmarks.

---

## 9. Known limitations

- **M1/M2 only for headline.** The committed paper figures are all
  produced on Apple Silicon with the Accelerate BLAS. An OpenBLAS /
  Linux comparison would require a second host and is currently
  scoped out — mentioned as future work in the paper's discussion.
- **R OOM at xlarge**: the legacy `benches/runners/r_runner.R` uses
  JSON transport which OOMs at n=100k, p=10k. v2 replaces this with
  Arrow IPC (feather) via `pyarrow` + the R `arrow` package. The fix
  is in `runners/_r_io.py` and the v2-only `r_runner.R`. The old
  scenarios continue to use the legacy transport at small/medium.
- **Multinomial nonconvex** and **MultiTask** have no external
  comparators at all — appendix-only with recovery metrics.
- **Cox Efron ties** landed as an M3.5 follow-up (M3 row in ROADMAP);
  comparator agreement is reported for both Breslow (default) and
  Efron variants.

---

## 10. Extension guide

### Add a new scenario

1. Add a `(datafit, penalty)` block to `config.yaml` under either
   `headline:` or `appendix:`.
2. Create `scenarios/<datafit>_<penalty>.py` exposing
   `run(*, runner, package, size, seed, tol, n_lambdas, trials)`.
   Use `scenarios/_template.py` as the starting point.
3. If the simulator doesn't exist, add `simulators/<family>_truth.py`.
4. If a new comparator is needed, add `runners/<package>_runner.py`
   and register it in `runners/registry.py`.
5. `snakemake -n` will show the new DAG nodes; run a single cell to
   verify, then full profile.

### Add a new comparator

1. Implement the `Runner` Protocol from `runners/__init__.py`
   (`is_available()` and `fit()` returning `RunResult`).
2. Register supported `(datafit, penalty)` tuples with their ladder
   level (`direct` / `surrogate`) in `runners/registry.py`.
3. Pin the package version in `envs/{skein-bench,r-bench}.lock`.

### Add a new figure or table

Figures and tables are pure functions of `results/scenarios/*.aggregate.json`:

1. Add a builder function to `report/figures.py` or `report/tables.py`.
2. Add a `make_figure` / `make_table` rule wiring in `Snakefile`.
3. Reference the new artifact path from `paper_bundle.py` so it lands
   in `paper/` deterministically.
