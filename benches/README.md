# skein benchmarks (M9)

Reproducible local-only benchmark suite for skein vs. comparator
packages. See `ROADMAP.md` § M9 for the full plan.

This is a separate tree from `crates/skein-core/benches/`, which holds
internal criterion microbenches (block-CD scenarios). The work here is
cross-package: speed and correctness against `glmnet`, `ncvreg`,
`grpreg`, `skglm`, `sklearn`, `celer`, and `pyglmnet`.

## Layout

```
benches/
  problems.py          shared synthetic problem generators
  run.py               driver: dispatches scenarios × packages × sizes
  runners/             one runner per package; common ABI in __init__.py
    skein_runner.py
    sklearn_runner.py
    skglm_runner.py
    celer_runner.py
    pyglmnet_runner.py
    r_runner.R         glmnet / ncvreg / grpreg via Rscript
  scenarios/           one driver per (penalty × datafit) combo
    lasso_ls.py
    …
  results/             committed JSON snapshots
  correctness/         cross-package agreement matrices
```

## Running

```bash
# install bench-only deps (skein itself comes from `maturin develop --release`)
pip install scikit-learn skglm celer pyglmnet matplotlib

# run a single scenario across all available packages
python benches/run.py --scenarios lasso_ls --sizes small

# run everything (slow; produces all the snapshots in results/)
python benches/run.py --scenarios all --packages all --sizes small,medium,large

# R comparators require a local R install with the listed packages:
#   install.packages(c("glmnet", "ncvreg", "grpreg", "jsonlite"))
# Missing R is non-fatal — the R runner is skipped with a warning.
```

## Methodology

- **Same convergence tolerance for everyone.** Each runner accepts a
  `tol` argument and is configured uniformly per scenario. Documented
  per-scenario.
- **Shared λ-grid.** skein computes the grid via its standard
  `lambda_max` + geometric path; the same grid is fed to every other
  package. Avoids the apples-to-oranges where one package converges
  earlier on a coarser grid.
- **skein reported with and without screening.** Several comparators
  have no equivalent. Reporting both means the chart is honest both
  ways.
- **Wall-clock fit time only** (not predict, not setup). Path solves
  measure the whole path, not single-λ.
- **Three sizes** per scenario: small (n=1k, p=100), medium (n=10k,
  p=1k), large (n=100k, p=10k). Skipped if a runner errors or OOMs.

## Result schema

One JSON file per scenario, append-style so partial runs are
recoverable:

```json
{
  "scenario": "lasso_ls",
  "host_id": "darwin-arm64-m2",
  "runs": [
    {
      "package": "skein",
      "version": "0.4.0",
      "n": 10000, "p": 1000, "n_groups": null,
      "lambda_grid_len": 100,
      "fit_time_s": 1.234,
      "n_iter": null,
      "final_obj": 0.0123,
      "active_set_size": 17,
      "screening": "strong",
      "timestamp": "2026-05-10T18:00:00Z"
    },
    …
  ]
}
```

## Snapshot host

Snapshots committed under `results/` are tagged with `host_id`. Today's
host is recorded at the top of each result file. Re-running on a
different host produces a new snapshot — do not overwrite without
noting the host change.

## Status

Skeleton only. Real bench numbers land as M9.3 progresses.
