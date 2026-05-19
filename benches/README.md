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
- **Five sizes** per scenario:
  - small (n=1k, p=100)
  - medium (n=10k, p=1k)
  - large (n=20k, p=2k) — 4× medium, fits on a 16 GB host single-process
  - xlarge (n=50k, p=5k) — matches v2's headline `large`; opt-in,
    single-process at this size pushes past 16 GB once comparator
    copies stack
  - xxlarge (n=100k, p=10k) — opt-in, requires 32+ GB; the legacy R
    runner (JSON transport) OOMs here (v2 ships an Arrow-IPC variant)

  Skipped if a runner errors or OOMs.

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

Live numbers committed for: `lasso_ls`, `lasso_ls_sparse`, `mcp_ls`,
`mcp_ls_sparse`, `scad_ls`, `scad_ls_sparse` at sizes **small** and
**medium**. See `results/*.json`.

## Open gaps (ROADMAP M12-P1)

- **`large` snapshots are skein-only.** On the canonical 16 GB Apple
  Silicon bench host, running all comparators at `large` in a single
  Python process pushes the OS into swap-thrashing because comparators
  retain internal X copies across calls. The v2 paper bundle uses
  per-cell subprocess isolation precisely to avoid this; v2 also has
  zero `large` cells committed at its larger `large = (50k, 5k)`. v1's
  contract is therefore narrower: commit **skein-only large snapshots**
  as a regression gate against future M13 perf changes. Cross-package
  comparison at scale lives in v2:

  ```bash
  python benches/run.py \
      --scenarios lasso_ls lasso_ls_sparse mcp_ls mcp_ls_sparse scad_ls scad_ls_sparse \
      --packages skein \
      --sizes large
  ```

- **`xlarge` / `xxlarge` are opt-in only.** xlarge (n=50k, p=5k) needs
  ~12 GB headroom for skein at deep-tail paths and was tried + rolled
  back during M12-P1 — pin it to a 32+ GB host. xxlarge (n=100k, p=10k)
  needs ~32 GB headroom and the legacy r_runner.R JSON transport OOMs
  on it; v2 ships an Arrow-IPC variant. Do **not** mix xlarge/xxlarge
  snapshots from a different host into the same `results/*.json` (file
  is `host_id`-tagged at the top level).

- **No glasso scenarios.** M11 (graphical models) shipped without
  bench coverage — `sklearn.covariance.GraphicalLasso` and R
  `glasso` would be the natural comparators. Adding this requires
  either extending the `Problem` dataclass to carry a covariance /
  precision-style problem (no `y`) or scaffolding a parallel
  `cov_problem.py`. Tracked as a sub-item of M12-P1; not yet started.
