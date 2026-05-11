# skein

Weighted structured nonconvex sparse models. Rust core + Python API.

> **Documentation:** the [docs site](docs/index.md) has the full
> conceptual reference (penalties, datafits, weights, backends),
> porting guides for `glmnet` / `ncvreg` / `grpreg`, worked examples,
> and an auto-generated API reference. Hosted on Read the Docs once
> the project is connected (config in `.readthedocs.yaml`); preview
> locally with `mkdocs serve`. CI builds it `--strict` on every PR.

`skein` targets a niche that's well-served in R (`grpreg`, `ncvreg`) but
missing in Python at production quality: nonconvex group-structured
penalties (group MCP, group SCAD, sparse-group nonconvex) with first-class
support for *weights along three axes* — per-sample, per-feature, and
per-group.

## Status

v0.1 development. Core algorithms and the headline GLM family are in
place; design-matrix backends (sparse, mmap, chunked) are next. See
[ROADMAP.md](ROADMAP.md) for the full plan.

**Done so far:**

- **Solvers** — production CD core (path solver, strong rule + KKT
  verification, gap-safe screening, Anderson acceleration); group block-CD
  with LLA outer loop for nonconvex group penalties; Rayon-parallel
  group sweeps; operator-norm Lipschitz via power iteration.
- **Datafits** — least squares, binomial logistic, Poisson (log link),
  Cox PH (Breslow ties). All glued together by a `GlmDatafit` trait that
  exposes a weighted-LS surrogate; the M1/M2 inner solvers absorb every
  GLM unchanged.
- **Penalties** — MCP, SCAD, group lasso, group MCP, sparse-group lasso,
  sparse-group MCP. Per-feature and per-group weights honored
  throughout.
- **Python** — sklearn-compatible estimators for every (datafit ×
  penalty) combination; type stubs; warm-started λ-paths; standardization
  with original-scale `coef_` / `intercept_` recovery (dense backend).

**M8 (Distribution & DX) is done:** CI + cibuildwheel + Read the Docs +
25-page mkdocs site (concepts + R-porting + extending + examples + API
ref) + R numerical regression suite vs glmnet/ncvreg/grpreg + stable
Rust API contract. The library is `pip install`-able once published,
documented end-to-end, and pinned against R reference fits so we don't
silently drift.

**Coming next:** algorithmic features — M5.x adaptive weights and
stability selection are the next high-value milestones; both leverage
the existing per-feature/per-group weight axes that are already wired
through every solver.

## Layout

```
crates/skein-core/   pure Rust: traits + algorithms (no Python)
crates/skein-py/     PyO3 bindings (cdylib → skein_glm._core)
python/skein/        sklearn-compatible estimators + ABCs for extensions
tests/               pytest smoke tests
benches/             criterion (Rust) + asv (Python)
```

The Rust traits (`DesignMatrix`, `Datafit`, `GlmDatafit`, `Penalty`,
`GroupPenalty`) and their Python ABC mirrors (`skein.penalties.Penalty`,
etc.) are the extension surface for downstream per-paper projects.

## Quick start

```python
import numpy as np
from skein import MCPPathRegressor, LogisticGroupMCPPathRegressor, CoxMCPRegressor

# Nonconvex sparse least squares with a λ-path.
rng = np.random.default_rng(0)
n, p = 200, 50
X = rng.standard_normal((n, p))
y = X[:, :3] @ np.array([1.5, -2.0, 0.8]) + 0.1 * rng.standard_normal(n)
model = MCPPathRegressor(gamma=3.0, n_lambdas=50, standardize=True).fit(X, y)
print(model.coefs_[-1, :5], model.intercepts_[-1])

# Logistic + group MCP via LLA, with sklearn-style predict/predict_proba.
groups = np.repeat(np.arange(p // 5), 5)  # 5 features per group
y_bin = (X[:, :3].sum(axis=1) > 0).astype(float)
clf = LogisticGroupMCPPathRegressor(groups=groups, gamma=3.0, n_lambdas=20).fit(X, y_bin)
proba = clf.predict_proba(X)  # shape (n, n_lambdas)

# Cox PH with right-censored survival data.
time = rng.exponential(1.0 / np.exp(X[:, :3].sum(axis=1)))
event = rng.uniform(size=n) < 0.7
cox = CoxMCPRegressor(lambda_=0.01, gamma=3.0).fit(X, time, event.astype(float))
risk = cox.predict(X)  # prognostic index η
```

Every regressor follows the same `(datafit) × (penalty)` × `({,Path}Regressor)`
naming scheme. The path variants warm-start across λ; their `coefs_` /
`intercepts_` (where applicable) are 2D arrays indexed by λ.

## Performance

skein is benchmarked against sklearn / skglm / celer / glmnet / ncvreg
on shared λ-grids via the harness under `benches/`. Headline numbers
(Apple M1, 16 GB; median of N timed trials after a warm-up):

| scenario | size | skein | next-fastest comparator |
|---|---|---|---|
| Lasso LS — deep   | medium (n=10k, p=1k)  | 1.17 s     | sklearn 0.125 s |
| Lasso LS — sparse | medium                | 0.78 s     | sklearn 0.099 s |
| MCP   LS — deep   | medium                | **1.37 s** | skglm 3.35 s    |
| MCP   LS — sparse | medium                | **0.75 s** | ncvreg 1.17 s   |
| MCP   LS — deep   | large (n=100k, p=10k) | **510 s**  | skglm 666 s     |
| MCP   LS — sparse | large                 | **497 s**  | skglm 702 s     |
| SCAD  LS — deep   | medium                | **1.78 s** | ncvreg 7.99 s   |
| SCAD  LS — sparse | medium                | **0.90 s** | ncvreg 1.86 s   |

skein is the fastest on every nonconvex row across every size; on
convex lasso/LS the sklearn Cython `lasso_path` remains the floor at
~8–9× faster on the medium bench. See
[`docs/benchmarks/mcp_ls.md`](docs/benchmarks/mcp_ls.md) and
[`docs/benchmarks/scad_ls.md`](docs/benchmarks/scad_ls.md) for the
full nonconvex write-ups (correctness matrices + methodology +
per-size tables) and
[`docs/perf/lasso_ls_profile.md`](docs/perf/lasso_ls_profile.md) for
the lasso/LS profiling work that drove M10.

Reproduce with `python benches/run.py --scenarios mcp_ls
mcp_ls_sparse --sizes small,medium`.

## Build

```bash
# Rust core only (fast iteration on algorithms)
cargo test -p skein-core

# Full Python package (requires maturin in your env)
maturin develop --release
pytest
```

## License

MIT.
