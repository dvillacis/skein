# skein

Weighted structured nonconvex sparse models. Rust core + Python API.

> **Documentation:** [the docs site](docs/index.md) has the full
> conceptual reference (penalties, datafits, weights, backends), a
> quick-start tour, installation notes, and the roadmap. `mkdocs serve`
> previews it locally; CI builds it `--strict` on every PR.

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

**Coming next:** docs site (mkdocs with a "porting from glmnet/ncvreg"
cheat sheet) and comparison benchmarks vs. glmnet/ncvreg/grpreg/skglm.
CI and wheel builds are in place; the library is now `pip install`-able
once published.

## Layout

```
crates/skein-core/   pure Rust: traits + algorithms (no Python)
crates/skein-py/     PyO3 bindings (cdylib → skein._core)
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
