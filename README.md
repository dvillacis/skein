# skein

Weighted structured nonconvex sparse models. Rust core + Python API.

> **Documentation:** the [docs site](docs/index.md) has the full
> conceptual reference (penalties, datafits, weights, backends),
> porting guides for `glmnet` / `ncvreg` / `grpreg`, worked examples,
> and an auto-generated API reference. Built with Sphinx + Furo and
> hosted on Read the Docs (config in `.readthedocs.yaml`); preview
> locally with `sphinx-build -b html docs docs/_build/html`. CI builds
> it with `-W` (warnings = errors) on every PR.

`skein` targets a niche that's well-served in R (`grpreg`, `ncvreg`) but
missing in Python at production quality: nonconvex group-structured
penalties (group MCP, group SCAD, sparse-group nonconvex) with first-class
support for *weights along three axes* — per-sample, per-feature, and
per-group.

## Status

v0.9 — the research-grade release. Closes the inference axis across
all four mainstream GLM families (debiased Cox lasso joins LS /
logistic / Poisson), adds edge-level FDR / FWER / MB stability
control on graphical models, ships polychoric / polyserial
preprocessing for ordinal Likert data, and finishes the M13 / M14c
perf work — every GLM × group penalty (plain + sparse-group) now
runs native, no LLA wrappers underneath any prox-Newton outer.
M13.8 (post-v0.9) ports celer's gap-safe screening + Anderson dual
extrapolation to the GLM prox-Newton surrogate, closing the
F-series gap that M10 left LS-only — 3–8× wall-clock on
`logistic_lasso` v2 cells. M5 model selection + inference +
threaded CV folds + M11 graphical lasso (single + joint) + M12
hardening all carried over from v0.8. See [ROADMAP.md](ROADMAP.md)
for the full plan and the open M14b software-paper milestone.

**Done so far:**

- **Solvers** — production CD core (path solver, strong rule + KKT
  verification, gap-safe screening, Anderson acceleration, M13.1
  saturation bypass, M13.2 cross-λ gradient cache); **GLM
  prox-Newton paths run the same celer-style screening on the
  weighted-LS surrogate** (M13.8: gap-safe sphere + Anderson dual
  extrapolation + adaptive `0.3 × prev_outer_pgd` inner tol +
  weighted strong-convexity correction `r²=2·gap·max(w)/n` —
  **2.2–8.2× wall-clock on `logistic_lasso` v2 cells**); group
  block-CD with native non-convex prox for group MCP (M13.4b for
  LS, M13.4c for logistic / Poisson / Cox) and an LLA outer loop
  for the remaining sparse-group MCP / SCAD families (M13.4 Phase
  2.3 weight-space short-circuit); Rayon-parallel group sweeps;
  operator-norm Lipschitz via power iteration.
- **Datafits** — least squares, binomial logistic, Poisson (log link,
  with offsets), Cox PH (Breslow + Efron ties), multinomial softmax,
  Huber. All glued together by a `GlmDatafit` trait that exposes a
  weighted-LS surrogate; the M1/M2 inner solvers absorb every GLM
  unchanged.
- **Penalties** — lasso, MCP, SCAD, elastic net, bridge `|β|^q`, group
  lasso, group MCP, group SCAD, group elastic net, sparse-group lasso,
  sparse-group MCP, sparse-group SCAD. Per-feature and per-group
  weights honored throughout.
- **Design-matrix backends** — `DenseMatrix`, `SparseCSC`, lazy
  `Standardized<D>`, `MmapMatrix` (f64 + f32), row-block `Chunked<C>`,
  `Augmented<D>`, `MultiTaskDesign<D>` — all behind one trait, freely
  composable.
- **Python** — sklearn-compatible estimators for every (datafit ×
  penalty) combination (~150 classes); type stubs; warm-started
  λ-paths; standardization with original-scale `coef_` / `intercept_`
  recovery on dense and sparse.
- **Model selection + inference (M5 + M14a)** — K-fold CV across
  every `*PathCV` class (threaded folds via PyO3 GIL release,
  ~2.3–2.5× speedup); AIC/BIC/EBIC tuning; stability selection (MB
  bootstrap); **debiased / desparsified lasso for LS + binomial +
  Poisson + Cox** with Wald CIs and p-values (Cox added in M14a — no
  mainstream Python package has it).
- **Graphical models (M11 + M14a)** — sparse precision matrix
  estimation (`GraphicalLasso` / `GraphicalMCP` / `GraphicalSCAD`)
  and joint estimation across `K` related populations
  (`JointGraphicalLasso` / `JointGraphicalMCP`, Danaher–Wang–Witten
  2014 group form via ADMM), with EBIC tuning, bootnet-style
  bootstrap edge stability, and **edge-level Benjamini–Hochberg FDR
  / Bonferroni / Holm FWER / Meinshausen–Bühlmann stability bound**
  (M14a — no other graphical-models package controls error rates
  at the edge level). Nonconvex penalties on edges close the
  shrinkage-bias gap that `sklearn.covariance.GraphicalLasso` and
  R's `glasso` / `qgraph` / `bootnet` leave open.
- **Network psychometrics pipeline (M14a)** — polychoric / polyserial
  correlations (Olsson 1979 two-step ML) for ordinal Likert data
  via `polychoric_correlation` / `polyserial_correlation` /
  `polychoric_covariance_matrix`. The end-to-end
  `polychoric_correlation` → `GraphicalMCP` (EBIC-tuned) →
  `GraphicalBootstrap.fdr_threshold(...)` worked example in
  `docs/examples/psychometrics.md` is the closeout for the M11.1
  psychometrics-replication exit criterion.
- **Distribution + docs (M8) + hardening (M12)** — CI + cibuildwheel +
  Read the Docs + Sphinx site (concepts + R-porting + extending +
  examples + API ref) + R numerical regression suite vs glmnet /
  ncvreg / grpreg + stable Rust API contract. M12 added penalty +
  datafit unit-test coverage, an integration test directory, a CI
  smoke job for the PyO3 layer, and an R-fixture gate.

**Coming next:** **M14b (software paper)** — run the full
`benches/v2` GLM + graphical headline matrix and draft the
JMLR-MLOSS / JOSS manuscript from the figures + tables that already
auto-generate. M14c shipped: scalar LLA weight short-circuit
(bridge / adaptive / multitask), native sparse-group MCP BCD for
logistic / Poisson / Cox, and an at-scale R-fixture tier (n=500,
p=100) for cross-package regression gating.

## Layout

```
crates/skein-core/   pure Rust: traits + algorithms (no Python)
crates/skein-py/     PyO3 bindings (cdylib → skein_glm._core)
python/skein_glm/    sklearn-compatible estimators + ABCs for extensions
tests/               pytest suite (Rust extension required)
benches/             v1 cross-package harness (skein vs sklearn / skglm / celer / glmnet / ncvreg / grpreg)
benches/v2/          publication-quality Snakemake suite backing the paper
crates/skein-core/benches/   internal Rust criterion microbenches
paper/               figure + table bundle regenerated by benches/v2
docs/                Sphinx site (Read the Docs)
```

The Rust traits (`DesignMatrix`, `Datafit`, `GlmDatafit`, `Penalty`,
`GroupPenalty`) and their Python ABC mirrors
(`skein_glm.penalties.Penalty`, etc.) are the extension surface for
downstream per-paper projects.

## Quick start

```python
import numpy as np
from skein_glm import MCPPathRegressor, LogisticGroupMCPPathRegressor, CoxMCPRegressor

# Nonconvex sparse least squares with a λ-path.
rng = np.random.default_rng(0)
n, p = 200, 50
X = rng.standard_normal((n, p))
y = X[:, :3] @ np.array([1.5, -2.0, 0.8]) + 0.1 * rng.standard_normal(n)
model = MCPPathRegressor(gamma=3.0, n_lambdas=50, standardize=True).fit(X, y)
print(model.coefs_[-1, :5], model.intercepts_[-1])

# Logistic + group MCP (native non-convex BCD), with sklearn-style predict/predict_proba.
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

Each scenario is run in two regimes that name what the *solution* does
at the tail of the λ-path, not the path geometry:

- **dense** — `λ_min/λ_max = 1e-3`; the active set saturates near the
  smallest λ (typical "I want the full path including the over-fit
  tail" usage).
- **sparse** — `λ_min/λ_max = 5e-2`; the path stops near support
  recovery, support stays small throughout.

| scenario | size | skein | next-fastest comparator |
|---|---|---|---|
| Lasso LS — dense  | medium (n=10k, p=1k)  | 1.17 s     | sklearn 0.125 s |
| Lasso LS — sparse | medium                | 0.78 s     | sklearn 0.099 s |
| MCP   LS — dense  | medium                | **1.37 s** | skglm 3.35 s    |
| MCP   LS — sparse | medium                | **0.75 s** | ncvreg 1.17 s   |
| MCP   LS — dense  | large (n=100k, p=10k) | **510 s**  | skglm 666 s     |
| MCP   LS — sparse | large                 | **497 s**  | skglm 702 s     |
| SCAD  LS — dense  | medium                | **1.78 s** | ncvreg 7.99 s   |
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
mcp_ls_sparse --sizes small,medium`. The publication-quality
benchmark suite under [`benches/v2/`](benches/v2/README.md) drives the
paper figures and tables; see
[`docs/benchmarks/index.md`](docs/benchmarks/index.md) for the layered
overview.

## Build

```bash
# Rust core only (fast iteration on algorithms)
cargo test -p skein-core --lib

# Full Python package (requires maturin in your env). Always pass the
# BLAS feature flag — without it ndarray's matvec / rmatvec / dot fall
# back to a naive Rust loop and the GLM hot path is ~3× slower. The
# shipped PyPI wheels are built this way; building from source without
# the flag will not match published benchmark numbers.
maturin develop --release --features=blas-accelerate   # macOS
maturin develop --release --features=blas-openblas     # Linux
pytest
```

See [`docs/installation.md`](docs/installation.md) for from-source and
development installs, and [`CLAUDE.md`](CLAUDE.md) for the contributor
quickstart (pre-PR checks, solver-change pre-flight protocol, etc.).

## License

MIT.
