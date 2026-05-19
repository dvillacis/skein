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

v0.10.0 — performance milestone shipping celer-style gap-safe
screening on the GLM prox-Newton surrogate (3–8× wall-clock on
`logistic_lasso` v2 cells). Post-v0.10.0 the working tree adds
**M14d/e/f** (untagged): the ncvreg-equivalent v-scaled MCP/SCAD
prox closes the GLM nonconvex active-set bloat (M14e — `logistic_mcp
medium-sparse` 842 → 107 active features, matching ncvreg exactly,
6× speedup); a fused IRLS+CD GLM solver mirroring `ncvreg::cdfit_glm`'s
lazy per-iter cost structure (M14f) closes the remaining wall-clock
gap — **`logistic_mcp medium-sparse` 19.7 s → 3.05 s, ~8 % ahead of
ncvreg** on the same shape. M14b (software paper) ships an empirical
run + a 909-line manuscript draft; remaining 1.0 work is a stable-API
audit, two small dispatch / regression fixes, and the manuscript
wrapper. M5 model selection + inference + threaded CV folds + M11
graphical lasso (single + joint) + M12 hardening all carried over
from v0.8. See [ROADMAP.md](ROADMAP.md) for the full plan.

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

**Coming next — the v1.0 punch-list:**

1. **Fix the `glasso_l1` benchmark-runner dispatch bug** — both the
   skein and sklearn runners in `benches/v2/runners/*` key the
   graphical-family fit on `penalty == "glasso"`, but the scenario
   passes `penalty = "lasso"`, so those cells silently ran the regular
   `lasso_path` on the n×p data matrix. Cells that were committed
   pre-fix should be regenerated; the R-glasso runner shipped this
   release does dispatch correctly.
2. **Investigate the `poisson_lasso` regression** — `medium / dense`
   moved from 29.0 s (pre-M14e baseline) to 41.7 s post-M14f
   (glmnet 2.5 s on the same cell). Convex Poisson is the standout
   weakness now that M14e/f closed the nonconvex GLM gap.
3. **Stable-Rust-API audit** — per `docs/extending/rust-api.md`, 1.0
   freezes the documented surface and forces every other `pub` item to
   either promote or move to `pub(crate)`. Mechanical: `cargo doc -p
   skein-core --no-deps` diff against the contract page.
4. **M14b manuscript wrapper** — empirical run + 909-line LaTeX draft
   landed; remaining work is folding the post-M14e/f numbers into
   §Results / §Ablation and the JMLR-MLOSS / JOSS submission pass.

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

Numbers below are the median of 5 timed trials (single warm-up) from
the [`benches/v2`](benches/v2/README.md) headline matrix on Apple M1
16 GB, `--features=blas-accelerate`, `tol=1e-7`, regenerated 2026-05-18
against the current working tree (M13.8 + M14d/e/f). Two regimes per
scenario, named by what the *solution* does at the tail of the
λ-path:

- **dense** — `λ_min/λ_max = 1e-3`, 100 λs; active set saturates at
  the small-λ end (typical "I want the full path including the
  over-fit tail" usage). Internal config key: `deep`.
- **sparse** — `λ_min/λ_max = 5e-2`, 50 λs; path stops near support
  recovery, support stays small throughout.

### Nonconvex penalties (skein leads)

| scenario | size | regime | skein | next-fastest | ratio |
|---|---|---|---:|---|---:|
| MCP LS  | medium (n=10k, p=1k)   | dense  | **1.70 s** | skglm 4.61 s    | 2.7× |
| MCP LS  | medium                 | sparse | **0.46 s** | ncvreg 1.32 s   | 2.9× |
| MCP LS  | large (n=50k, p=5k)    | dense  | **31.3 s** | skglm 73.5 s    | 2.3× |
| MCP LS  | large                  | sparse | **13.6 s** | ncvreg 24.9 s   | 1.8× |
| SCAD LS | medium                 | dense  | **1.57 s** | ncvreg 7.82 s   | 5.0× |
| SCAD LS | medium                 | sparse | **0.37 s** | ncvreg 1.33 s   | 3.6× |
| SCAD LS | large                  | dense  | **30.6 s** | ncvreg 186 s    | 6.1× |
| Group lasso     | medium (n=10k, J=100) | dense  | **5.33 s** | grpreg 11.4 s  | 2.1× |
| Group MCP       | medium                | dense  | **6.57 s** | grpreg 12.6 s  | 1.9× |
| Logistic MCP    | medium (n=10k, p=1k)  | dense  | **19.6 s** | ncvreg 95.1 s  | 4.9× |
| Logistic MCP    | medium                | sparse | **1.77 s** | ncvreg 3.29 s  | 1.9× |

### Convex penalties (mixed)

| scenario | size | regime | skein | leader | notes |
|---|---|---|---:|---|---|
| Lasso LS         | medium | dense  | **1.13 s** | sklearn 0.20 s | beats celer 3.05 s / glmnet 1.64 s / skglm 4.80 s |
| Lasso LS         | medium | sparse | 0.37 s | celer 0.17 s   | beats glmnet 1.36 s, sklearn 0.12 s wins |
| Lasso LS         | large  | dense  | 25.5 s | sklearn 10.0 s | beats celer 36.4 s / skglm 76.8 s; glmnet 20.3 s |
| ElasticNet LS    | medium | dense  | **1.40 s** | glmnet 1.71 s  | beats glmnet; sklearn 0.28 s wins overall |
| Cox lasso        | medium | dense  | 3.82 s | glmnet 2.24 s  | within 1.7× of glmnet |
| Logistic lasso   | medium | dense  | 108 s  | glmnet 7.9 s   | **14× behind glmnet** — convex GLM is the open gap |
| Poisson lasso    | medium | dense  | 41.7 s | glmnet 2.5 s   | **17× behind glmnet** — regressed from v0.10.0 (M14d/e/f untouched the convex Poisson path) |

skein is now the fastest public option for nonconvex penalties (MCP /
SCAD / their group + sparse-group variants) across every size, and
competitive-to-leading on convex group penalties. The two standing
weaknesses are convex Lasso vs sklearn's Cython `lasso_path` at small
scales (sklearn's coordinate descent kernel is hard to beat), and the
convex GLM lasso paths (logistic / Poisson) where glmnet's specialized
weighted-LS path stays well ahead. The nonconvex GLM gap that M13.8
left open closed in M14e (v-scaled prox) + M14f (fused IRLS+CD): both
already merged in the working tree.

See [`docs/benchmarks/mcp_ls.md`](docs/benchmarks/mcp_ls.md) and
[`docs/benchmarks/scad_ls.md`](docs/benchmarks/scad_ls.md) for the
detailed nonconvex write-ups,
[`docs/perf/lasso_ls_profile.md`](docs/perf/lasso_ls_profile.md) for
the lasso/LS profiling work that drove M10, and
[`paper/tables/T2_headline_timings.md`](paper/tables/T2_headline_timings.md)
for the complete v2 table this section is condensed from.

Reproduce with `pip install -e '.[bench]' && cd benches/v2 &&
snakemake --profile profiles/m1-headline`. The full matrix is ~12 h
on M1; for a fast LS-only shakedown use the `ls_headline` target.

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
