# Changelog

All notable changes to `skein-glm` are recorded here. The project follows
semantic versioning, with the pre-1.0 minor-bump-on-feature policy
documented in `docs/extending/rust-api.md`.

## [0.3.0] — 2026-05-09

Adds a new GLM family (multinomial / softmax), a new penalty (bridge
`|β|^q`), 28 new adaptive estimators across LS / group / GLM datafits,
and closes two `glmnet` / `survival::coxph` parity gaps (Poisson
offsets, Cox Efron tie handling). 50+ new sklearn estimators total.

### Added

- **Multinomial / softmax classification (M3.6).** New `MultinomialLogit`
  GLM datafit using Böhning's diagonal majorization (matches
  `glmnet(family="multinomial", type.multinomial="grouped")`). 12 new
  sklearn classes — `Multinomial{Lasso,MCP,SCAD,ElasticNet}{Classifier,
  PathClassifier,PathCV}` — with `coef_ (K, p)`, `predict_proba (n, K)`,
  arbitrary-dtype label support (integer / string).
- **Bridge / ℓ_q penalty `λ · Σ_j w_j |β_j|^q`, `q ∈ (0, 1]` (M6.x).**
  New scalar LLA path solver in `crates/skein-core/src/solver/path_lla.rs`
  + `surrogate_weights_bridge` helper. 3 sklearn classes
  (`BridgeRegressor`, `BridgePathRegressor`, `BridgePathCV`); closes a
  `grpreg` parity gap.
- **SparseGroupSCAD end-to-end (M6.x).** The M2.7 surrogate helper is
  now wired through PyO3 + sklearn estimators across LS + 3 GLMs. 12
  new classes: `{,Logistic,Poisson,Cox}SparseGroupSCAD{Regressor,
  PathRegressor,PathCV}`.
- **Adaptive {Lasso, MCP, SCAD} two-stage estimators (M6.x).** 28 new
  sklearn classes total:
  - 6 LS scalar (`Adaptive{Lasso,MCP,SCAD}{PathRegressor,PathCV}`).
  - 4 LS group (`AdaptiveGroup{Lasso,MCP}{PathRegressor,PathCV}` —
    plain `GroupSCAD` deferred).
  - 18 GLM (`Adaptive{Logistic,Poisson,Cox}{Lasso,MCP,SCAD}{Path,
    PathCV}`). Pure Python composition; the per-feature `weights=`
    parameter does the work. `coef_pilot_` and `weights_` exposed
    for inspection.
- **Poisson offsets (M3.x).** `PoissonLog::with_offset(y, offset)`
  threads a per-sample log-exposure through `surrogate_at` and
  `loss`. Every Poisson estimator (14 + 7 PathCV) accepts
  `offset=None`; CV slices `offset[train_idx]` per fold. Standard
  rate-model use case.
- **Cox Efron ties (M3.x).** New `TieHandling::{Breslow, Efron}` enum;
  `CoxPH::with_ties(time, event, ties)`. `loss` and the cumulative-
  hazard accumulation handle Efron's per-event reduced risk set
  `S_eff_i = S(t) − (i/k)·S_D(t)`. Reduces exactly to Breslow when
  no ties. Every Cox estimator (14 + 7 PathCV) accepts
  `ties="breslow"|"efron"`. Default stays Breslow for back-compat;
  matches `glmnet(family="cox", ties="efron")` and is more accurate
  when ties are heavy (R `survival::coxph`'s default).

### Changed

- `_glm_dispatch_inputs` reads `getattr(estimator, 'offset', None)`
  to thread Poisson offsets through the existing PyO3 path; logistic
  / Cox are unaffected (no `offset` attr).
- `_PathCVMixin.fit` slices `offset[train_idx]` per fold for any
  estimator carrying an `offset` attribute.
- `CoxPH::new(time, event)` still works — defaults to Breslow ties.

### Estimator counts (cumulative)

108 estimators in v0.2.0 → **136 estimators in v0.3.0**, 51 → **58**
`*PathCV` cross-validation wrappers. Every datafit × penalty
combination is wired end-to-end with sklearn-style `fit` / `predict` /
`predict_proba` / `score`.

### Tests

**265 cargo + 261 pytest, all green.**
- v0.2.0 baseline: 254 cargo + 197 pytest.
- New tests this release: +11 cargo (multinomial 8 + Poisson offset 4
  + Cox Efron 4 — minus one shared) and +64 pytest spanning every new
  family.

### Deprecation / breaking

None. v0.3.0 is fully backward-compatible with v0.2.0; the
`CoxPH::new` constructor still defaults to Breslow ties, every
existing PyO3 entry preserves its v0.2.0 keyword surface, and the
new `offset=None` and `ties="breslow"` defaults match v0.2.0
behavior on the GLM path.

### Roadmap status

- M3 — GLM datafits: multinomial done, Poisson offsets done, Cox Efron
  done. M3.7 (negative binomial / Huber / quantile) is the only
  remaining open item.
- M6 — Penalty zoo: SparseGroupSCAD, bridge, adaptive (scalar / group /
  GLM) done. Overlapping group lasso, fused lasso, constrained
  variants pending.

## [0.2.0] — earlier release

Initial public release. Multi-task LS, sparse + dense + mmap +
chunked backends, full GLM coverage (binomial, Poisson, Cox Breslow),
group penalty zoo, CV + IC selection, R numerical regression suite.
See `ROADMAP.md` for the milestone breakdown.
