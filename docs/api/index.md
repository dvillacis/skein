# API reference

Auto-generated from Python docstrings. Pages are organized by
family rather than alphabetically — most users want "all the
logistic estimators" or "all the LS estimators with groups", not
a linear scan through 80 classes.

## Estimators

| Family   | Page                                          |
|----------|-----------------------------------------------|
| Gaussian (LS) — scalar penalties (MCP, SCAD, ElasticNet)  | [LS](estimators-ls.md)        |
| Gaussian (LS) — group penalties (incl. GroupElasticNet)   | [LS](estimators-ls.md)        |
| Multi-task LS — 2D `Y`, joint feature selection           | [Multi-task](estimators-multitask.md) |
| Binomial logistic                             | [Logistic](estimators-logistic.md) |
| Multinomial / softmax (K classes, row-grouped) | [Multinomial](estimators-multinomial.md) |
| Poisson                                       | [Poisson](estimators-poisson.md)   |
| Cox proportional hazards                      | [Cox](estimators-cox.md)           |

Every GLM family follows the same pattern: scalar (`MCP`, `SCAD`)
and group (`GroupLasso`, `GroupMCP`, `SparseGroupLasso`,
`SparseGroupMCP`) penalties; each penalty has a single-λ `Regressor`
and a full-path `PathRegressor`. The Gaussian LS family also adds
`ElasticNet` (scalar) and `GroupElasticNet` (group). The multi-task
LS family adds 8 more (`MultiTaskLasso/MCP/SCAD/ElasticNet ×
{single-λ, Path}`).

## Cross-validation and IC selection

| Module                       | Page                |
|------------------------------|---------------------|
| `*PathCV` cross-validation   | [CV](cv.md)         |
| `select_by_ic` (AIC/BIC/EBIC) | [IC](ic.md)         |

## Design-matrix helpers

| Module           | Page              |
|------------------|-------------------|
| `MmapDesignF64`, `MmapDesignF32`, `ChunkedDesignF64`, `ChunkedDesignF32` | [Design](design.md)                    |

## Extension ABCs

| Module                    | Page                |
|---------------------------|---------------------|
| `skein_glm.penalties.Penalty`, `skein_glm.penalties.GroupPenalty` | [ABCs](abcs.md)              |
| `skein_glm.datafits.Datafit`  | [ABCs](abcs.md)     |

## A note on inherited members

Estimator subclasses share most of their behavior through base
classes (`_PathRegressorBase`, `_LogisticPathRegressorBase`, etc.).
The auto-generated docs show each concrete class with a brief
docstring; for the meatier "what does `fit` actually do" content,
look at the base class section near the top of each family page.

Some methods (sklearn's `BaseEstimator.get_params`, `set_params`,
`__sklearn_tags__`) are inherited from sklearn itself. We don't
re-document those — see [sklearn's docs](https://scikit-learn.org/)
for the inheritance chain.
