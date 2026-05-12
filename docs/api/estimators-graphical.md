# Graphical estimators

Sparse inverse-covariance (precision matrix) estimators, single- and
multi-population.

For the conceptual model — what a precision matrix is, why zeros
correspond to conditional independence, when to choose nonconvex
penalties over L1 — see [Graphical models](../concepts/graphical_models.md).

## Single-population

All three accept either raw `X (n, p)` or a precomputed `(p, p)`
symmetric covariance (sniffed automatically). Fitted attributes:
`precision_`, `covariance_`, `info_`, `n_features_in_`.

```{eval-rst}
.. autoclass:: skein_glm.estimators.GraphicalLasso
   :members:

.. autoclass:: skein_glm.estimators.GraphicalMCP
   :members:

.. autoclass:: skein_glm.estimators.GraphicalSCAD
   :members:
```

## Joint estimation across populations

`JointGraphicalLasso` and `JointGraphicalMCP` fit `K` precision
matrices simultaneously with a group penalty on each edge across
populations. Accept a *list* of arrays (one per population), each
raw or precomputed cov. Fitted attributes: `precisions_`, `info_`,
`n_features_in_`, `n_populations_`.

```{eval-rst}
.. autoclass:: skein_glm.estimators.JointGraphicalLasso
   :members:

.. autoclass:: skein_glm.estimators.JointGraphicalMCP
   :members:
```

## Choosing α (and λ for joint)

For single-population, see
[`ebic_path`](graph_selection.md) — the field-standard
Extended-BIC tuner from Foygel & Drton (2010). For joint,
[`joint_ebic_path`](graph_selection.md) walks the `λ_2` coupling
grid.
