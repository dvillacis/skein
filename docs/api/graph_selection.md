# Graph model selection

Tuning rules for graphical models. The headline export is
[`ebic_path`](#skein_glm.graph_selection.ebic_path), the
field-standard Extended Bayesian Information Criterion (Foygel & Drton
2010) for single-population graphical lasso. The joint analogue
([`joint_ebic_path`](#skein_glm.graph_selection.joint_ebic_path)) sums
per-population log-likelihoods and counts the union of active edges
across populations.

## EBIC

The EBIC formula for a single-population precision estimate `Θ̂(α)` is

$$
\text{EBIC}(\alpha) = -2\,\hat\ell\,(\hat\Theta(\alpha); S) + |\hat E(\alpha)| \log n + 4\gamma|\hat E(\alpha)| \log p,
$$

where `|Ê(α)|` is the number of nonzero off-diagonal entries and
`γ ∈ [0, 1]` controls the strength of the EBIC correction. `γ = 0`
gives plain BIC. `γ = 0.5` is the field default for graphical models
and the value `bootnet` / `qgraph` use out of the box.

```{eval-rst}
.. autofunction:: skein_glm.graph_selection.ebic_path

.. autoclass:: skein_glm.graph_selection.EBICPathResult
   :members:

.. autofunction:: skein_glm.graph_selection.joint_ebic_path

.. autoclass:: skein_glm.graph_selection.JointEBICPathResult
   :members:
```
