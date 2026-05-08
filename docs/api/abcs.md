# Extension ABCs

The Python ABCs in `skein_glm.penalties` and `skein_glm.datafits` mirror the
Rust traits in `skein-core`. Implement these to prototype new
penalties / datafits in Python before porting to Rust.

See [Extending: custom penalties](../extending/penalty.md) and
[Extending: custom datafits](../extending/datafit.md) for worked
examples.

## Penalty ABCs

```{eval-rst}
.. autoclass:: skein_glm.penalties.Penalty
   :members:

.. autoclass:: skein_glm.penalties.GroupPenalty
   :members:
```

## Datafit ABC

```{eval-rst}
.. autoclass:: skein_glm.datafits.Datafit
   :members:
```
