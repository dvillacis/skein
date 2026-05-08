# Design-matrix helpers

Python helper classes for the out-of-RAM backends. Used in place of
a numpy array when fitting:

```python
design = skein_glm.MmapDesignF64("X.bin", n_rows=n, n_cols=p)
model = skein_glm.MCPPathRegressor(...).fit(design, y)
```

Estimators with mmap / chunked support sniff `isinstance(x, ...)` to
route through to the corresponding `_mmap` / `_chunked` PyO3 entry
points. v1 estimator coverage: `MCPPathRegressor` and
`LogisticMCPPathRegressor`. Other estimators raise a clear error
if handed a `Mmap*` or `Chunked*` design — expanding coverage is
mechanical and tracked on the M4.x roadmap.

See [Concepts: Backends](../concepts/backends.md) for the storage
model and when to use each helper.

## Memory-mapped (single file)

::: skein_glm.mmap.MmapDesignF64

::: skein_glm.mmap.MmapDesignF32

## Row-block-chunked (multiple files)

::: skein_glm.mmap.ChunkedDesignF64

::: skein_glm.mmap.ChunkedDesignF32
