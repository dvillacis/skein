"""Cross-package agreement checks at bench scale.

Lives outside `tests/` on purpose: optional comparator deps (skglm, R)
should not be required to run pytest, and cross-package agreement on
nonconvex problems is treated as a *benchmark output* (M9.4), not a
correctness gate on skein.
"""
