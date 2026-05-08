# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`skein` provides weighted structured nonconvex sparse models (MCP, SCAD, group MCP, group lasso, sparse-group nonconvex) with per-sample, per-feature, and per-group weights. Rust core + PyO3 bindings + sklearn-compatible Python estimators.

Status: v0.1 scaffold. The trait surface is set and a minimal cyclic CD solver validates it on MCP/SCAD + LS. The headline algorithm (LLA outer + group block-CD inner with a working set, parallelized across groups via Rayon) is the next milestone — keep new code compatible with this direction.

## Layout

```
crates/skein-core/   pure Rust: traits + algorithms (no Python deps)
crates/skein-py/     PyO3 bindings, builds cdylib `skein_glm._core`
python/skein/        sklearn-compatible estimators + Python ABCs
tests/               pytest smoke tests (require maturin develop first)
benches/             criterion (Rust) + asv (Python) — empty in v0.1
```

## Build & test

```bash
# Rust core only — fast iteration on algorithms
cargo test -p skein-core

# Single Rust test
cargo test -p skein-core --lib cd_recovers_zero_solution_under_strong_penalty

# Build + install Python extension into the active env (requires maturin)
maturin develop --release

# Python tests (require maturin develop first; test_smoke skips otherwise)
pytest
pytest tests/test_smoke.py::test_mcp_regressor_fits_and_zeros_noise_features
```

Lint/format: `ruff` (line-length 100, py310 target) and `mypy` are configured as dev extras but no CI is wired up yet.

## Architecture

The whole codebase is organized around four Rust traits in `skein-core` that are also mirrored as Python ABCs. Anything new (datafit, penalty, design backend) plugs in by implementing the trait, and the solver doesn't need to know which concrete type it has.

- `DesignMatrix` (`crates/skein-core/src/design.rs`): the only way the solver touches `X`. `col_dot` and `col_sq_norm` are the hot paths for CD; `columns(&[usize])` returns blocks for group block-CD. `DenseMatrix` is the only impl in v0.1; sparse / mmap backends slot in here.
- `Datafit` (`crates/skein-core/src/datafit/mod.rs`): structured around an explicit residual `r = Xβ − y` so coordinate updates can patch `r` in place instead of recomputing `Xβ`. `coord_lipschitz` returns `‖X[:, j]‖²` for LS. `sample_weights` is the per-sample weight axis.
- `Penalty` / `GroupPenalty` (`crates/skein-core/src/penalty/mod.rs`): two traits because prox signatures differ. Both expose `weights()` so the solver is agnostic to whether weights are uniform, adaptive, or externally supplied — that's the per-feature / per-group weight axis.
- `Groups` (`crates/skein-core/src/groups.rs`): CSR-style `(ptr, idx)` so groups can overlap and have unequal sizes. `singletons` and `contiguous_blocks` are the two convenience constructors.

Prox primitives (`crates/skein-core/src/prox.rs`) follow a fixed convention: `prox solves argmin_x { (1/(2·step))(x−z)² + p(x) }`, and every weighted variant routes through a single `weight` multiplier. Don't introduce a parallel weighting hook — extend through this one.

All trait objects are `Sync + Send`; the solver will dispatch group-wise work across Rayon threads, so don't add `!Sync` state.

Solver entry point is `cd_solve` in `crates/skein-core/src/solver/cd.rs`. It's intentionally minimal (no working set, no acceleration) and exists as a smoke test for the trait wiring. The production solver will be added alongside it under `solver/` rather than replacing this one.

## Python ↔ Rust binding

`crates/skein-py/src/lib.rs` is the PyO3 layer. `[tool.maturin]` in `pyproject.toml` builds it as the `skein_glm._core` module (python source lives in `python/skein_glm/`, manifest lives at the workspace path). The Python-facing surface is deliberately thin: `_core.solve_mcp_ls` and `_core.solve_scad_ls` take numpy arrays + scalars and return `(coef, info_dict)`. The sklearn wrappers in `python/skein_glm/estimators.py` (`MCPRegressor`, `SCADRegressor`) are `BaseEstimator + RegressorMixin` and store results on `coef_`, `info_`, `n_features_in_`.

When adding a new solver to Rust, surface it by adding a `#[pyfunction]` to `crates/skein-py/src/lib.rs`, register it in `_core(...)`, and write a thin sklearn estimator in `python/skein_glm/estimators.py` that calls it.

The Python ABCs in `python/skein/penalties.py` and `python/skein/datafits.py` are the extension surface for downstream per-paper projects to prototype custom penalties / datafits in Python before optionally porting to Rust. Keep them in lockstep with the Rust traits.
