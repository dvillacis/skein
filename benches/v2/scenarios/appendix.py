"""Appendix-matrix generator — produces a (datafit, penalty) cell list
covering every public skein estimator at one (small size, seed 0) cell.

The Snakemake `appendix` profile expands this list into cells. Each
appendix cell runs skein only (no comparator) at small size, deep
regime, seed 0 — enough to populate T1's count column with live data
and to surface any "this estimator has no scenario wiring" gaps.

A scenario module is *not* registered for every cell; instead this
list drives an alternate cell driver that calls `make_problem` with
heuristic defaults per family.
"""
from __future__ import annotations

# (scenario_id, datafit, penalty, group_size_for_group_penalties)
APPENDIX_CELLS: list[tuple[str, str, str, int]] = [
    # Gaussian
    ("appendix_gaussian_lasso",   "gaussian", "lasso",         0),
    ("appendix_gaussian_en",      "gaussian", "elastic_net",   0),
    ("appendix_gaussian_mcp",     "gaussian", "mcp",           0),
    ("appendix_gaussian_scad",    "gaussian", "scad",          0),
    ("appendix_gaussian_grlasso", "gaussian", "group_lasso",   5),
    ("appendix_gaussian_grmcp",   "gaussian", "group_mcp",     5),
    ("appendix_gaussian_grscad",  "gaussian", "group_scad",    5),
    # Logistic
    ("appendix_logistic_lasso",   "logistic", "lasso",         0),
    ("appendix_logistic_mcp",     "logistic", "mcp",           0),
    ("appendix_logistic_scad",    "logistic", "scad",          0),
    ("appendix_logistic_en",      "logistic", "elastic_net",   0),
    ("appendix_logistic_grlasso", "logistic", "group_lasso",   5),
    ("appendix_logistic_grmcp",   "logistic", "group_mcp",     5),
    # Poisson
    ("appendix_poisson_lasso",    "poisson",  "lasso",         0),
    ("appendix_poisson_mcp",      "poisson",  "mcp",           0),
    ("appendix_poisson_scad",     "poisson",  "scad",          0),
    ("appendix_poisson_en",       "poisson",  "elastic_net",   0),
    ("appendix_poisson_grlasso",  "poisson",  "group_lasso",   5),
    ("appendix_poisson_grmcp",    "poisson",  "group_mcp",     5),
    # Cox
    ("appendix_cox_lasso",        "cox",      "lasso",         0),
    ("appendix_cox_mcp",          "cox",      "mcp",           0),
    ("appendix_cox_scad",         "cox",      "scad",          0),
    ("appendix_cox_grlasso",      "cox",      "group_lasso",   5),
    ("appendix_cox_grmcp",        "cox",      "group_mcp",     5),
]
