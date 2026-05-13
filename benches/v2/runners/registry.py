"""Comparator ladder: which packages support which (datafit, penalty)
cells, and whether they are direct or surrogate comparators.

The ladder propagates into JSONL output and figure legends so the
reader can tell "ncvreg MCP" (direct) apart from "glmnet Lasso vs
skein Logistic-MCP" (surrogate, not equivalent).
"""
from __future__ import annotations

from typing import Literal

LadderLevel = Literal["direct", "surrogate", "none"]


# Keys are (datafit, penalty); values map package -> level.
LADDER: dict[tuple[str, str], dict[str, LadderLevel]] = {
    # ----- Gaussian LS -----
    ("gaussian", "lasso"):       {"sklearn": "direct", "celer": "direct",
                                  "skglm": "direct", "glmnet": "direct"},
    ("gaussian", "elastic_net"): {"sklearn": "direct", "glmnet": "direct"},
    ("gaussian", "mcp"):         {"skglm": "direct", "ncvreg": "direct"},
    ("gaussian", "scad"):        {"ncvreg": "direct"},
    ("gaussian", "group_lasso"): {"grpreg": "direct"},
    ("gaussian", "group_mcp"):   {"grpreg": "direct"},
    ("gaussian", "group_scad"):  {"grpreg": "direct"},
    ("gaussian", "sparse_group_mcp"): {},     # internal-only

    # ----- Logistic -----
    ("logistic", "lasso"):        {"glmnet": "direct", "sklearn": "direct"},
    ("logistic", "elastic_net"):  {"glmnet": "direct"},
    ("logistic", "mcp"):          {"ncvreg": "direct", "glmnet": "surrogate"},
    ("logistic", "scad"):         {"ncvreg": "direct"},
    ("logistic", "group_lasso"):  {},
    ("logistic", "group_mcp"):    {},

    # ----- Poisson -----
    ("poisson", "lasso"):       {"glmnet": "direct"},
    ("poisson", "elastic_net"): {"glmnet": "direct"},
    ("poisson", "mcp"):         {"glmnet": "surrogate"},
    ("poisson", "scad"):        {},

    # ----- Cox -----
    ("cox", "lasso"):       {"glmnet": "direct", "lifelines": "direct"},
    ("cox", "mcp"):         {"glmnet": "surrogate"},
    ("cox", "group_lasso"): {},
    ("cox", "group_mcp"):   {},

    # ----- Graphical (Σ⁻¹ inverse covariance) -----
    ("gaussian_inv_cov", "lasso"): {"sklearn": "direct", "glasso": "direct"},
    ("gaussian_inv_cov", "mcp"):   {},
}

# Unpenalized GLM baselines (statsmodels) — not in the LADDER because
# they aren't comparators in the sparse-modeling sense, but they're a
# useful anchor for the deviance scale in the appendix. The cell
# driver dispatches them like any other runner; figures suppress them
# unless explicitly requested.
UNPENALIZED_BASELINES = {
    ("logistic", "lasso"): {"statsmodels": "baseline"},
    ("poisson",  "lasso"): {"statsmodels": "baseline"},
}


def level_for(datafit: str, penalty: str, package: str) -> LadderLevel:
    """Return the ladder level for one cell, or 'none' if unsupported."""
    if package == "skein":
        return "direct"
    return LADDER.get((datafit, penalty), {}).get(package, "none")
