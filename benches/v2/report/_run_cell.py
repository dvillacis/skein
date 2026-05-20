"""Cell driver — runs one (scenario, size, regime, seed, package) tuple.

Writes one JSONL row + an env.json sidecar. Invoked from the Snakemake
`run_cell` rule. Stays small on purpose: scenario modules build the
problem, runners do the fit, metrics compute accuracy, and this driver
is just glue.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from benches.v2.report import capture_env
from benches.v2.runners.registry import level_for


SCENARIO_PKG = "benches.v2.scenarios"
RUNNER_PKG_V2 = "benches.v2.runners"

# config.yaml package names → benches/v2/runners/<module>.py
RUNNER_ALIASES = {
    "skein":       "skein_runner",
    "sklearn":     "sklearn_runner",
    "celer":       "celer_runner",
    "skglm":       "skglm_runner",
    "glmnet":      "glmnet_runner",
    "ncvreg":      "ncvreg_runner",
    "grpreg":      "grpreg_runner",
    "glasso":      "glasso_runner",
    "lifelines":   "lifelines_runner",
    "statsmodels": "statsmodels_runner",
}


def _load_runner(package: str):
    """Return a runner module exposing fit() and is_available()."""
    if package not in RUNNER_ALIASES:
        raise NotImplementedError(
            f"package {package!r} not registered in RUNNER_ALIASES "
            f"(known: {sorted(RUNNER_ALIASES)})"
        )
    return importlib.import_module(f"{RUNNER_PKG_V2}.{RUNNER_ALIASES[package]}")


def _resolve_size(cfg: dict, size_name: str) -> tuple[int, int]:
    sz = cfg["sizes"][size_name]
    return int(sz["n"]), int(sz["p"])


def _make_problem(scenario_id: str, n: int, p: int, seed: int):
    """Build the problem at the requested (n, p, seed)."""
    mod = importlib.import_module(f"{SCENARIO_PKG}.{scenario_id}")
    if hasattr(mod, "make_problem_explicit"):
        return mod, mod.make_problem_explicit(n, p, seed)
    return mod, mod.make_problem(size_name_for(n, p), seed)


def size_name_for(n: int, p: int) -> str:
    # Reverse lookup for legacy SIZES dict (small/medium/large).
    from benches.problems import SIZES
    for name, sz in SIZES.items():
        if sz.n == n and sz.p == p:
            return name
    raise KeyError((n, p))


def _lambda_grid(
    x: np.ndarray, y: np.ndarray, regime_cfg: dict, datafit: str = "gaussian"
) -> np.ndarray:
    """Geometric λ-grid descending from a datafit-appropriate `λ_max`.

    For regression-style datafits (gaussian / logistic / poisson / cox)
    we use the KKT-at-zero bound `max |X^T y| / n`. For graphical
    models (`gaussian_inv_cov`) the simulator returns a placeholder
    y = 0 vector — the relevant KKT bound is on the off-diagonal of
    the sample covariance `S = X^T X / n` instead.
    """
    n = x.shape[0]
    if datafit == "gaussian_inv_cov":
        s = (x.T @ x) / n
        s_off = s - np.diag(np.diag(s))
        lambda_max = float(np.max(np.abs(s_off)))
    else:
        lambda_max = float(np.max(np.abs(x.T @ y)) / n)
    return np.geomspace(lambda_max,
                        lambda_max * regime_cfg["lambda_min_ratio"],
                        regime_cfg["n_lambdas"])


def run_cell(*, scenario: str, size: str, regime: str, seed: int,
             package: str, config_path: Path, out: Path,
             env_out: Path, trials: int | None = None) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    n, p = _resolve_size(cfg, size)
    regime_cfg = cfg["regimes"][regime]
    tol = float(cfg["defaults"]["tol"])
    if trials is None:
        trials = int(cfg["defaults"]["trials"])

    # Environment capture happens first so the file exists even if the fit
    # blows up later (helps debug "what did this host have installed").
    env = capture_env.write(env_out, extra={
        "scenario": scenario, "size": size, "regime": regime,
        "seed": seed, "package": package, "n": n, "p": p,
    })

    scenario_mod, problem = _make_problem(scenario, n, p, seed)
    grid = _lambda_grid(
        problem.x, problem.y, regime_cfg,
        datafit=scenario_mod.SPEC.get("datafit", "gaussian"),
    )
    runner = _load_runner(package)

    if not runner.is_available():
        # Empty cell — write a row marking it as skipped.
        row = {
            "scenario": scenario, "size": size, "regime": regime,
            "seed": seed, "package": package, "status": "skipped",
            "reason": f"{package} not installed",
            "host_id": env["host_id"],
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row) + "\n")
        return row

    penalty = scenario_mod.SPEC["penalty"]
    datafit = scenario_mod.SPEC["datafit"]

    # 1 warmup + N timed trials, mirroring benches.scenarios._common.
    runner.fit(problem, penalty=penalty, lambda_grid=grid, tol=tol)
    per_trial: list[float] = []
    last = None
    for _ in range(trials):
        t0 = time.perf_counter()
        result = runner.fit(problem, penalty=penalty, lambda_grid=grid, tol=tol)
        per_trial.append(time.perf_counter() - t0)
        last = result

    coef_path = (np.asarray(last.coef_path)
                 if getattr(last, "coef_path", None) is not None
                 else None)

    # Recovery + IC selection metrics — cheap, always compute when
    # β_true is known.
    from benches.v2.metrics import recovery as rec
    recovery_metrics: dict | None = None
    selection_metrics: dict | None = None
    if coef_path is not None and getattr(problem, "beta_true", None) is not None:
        beta_true = np.asarray(problem.beta_true)
        # Graphical scenarios stash Ω as the truth; per-λ recovery against
        # a flattened p×p truth needs special handling (Phase D), so for
        # now skip the recovery panel for them but still record the cell.
        if datafit != "gaussian_inv_cov":
            recovery_metrics = rec.per_lambda(coef_path, beta_true)
            try:
                from benches.v2.metrics import selection as sel
                event = problem.meta.get("event") if problem.meta else None
                selection_metrics = sel.ic_selection_accuracy(
                    coef_path, beta_true, problem.x, problem.y, datafit,
                    event=np.asarray(event, dtype=np.int64) if event is not None else None,
                )
            except Exception as e:
                selection_metrics = {"error": f"{type(e).__name__}: {e}"}

    row = {
        "scenario": scenario, "size": size, "regime": regime,
        "seed": seed, "package": package,
        "status": "ok",
        "datafit": datafit, "penalty": penalty,
        "ladder_level": level_for(datafit, penalty, package),
        "n": n, "p": p,
        "lambda_min_ratio": regime_cfg["lambda_min_ratio"],
        "n_lambdas": regime_cfg["n_lambdas"],
        "tol": tol,
        "trials": per_trial,
        "fit_time_s": float(np.median(per_trial)),
        "fit_time_min_s": float(np.min(per_trial)),
        "fit_time_max_s": float(np.max(per_trial)),
        "version": last.version,
        "n_iter": last.n_iter,
        "final_obj": last.final_obj,
        "active_set_size": int(last.active_set_size),
        "recovery": recovery_metrics,
        "selection": selection_metrics,
        "host_id": env["host_id"],
        "git_rev": env["git_rev"],
        "extra": dict(last.extra),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row) + "\n")

    # Persist coefficient path as a .npy sidecar so the aggregator can
    # compute cross-package agreement without re-fitting. ~8 MB at
    # n_lambdas=100, p=10000 — bigger than the JSONL but bounded.
    if coef_path is not None:
        coef_sidecar = out.with_suffix(".coefs.npy")
        np.save(coef_sidecar, coef_path)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env-out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=None,
                    help="Override `defaults.trials` from config. Used by "
                         "bench-smoke to keep the per-PR at-scale cell under "
                         "the 15 min wall-clock budget.")
    a = ap.parse_args()
    row = run_cell(
        scenario=a.scenario, size=a.size, regime=a.regime, seed=a.seed,
        package=a.package, config_path=a.config, out=a.out, env_out=a.env_out,
        trials=a.trials,
    )
    print(json.dumps({k: row.get(k) for k in
                      ("status", "fit_time_s", "active_set_size", "host_id")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
