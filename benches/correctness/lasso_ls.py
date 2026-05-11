"""Cross-package agreement on lasso / Gaussian LS.

Convex companion to the `mcp_ls` / `scad_ls` scripts. Lasso has a
unique global minimum, so every package on a sufficiently tight λ-grid
and tolerance should land within numerical roundoff — this is a tight
correctness gate, not the tolerant nonconvex framing. Drift here would
flag a real bug.

Comparators: skein, sklearn (`lasso_path`), skglm (`Lasso.path`),
celer (`celer_path`), R/glmnet (`glmnet(..., alpha=1)`). All five
implement the same `(1/2n) ‖y − Xβ‖² + λ‖β‖₁` problem on the same
λ-grid; we compare the *slope* coefficients only (each package fits
an intercept on its own centering, which we don't compare across).

Run as a script:

    python benches/correctness/lasso_ls.py --size small
    python benches/correctness/lasso_ls.py --size small --n-lambdas 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benches.correctness import _common as cc  # noqa: E402  — sys.path tweak above
from benches.problems import SIZES, gaussian_lasso  # noqa: E402
from benches.scenarios import _common as sc  # noqa: E402

logger = logging.getLogger("benches.correctness.lasso_ls")

PENALTY = "lasso"
FAMILY = "gaussian"


def _fit_skein(problem, lambdas, tol):
    from benches.runners import skein_runner

    if not skein_runner.is_available():
        return None
    result = skein_runner.fit(problem, penalty=PENALTY, lambda_grid=lambdas, tol=tol)
    return ("skein", result.version, np.asarray(result.coef_path))


def _fit_sklearn(problem, lambdas, tol):
    from benches.runners import sklearn_runner

    if not sklearn_runner.is_available():
        return None
    result = sklearn_runner.fit(problem, penalty=PENALTY, lambda_grid=lambdas, tol=tol)
    return ("sklearn", result.version, np.asarray(result.coef_path))


def _fit_skglm(problem, lambdas, tol):
    from benches.runners import skglm_runner

    if not skglm_runner.is_available():
        return None
    result = skglm_runner.fit(problem, penalty=PENALTY, lambda_grid=lambdas, tol=tol)
    return ("skglm", result.version, np.asarray(result.coef_path))


def _fit_celer(problem, lambdas, tol):
    from benches.runners import celer_runner

    if not celer_runner.is_available():
        return None
    result = celer_runner.fit(problem, penalty=PENALTY, lambda_grid=lambdas, tol=tol)
    return ("celer", result.version, np.asarray(result.coef_path))


def _fit_glmnet(problem, lambdas, tol):
    if not sc.has_rscript():
        return None
    out = sc.run_r(
        package="glmnet",
        penalty=PENALTY,
        family=FAMILY,
        problem=problem,
        lambda_grid=lambdas,
        tol=tol,
    )
    return ("glmnet", out.get("version", "unknown"), np.asarray(out["coef_path"]))


def run(size: str, n_lambdas: int, tol: float, lambda_min_ratio: float) -> dict:
    problem = gaussian_lasso(SIZES[size])
    lambdas = sc.lambda_grid(problem.x, problem.y, n_lambdas, lambda_min_ratio)

    logger.info(
        "problem: size=%s n=%d p=%d k_active=%d  λ-grid=[%.3g..%.3g]×%d  tol=%g",
        size, problem.x.shape[0], problem.x.shape[1],
        int(np.count_nonzero(problem.beta_true)),
        lambdas[0], lambdas[-1], n_lambdas, tol,
    )

    fits: list[tuple[str, str, np.ndarray]] = []
    runners = (
        ("skein", _fit_skein),
        ("sklearn", _fit_sklearn),
        ("skglm", _fit_skglm),
        ("celer", _fit_celer),
        ("glmnet", _fit_glmnet),
    )
    for label, fn in runners:
        try:
            res = fn(problem, lambdas, tol)
        except Exception as exc:  # noqa: BLE001 — bench script, log + skip
            logger.warning("%s: fit failed (%s); skipping", label, exc)
            continue
        if res is None:
            logger.info("%s: not available; skipping", label)
            continue
        name, version, coef = res
        logger.info("%s %s: coef_path shape=%s", name, version, coef.shape)
        fits.append(res)

    if len(fits) < 2:
        raise RuntimeError(f"need at least 2 fitted comparators; got {len(fits)}")

    pairs: list[cc.PairAgreement] = []
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            a_name, _av, a = fits[i]
            b_name, _bv, b = fits[j]
            pairs.append(cc.pair_agreement(a_name, a, b_name, b))

    print(cc.format_summary_table(pairs))

    return {
        "scenario": "lasso_ls",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "size": size,
        "n": int(problem.x.shape[0]),
        "p": int(problem.x.shape[1]),
        "n_lambdas": n_lambdas,
        "lambda_min_ratio": lambda_min_ratio,
        "tol": tol,
        "comparators": [
            {"package": name, "version": version} for name, version, _ in fits
        ],
        "pairs": [p.to_dict() for p in pairs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="small", choices=list(SIZES))
    parser.add_argument("--n-lambdas", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--lambda-min-ratio", type=float, default=1e-2)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    payload = run(args.size, args.n_lambdas, args.tol, args.lambda_min_ratio)
    out = cc.write_results("lasso_ls", payload)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
