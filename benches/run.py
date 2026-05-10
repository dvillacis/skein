"""Driver for the M9 bench suite.

Dispatches scenarios × packages × sizes; writes JSON snapshots under
benches/results/. Runners that fail to import are skipped with a
warning — the suite is intentionally tolerant of missing optional
comparator deps.

Usage:

    python benches/run.py --scenarios lasso_ls --sizes small
    python benches/run.py --scenarios all --packages all --sizes small,medium,large
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("benches.run")

SCENARIOS_DIR = REPO_ROOT / "benches" / "scenarios"
RESULTS_DIR = REPO_ROOT / "benches" / "results"

ALL_PACKAGES = ("skein", "sklearn", "skglm", "celer", "pyglmnet", "r")


def _host_id() -> str:
    return f"{platform.system().lower()}-{platform.machine()}-{socket.gethostname()}"


def _discover_scenarios() -> list[str]:
    if not SCENARIOS_DIR.exists():
        return []
    return sorted(
        p.stem for p in SCENARIOS_DIR.glob("*.py") if not p.stem.startswith("_")
    )


def _load_scenario(name: str):
    return importlib.import_module(f"benches.scenarios.{name}")


def _load_runner(package: str):
    if package == "r":
        # The R runner is invoked by scenarios via subprocess against runners/r_runner.R;
        # there is no Python module to load.
        return None
    try:
        return importlib.import_module(f"benches.runners.{package}_runner")
    except ImportError as exc:
        logger.warning("runner %s not importable (%s); skipping", package, exc)
        return None


def _resolve(items: Iterable[str], universe: Iterable[str]) -> list[str]:
    items = list(items)
    if items == ["all"] or not items:
        return list(universe)
    return items


def _append_result(scenario: str, run: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{scenario}.json"
    payload = {"scenario": scenario, "host_id": _host_id(), "runs": []}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("results file %s was malformed; starting fresh", path)
    payload["runs"].append(run)
    path.write_text(json.dumps(payload, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["all"],
        help="scenario names from benches/scenarios/, or 'all'",
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        default=["all"],
        help=f"subset of {ALL_PACKAGES}, or 'all'",
    )
    parser.add_argument(
        "--sizes",
        default="small",
        help="comma-separated list of small,medium,large",
    )
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--n-lambdas", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5, help="measured trials per (scenario, package, size)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    scenarios = _resolve(args.scenarios, _discover_scenarios())
    packages = _resolve(args.packages, ALL_PACKAGES)
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]

    if not scenarios:
        logger.error("no scenarios found under %s", SCENARIOS_DIR)
        return 2

    timestamp = datetime.now(timezone.utc).isoformat()
    for scenario_name in scenarios:
        logger.info("=== scenario: %s ===", scenario_name)
        scenario = _load_scenario(scenario_name)
        for size in sizes:
            for package in packages:
                runner = _load_runner(package)
                if runner is None and package != "r":
                    continue
                if runner is not None and not runner.is_available():
                    logger.info("  %s/%s: package not available; skipping", size, package)
                    continue
                try:
                    run = scenario.run(
                        runner=runner,
                        package=package,
                        size=size,
                        tol=args.tol,
                        n_lambdas=args.n_lambdas,
                        trials=args.trials,
                    )
                except NotImplementedError as exc:
                    logger.info("  %s/%s: %s", size, package, exc)
                    continue
                except Exception as exc:  # noqa: BLE001 — bench script, log + continue
                    logger.exception("  %s/%s: failed (%s)", size, package, exc)
                    continue
                run["timestamp"] = timestamp
                _append_result(scenario_name, run)
                logger.info("  %s/%s: %.3fs (active=%d)", size, package, run["fit_time_s"], run["active_set_size"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
