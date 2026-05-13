"""Per-(datafit, penalty) scenario modules.

Each module exposes:
    run(*, package, size, regime, seed, tol, trials) -> dict
returning the JSONL row to commit. The dispatcher in
benches.v2.report._run_cell imports the module by id from config.yaml.
"""
