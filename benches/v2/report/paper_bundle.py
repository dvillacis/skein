"""Collate the v2 outputs into the publication-ready `paper/` directory.

What this does:
  1. Verify every expected figure (F1..F10) and table (T1..T6) exists.
  2. Write a `paper/manifest.json` describing every artifact: source
     aggregates, host(s) used, git rev, missing pieces.
  3. Write a `paper/README.md` overview that maps figure/table → paper
     section + the driver that produced it.
  4. Optionally tar the directory into `paper/skein-paper-bundle-<git>.tar.gz`
     for archival.

Snakemake's `all` rule already enforces the artifact existence; this
script adds the provenance and human-readable mapping.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def _git_rev(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, check=False, capture_output=True, text=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_dirty(root: Path) -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, check=False, capture_output=True, text=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


# Map (artifact path → source aggregates / driver script). Pure metadata.
ARTIFACT_PROVENANCE = {
    "figures/F1_coverage_matrix.pdf": {
        "source": "benches/v2/runners/registry.py (LADDER)",
        "driver": "report/coverage_matrix.py",
        "paper_section": "§Introduction",
    },
    "figures/F2_headline_timing.pdf": {
        "source": "results/scenarios/*.aggregate.json (size=medium, regime=deep)",
        "driver": "report/figures.py headline_timing",
        "paper_section": "§Results",
    },
    "figures/F3_scaling_curves.pdf": {
        "source": "results/scenarios/*.aggregate.json (all sizes, regime=deep)",
        "driver": "report/figures.py scaling_curves",
        "paper_section": "§Results",
    },
    "figures/F4_agreement.pdf": {
        "source": "results/scenarios/*.aggregate.json (cells with agreement_vs_skein_mean)",
        "driver": "report/figures.py agreement",
        "paper_section": "§Correctness",
    },
    "figures/F5_recovery_curves.pdf": {
        "source": "results/scenarios/*.aggregate.json (recovery_per_lambda_mean)",
        "driver": "report/figures.py recovery_curves",
        "paper_section": "§Recovery",
    },
    "figures/F6_realdata_boxplots.pdf": {
        "source": "results/realdata/*.jsonl",
        "driver": "report/run_realdata.py → figures.py realdata_boxplots",
        "paper_section": "§Applications",
    },
    "figures/F7_ic_selection_accuracy.pdf": {
        "source": "results/scenarios/*.aggregate.json (selection_mean)",
        "driver": "report/figures.py ic_selection",
        "paper_section": "§Selection",
    },
    "figures/F8_stability_fdr_power.pdf": {
        "source": "results/stability/stability.jsonl",
        "driver": "report/run_stability.py → figures.py stability_fdr_power",
        "paper_section": "§Selection",
    },
    "figures/F9_screening_parallel.pdf": {
        "source": "results/criterion.jsonl",
        "driver": "cargo bench → report/ingest_criterion.py → figures.py screening_parallel",
        "paper_section": "§Ablation",
    },
    "figures/F10_cv_parallel_speedup.pdf": {
        "source": "results/cv_parallel/cv_parallel.jsonl",
        "driver": "report/run_cv_parallel.py → figures.py cv_parallel_speedup",
        "paper_section": "§Ablation",
    },
    "tables/T1_estimator_surface.tex": {
        "source": "skein_glm.* (introspection)",
        "driver": "report/tables.py estimator_surface",
        "paper_section": "§Introduction",
    },
    "tables/T2_headline_timings.tex": {
        "source": "results/scenarios/*.aggregate.json (size=medium, regime=deep)",
        "driver": "report/tables.py headline_timings",
        "paper_section": "§Results",
    },
    "tables/T4_recovery.tex": {
        "source": "results/scenarios/*.aggregate.json (selection_mean, regime=deep)",
        "driver": "report/tables.py recovery",
        "paper_section": "§Recovery",
    },
    "tables/T5_realdata.tex": {
        "source": "results/realdata/*.jsonl",
        "driver": "report/tables.py realdata",
        "paper_section": "§Applications",
    },
    "tables/T6_environment.tex": {
        "source": "results/cells/*.env.json (host_id, BLAS, package versions)",
        "driver": "report/tables.py environment",
        "paper_section": "§Reproducibility",
    },
}


# H1 — at-scale (n=100k, p=10k) headline cells run with a reduced
# comparator set because several Python and R packages OOM or exceed
# the per-cell wall budget before the size ceiling. The asymmetry is
# captured here so downstream paper figures can flag the gap rather
# than silently dropping the comparator. Keep aligned with the
# `xlarge` entries in `benches/v2/config.yaml`.
AT_SCALE_COMPARATOR_GAP = {
    "xlarge": {
        "ls_lasso":        {"included": ["skein", "celer", "skglm"],
                            "excluded": {"sklearn": "coordinate_descent OOM on dense 8 GB X",
                                         "glmnet":  "32-bit nlam × nvar index space"}},
        "ls_mcp":          {"included": ["skein", "skglm"],
                            "excluded": {"ncvreg":  "p × p `XX` intermediate ≈ 800 MB at p=10k"}},
        "logistic_lasso":  {"included": ["skein"],
                            "excluded": {"glmnet":  "binomial path exceeds per-cell wall (~hour)"}},
        "ls_group_lasso":  {"included": ["skein"],
                            "excluded": {"grpreg":  "Fortran core copies X (peak RSS ≈ 3× X size)"}},
    },
}


def build_manifest(paper_dir: Path, project_root: Path) -> dict:
    files: dict[str, dict] = {}
    missing: list[str] = []
    for rel, meta in ARTIFACT_PROVENANCE.items():
        path = paper_dir / rel
        if not path.exists():
            missing.append(rel)
            continue
        files[rel] = {
            **meta,
            "size_bytes": path.stat().st_size,
            "mtime":      path.stat().st_mtime,
        }
    return {
        "generated_at": int(time.time()),
        "git_rev":      _git_rev(project_root),
        "git_dirty":    _git_dirty(project_root),
        "files":        files,
        "missing":      missing,
        "expected":     sorted(ARTIFACT_PROVENANCE.keys()),
        "at_scale_comparator_gap": AT_SCALE_COMPARATOR_GAP,
    }


def write_overview(paper_dir: Path, manifest: dict) -> None:
    sections: dict[str, list[tuple[str, dict]]] = {}
    for rel, meta in manifest["files"].items():
        sections.setdefault(meta["paper_section"], []).append((rel, meta))
    for k in sections:
        sections[k].sort(key=lambda r: r[0])

    lines = [
        "# Paper artifact bundle",
        "",
        f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(manifest['generated_at']))}.",
        f"Git rev: `{manifest['git_rev']}`"
        + (" (dirty working tree)" if manifest['git_dirty'] else ""),
        "",
        "Each artifact under `figures/` and `tables/` is the build output of a "
        "deterministic driver — never edit them by hand. Re-run the relevant "
        "Snakemake rule or driver script to regenerate.",
        "",
    ]
    for section in sorted(sections):
        lines.append(f"## {section}")
        lines.append("")
        lines.append("| Artifact | Driver | Source |")
        lines.append("|---|---|---|")
        for rel, meta in sections[section]:
            lines.append(f"| `{rel}` | `{meta['driver']}` | {meta['source']} |")
        lines.append("")
    if manifest["missing"]:
        lines.append("## Missing artifacts")
        lines.append("")
        for m in manifest["missing"]:
            lines.append(f"- `{m}`")
        lines.append("")
        lines.append("Run the appropriate driver to populate these.")
        lines.append("")
    (paper_dir / "BUNDLE.md").write_text("\n".join(lines))


def maybe_tarball(paper_dir: Path, git_rev: str, out_dir: Path) -> Path | None:
    """Write a `paper-bundle-<git>.tar.gz` next to `paper/` for archival."""
    out_dir.mkdir(parents=True, exist_ok=True)
    short = (git_rev or "unknown")[:8]
    tar_path = out_dir / f"skein-paper-bundle-{short}.tar.gz"
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(paper_dir, arcname=paper_dir.name,
               filter=lambda ti: ti if not ti.name.endswith(".tar.gz") else None)
    return tar_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", type=Path,
                    default=Path(__file__).resolve().parents[3] / "paper")
    ap.add_argument("--project-root", type=Path,
                    default=Path(__file__).resolve().parents[3])
    ap.add_argument("--tarball", action="store_true",
                    help="Also write a .tar.gz next to the paper/ dir.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit 1 if any expected artifact is missing.")
    a = ap.parse_args()

    if not a.paper_dir.exists():
        print(f"paper-dir does not exist: {a.paper_dir}", file=sys.stderr)
        return 2

    manifest = build_manifest(a.paper_dir, a.project_root)
    (a.paper_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    write_overview(a.paper_dir, manifest)

    print(f"manifest:  {a.paper_dir / 'manifest.json'}")
    print(f"overview:  {a.paper_dir / 'BUNDLE.md'}")
    print(f"present:   {len(manifest['files'])}/{len(manifest['expected'])} expected")
    if manifest["missing"]:
        print("missing:")
        for m in manifest["missing"]:
            print(f"  - {m}")

    if a.tarball:
        tar = maybe_tarball(a.paper_dir, manifest["git_rev"],
                            a.project_root / "dist")
        if tar:
            print(f"tarball:   {tar}  ({tar.stat().st_size // 1024} KB)")

    if a.strict and manifest["missing"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
