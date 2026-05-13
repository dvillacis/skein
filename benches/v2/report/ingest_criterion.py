"""Ingest the criterion JSON tree at `target/criterion/` and emit a
tidy JSONL the F9 figure builder consumes.

criterion's layout per benchmark:
    target/criterion/<group>/<bench>/new/raw.csv      # all samples
    target/criterion/<group>/<bench>/new/estimates.json  # median, mean, std, etc.

We pick `estimates.json` and pull the median.point_estimate field
(nanoseconds). The benches in `crates/skein-core/benches/block_cd.rs`
have groups like `serial_vs_parallel/{8,32,128}` and
`screening_modes/{off,strong,gap_safe}`.

Usage:
    cargo bench -p skein-core
    python -m benches.v2.report.ingest_criterion \\
        --criterion-dir target/criterion \\
        --out benches/v2/results/criterion.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def ingest(criterion_dir: Path) -> list[dict]:
    """Walk the criterion tree and emit (group, mode, median_ns).

    Real layouts seen:
      target/criterion/<group>/<bench>/new/estimates.json
      target/criterion/<group>/<subgroup>/<bench>/new/estimates.json
    """
    rows: list[dict] = []
    if not criterion_dir.exists():
        return rows
    for est_path in criterion_dir.rglob("estimates.json"):
        if est_path.parent.name != "new":
            continue
        try:
            data = json.loads(est_path.read_text())
        except Exception:
            continue
        rel = est_path.relative_to(criterion_dir)
        parts = rel.parts        # e.g. ('serial_vs_parallel', 'parallel', '32', 'new', 'estimates.json')
        # All but the trailing ('new', 'estimates.json') describe identity.
        ident = parts[:-2]
        if len(ident) < 2:
            continue
        group = ident[0]
        # mode = the remaining components joined with "/".
        mode = "/".join(ident[1:])
        median_ns = float(
            (data.get("median") or data.get("Median") or {}).get(
                "point_estimate", 0.0))
        rows.append({
            "group":   group,
            "mode":    mode,
            "median_ns": median_ns,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion-dir", type=Path,
                    default=Path("target/criterion"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = ingest(a.criterion_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"ingested {len(rows)} criterion benchmarks from {a.criterion_dir} → {a.out}")
    if not rows and a.criterion_dir.exists():
        print("  (no 'new/estimates.json' files found; "
              "did you run `cargo bench -p skein-core`?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
