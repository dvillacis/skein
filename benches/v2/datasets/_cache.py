"""Shared cache helpers for real-dataset loaders.

Each loader caches to `~/.cache/skein-bench/<dataset_id>/` and verifies
a SHA-256 manifest. The Problem dataclass it returns is the same one
the rest of the v2 pipeline already consumes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def cache_root() -> Path:
    root = Path(os.environ.get(
        "SKEIN_BENCH_CACHE",
        Path.home() / ".cache" / "skein-bench",
    ))
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_dir(dataset_id: str) -> Path:
    d = cache_root() / dataset_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(d: Path, manifest: dict[str, Any]) -> None:
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def read_manifest(d: Path) -> dict[str, Any] | None:
    p = d / "manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def verify_or_drop(d: Path) -> bool:
    """Return True if every file in manifest matches its recorded sha256.
    Drops broken caches so the loader re-fetches cleanly."""
    m = read_manifest(d)
    if not m:
        return False
    for relpath, want in (m.get("files") or {}).items():
        p = d / relpath
        if not p.exists() or sha256(p) != want:
            # Cache is broken — let the caller re-fetch.
            return False
    return True
