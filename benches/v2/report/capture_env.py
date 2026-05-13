"""Environment capture — runs at the start of every Snakemake cell.

Writes a dict with everything needed to reproduce the run:
- OS / CPU model / BLAS link
- Python / R interpreter versions
- pip freeze (filtered to the bench-relevant packages)
- git rev + dirty-tree flag
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


BENCH_PACKAGES = {
    "skein-glm", "scikit-learn", "skglm", "celer", "statsmodels",
    "lifelines", "numpy", "scipy", "pandas", "pyarrow", "snakemake",
}


def _run(cmd: list[str], timeout: int = 10) -> str:
    try:
        out = subprocess.run(cmd, check=False, capture_output=True,
                             text=True, timeout=timeout)
        return (out.stdout or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _blas_link() -> str:
    """Best-effort BLAS detection from numpy.

    Numpy 2.x exposes show_config(mode='dicts'). On macOS Accelerate
    is the link target; on Linux it's typically OpenBLAS or MKL.
    """
    try:
        import numpy as np
        if hasattr(np.__config__, "show"):
            try:
                cfg = np.__config__.show(mode="dicts")
            except TypeError:
                cfg = None
            if isinstance(cfg, dict):
                build = cfg.get("Build Dependencies", cfg)
                blas = build.get("blas") if isinstance(build, dict) else None
                if isinstance(blas, dict):
                    name = blas.get("name") or blas.get("found", {}).get("name")
                    if name:
                        return str(name)
                for k in ("blas_info", "openblas_info", "accelerate_info", "blas_opt_info"):
                    if k in cfg and cfg[k]:
                        return str(cfg[k].get("libraries") or k)
        # Fallback: probe the loaded numpy core for Accelerate (macOS).
        if platform.system() == "Darwin":
            core = Path(np.__file__).parent / "_core"
            for so in core.glob("*multiarray*.so"):
                links = _run(["otool", "-L", str(so)])
                if "Accelerate" in links:
                    return "accelerate"
                if "openblas" in links.lower():
                    return "openblas"
        return "unknown"
    except Exception:
        return "unknown"


def _pip_freeze() -> dict[str, str]:
    """Map of bench-relevant packages → version.

    Prefer importlib.metadata so editable installs (maturin develop)
    are picked up even when not in `pip freeze` output.
    """
    out: dict[str, str] = {}
    try:
        from importlib import metadata as md
        for name in BENCH_PACKAGES:
            try:
                out[name] = md.version(name)
            except md.PackageNotFoundError:
                pass
    except Exception:
        pass
    # Fall back to pip freeze for anything still missing.
    raw = _run([sys.executable, "-m", "pip", "freeze"], timeout=30)
    for line in raw.splitlines():
        if "==" in line:
            name, ver = line.split("==", 1)
            if name.lower() in BENCH_PACKAGES and name not in out:
                out[name] = ver
    return out


def _r_session() -> str:
    rscript = shutil.which("Rscript")
    if not rscript:
        return ""
    return _run([rscript, "-e",
                 "cat(R.version.string); cat('\\n'); "
                 "for (p in c('glmnet','ncvreg','grpreg','arrow')) "
                 "tryCatch(cat(p,'==',as.character(packageVersion(p)),'\\n'), error=function(e) NULL)"],
                timeout=30)


def capture(extra: dict[str, object] | None = None) -> dict[str, object]:
    env: dict[str, object] = {
        "os":       platform.platform(),
        "system":   platform.system(),
        "machine":  platform.machine(),
        "cpu":      _cpu_model(),
        "cpu_count": os.cpu_count(),
        "blas":     _blas_link(),
        "python":   sys.version.split()[0],
        "git_rev":  _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "pip":      _pip_freeze(),
        "r":        _r_session(),
    }
    env["host_id"] = hashlib.sha256(
        f"{env['cpu']}|{env['blas']}|{env['cpu_count']}".encode()
    ).hexdigest()[:12]
    if extra:
        env.update(extra)
    return env


def write(out: Path, extra: dict[str, object] | None = None) -> dict[str, object]:
    env = capture(extra=extra)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2, sort_keys=True))
    return env


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    env = write(args.out)
    print(json.dumps({"host_id": env["host_id"], "git_rev": env["git_rev"]}, indent=2))
