"""Config loading and paths. Everything is relative to the AEROBLADE/ folder."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "aeroblade_tgif.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def rpath(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def cache_dir(cfg: dict[str, Any]) -> Path:
    """Reconstructions, LPIPS maps and per-image records of one (image size, seed) setting."""
    m = cfg["model"]
    return ROOT / "cache" / f"{m['image_size']}_seed{m['seed']}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance only
        return None


def environment_info() -> dict[str, Any]:
    """Software and hardware the numbers were produced on (for the timing context)."""
    import numpy
    import pandas
    import torch

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit(),
    }
    for mod in ("diffusers", "lpips", "sklearn", "skimage"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        cpu = [ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")]
        info["cpu"] = cpu[0] if cpu else platform.processor()
        info["cpu_logical_cores"] = len(cpu)
    except OSError:
        info["cpu"] = platform.processor()
    try:
        mem = [ln for ln in open("/proc/meminfo") if ln.startswith("MemTotal")][0]
        info["ram_gb"] = round(int(mem.split()[1]) / 1024**2, 1)
    except (OSError, IndexError):
        pass
    return info
