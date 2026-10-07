"""Check that this machine can run the benchmark, before starting the long run.

    python scripts/check_env.py            # packages, GPU, data, disk
    python scripts/check_env.py --bench    # also time 3 images with the real SD2 autoencoder
                                           # (downloads the weights) and estimate the full run

Problems that would stop the run are marked FAIL (exit code 1), the rest WARN.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PINNED = {}
for line in (ROOT / "requirements.txt").read_text().splitlines():
    m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line)
    if m:
        PINNED[m[1].lower()] = m[2]
MODULES = {"torch": "torch", "torchvision": "torchvision", "diffusers": "diffusers",
           "transformers": "transformers", "huggingface-hub": "huggingface_hub", "lpips": "lpips",
           "numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow", "pillow": "PIL",
           "scipy": "scipy", "scikit-image": "skimage", "scikit-learn": "sklearn",
           "matplotlib": "matplotlib", "pyyaml": "yaml", "tqdm": "tqdm", "safetensors": "safetensors",
           "accelerate": "accelerate"}
problems = {"FAIL": 0, "WARN": 0}


def report(level: str, msg: str) -> None:
    if level in problems:
        problems[level] += 1
    print(f"  [{level:4s}] {msg}")


def check_python() -> None:
    print("Python")
    v = sys.version_info
    report("OK" if (v.major, v.minor) == (3, 10) else "WARN",
           f"{sys.version.split()[0]} ({sys.executable})"
           + ("" if (v.major, v.minor) == (3, 10) else "; the pinned versions were tested with 3.10"))


def check_packages() -> bool:
    print("Packages")
    ok = True
    for name, mod in MODULES.items():
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            report("FAIL", f"{name}: not importable ({type(e).__name__}: {e})")
            ok = False
            continue
        try:
            have = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            have = getattr(m, "__version__", "?")
        have = have.split("+")[0]
        want = PINNED.get(name)
        if want and have != want:
            report("WARN", f"{name} {have} (requirements.txt pins {want})")
    if ok:
        report("OK", "all packages import")
    return ok


def check_gpu() -> str:
    """Returns the device the run will use."""
    import torch

    print("GPU")
    built = torch.version.cuda
    report("OK" if built else "WARN",
           f"torch {torch.__version__}, " + (f"built for CUDA {built}" if built else "CPU-only build"))
    has_smi = shutil.which("nvidia-smi") is not None
    if not torch.cuda.is_available():
        if has_smi:
            report("FAIL" if not built else "WARN",
                   "an NVIDIA driver is installed but PyTorch sees no GPU. "
                   + ("Reinstall PyTorch with:  pip install -r requirements-gpu.txt" if not built
                      else "Check the driver version (nvidia-smi) supports this CUDA build."))
        else:
            report("WARN", "no NVIDIA GPU found; the run will use the CPU (~17 s per image)")
        return "cpu"
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        report("OK", f"cuda:{i} {p.name}, compute capability {p.major}.{p.minor}, "
                     f"{p.total_memory / 1024**3:.1f} GB")
        if p.total_memory < 3.5 * 1024**3:
            report("WARN", "under 4 GB of GPU memory: the 512x512 VAE decode may run out of memory")
    arch = f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"
    supported = torch.cuda.get_arch_list()
    if arch not in supported:
        report("WARN", f"this PyTorch build has kernels for {', '.join(supported)}, not {arch}")
    try:
        x = torch.randn(1, 3, 64, 64, device="cuda")
        w = torch.randn(8, 3, 3, 3, device="cuda")
        torch.nn.functional.conv2d(x, w).sum().item()
        report("OK", "a convolution runs on the GPU")
    except RuntimeError as e:
        report("FAIL", f"the GPU cannot run this PyTorch build ({str(e).splitlines()[0]}). For an "
                       "RTX 50-series GPU see README, 'Run on a GPU'.")
        return "cpu"
    return "cuda"


def check_data() -> None:
    import pandas as pd

    from aeroblade_bench.config import load_config, rpath

    print("Data")
    cfg = load_config()
    root = rpath(cfg["data"]["root"])
    if not root.is_dir():
        report("FAIL", f"{root} does not exist; put the TGIF testing files there (README, 'Data')")
        return
    index_csv = rpath(cfg["data"]["index_csv"])
    if not index_csv.exists():
        report("WARN", f"no {index_csv.name} yet; scripts/build_index.py creates it")
        return
    index = pd.read_csv(index_csv)
    files = list(index["file"]) + [f for f in index["mask_file"].dropna()]
    missing = [f for f in files if not (root / f).exists()]
    counts = index["generator"].value_counts().to_dict()
    if missing:
        report("FAIL", f"{len(missing)} of {len(files)} images/masks listed in the index are missing "
                       f"under {root}, e.g. {missing[0]}")
    else:
        report("OK", f"all {len(index)} images ({counts}) and their masks are under {root}")


def check_disk() -> None:
    print("Disk")
    free = shutil.disk_usage(ROOT).free / 1024**3
    report("OK" if free > 5 else "WARN", f"{free:.0f} GB free next to the cache (the full run writes ~3 GB)")


def bench(device: str) -> None:
    import torch

    from aeroblade_bench.autoencoders import load_autoencoder, reconstruct
    from aeroblade_bench.lpips_map import LPIPSLayerMap

    print(f"Benchmark ({device}, SD2 autoencoder, 512x512; downloads the weights on first use)")
    ae = load_autoencoder("SD2", device)
    lp = LPIPSLayerMap(2, pad=16).to(device)
    x = torch.rand(1, 3, 512, 512)
    times = []
    for i in range(4):
        t = time.perf_counter()
        rec = reconstruct(ae, x.to(device), 1)
        lp(x.to(device), rec, pad=0)
        lp(x.to(device), rec, pad=16)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if i:  # the first pass includes one-time warm-up
            times.append(time.perf_counter() - t)
    s = sum(times) / len(times)
    report("OK", f"{s:.2f} s per image for the model alone; 4459 images = about {4459 * s / 3600:.1f} h "
                 f"plus file reading/writing")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", action="store_true", help="time the SD2 autoencoder on this machine")
    args = p.parse_args()
    check_python()
    if not check_packages():
        print("\nInstall the packages first: pip install -r requirements-gpu.txt (NVIDIA GPU) "
              "or pip install -r requirements.txt (CPU)")
        sys.exit(1)
    device = check_gpu()
    check_data()
    check_disk()
    if args.bench:
        bench(device)
    print(f"\nThe run will use: {device}.  {problems['FAIL']} FAIL, {problems['WARN']} WARN")
    if problems["FAIL"]:
        sys.exit(1)
    print("Next: python scripts/run_all.py")


if __name__ == "__main__":
    main()
