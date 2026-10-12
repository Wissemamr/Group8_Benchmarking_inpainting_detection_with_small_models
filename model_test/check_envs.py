"""Find (or prepare) a working conda environment for each of the 4 models. Standard library only:
run it from the Anaconda Prompt with any Python.

    python model_test/check_envs.py            # probe every env, print the plan, write envs.json
    python model_test/check_envs.py --apply    # also carry out the plan (pip install / conda create)

For each model, in this order:
  1. its preferred env (AEROBLADE: Aeroblade, PSCC-Net: pcss, ...) if it already has everything
  2. any other env that already has everything, with a working CUDA PyTorch
  3. ADD-ON: an env with a working CUDA PyTorch that only lacks small pip packages (timm, yacs,
     fvcore, opencv, pyyaml, einops). Only missing packages are installed, never upgraded or downgraded,
     so existing projects keep working. Our own AEROBLADE env is preferred for this.
  4. CREATE: a new env "<model>-gpu" (Python 3.10, PyTorch 2.1.2 + CUDA 12.1, the rest pinned).
     This downloads ~2.5 GB of PyTorch and can take 10-20 minutes.
The choice is written to model_test/envs.json, which run_all.py reads.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PROBE = r"""
import json, importlib, sys
out = {"python": sys.version.split()[0], "modules": {}}
for name in ["torch", "torchvision", "numpy", "PIL", "timm", "yacs", "fvcore", "cv2", "yaml", "diffusers",
             "lpips", "huggingface_hub", "safetensors", "transformers", "skimage", "matplotlib", "psutil", "triton",
             "einops"]:
    try:
        m = importlib.import_module(name)
        out["modules"][name] = str(getattr(m, "__version__", "?"))
    except Exception as e:
        out["modules"][name] = None
try:
    import torch
    out["torch_cuda_build"] = torch.version.cuda
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
        out["arch_list"] = torch.cuda.get_arch_list()
        x = torch.randn(1, 3, 32, 32, device="cuda")
        torch.nn.functional.conv2d(x, torch.randn(4, 3, 3, 3, device="cuda")).sum().item()
        out["cuda_works"] = True
except Exception as e:
    out["cuda_works"] = False
    out["cuda_error"] = str(e).splitlines()[0][:200]
print("PROBE_JSON" + json.dumps(out))
"""

# pip pins for the small packages (never installed over an existing version)
LIGHT = {"timm": "timm==0.5.4", "yacs": "yacs==0.1.8", "fvcore": "fvcore==0.1.5.post20221221",
         "cv2": "opencv-python-headless==4.10.0.84", "yaml": "pyyaml", "PIL": "pillow", "einops": "einops==0.8.0"}
BASE = ["torch", "torchvision", "numpy", "PIL"]
MODELS = {
    "aeroblade": {"label": "AEROBLADE", "prefer": ["aeroblade"],
                  "needs": BASE + ["diffusers", "lpips", "huggingface_hub", "safetensors", "transformers", "skimage"],
                  "create": ["-r", str(ROOT / "AEROBLADE" / "requirements-gpu.txt")]},
    "pscc": {"label": "PSCC-Net", "prefer": ["pcss", "pscc", "pscc-net"], "needs": BASE + ["yacs"]},
    "trufor": {"label": "TruFor", "prefer": ["trufor"], "needs": BASE + ["timm", "yacs", "cv2"]},
    "forma": {"label": "ForMa", "prefer": ["forma"], "needs": BASE + ["timm", "fvcore", "yaml", "cv2", "einops"]},
}
TORCH_CUDA = ["--extra-index-url", "https://download.pytorch.org/whl/cu121", "torch==2.1.2+cu121",
              "torchvision==0.16.2+cu121", "numpy==1.26.4", "pillow"]


def conda_exe() -> str:
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda") or shutil.which("conda.bat")
    if not exe:
        sys.exit("conda not found: run this from the Anaconda Prompt (or a terminal where `conda` works)")
    return exe


def conda(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([conda_exe(), *args], capture_output=capture, text=True)


def list_envs() -> list[dict]:
    info = json.loads(conda("info", "--json").stdout)
    root = os.path.normcase(os.path.normpath(info.get("root_prefix", "")))
    envs = []
    for p in json.loads(conda("env", "list", "--json").stdout)["envs"]:
        name = "base" if os.path.normcase(os.path.normpath(p)) == root else Path(p).name
        envs.append({"name": name, "prefix": p})
    return envs


def probe(prefix: str) -> dict:
    # a script file, not `python -c`: conda run on Windows breaks multi-line -c arguments
    script = Path(tempfile.gettempdir()) / "model_test_probe_env.py"
    script.write_text(PROBE, encoding="utf-8")
    r = conda("run", "-p", prefix, "python", str(script))
    for line in (r.stdout or "").splitlines():
        if line.startswith("PROBE_JSON"):
            return json.loads(line[len("PROBE_JSON"):])
    return {"error": (r.stderr or r.stdout or "no python in this env").strip().splitlines()[-1][:200]
            if (r.stderr or r.stdout) else "no python in this env"}


def missing(p: dict, needs: list[str]) -> list[str]:
    return [m for m in needs if not (p.get("modules") or {}).get(m)]


ALLOW_CPU = False  # --allow-cpu: accept envs whose PyTorch has no CUDA (CPU-only runs, testing)


def gpu_ok(p: dict) -> bool:
    if ALLOW_CPU:
        return bool((p.get("modules") or {}).get("torch"))
    return bool(p.get("cuda_works"))


def plan_model(key: str, spec: dict, envs: list[dict], probes: dict) -> dict:
    def rank(e):
        n = e["name"].lower()
        pref = spec["prefer"].index(n) if n in spec["prefer"] else 99
        py310 = 0 if (probes[e["prefix"]].get("python") or "").startswith("3.10") else 1
        return (pref, py310, e["name"])

    ready = [e for e in envs if gpu_ok(probes[e["prefix"]]) and not missing(probes[e["prefix"]], spec["needs"])]
    if ready:
        e = sorted(ready, key=rank)[0]
        return {"action": "use", "env": e["name"], "prefix": e["prefix"], "install": []}
    addon = [e for e in envs if gpu_ok(probes[e["prefix"]])
             and set(missing(probes[e["prefix"]], spec["needs"])) <= set(LIGHT)]
    if addon:
        aero = [e for e in addon if e["name"].lower() in MODELS["aeroblade"]["prefer"]]
        e = aero[0] if aero else sorted(addon, key=rank)[0]
        pkgs = [LIGHT[m] for m in missing(probes[e["prefix"]], spec["needs"])]
        return {"action": "pip", "env": e["name"], "prefix": e["prefix"], "install": pkgs}
    name = f"{key}-gpu"
    pkgs = spec.get("create") or (TORCH_CUDA + [LIGHT[m] for m in spec["needs"] if m in LIGHT])
    return {"action": "create", "env": name, "prefix": None, "install": pkgs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="install / create as planned")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--allow-cpu", action="store_true", help="accept PyTorch without CUDA (CPU-only runs)")
    args = ap.parse_args()
    global ALLOW_CPU
    ALLOW_CPU = args.allow_cpu

    envs = list_envs()
    print(f"probing {len(envs)} conda envs (a few seconds each)...")
    probes = {}
    for e in envs:
        p = probes[e["prefix"]] = probe(e["prefix"])
        mods = p.get("modules") or {}
        if not mods.get("torch"):
            gpu = "no torch" if p.get("python") else ("not a Python env" if p.get("error") else "no torch")
        elif p.get("gpu"):
            gpu = p["gpu"] + ("" if p.get("cuda_works") else " (CUDA test failed: " + p.get("cuda_error", "?") + ")")
        else:
            gpu = "PyTorch without CUDA" if not p.get("torch_cuda_build") else "CUDA build, but no GPU visible"
        print(f"  {e['name']:18s} python {p.get('python', '?'):8s} torch {mods.get('torch') or '-':16s} {gpu}")
        if p.get("capability") and p.get("arch_list") and p["capability"] not in p["arch_list"]:
            print(f"  {'':18s} WARNING: this PyTorch has no kernels for {p['capability']} "
                  f"(RTX 50-series?); see AEROBLADE/README.md, 'Run on a GPU'")

    plan = {}
    print("\nplan:")
    for key in args.models:
        spec = MODELS[key]
        pl = plan[key] = plan_model(key, spec, envs, probes)
        what = {"use": "use as is", "pip": "add " + " ".join(pl["install"]),
                "create": "create a new env (Python 3.10 + PyTorch 2.1.2 CUDA 12.1; ~10-20 min)"}[pl["action"]]
        print(f"  {spec['label']:10s} -> {pl['env']:18s} {what}")

    if args.apply:
        for key, pl in plan.items():
            if pl["action"] == "pip":
                print(f"\ninstalling into {pl['env']}: {' '.join(pl['install'])}")
                if conda("run", "-p", pl["prefix"], "--no-capture-output", "python", "-m", "pip", "install",
                         *pl["install"], capture=False).returncode:
                    sys.exit(f"pip install failed in {pl['env']}")
            elif pl["action"] == "create":
                existing = [e for e in envs if e["name"] == pl["env"]]
                if not existing:
                    print(f"\ncreating env {pl['env']} (Python 3.10)")
                    if conda("create", "-y", "-n", pl["env"], "python=3.10", capture=False).returncode:
                        sys.exit(f"conda create {pl['env']} failed")
                envs = list_envs()
                pl["prefix"] = next(e["prefix"] for e in envs if e["name"] == pl["env"])
                print(f"installing into {pl['env']}: {' '.join(pl['install'])}")
                if conda("run", "-p", pl["prefix"], "--no-capture-output", "python", "-m", "pip", "install",
                         *pl["install"], capture=False).returncode:
                    sys.exit(f"pip install failed in {pl['env']}")
            if pl["action"] != "use":
                p = probe(pl["prefix"])
                still = missing(p, MODELS[key]["needs"])
                if still or not gpu_ok(p):
                    sys.exit(f"{pl['env']} still not ready for {MODELS[key]['label']}: missing {still}, "
                             f"CUDA ok {gpu_ok(p)}")
                pl["action"] = "use"
        print("\nall environments ready")
    elif any(pl["action"] != "use" for pl in plan.values()):
        print("\nnothing installed yet: re-run with --apply to carry out the plan")

    report_env = plan.get("aeroblade") or next(iter(plan.values()))
    out = {"models": {k: {"env": v["env"], "prefix": v["prefix"], "ready": v["action"] == "use"} for k, v in plan.items()},
           "report_env": {"env": report_env["env"], "prefix": report_env["prefix"]}}
    (HERE / "envs.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {HERE / 'envs.json'}")


if __name__ == "__main__":
    main()
