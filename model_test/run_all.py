"""Run the 4 models, each in its own conda env (from envs.json), then build the report.
Standard library only: run it from the Anaconda Prompt with any Python.

    python model_test/check_envs.py --apply       # once: pick / prepare the envs
    python model_test/run_all.py                  # all models on the GPU, then the report
    python model_test/run_all.py --models trufor pscc --limit 5     # quick partial run
    python model_test/run_all.py --report-only

Extra options (--device, --limit, --warmup, --profile-images, --flops-images, --no-trace,
--pscc-root, --pscc-threshold) are passed to run_model.py. A failing model does not stop the
others; the report covers whatever finished.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["aeroblade", "trufor", "forma", "pscc"]


def conda_exe() -> str:
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda") or shutil.which("conda.bat")
    if not exe:
        sys.exit("conda not found: run this from the Anaconda Prompt")
    return exe


def in_env(env: dict, *args: str) -> int:
    target = ["-p", env["prefix"]] if env.get("prefix") else ["-n", env["env"]]
    cmd = [conda_exe(), "run", "--no-capture-output", *target, "python", *args]
    return subprocess.run(cmd, cwd=HERE.parent, env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=ORDER, default=ORDER)
    ap.add_argument("--report-only", action="store_true")
    args, passthrough = ap.parse_known_args()

    envs_file = HERE / "envs.json"
    if not envs_file.exists():
        sys.exit("no model_test/envs.json: run  python model_test/check_envs.py --apply  first")
    envs = json.loads(envs_file.read_text())
    status = {}
    if not args.report_only:
        for m in args.models:
            env = envs["models"].get(m)
            if not env or not env.get("ready"):
                print(f"\n=== {m}: no ready env in envs.json (run check_envs.py --apply); skipped")
                status[m] = "skipped"
                continue
            print(f"\n=== {m} in env {env['env']}", flush=True)
            t = time.time()
            code = in_env(env, "model_test/run_model.py", "--model", m, *passthrough)
            status[m] = "ok" if code == 0 else f"failed (exit {code})"
            print(f"=== {m}: {status[m]} in {time.time() - t:.0f} s", flush=True)
    print("\n=== report", flush=True)
    code = in_env(envs["report_env"], "model_test/make_report.py")
    for m, s in status.items():
        print(f"  {m:10s} {s}")
    sys.exit(code)


if __name__ == "__main__":
    main()
