"""Whole AEROBLADE benchmark in one command, on Windows, Linux or macOS (same steps as run_all.sh):
index -> resamples (if missing) -> AEROBLADE over every image -> dataframes/tables.

    python scripts/run_all.py                    # GPU if PyTorch sees one, else CPU
    python scripts/run_all.py --device cuda      # extra arguments go to run_aeroblade.py
    python scripts/run_all.py --max-sources 5    # quick partial run (65 images)

Resumable: run it again after an interruption and it continues where it stopped. The evaluation
step runs only when the compute step finished without being interrupted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def step(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print(">>", " ".join([Path(sys.executable).name, f"scripts/{script}", *args]), flush=True)
    try:
        code = subprocess.run(cmd, cwd=ROOT).returncode
    except KeyboardInterrupt:
        code = 130
    if code:
        print(f"stopped: scripts/{script} exited with code {code}; run the same command again to "
              f"continue", flush=True)
        sys.exit(code)


def main() -> None:
    step("build_index.py")
    if not (ROOT / "resamples" / "resample_1.csv").exists():
        step("make_resamples.py")
    step("run_aeroblade.py", *sys.argv[1:])
    step("evaluate.py")


if __name__ == "__main__":
    main()
