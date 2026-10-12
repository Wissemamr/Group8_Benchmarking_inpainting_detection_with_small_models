"""Fetch the third-party code and weights the 4 models need (Python standard library only).

    python model_test/setup_models.py              # everything that is missing
    python model_test/setup_models.py --only trufor forma

AEROBLADE  nothing to do: its autoencoder and VGG16 weights download on first use (Hugging Face,
           download.pytorch.org)
TruFor     official code (github.com/grip-unina/TruFor, commit pinned by TruFor/README.md) into
           TruFor/TruFor/, weights zip (261 MB, MD5-checked) unzipped into TruFor/weights/extracted/
ForMa      checkpoint from the authors' Google Drive link into ForMa/ForMa_weights/ (SHA-256-checked)
PSCC-Net   uses an existing PSCC-Net folder when one is found (env var PSCC_ROOT, or a folder named
           PSCC-TGIF / PSCC-Net inside or next to this repo, or ~/projects/PSCC-TGIF); otherwise
           clones the official repo (its checkpoints are in git) into model_test/third_party/PSCC-Net
All of these locations are git-ignored. Re-running skips what is already there.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

TRUFOR_REPO = "https://github.com/grip-unina/TruFor"
TRUFOR_COMMIT = "ae54475df6f41a491d7615100feb19263dec13f7"
TRUFOR_ZIP = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
TRUFOR_ZIP_MD5 = "7bee48f3476c75616c3c5721ab256ff8"
FORMA_URL = "https://drive.usercontent.google.com/download?id=15NcJQr5kxhqC4V-WjyJhAqIsobI6u-yh&export=download&confirm=t"
FORMA_SHA256 = "3c28719a48848411c4647e2460887d9a5d6eb64f2c3e3a77f5061c686925c2ee"
PSCC_REPO = "https://github.com/proteus1991/PSCC-Net"


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def git(*args: str) -> None:
    if shutil.which("git") is None:
        sys.exit("git is not installed; install Git for Windows (git-scm.com) and re-run")
    subprocess.run(["git", *args], check=True)


def find_pscc_root():
    """An existing PSCC-Net checkout with its three checkpoints, or None."""
    cands = []
    if os.environ.get("PSCC_ROOT"):
        cands.append(Path(os.environ["PSCC_ROOT"]))
    for base in (ROOT, ROOT.parent):
        for name in ("PSCC-TGIF", "PSCC-Net", "PSCC"):
            cands.append(base / name)
    cands += [Path.home() / "projects" / "PSCC-TGIF", HERE / "third_party" / "PSCC-Net"]
    for c in cands:
        if (c / "checkpoint" / "HRNet_checkpoint" / "HRNet.pth").is_file() and (c / "models" / "seg_hrnet.py").is_file():
            return c.resolve()
    return None


def setup_trufor() -> None:
    print("TruFor")
    code = ROOT / "TruFor" / "TruFor"
    if not (code / "TruFor_train_test").is_dir():
        git("clone", "--quiet", TRUFOR_REPO, str(code))
        git("-C", str(code), "checkout", "--quiet", TRUFOR_COMMIT)
    print(f"  code: {code} (commit {TRUFOR_COMMIT[:8]})")
    ckpt = ROOT / "TruFor" / "weights" / "extracted" / "weights" / "trufor.pth.tar"
    if not ckpt.is_file():
        z = ROOT / "TruFor" / "weights" / "TruFor_weights.zip"
        if not z.is_file():
            download(TRUFOR_ZIP, z)
        if digest(z, "md5") != TRUFOR_ZIP_MD5:
            sys.exit(f"  {z}: MD5 mismatch; delete it and re-run")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ROOT / "TruFor" / "weights" / "extracted")
    print(f"  weights: {ckpt}")


def setup_forma() -> None:
    print("ForMa")
    w = ROOT / "ForMa" / "ForMa_weights" / "ForMa_weights.pth"
    if not w.is_file():
        download(FORMA_URL, w)
    if digest(w, "sha256") != FORMA_SHA256:
        sys.exit(f"  {w}: SHA-256 mismatch (Google Drive may have returned an HTML page); delete it and "
                 f"download manually from https://drive.google.com/file/d/15NcJQr5kxhqC4V-WjyJhAqIsobI6u-yh")
    print(f"  weights: {w}")


def setup_pscc() -> None:
    print("PSCC-Net")
    root = find_pscc_root()
    if root is None:
        dest = HERE / "third_party" / "PSCC-Net"
        print(f"  no existing PSCC-Net folder found; cloning the official repo into {dest}")
        git("clone", "--quiet", "--depth", "1", PSCC_REPO, str(dest))
        root = find_pscc_root()
    print(f"  code + checkpoints: {root}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=["trufor", "forma", "pscc"])
    args = p.parse_args()
    steps = {"trufor": setup_trufor, "forma": setup_forma, "pscc": setup_pscc}
    for name, fn in steps.items():
        if not args.only or name in args.only:
            fn()
    print("AEROBLADE\n  nothing to fetch: weights download on first run")
    print("done")


if __name__ == "__main__":
    main()
