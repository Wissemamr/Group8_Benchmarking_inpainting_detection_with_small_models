"""Build the small committed test set: one source photo per TGIF category, 5 images from it.

    python model_test/make_testset.py --tgif-root AEROBLADE/data/tgif

Per category (13): the authentic original, SD2-sp bbox and segm, PS-sp bbox and segm (generation
variant 0), plus the 4 ground-truth masks; 65 images in total. The source photo is drawn per
category with a fixed seed. Files keep TGIF's layout under model_test/testset/, and
model_test/testset/manifest.csv lists them (paths relative to testset/).
Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "testset"
INDEX = ROOT / "AEROBLADE" / "results" / "index" / "all_images.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tgif-root", type=Path, required=True, help="folder holding orig/ sd2-sp/ ps-sp/ masks/")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    rows = list(csv.DictReader(open(INDEX, newline="")))
    by_cat = defaultdict(set)
    for r in rows:
        by_cat[r["category"]].add(int(r["source_id"]))
    rng = random.Random(args.seed)
    chosen = {cat: rng.choice(sorted(ids)) for cat, ids in sorted(by_cat.items())}

    keep = []
    for r in rows:
        if chosen.get(r["category"]) != int(r["source_id"]):
            continue
        if r["label"] == "0" or r["variant"] == "0":
            keep.append(r)
    keep.sort(key=lambda r: (r["category"], r["label"], r["generator"], r["mask_type"]))
    assert len(keep) == 5 * len(chosen), len(keep)

    for r in keep:
        for col in ("file", "mask_file"):
            if r[col]:
                dest = OUT / r[col]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.tgif_root / r[col], dest)
    cols = ["image_id", "source_id", "category", "generator", "mask_type", "variant", "label",
            "width", "height", "file", "mask_file"]
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)
    n_masks = sum(1 for r in keep if r["mask_file"])
    print(f"{len(keep)} images ({n_masks} with masks) from {len(chosen)} source photos -> {OUT}")
    for cat, sid in chosen.items():
        print(f"  {cat:14s} source {sid}")


if __name__ == "__main__":
    main()
