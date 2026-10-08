"""Create the shareable resamples of the TGIF testing set.

A resample is a random draw of *source images* (COCO ids), stratified by category and made
without replacement. It keeps each drawn source image's whole family: the authentic image plus
its 6 SD and 6 PS inpaintings. Resamples are therefore self-contained detection and localization
benchmarks, and a model never sees a source image's authentic version without its inpaintings
(or the reverse) in the same resample.

Outputs, all small enough to commit and send to colleagues:
  resamples/resample_{k}.csv      one row per image of resample k (same columns as the index)
  resamples/resample_sources.csv  one row per source image, with a 0/1 flag per resample
  resamples/resamples.json        parameters, seeds, sizes, pairwise overlaps, checksums

`--materialize DIR` additionally copies the image and mask files of each resample into
DIR/resample_{k}/ (hard links when possible), with the TGIF folder layout, ready to zip.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aeroblade_bench.config import DEFAULT_CONFIG, ROOT, load_config, rpath, sha256_file  # noqa: E402


def draw(sources: pd.DataFrame, fraction: float, seed: int) -> set[int]:
    """Stratified draw: round(fraction * n) sources per category (at least 1), no replacement."""
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    for _, grp in sources.groupby("category"):
        ids = np.sort(grp["source_id"].to_numpy())
        k = min(len(ids), max(1, math.floor(fraction * len(ids) + 0.5)))
        chosen.update(int(i) for i in rng.choice(ids, size=k, replace=False))
    return chosen


def materialize(df: pd.DataFrame, data_root: Path, dest: Path) -> None:
    files = set(df["file"]) | set(df["mask_file"].dropna())
    for rel in sorted(files):
        src, dst = data_root / rel, dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--n", type=int, help="number of resamples (config: resamples.n)")
    p.add_argument("--fraction", type=float, help="share of source images per resample")
    p.add_argument("--seed", type=int, help="base seed; resample k uses seed + k")
    p.add_argument("--out-dir", type=Path, default=ROOT / "resamples")
    p.add_argument("--materialize", type=Path, help="also copy/link the files into this folder")
    args = p.parse_args()
    cfg = load_config(args.config)
    rs = cfg["resamples"]
    n = args.n or rs["n"]
    fraction = args.fraction or rs["fraction"]
    seed = args.seed if args.seed is not None else rs["seed"]

    index = pd.read_csv(rpath(cfg["data"]["index_csv"]))
    sources = index.drop_duplicates("source_id")[["source_id", "category"]].sort_values("source_id")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    drawn: dict[int, set[int]] = {}
    meta = {
        "scheme": "stratified by category, source images drawn without replacement, "
                  "whole families (1 authentic + 6 SD + 6 PS) kept",
        "fraction": fraction, "base_seed": seed, "n_source_images": len(sources),
        "index_csv": str(rpath(cfg["data"]["index_csv"]).relative_to(ROOT)),
        "index_csv_sha256": sha256_file(rpath(cfg["data"]["index_csv"])),
        "resamples": {},
    }
    for k in range(1, n + 1):
        ids = draw(sources, fraction, seed + k)
        drawn[k] = ids
        df = index[index["source_id"].isin(ids)].copy()
        df.insert(0, "resample_id", k)
        path = args.out_dir / f"resample_{k}.csv"
        df.to_csv(path, index=False)
        meta["resamples"][str(k)] = {
            "seed": seed + k,
            "n_source_images": len(ids),
            "n_images": len(df),
            "by_generator": df["generator"].value_counts().to_dict(),
            "sources_per_category": df.groupby("category")["source_id"].nunique().to_dict(),
            "csv": path.name,
            "csv_sha256": sha256_file(path),
        }
        if args.materialize:
            materialize(df, rpath(cfg["data"]["root"]), args.materialize / f"resample_{k}")
        print(f"resample {k}: {len(ids)} source images, {len(df)} images -> {path.name}")

    flags = sources.copy()
    for k, ids in drawn.items():
        flags[f"in_resample_{k}"] = flags["source_id"].isin(ids).astype(int)
    flags.to_csv(args.out_dir / "resample_sources.csv", index=False)
    meta["pairwise_source_overlap"] = {
        f"{a}&{b}": len(drawn[a] & drawn[b]) for a in drawn for b in drawn if a < b
    }
    meta["sources_in_no_resample"] = int((flags[[f"in_resample_{k}" for k in drawn]].sum(axis=1) == 0).sum())
    (args.out_dir / "resamples.json").write_text(json.dumps(meta, indent=1))
    print("pairwise overlap of source images:", meta["pairwise_source_overlap"])


if __name__ == "__main__":
    main()
