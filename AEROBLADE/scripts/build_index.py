"""Index every TGIF testing image (authentic + SD2 + Photoshop) and its mask into one CSV.

Writes results/index/all_images.csv (one row per image) and index_summary.json (counts per
generator / category / mask type, image sizes). Fails loudly if a mask is missing, an image and
its mask differ in size, or a source image is not a complete 1 authentic + 6 SD + 6 PS family.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aeroblade_bench.config import DEFAULT_CONFIG, load_config, rpath, sha256_file  # noqa: E402
from aeroblade_bench.dataset import build_index, check_families  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    cfg = load_config(args.config)
    data_root = rpath(cfg["data"]["root"])
    out_csv = rpath(cfg["data"]["index_csv"])

    index = build_index(data_root)
    check_families(index)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(out_csv, index=False)

    summary = {
        "data_root": str(data_root),
        "n_images": len(index),
        "n_source_images": int(index["source_id"].nunique()),
        "by_generator": index["generator"].value_counts().to_dict(),
        "by_mask_type": index["mask_type"].value_counts().to_dict(),
        "by_category": index.groupby("category")["source_id"].nunique().to_dict(),
        "by_category_generator": {
            c: g["generator"].value_counts().to_dict() for c, g in index.groupby("category")
        },
        "image_width": index["width"].describe().to_dict(),
        "image_height": index["height"].describe().to_dict(),
        "n_distinct_sizes": int(index.groupby(["width", "height"]).ngroups),
        "index_csv_sha256": sha256_file(out_csv),
    }
    (out_csv.parent / "index_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(index)} images from {summary['n_source_images']} source images -> {out_csv}")
    print("generators:", summary["by_generator"], "| mask types:", summary["by_mask_type"])
    print("sources per category:", summary["by_category"])


if __name__ == "__main__":
    main()
