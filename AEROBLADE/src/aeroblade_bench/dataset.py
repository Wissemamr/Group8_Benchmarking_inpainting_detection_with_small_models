"""Index of the TGIF *testing* images used by the AEROBLADE benchmark.

Three groups of files, all from TGIF's testing split (343 COCO source images):

    orig/testing/<category>/{id}_orig.png                             authentic       (label 0)
    sd2-sp/testing/<category>/{id}_mask_{bbox|segm}.png_ps_mask.png_sd2_{0|1|2}.png   SD2 inpainting  (label 1)
    ps-sp/testing/<category>/{id}_mask_{bbox|segm}.png_ps_{0|1|2}.png                  Photoshop       (label 1)

The ground truth of both spliced sets is `masks/testing/<category>/{id}_mask_{type}.png_ps_mask.png`
(TGIF's "Photoshop adaptation" of the mask; the three variants of an image share it). Files such
as `*_orig_512.png` / `*_orig_1024.png` belong to TGIF's other subsets and are ignored.

`generator` is "SD" (Stable Diffusion 2 inpainting), "PS" (Adobe Photoshop generative fill) or
"real" (authentic). `variant` is the generation index 0..2, -1 for authentic images.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd
from PIL import Image

SD_RE = re.compile(
    r"^(?P<id>\d+)_mask_(?P<mt>bbox|segm)\.png_ps_mask\.png_sd2_(?P<v>\d+)\.png$"
)
PS_RE = re.compile(r"^(?P<id>\d+)_mask_(?P<mt>bbox|segm)\.png_ps_(?P<v>\d+)\.png$")
ORIG_RE = re.compile(r"^(?P<id>\d+)_orig\.png$")

SOURCE_DIRS = {"real": "orig", "SD": "sd2-sp", "PS": "ps-sp"}
GENERATOR_DETAIL = {
    "real": "authentic (MS-COCO)",
    "SD": "Stable Diffusion 2 inpainting (TGIF sd2-sp)",
    "PS": "Adobe Photoshop generative fill (TGIF ps-sp)",
}
INDEX_COLUMNS = [
    "image_id", "source_id", "category", "generator", "mask_type", "variant", "label",
    "width", "height", "file", "mask_file",
]


def mask_filename(source_id: int | str, mask_type: str) -> str:
    return f"{source_id}_mask_{mask_type}.png_ps_mask.png"


def build_index(data_root: Path) -> pd.DataFrame:
    """One row per testing image (authentic + inpainted), sorted by (source_id, generator, ...).

    `file` and `mask_file` are relative to `data_root`. Raises if an inpainted image has no mask
    or if image and mask sizes disagree, because every metric depends on that pairing.
    """
    rows = []
    for generator, folder in SOURCE_DIRS.items():
        for path in sorted((data_root / folder / "testing").rglob("*.png")):
            category = path.parent.name
            if generator == "real":
                m = ORIG_RE.match(path.name)
                if m is None:
                    continue  # *_orig_512.png, *_orig_1024.png: other TGIF subsets
                mask_type, variant, mask_file = "none", -1, None
                image_id = f"real_{m['id']}"
            else:
                m = (SD_RE if generator == "SD" else PS_RE).match(path.name)
                if m is None:
                    raise ValueError(f"unexpected file name in {folder}: {path.name}")
                mask_type, variant = m["mt"], int(m["v"])
                mask = data_root / "masks" / "testing" / category / mask_filename(m["id"], mask_type)
                if not mask.is_file():
                    raise FileNotFoundError(f"no ground-truth mask for {path.name}: {mask}")
                mask_file = mask.relative_to(data_root).as_posix()
                image_id = f"{generator.lower()}_{m['id']}_{mask_type}_{variant}"
            with Image.open(path) as im:
                width, height = im.size
            if mask_file is not None:
                with Image.open(data_root / mask_file) as mk:
                    if mk.size != (width, height):
                        raise ValueError(f"mask/image size mismatch for {path.name}")
            rows.append({
                "image_id": image_id, "source_id": int(m["id"]), "category": category,
                "generator": generator, "mask_type": mask_type, "variant": variant,
                "label": 0 if generator == "real" else 1, "width": width, "height": height,
                "file": path.relative_to(data_root).as_posix(), "mask_file": mask_file,
            })
    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    if df["image_id"].duplicated().any():
        raise ValueError("duplicate image_id in index")
    order = {"real": 0, "SD": 1, "PS": 2}
    df = df.assign(_g=df["generator"].map(order)).sort_values(
        ["source_id", "_g", "mask_type", "variant"]
    ).drop(columns="_g")
    return df.reset_index(drop=True)


def check_families(index: pd.DataFrame) -> None:
    """Every source image must come with 1 authentic + 6 SD + 6 PS images (2 mask types x 3)."""
    per_source = index.groupby(["source_id", "generator"]).size().unstack(fill_value=0)
    bad = per_source[(per_source["real"] != 1) | (per_source["SD"] != 6) | (per_source["PS"] != 6)]
    if len(bad):
        raise ValueError(f"{len(bad)} source images are not complete 1+6+6 families:\n{bad.head()}")
    cats = index.groupby("source_id")["category"].nunique()
    if (cats != 1).any():
        raise ValueError("a source image appears in several categories")


def family_order(index: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Processing order: source images shuffled with `seed`, each one complete (authentic, then
    SD, then PS) before the next. Any prefix of a run is then a set of complete families, i.e. a
    usable benchmark on its own."""
    sources = sorted(index["source_id"].unique())
    random.Random(seed).shuffle(sources)
    rank = {s: i for i, s in enumerate(sources)}
    return index.assign(_r=index["source_id"].map(rank)).sort_values(
        ["_r"], kind="stable"
    ).drop(columns="_r").reset_index(drop=True)
