"""Phase A: validate the TGIF resample CSVs against the extracted data and build the
manifest of unique image files. Reads data only; never modifies it.

Run from the project root:  .venv-data/bin/python -I code/build_manifest.py
"""
import collections
import csv
import hashlib
import json
import logging
import os
import re
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG = json.load(open(os.path.join(ROOT, "config", "config.json")))
TGIF = os.path.join(ROOT, CFG["tgif_root"])
RDIR = os.path.join(ROOT, CFG["resamples_dir"])
EXP = CFG["expected"]
Image.MAX_IMAGE_PIXELS = None

os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
os.makedirs(os.path.dirname(os.path.join(ROOT, CFG["manifest_csv"])), exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[logging.FileHandler(os.path.join(ROOT, "logs", "build_manifest.log"), mode="w"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger()
problems = []


def check(cond, msg):
    log.info(("  [OK]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        problems.append(msg)
    return cond


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 1. CSV integrity
log.info("== 1. Resample CSVs: sha256 and row counts vs resamples.json")
meta = json.load(open(os.path.join(RDIR, CFG["resamples_json"])))
rows = {}
for rid, info in meta["resamples"].items():
    path = os.path.join(RDIR, info["csv"])
    check(sha256(path) == info["csv_sha256"], f"resample_{rid}: sha256 matches resamples.json")
    with open(path, newline="") as f:
        rows[rid] = list(csv.DictReader(f))
    check(len(rows[rid]) == info["n_images"] == EXP["rows_per_resample"],
          f"resample_{rid}: {len(rows[rid])} rows (json says {info['n_images']})")

# ---------------------------------------------------------------- 2. Per-resample composition
log.info("== 2. Per-resample composition")
FAMILY = {("real", "none", "-1")} | {(g, m, v) for g in ("SD", "PS") for m in ("bbox", "segm") for v in "012"}
for rid, rs in rows.items():
    info = meta["resamples"][rid]
    check(all(r["resample_id"] == rid for r in rs), f"resample_{rid}: resample_id column == {rid}")
    check(len({r["file"] for r in rs}) == len(rs), f"resample_{rid}: no duplicate files")
    check(len({r["image_id"] for r in rs}) == len(rs), f"resample_{rid}: no duplicate image_ids")
    gen = collections.Counter(r["generator"] for r in rs)
    gm = collections.Counter((r["generator"], r["mask_type"]) for r in rs)
    lab = collections.Counter(r["label"] for r in rs)
    log.info(f"  resample_{rid}: generator={dict(gen)} gen×mask={dict(gm)} label={dict(lab)}")
    check(gen == {"real": EXP["real"], "SD": EXP["SD"], "PS": EXP["PS"]}, f"resample_{rid}: 275 real / 1650 SD / 1650 PS")
    check(all(gm[(g, m)] == EXP["per_generator_mask_type"] for g in ("SD", "PS") for m in ("bbox", "segm")),
          f"resample_{rid}: 825 per SD/PS × bbox/segm")
    check(lab == {"0": EXP["real"], "1": EXP["SD"] + EXP["PS"]}, f"resample_{rid}: labels 275×0 / 3300×1")
    check(all((r["label"] == "0") == (r["generator"] == "real") for r in rs), f"resample_{rid}: label 0 iff generator real")
    check(all((r["mask_file"] == "") == (r["generator"] == "real") for r in rs), f"resample_{rid}: mask_file empty iff real")
    fams = collections.defaultdict(set)
    for r in rs:
        fams[r["source_id"]].add((r["generator"], r["mask_type"], r["variant"]))
    check(len(fams) == info["n_source_images"], f"resample_{rid}: {len(fams)} source images (json {info['n_source_images']})")
    check(all(f == FAMILY for f in fams.values()), f"resample_{rid}: every family complete (1 real + SD/PS × bbox/segm × variants 0,1,2)")
    cats = collections.Counter(r["category"] for r in rs if r["generator"] == "real")  # one real per source
    check(dict(cats) == info["sources_per_category"], f"resample_{rid}: sources per category match json")

srcs = {rid: {r["source_id"] for r in rs} for rid, rs in rows.items()}
for pair, n in meta["pairwise_source_overlap"].items():
    a, b = pair.split("&")
    check(len(srcs[a] & srcs[b]) == n, f"source overlap {pair}: {len(srcs[a] & srcs[b])} (json {n})")
with open(os.path.join(RDIR, "resample_sources.csv"), newline="") as f:
    rsrc = list(csv.DictReader(f))
check(len(rsrc) == meta["n_source_images"], f"resample_sources.csv: {len(rsrc)} sources (json {meta['n_source_images']})")
check(all({rid for rid in "123" if s[f"in_resample_{rid}"] == "1"} == {rid for rid in "123" if s["source_id"] in srcs[rid]} for s in rsrc),
      "resample_sources.csv membership flags agree with resample CSVs")
none = sum(all(s[f"in_resample_{k}"] == "0" for k in "123") for s in rsrc)
check(none == meta["sources_in_no_resample"], f"sources in no resample: {none} (json {meta['sources_in_no_resample']})")

# ---------------------------------------------------------------- 3. Unique files
log.info("== 3. Unique files across the three resamples")
uniq = {}
FIELDS = ["image_id", "source_id", "category", "generator", "mask_type", "variant", "label", "width", "height", "file", "mask_file"]
consistent = True
for rid, rs in rows.items():
    for r in rs:
        rec = {k: r[k] for k in FIELDS}
        if r["file"] in uniq:
            consistent &= all(uniq[r["file"]][k] == rec[k] for k in FIELDS)
            uniq[r["file"]]["resamples"] += rid
        else:
            uniq[r["file"]] = dict(rec, resamples=rid)
check(len(uniq) == EXP["unique_files"], f"unique image files: {len(uniq)} (expected {EXP['unique_files']})")
check(consistent, "a file shared by several resamples has identical metadata in each")
log.info(f"  unique by generator: {dict(collections.Counter(u['generator'] for u in uniq.values()))}")
check(all(not os.path.isabs(p) and ".." not in p.split("/") for u in uniq.values() for p in (u["file"], u["mask_file"]) if p),
      "all CSV paths are relative and stay inside the TGIF root")

# Filename pattern: does the path agree with the metadata columns?
PAT = {"real": r"orig/testing/{c}/{s}_orig\.png",
       "SD": r"sd2-sp/testing/{c}/{s}_mask_{m}\.png_ps_mask\.png_sd2_{v}\.png",
       "PS": r"ps-sp/testing/{c}/{s}_mask_{m}\.png_ps_{v}\.png"}
MPAT = r"masks/testing/{c}/{s}_mask_{m}\.png_ps_mask\.png"
ok_name = ok_mname = True
for u in uniq.values():
    kw = dict(c=re.escape(u["category"]), s=u["source_id"], m=u["mask_type"], v=u["variant"])
    ok_name &= re.fullmatch(PAT[u["generator"]].format(**kw), u["file"]) is not None
    if u["generator"] != "real":
        ok_mname &= re.fullmatch(MPAT.format(**kw), u["mask_file"]) is not None
check(ok_name, "image path agrees with category/source_id/generator/mask_type/variant columns")
check(ok_mname, "mask path agrees with category/source_id/mask_type (same source & mask type as its image)")

# ---------------------------------------------------------------- 4. Files on disk
log.info("== 4. Existence, dimensions, mask values, image↔mask pairing (pixel check)")
by_source = collections.defaultdict(list)
for u in uniq.values():
    by_source[(u["category"], u["source_id"])].append(u)


def load(rel, mode=None):
    im = Image.open(os.path.join(TGIF, rel))
    im.load()
    return im if mode is None else im.convert(mode)


mask_cache = {}
per_file = []
for (cat, sid), items in tqdm(sorted(by_source.items()), desc="verify sources", unit="src"):
    orig_rel = f"orig/testing/{cat}/{sid}_orig.png"
    orig = np.asarray(load(orig_rel, "RGB")).astype(np.int16) if os.path.isfile(os.path.join(TGIF, orig_rel)) else None
    for u in items:
        rec = dict(file=u["file"], generator=u["generator"], exists=os.path.isfile(os.path.join(TGIF, u["file"])))
        if not rec["exists"]:
            per_file.append(rec); continue
        im = load(u["file"])
        rec.update(img_w=im.width, img_h=im.height, img_mode=im.mode,
                   dims_ok=(im.width, im.height) == (int(u["width"]), int(u["height"])))
        if u["generator"] != "real":
            mrel = u["mask_file"]
            rec["mask_exists"] = os.path.isfile(os.path.join(TGIF, mrel))
            if rec["mask_exists"]:
                if mrel not in mask_cache:
                    mim = load(mrel)
                    raw = np.asarray(mim)
                    mask_cache[mrel] = (mim.mode, mim.size, tuple(np.unique(raw).tolist()), raw if raw.ndim == 2 else raw[..., 0],
                                        raw.ndim == 2 or all((raw[..., k] == raw[..., 0]).all() for k in range(raw.shape[2])))
                mmode, msize, mvals, m2d, mchan_eq = mask_cache[mrel]
                rec.update(mask_mode=mmode, mask_w=msize[0], mask_h=msize[1], mask_values=" ".join(map(str, mvals)),
                           mask_channels_equal=mchan_eq,
                           mask_dims_ok=msize == (int(u["width"]), int(u["height"])))
                if orig is not None and rec["dims_ok"] and rec["mask_dims_ok"] and orig.shape[:2] == m2d.shape:
                    changed = (np.abs(np.asarray(im.convert("RGB")).astype(np.int16) - orig).max(axis=2) > 0)
                    hi = m2d == m2d.max()
                    rec["changed_frac"] = float(changed.mean())
                    rec["mask_hi_frac"] = float(hi.mean())
                    # share of changed pixels that lie on the high-valued mask region, and vice-versa
                    rec["changed_in_hi"] = float((changed & hi).sum() / max(changed.sum(), 1))
                    rec["changed_in_lo"] = float((changed & ~hi).sum() / max(changed.sum(), 1))
                    rec["hi_covered_by_change"] = float((changed & hi).sum() / max(hi.sum(), 1))
                    # compare with the sibling mask that lacks the "_ps_mask" suffix
                    sib = mrel.replace(".png_ps_mask.png", ".png")
                    if os.path.isfile(os.path.join(TGIF, sib)):
                        s2 = np.asarray(load(sib, "L"))
                        if s2.shape == hi.shape:
                            sh = s2 == s2.max()
                            rec["sibling_iou"] = float((sh & hi).sum() / max((sh | hi).sum(), 1))
                            rec["changed_in_sibling_hi"] = float((changed & sh).sum() / max(changed.sum(), 1))
        per_file.append(rec)

def col(k, gen=None):
    return [r[k] for r in per_file if k in r and (gen is None or r["generator"] == gen)]

missing_img = [r["file"] for r in per_file if not r["exists"]]
check(not missing_img, f"all {len(uniq)} referenced images exist (missing: {len(missing_img)}) {missing_img[:5]}")
mask_refs = {u["mask_file"] for u in uniq.values() if u["mask_file"]}
missing_m = sorted(m for m in mask_refs if not os.path.isfile(os.path.join(TGIF, m)))
check(not missing_m, f"all {len(mask_refs)} referenced masks exist (missing: {len(missing_m)}) {missing_m[:5]}")
bad_dims = [r["file"] for r in per_file if r.get("dims_ok") is False]
check(not bad_dims, f"image size == CSV width×height for all images (mismatch: {len(bad_dims)}) {bad_dims[:5]}")
bad_mdims = [r["file"] for r in per_file if r.get("mask_dims_ok") is False]
check(not bad_mdims, f"mask size == CSV width×height for all manipulated images (mismatch: {len(bad_mdims)}) {bad_mdims[:5]}")
log.info(f"  image modes: {dict(collections.Counter(col('img_mode')))}")
log.info(f"  mask modes: {dict(collections.Counter(v[0] for v in mask_cache.values()))}; "
         f"value sets: {dict(collections.Counter(v[2] for v in mask_cache.values()))}")
check(all(v[4] for v in mask_cache.values()), "multi-channel masks (if any) have identical channels")
check(all(set(v[2]) <= {0, 255} and len(v[2]) == 2 for v in mask_cache.values()), "every mask is binary {0,255} with both values present")
log.info(f"  mask positive (255) area fraction: min={min(col('mask_hi_frac')):.4f} median={np.median(col('mask_hi_frac')):.4f} max={max(col('mask_hi_frac')):.4f}")
for g in ("SD", "PS"):
    a, b, c = np.array(col("changed_in_hi", g)), np.array(col("hi_covered_by_change", g)), np.array(col("changed_frac", g))
    log.info(f"  {g} vs authentic: share of changed pixels inside mask==255: min={a.min():.4f} p5={np.percentile(a,5):.4f} median={np.median(a):.4f}; "
             f"share of mask==255 actually changed: p5={np.percentile(b,5):.4f} median={np.median(b):.4f}; changed area frac median={np.median(c):.4f} (n={len(a)})")
    sib, sibc = np.array(col("sibling_iou", g)), np.array(col("changed_in_sibling_hi", g))
    if len(sib):
        log.info(f"  {g}: sibling mask (no '_ps_mask') IoU with referenced GT: median={np.median(sib):.4f} min={sib.min():.4f}; "
                 f"changed pixels inside sibling: median={np.median(sibc):.4f} min={sibc.min():.4f}")

# ---------------------------------------------------------------- 5. Unreferenced files
log.info("== 5. Files on disk not referenced by any resample (count only)")
ref = set(uniq) | mask_refs
for d in ("orig", "sd2-sp", "ps-sp", "masks"):
    on_disk = {os.path.relpath(os.path.join(dp, f), TGIF) for dp, _, fs in os.walk(os.path.join(TGIF, d)) for f in fs}
    log.info(f"  {d:7s}: on disk={len(on_disk):5d} referenced={len(on_disk & ref):5d} unreferenced={len(on_disk - ref):5d} "
             f"referenced-but-absent={len({r for r in ref if r.startswith(d + '/')} - on_disk)}")

# ---------------------------------------------------------------- 6. Write outputs
man_path = os.path.join(ROOT, CFG["manifest_csv"])
with open(man_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS + ["resamples"])
    w.writeheader()
    for k in sorted(uniq):
        w.writerow(uniq[k])
keys = sorted({k for r in per_file for k in r}, key=lambda k: (k != "file", k))
with open(os.path.join(ROOT, CFG["validation_csv"]), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(per_file)
json.dump({"n_unique": len(uniq), "problems": problems}, open(os.path.join(ROOT, CFG["validation_summary"]), "w"), indent=1)
log.info(f"== Manifest: {man_path} ({len(uniq)} rows)")
log.info(f"== RESULT: {'ALL CHECKS PASSED' if not problems else f'{len(problems)} CHECK(S) FAILED'}")
for p in problems:
    log.info(f"   - {p}")
