"""Characterise non-binary GT masks: intermediate-value share, location (near the 0/255 boundary?),
and agreement of thresholds (>0, >=128, ==255) with pixels that actually differ from the authentic image."""
import csv, os, sys, collections
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

root, tgif = sys.argv[1], os.path.join(sys.argv[1], "data/tgif")
man = list(csv.DictReader(open(os.path.join(root, "outputs/manifest/manifest_unique.csv"))))
masks = sorted({r["mask_file"] for r in man if r["mask_file"]})
stats = []
for m in tqdm(masks, desc="masks"):
    a = np.asarray(Image.open(os.path.join(tgif, m)).convert("L"))
    mid = (a > 0) & (a < 255)
    # pixels within 3 px of the 255 region (dilation via MaxFilter)
    near = np.asarray(Image.fromarray(((a == 255) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))) > 0
    stats.append(dict(mask=m, binary=not mid.any(), mid_frac_img=mid.mean(),
                      mid_over_pos=mid.sum() / max((a > 0).sum(), 1), mid_near_255=(mid & near).sum() / max(mid.sum(), 1),
                      n_vals=len(np.unique(a))))
b = sum(s["binary"] for s in stats)
print(f"masks strictly binary {{0,255}}: {b}/{len(stats)}")
for k in ("mid_frac_img", "mid_over_pos", "mid_near_255", "n_vals"):
    v = np.array([s[k] for s in stats if not s["binary"]])
    if len(v): print(f"  non-binary masks, {k}: min={v.min():.4f} median={np.median(v):.4f} p95={np.percentile(v,95):.4f} max={v.max():.4f}")
print("by mask type:", {t: f"{sum(s['binary'] for s in stats if f'_mask_{t}' in s['mask'])}/{sum(f'_mask_{t}' in s['mask'] for s in stats)} binary" for t in ("bbox", "segm")})

# agreement with actual change, for a few thresholds
rows = [r for r in man if r["generator"] != "real"]
res = collections.defaultdict(lambda: collections.defaultdict(list))
cache = {}
for r in tqdm(rows, desc="pairs"):
    o = f"orig/testing/{r['category']}/{r['source_id']}_orig.png"
    if o not in cache:
        cache.clear(); cache[o] = np.asarray(Image.open(os.path.join(tgif, o)).convert("RGB")).astype(np.int16)
    im = np.asarray(Image.open(os.path.join(tgif, r["file"])).convert("RGB")).astype(np.int16)
    ch = np.abs(im - cache[o]).max(2) > 0
    a = np.asarray(Image.open(os.path.join(tgif, r["mask_file"])).convert("L"))
    for name, gt in (("gt>0", a > 0), ("gt>=128", a >= 128), ("gt==255", a == 255)):
        tp = (ch & gt).sum()
        res[r["generator"]][name].append((tp / max(ch.sum(), 1), tp / max(gt.sum(), 1), tp / max((ch | gt).sum(), 1)))
for g in res:
    for name, v in res[g].items():
        v = np.array(v)
        print(f"{g} {name:8s}: changed∩GT/changed median={np.median(v[:,0]):.4f} p5={np.percentile(v[:,0],5):.4f} | "
              f"changed∩GT/GT median={np.median(v[:,1]):.4f} p5={np.percentile(v[:,1],5):.4f} | IoU(change,GT) median={np.median(v[:,2]):.4f}")
