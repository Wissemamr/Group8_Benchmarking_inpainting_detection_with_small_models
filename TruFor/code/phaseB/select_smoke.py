"""Phase B: pick the 6 smoke-test images from resample_1 (2 real, 2 SD, 2 PS), deterministically from config seed.
For SD and PS, one bbox and one segm image are taken so both mask types are exercised.
Usage: python -I select_smoke.py <project_root>"""
import csv, json, os, random, sys

root = os.path.abspath(sys.argv[1])
cfg = json.load(open(os.path.join(root, "config/config.json")))
rows = list(csv.DictReader(open(os.path.join(root, cfg["resamples_dir"], cfg["resample_csvs"][0]), newline="")))
rng = random.Random(cfg["seed"])
pick = rng.sample([r for r in rows if r["generator"] == "real"], 2)
for g in ("SD", "PS"):
    for m in ("bbox", "segm"):
        pick.append(rng.choice([r for r in rows if r["generator"] == g and r["mask_type"] == m]))
out = os.path.join(root, "outputs/phaseB_smoke")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "selection.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(pick)
for r in pick:
    print(r["image_id"], r["generator"], r["mask_type"], r["width"], r["height"], r["file"], r["mask_file"], sep=" | ")
