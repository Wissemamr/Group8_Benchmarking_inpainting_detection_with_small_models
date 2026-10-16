"""
PSCC-Net inference (pretrained, no training) -> two appendable result tables.

    detection.csv     one row per image (forged + authentic)
    localization.csv  one row per image (forged + authentic)  [no gating]

No metrics are aggregated here; the tables hold everything needed to compute them later.
For the gated protocol, join the two tables on "filename" and keep pred == 1; forged images
the detector missed (pred = 0) then count as fully missed (IoU = 0, FN = gt_pixels).

Run from the repository root. The defaults are the TGIF testing folders under tgif/ and the official
weights under checkpoint/, so a full run at localization threshold 0.5 is just:

    python infer.py --out results/thr05 --loc-threshold 0.5

Quick test on a few named images (searched in the --forged / --authentic folders):

    python infer.py --out tmp_check --files 509735_mask_segm.png_ps_1.png 409211_orig.png

Running it again APPENDS new images to the same tables; images already in detection.csv are
skipped. Rows are saved one by one, so you can stop with Ctrl+C and continue later.
"""
import argparse
import csv
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.config import get_pscc_args

CKPTS = {  # relative to --weights
    "HRNet": "HRNet_checkpoint/HRNet.pth",
    "NLCDetection": "NLCDetection_checkpoint/NLCDetection.pth",
    "DetectionHead": "DetectionHead_checkpoint/DetectionHead.pth",
}
DEFAULT_FORGED = ["tgif/ps-sp_testing/testing", "tgif/sd2-sp_testing/testing"]
DEFAULT_AUTHENTIC = ["tgif/orig_testing/testing"]
DEFAULT_MASKS = "tgif/masks_testing/testing"

META = ["filename", "image_id", "category", "generator", "mask_type", "variant"]
DET_COLUMNS = META + ["orig_size", "label", "score", "pred", "gt_pixels", "height", "width", "run"]
LOC_COLUMNS = META + ["label", "TP", "FP", "FN", "TN",
                      "iou", "dice", "pix_precision", "pix_recall", "false_alarm", "run"]

# Forged, e.g. 1761_mask_bbox.png_ps_0.png   or   1761_mask_segm.png_ps_mask.png_sd2_2.png
FORGED_RE = re.compile(r"^(\d+)_mask_([a-z]+)(?:_\d+)?\.png.*?_(ps|sd2|sdxl)_(\d+)\.png$", re.I)
# Authentic, e.g. 1761_orig.png   or   1761_orig_512.png
AUTH_RE = re.compile(r"^(\d+)_orig(?:_(\d+))?\.png$", re.I)
# Prefix used to find the ground-truth mask
MASK_PREFIX = re.compile(r"^(\d+_mask_(?:bbox|segm|random)(?:_\d+)?\.png)")
GEN_NAMES = {"ps": "PS", "sd2": "SD2", "sdxl": "SDXL"}


# ----------------------------------------------------------------------------- model
def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(net, path, device):
    # Checkpoints were saved from nn.DataParallel -> strip the "module." prefix
    sd = torch.load(path, map_location="cpu")
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd)
    return net.to(device).eval()


# ----------------------------------------------------------------------------- data
def list_files(folder, limit, seed=0):
    """All files in folder and its subfolders (e.g. testing/airplane, testing/bus, ...)."""
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {os.path.abspath(folder)}")
    paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs)
    if limit and limit < len(paths):
        random.seed(seed)
        paths = sorted(random.sample(paths, limit))
    return paths


def index_masks(folder):
    idx = {}
    for r, _, fs in os.walk(folder):
        for f in fs:
            idx.setdefault(f, os.path.join(r, f))
    return idx


def find_gt_mask(filename, mask_index):
    m = MASK_PREFIX.match(filename)
    if not m:
        return None
    base = m.group(1)
    for cand in (base + "_ps_mask.png", base):  # *-sp subsets use the ps_mask (extra border)
        if cand in mask_index:
            return mask_index[cand]
    return None


def parse_name(filename, label):
    """-> image_id, generator, mask_type, variant, orig_size ('' if unknown)."""
    if label == 1:
        m = FORGED_RE.match(filename)
        if m:
            return m.group(1), GEN_NAMES[m.group(3).lower()], m.group(2).lower(), m.group(4), ""
        return "", "", "", "", ""
    m = AUTH_RE.match(filename)
    if m:
        return m.group(1), "none", "none", "", m.group(2) or ""
    return "", "none", "none", "", ""


def open_table(path, columns):
    """Open a CSV for appending (header written if new). Returns (file, writer, existing filenames)."""
    done = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != columns:
                raise SystemExit(f"{path} has different columns than this script expects.\n"
                                 f"Rename/move the old file or choose another --out folder.")
            done = {row["filename"] for row in reader}
        f = open(path, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=columns)
    else:
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
    return f, w, done


def drop_orphan_rows(path, columns, done):
    """If a previous run was interrupted between writing the localization row and the detection row,
    that image will be processed again -> remove its unfinished localization row to avoid duplicates."""
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            return  # open_table will report the problem
        rows = list(reader)
    keep = [r for r in rows if r["filename"] in done]
    if len(keep) != len(rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(keep)
        print(f"Removed {len(rows) - len(keep)} unfinished row(s) from {path}")


def save_visual(img, prob, gt, out_path, mask_thr):
    """input | ground truth | predicted mask | overlay (green = correct, red = false alarm, blue = missed)"""
    rgb = np.asarray(img, dtype=np.float32)
    pred = prob > mask_thr
    over = rgb.copy()
    for m, col in ((pred & gt, (0, 255, 0)), (pred & ~gt, (255, 0, 0)), (~pred & gt, (0, 90, 255))):
        over[m] = 0.45 * rgb[m] + 0.55 * np.array(col, dtype=np.float32)
    gap = np.full((rgb.shape[0], 8, 3), 255.0)
    gt_p = np.repeat((gt * 255.0)[..., None], 3, axis=2)
    pr_p = np.repeat((prob.clip(0, 1) * 255.0)[..., None], 3, axis=2)
    panel = np.concatenate([rgb, gap, gt_p, gap, pr_p, gap, over], axis=1).astype(np.uint8)
    Image.fromarray(panel).save(out_path)


def r6(v):
    return round(v, 6)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forged", nargs="*", default=DEFAULT_FORGED, help="folder(s) of manipulated images")
    p.add_argument("--authentic", nargs="*", default=DEFAULT_AUTHENTIC, help="folder(s) of authentic images")
    p.add_argument("--masks", default=DEFAULT_MASKS, help="folder with ground-truth masks")
    p.add_argument("--weights", default="checkpoint", help="folder with the official PSCC-Net checkpoints")
    p.add_argument("--out", default="results/thr05", help="folder for detection.csv and localization.csv")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--limit", type=int, default=0, help="random N files per folder (0 = all)")
    p.add_argument("--files", nargs="*", default=[], help="only these file names (quick test)")
    p.add_argument("--threshold", type=float, default=0.5, help="image score threshold for 'pred'")
    p.add_argument("--loc-threshold", "--mask-threshold", dest="mask_threshold", type=float, default=0.5,
                   help="pixel threshold for the mask")
    p.add_argument("--save-vis", action="store_true", help="save comparison images for localized images")
    a = p.parse_args()
    if not a.forged and not a.authentic:
        raise SystemExit("Give at least one --forged or --authentic folder.")
    for name, path in CKPTS.items():
        if not os.path.isfile(os.path.join(a.weights, path)):
            raise SystemExit(f"Missing {name} weights: {os.path.abspath(os.path.join(a.weights, path))}")

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(a.out, exist_ok=True)
    det_f, det_w, done = open_table(os.path.join(a.out, "detection.csv"), DET_COLUMNS)
    loc_path = os.path.join(a.out, "localization.csv")
    drop_orphan_rows(loc_path, LOC_COLUMNS, done)
    loc_f, loc_w, _ = open_table(loc_path, LOC_COLUMNS)

    jobs = [(path, 1) for f in a.forged for path in list_files(f, a.limit)]
    jobs += [(path, 0) for f in a.authentic for path in list_files(f, a.limit)]
    if a.files:
        wanted = set(a.files)
        jobs = [j for j in jobs if os.path.basename(j[0]) in wanted]
        missing = wanted - {os.path.basename(j[0]) for j in jobs}
        if missing:
            raise SystemExit(f"Not found in the --forged / --authentic folders: {sorted(missing)}")
    todo = [j for j in jobs if os.path.basename(j[0]) not in done]
    print(f"{len(jobs)} files found, {len(jobs) - len(todo)} already in results -> {len(todo)} to process")
    if not todo:
        det_f.close()
        loc_f.close()
        return

    device = pick_device(a.device)
    print(f"Using device: {device} | saving to {os.path.abspath(a.out)}")
    args = get_pscc_args()
    ckpt = {k: os.path.join(a.weights, v) for k, v in CKPTS.items()}
    FENet = load(get_seg_model(get_hrnet_cfg()), ckpt["HRNet"], device)
    SegNet = load(NLCDetection(args), ckpt["NLCDetection"], device)
    ClsNet = load(DetectionHead(args), ckpt["DetectionHead"], device)
    mask_index = index_masks(a.masks)
    vis_dir = os.path.join(a.out, "visuals")

    added, localized, problems = 0, 0, []
    try:
        for i, (path, label) in enumerate(todo, 1):
            name = os.path.basename(path)
            try:
                img = Image.open(path).convert("RGB")  # handles RGBA / grayscale / palette
            except Exception:
                problems.append(f"not an image, skipped: {path}")
                continue

            image_id, generator, mask_type, variant, orig_size = parse_name(name, label)
            if not image_id:
                problems.append(f"unrecognised filename (metadata left empty): {path}")
            meta = {"filename": name, "image_id": image_id, "category": os.path.basename(os.path.dirname(path)),
                    "generator": generator, "mask_type": mask_type, "variant": variant}

            # ground truth: forged -> TGIF mask, authentic -> nothing edited
            if label == 1:
                gt_path = find_gt_mask(name, mask_index)
                if gt_path:
                    g = Image.open(gt_path).convert("L")
                    if g.size != img.size:
                        g = g.resize(img.size, Image.NEAREST)
                    gt = np.asarray(g) > 127
                else:
                    gt = None
                    problems.append(f"no ground-truth mask (gt_pixels / localization left empty): {path}")
            else:
                gt = np.zeros((img.height, img.width), dtype=bool)

            # ---- inference (no gate: both heads run on every image)
            x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = FENet(x)
                score = torch.softmax(ClsNet(feat), dim=1)[0, 1].item()
                pred = int(score > a.threshold)
                mask = SegNet(feat)[0]
                mask = F.interpolate(mask, size=x.shape[2:], mode="bilinear", align_corners=True)
                prob = mask[0, 0].cpu().numpy()

            # ---- localization row (written first; detection.csv marks the image as done)
            lrow = None
            if prob is not None:
                lrow = {**meta, "label": label, "TP": "", "FP": "", "FN": "", "TN": "", "iou": "", "dice": "",
                        "pix_precision": "", "pix_recall": "", "false_alarm": "", "run": run_id}
                if gt is not None:
                    pb = prob > a.mask_threshold
                    tp, fp = int((pb & gt).sum()), int((pb & ~gt).sum())
                    fn, tn = int((~pb & gt).sum()), int((~pb & ~gt).sum())
                    lrow.update(TP=tp, FP=fp, FN=fn, TN=tn)
                    if label == 1:
                        lrow.update(iou=r6(tp / (tp + fp + fn)) if tp + fp + fn else "",
                                    dice=r6(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else "",
                                    pix_precision=r6(tp / (tp + fp)) if tp + fp else 0.0,  # nothing marked -> 0
                                    pix_recall=r6(tp / (tp + fn)) if tp + fn else "")
                    else:
                        lrow["false_alarm"] = r6(fp / (fp + tn)) if fp + tn else ""
                    if a.save_vis:
                        sub = os.path.join(vis_dir, generator or "unknown")
                        os.makedirs(sub, exist_ok=True)
                        stem = name[:-4] if name.lower().endswith(".png") else name
                        save_visual(img, prob, gt, os.path.join(sub, stem + "_vis.png"), a.mask_threshold)
                loc_w.writerow(lrow)
                loc_f.flush()
                localized += 1

            # ---- detection row
            det_w.writerow({**meta, "orig_size": orig_size, "label": label, "score": r6(score), "pred": pred,
                            "gt_pixels": int(gt.sum()) if gt is not None else "",
                            "height": img.height, "width": img.width, "run": run_id})
            det_f.flush()
            added += 1

            info = ""
            if lrow is not None:
                if lrow["iou"] != "":
                    info = f" IoU={lrow['iou']:.3f}"
                elif lrow["false_alarm"] != "":
                    info = f" false_alarm={lrow['false_alarm']:.3f}"
            print(f"[{i}/{len(todo)}] {name[:55]:55s} label={label} score={score:.3f} pred={pred}{info}")
    except KeyboardInterrupt:
        print("\nStopped by user - everything processed so far is saved.")
    finally:
        det_f.close()
        loc_f.close()

    print(f"\nAdded {added} rows to detection.csv and {localized} rows to localization.csv in {a.out}")
    if problems:
        print(f"{len(problems)} warning(s):")
        for msg in problems[:10]:
            print("  " + msg)
        if len(problems) > 10:
            print(f"  ... and {len(problems) - 10} more")


if __name__ == "__main__":
    main()