"""Phase B smoke test: metric sanity checks, verification list, float16-vs-float32 effect, figures, timing.
Reads the outputs of run_trufor.py (results.csv, maps/, maps_f32/) and the official test.py outputs.
GT rule: manipulated = mask_L >= 128. Prediction rule: map >= 0.5.
Usage: python smoke_eval.py <project_root>"""
import csv, json, os, re, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

root = os.path.abspath(sys.argv[1])
cfg = json.load(open(os.path.join(root, "config/config.json")))
tgif = os.path.join(root, cfg["tgif_root"])
S = os.path.join(root, "outputs/phaseB_smoke")
GT_T, PR_T = cfg["gt_mask"]["main_threshold"], cfg["prediction_threshold"]
sel = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(S, "selection.csv"), newline=""))}
res = list(csv.DictReader(open(os.path.join(S, "results.csv"), newline="")))
report = {}


def iou(p, g):
    u = (p | g).sum()
    return 1.0 if u == 0 else (p & g).sum() / u


def dice(p, g):
    d = p.sum() + g.sum()
    return 1.0 if d == 0 else 2 * (p & g).sum() / d


def load_mask(rel):
    return np.asarray(Image.open(os.path.join(tgif, rel)).convert("L"))


def load_rgb(rel):
    return np.asarray(Image.open(os.path.join(tgif, rel)).convert("RGB"))


# ------------------------------------------------------------------ 1. metric sanity checks
san = []
for r in res:
    s = sel[r["image_id"]]
    if not s["mask_file"]:
        continue
    g = load_mask(s["mask_file"]) >= GT_T
    z = np.zeros_like(g)
    san.append(dict(image_id=r["image_id"], gt_vs_gt_iou=iou(g, g), gt_vs_gt_dice=dice(g, g),
                    empty_pred_iou=iou(z, g), empty_pred_dice=dice(z, g),
                    inverted_gt_iou=iou(~g, g), full_pred_iou=iou(~z, g), full_pred_iou_expected=g.mean()))
report["sanity"] = san
assert all(x["gt_vs_gt_iou"] == 1 and x["gt_vs_gt_dice"] == 1 for x in san)
assert all(x["empty_pred_iou"] == 0 and x["empty_pred_dice"] == 0 and x["inverted_gt_iou"] == 0 for x in san)
assert all(abs(x["full_pred_iou"] - x["full_pred_iou_expected"]) < 1e-12 for x in san)

# ------------------------------------------------------------------ 2. per-image table, localization, float16 effect
rows, f16 = [], []
for r in res:
    s = sel[r["image_id"]]
    m16 = np.load(os.path.join(root, r["npz"]))
    m32 = np.load(os.path.join(S, "maps_f32", r["file"] + ".npz"))
    a32, a16 = m32["map"], m16["map"].astype(np.float32)
    c32, c16 = m32["conf"], m16["conf"].astype(np.float32)
    row = dict(image_id=r["image_id"], file=r["file"], label=r["label"], generator=r["generator"],
               mask_type=r["mask_type"], score=float(r["score"]), orig=f'{r["orig_w"]}x{r["orig_h"]}',
               processed=f'{r["proc_w"]}x{r["proc_h"]}', map=f'{r["map_w"]}x{r["map_h"]}',
               t_infer_s=float(r["t_infer_s"]), t_total_s=float(r["t_load_s"]) + float(r["t_infer_s"]) + float(r["t_save_s"]))
    e = dict(image_id=r["image_id"], n_pixels=a32.size, map_max_abs_err=float(np.abs(a32 - a16).max()),
             conf_max_abs_err=float(np.abs(c32 - c16).max()))
    for op, f in ((">=", np.greater_equal), (">", np.greater)):
        d = f(a32, PR_T) != f(a16, PR_T)
        e[f"bin{op}0.5_px_differ"] = int(d.sum())
        e[f"bin{op}0.5_frac_differ"] = float(d.mean())
    e["conf_bin>=0.5_px_differ"] = int(((c32 >= PR_T) != (c16 >= PR_T)).sum())
    if s["mask_file"]:
        a = load_mask(s["mask_file"])
        g = a >= GT_T
        p32, p16 = a32 >= PR_T, a16 >= PR_T
        row.update(gt_frac=float(g.mean()), iou=float(iou(p16, g)), dice=float(dice(p16, g)),
                   iou_gt_gt0=float(iou(p16, a > 0)), iou_gt_eq255=float(iou(p16, a == 255)))
        e.update(iou_f32=float(iou(p32, g)), iou_f16=float(iou(p16, g)), dice_f32=float(dice(p32, g)),
                 dice_f16=float(dice(p16, g)))
        e["iou_abs_diff"] = abs(e["iou_f32"] - e["iou_f16"])
        e["dice_abs_diff"] = abs(e["dice_f32"] - e["dice_f16"])
    rows.append(row)
    f16.append(e)
report["per_image"], report["float16"] = rows, f16

# ------------------------------------------------------------------ 3. verification list
ver = {}
# 3a. score direction: official sample images with known labels, and the code path
samp = {}
for n in ("pristine1.jpg", "pristine2.jpg", "tampered1.png", "tampered2.png"):
    samp[n] = float(np.load(os.path.join(S, "official/samples", n + ".npz"))["score"])
ver["score_direction"] = dict(
    official_samples=samp,
    min_tampered_gt_max_pristine=min(samp["tampered1.png"], samp["tampered2.png"]) > max(samp["pristine1.jpg"], samp["pristine2.jpg"]),
    tgif_mean_score_real=float(np.mean([x["score"] for x in rows if x["label"] == "0"])),
    tgif_mean_score_fake=float(np.mean([x["score"] for x in rows if x["label"] == "1"])),
    code="score = sigmoid(det_logit), as in test.py; map = softmax(pred)[1] = P(manipulated)")
# 3b. our runner reproduces the official test.py output (same image, run separately)
off = np.load(os.path.join(S, "official/tgif/24919_mask_bbox.png_ps_mask.png_sd2_2.png.npz"))
r0 = next(r for r in res if r["image_id"] == "sd_24919_bbox_2")
o32 = np.load(os.path.join(S, "maps_f32", r0["file"] + ".npz"))
ver["matches_official_test_py"] = dict(image_id=r0["image_id"], map_max_abs_diff=float(np.abs(off["map"] - o32["map"]).max()),
                                       conf_max_abs_diff=float(np.abs(off["conf"] - o32["conf"]).max()),
                                       score_official=float(off["score"]), score_ours=float(r0["score"]),
                                       official_keys=sorted(off.files))
# 3c-3f per manipulated image
per = []
for r in res:
    s = sel[r["image_id"]]
    H, W = int(r["orig_h"]), int(r["orig_w"])
    v = dict(image_id=r["image_id"], size_ok=(int(r["proc_h"]), int(r["proc_w"]), int(r["map_h"]), int(r["map_w"])) == (H, W, H, W))
    if s["mask_file"]:
        a = load_mask(s["mask_file"])
        g = a >= GT_T
        m = np.load(os.path.join(root, r["npz"]))["map"].astype(np.float32)
        v["mask_shape_eq_image"] = a.shape == (H, W)
        # orientation: identity should beat flips/rotation for agreement of the map with the GT
        cand = {"identity": m, "flipud": m[::-1], "fliplr": m[:, ::-1], "rot180": m[::-1, ::-1]}
        if H == W:
            cand["transpose"] = m.T
        v["map_mean_in_gt"], v["map_mean_out_gt"] = float(m[g].mean()), float(m[~g].mean())
        v["iou_by_orientation"] = {k: float(iou(x >= PR_T, g)) for k, x in cand.items()}
        v["mean_in_gt_by_orientation"] = {k: float(x[g].mean()) for k, x in cand.items()}
        # GT alignment and polarity, independent of the model: pixels changed vs the authentic image
        o = load_rgb(f"orig/testing/{s['category']}/{s['source_id']}_orig.png").astype(np.int16)
        im = load_rgb(s["file"]).astype(np.int16)
        ch = np.abs(im - o).max(2) > 0
        v["orig_shape_eq_image"] = o.shape[:2] == (H, W)
        v["gt_frac"] = float(g.mean())
        v["p_changed_given_gt1"], v["p_changed_given_gt0"] = float(ch[g].mean()), float(ch[~g].mean())
        v["frac_changed_inside_gt"] = float((ch & g).sum() / max(ch.sum(), 1))
        v["iou_changed_vs_gt"] = float(iou(ch, g))
        v["iou_changed_vs_inverted_gt"] = float(iou(ch, ~g))
        # pairing: same source id, mask type and category in file and mask names
        mf = re.match(r"masks/testing/(.+)/(\d+)_mask_(bbox|segm)\.png_ps_mask\.png$", s["mask_file"])
        ff = re.match(r"(sd2-sp|ps-sp)/testing/(.+)/(\d+)_mask_(bbox|segm)\.png_", s["file"])
        v["pairing_ok"] = bool(mf and ff and mf.group(1) == ff.group(2) == s["category"] and
                               mf.group(2) == ff.group(3) == s["source_id"] and mf.group(3) == ff.group(4) == s["mask_type"]
                               and {"SD": "sd2-sp", "PS": "ps-sp"}[s["generator"]] == ff.group(1))
    per.append(v)
ver["per_image"] = per
src = open(os.path.join(root, "code/phaseB/run_trufor.py")).read()
ver["no_gt_leak"] = dict(
    runner_drops_mask_columns='MASK_COLS = {"mask_file"}' in src,
    runner_mentions_masks_only_in_guards=[l.strip() for l in src.splitlines() if "mask" in l.lower() and "mask_type" not in l],
    model_inputs_contain_masks_dir=any("masks/" in r["file"] for r in res),
    official_test_py_identical="see matches_official_test_py: test.py never receives a mask path")
report["verification"] = ver

# ------------------------------------------------------------------ 4. timing and estimate
man = list(csv.DictReader(open(os.path.join(root, cfg["manifest_csv"]), newline="")))
mpx_total = sum(int(x["width"]) * int(x["height"]) for x in man) / 1e6
warm = [x for x in rows[1:]]  # first image includes one-off warm-up
t_img = float(np.mean([x["t_total_s"] for x in warm]))
mpx = [int(r["orig_w"]) * int(r["orig_h"]) / 1e6 for r in res[1:]]
t_mpx = float(sum(x["t_total_s"] for x in warm) / sum(mpx))
report["timing"] = dict(
    per_image_total_s=[x["t_total_s"] for x in rows], per_image_infer_s=[x["t_infer_s"] for x in rows],
    mean_s_per_image_excl_first=t_img, s_per_mpx_excl_first=t_mpx, n_images=len(man), mpx_total=mpx_total,
    est_hours_by_image=t_img * len(man) / 3600, est_hours_by_mpx=t_mpx * mpx_total / 3600)
json.dump(report, open(os.path.join(S, "smoke_report.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ------------------------------------------------------------------ 5. figures
BLUE = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ORANGE = LinearSegmentedColormap.from_list("orange", ["#ffffff", "#fbe0d3", "#f3a17e", "#eb6834", "#b84a1f", "#7a2e10"])
OUTC = ListedColormap(["#ffffff", "#2a78d6", "#eb6834", "#1baf7a"])  # TN, TP, FP, FN
os.makedirs(os.path.join(S, "figures"), exist_ok=True)
fig, ax = plt.subplots(len(res), 5, figsize=(17, 3.3 * len(res)))
for i, r in enumerate(res):
    s = sel[r["image_id"]]
    im = load_rgb(s["file"])
    m = np.load(os.path.join(root, r["npz"]))
    a, c = m["map"].astype(np.float32), m["conf"].astype(np.float32)
    g = load_mask(s["mask_file"]) >= GT_T if s["mask_file"] else np.zeros(a.shape, bool)
    p = a >= PR_T
    out = np.zeros(a.shape, np.uint8)
    out[p & g], out[p & ~g], out[~p & g] = 1, 2, 3
    ax[i, 0].imshow(im); ax[i, 0].set_title(f"{r['image_id']}\nscore = {float(r['score']):.4f}", fontsize=9)
    ax[i, 1].imshow(g, cmap="gray", vmin=0, vmax=1); ax[i, 1].set_title(f"GT (mask ≥ {GT_T})" if s["mask_file"] else "GT: none (real)", fontsize=9)
    h = ax[i, 2].imshow(a, cmap=BLUE, vmin=0, vmax=1); ax[i, 2].set_title("anomaly map (float16)", fontsize=9)
    fig.colorbar(h, ax=ax[i, 2], fraction=0.04)
    h = ax[i, 3].imshow(c, cmap=ORANGE, vmin=0, vmax=1); ax[i, 3].set_title("confidence map (float16)", fontsize=9)
    fig.colorbar(h, ax=ax[i, 3], fraction=0.04)
    ax[i, 4].imshow(out, cmap=OUTC, vmin=0, vmax=3, interpolation="nearest")
    ttl = f"map ≥ {PR_T} vs GT"
    if s["mask_file"]:
        ttl += f"\nIoU {iou(p, g):.3f}  Dice {dice(p, g):.3f}"
    else:
        ttl += f"\nflagged px {p.mean():.1%}"
    ax[i, 4].set_title(ttl, fontsize=9)
    for x in ax[i]:
        x.set_xticks([]); x.set_yticks([])
fig.legend(handles=[Patch(color="#2a78d6", label="TP"), Patch(color="#eb6834", label="FP"),
                    Patch(color="#1baf7a", label="FN"), Patch(facecolor="#ffffff", edgecolor="#999", label="TN")],
           loc="upper right", ncol=4, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(os.path.join(S, "figures/smoke_grid.png"), dpi=80)
print(json.dumps({k: report[k] for k in ("per_image", "float16", "timing")}, indent=1, default=str))
print(json.dumps(report["verification"], indent=1, default=str))
print(json.dumps(report["sanity"], indent=1, default=str))
