"""Localization example figures from the SAVED TruFor maps (no inference). Read-only on all inputs.
For each selected image: verifies alignment and that map > 0.5 vs GT >= 128 reproduces the per-image CSV metrics,
then draws input | GT | probability map | binary prediction | prediction overlay | TP/FP/FN error map.
Usage: .venv-trufor/bin/python code/phaseD/localization_examples.py <project_root> <image_id> [<image_id> ...]"""
import csv, json, os, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

PR_T, GT_T = 0.5, 128  # pixel-level thresholds used in the tables (not the image-level detection threshold)
BLUE = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
TP_C, FP_C, FN_C = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3 (dataviz validate_palette.js)
PRED_C = "#eb6834"


def rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) / 255


def ratio(n, d, e):
    return n / d if d > 0 else (1.0 if e else 0.0)


def main():
    root = os.path.abspath(sys.argv[1])
    ids = sys.argv[2:]
    cfg = json.load(open(os.path.join(root, "config/config.json")))
    tgif = os.path.join(root, cfg["tgif_root"])
    out_dir = os.path.join(root, "results/localization_examples")
    os.makedirs(out_dir, exist_ok=True)
    per = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(root, "outputs/tables/trufor_per_image.csv")))}
    npz = {r["image_id"]: r["npz"] for r in csv.DictReader(open(os.path.join(root, "outputs/inference/results.csv")))}
    report = {}
    for iid in ids:
        r = per[iid]
        W, H = int(r["width"]), int(r["height"])
        im_pil = Image.open(os.path.join(tgif, r["file"]))
        mk_pil = Image.open(os.path.join(tgif, r["mask_file"]))
        exif_orient = im_pil.getexif().get(274)  # 274 = Orientation; inference read pixels as stored
        im = np.asarray(im_pil.convert("RGB"))
        mk_rgb = np.asarray(mk_pil.convert("RGB"))
        mk = np.asarray(mk_pil.convert("L"))
        z = np.load(os.path.join(root, npz[iid]))
        m = z["map"].astype(np.float32)
        g = mk >= GT_T
        p = m > PR_T
        # ---------------------------------------------------------------- verification
        v = dict(image_size_WH=im_pil.size, mask_size_WH=mk_pil.size, map_shape_HW=m.shape,
                 npz_imgsize=z["imgsize"].tolist(), csv_size_WH=(W, H), exif_orientation=exif_orient,
                 mask_mode=mk_pil.mode, mask_channels_identical=bool((mk_rgb[..., 0] == mk_rgb[..., 1]).all()
                                                                    and (mk_rgb[..., 1] == mk_rgb[..., 2]).all()),
                 map_dtype=str(z["map"].dtype), map_min=float(m.min()), map_max=float(m.max()))
        assert im.shape[:2] == mk.shape == m.shape == (H, W), v
        assert tuple(z["imgsize"]) == (H, W) and exif_orient in (None, 1), v
        tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
        tn = p.size - tp - fp - fn
        e = (tp + fp == 0) and (tp + fn == 0)
        mine = dict(iou=ratio(tp, tp + fp + fn, e), pixel_f1=ratio(2 * tp, 2 * tp + fp + fn, e),
                    pixel_precision=ratio(tp, tp + fp, e), pixel_recall=ratio(tp, tp + fn, e))
        csv_m = dict(iou=float(r["loc_iou"]), pixel_f1=float(r["loc_pixel_f1"]),
                     pixel_precision=float(r["loc_pixel_precision"]), pixel_recall=float(r["loc_pixel_recall"]))
        counts_ok = (tp, fp, fn, tn) == tuple(int(r[f"loc_{k}"]) for k in ("tp", "fp", "fn", "tn"))
        max_err = max(abs(mine[k] - csv_m[k]) for k in mine)
        v.update(counts_recomputed=dict(tp=tp, fp=fp, fn=fn, tn=tn), counts_match_csv_exactly=counts_ok,
                 metrics_max_abs_diff_vs_csv=max_err, n_map_px_exactly_0_5=int((z["map"] == np.float16(0.5)).sum()),
                 gt_px=int(g.sum()), gt_fraction=float(g.mean()), pred_px=int(p.sum()),
                 mask_soft_edge_px=int(((mk > 0) & (mk < 255)).sum()),
                 map_mean_inside_gt=float(m[g].mean()), map_mean_outside_gt=float(m[~g].mean()),
                 fn_map_mean=float(m[~p & g].mean()) if fn else None)
        assert counts_ok and max_err < 1e-9, v  # CSV metrics are stored with 10 significant digits
        print(f"[VERIFIED] {iid}: sizes {W}x{H} aligned, counts identical to CSV, metric diff {max_err:.1e}")
        report[iid] = dict(csv={k: r[k] for k in ("generator", "mask_type", "category", "det_score", "det_pred")},
                           metrics=csv_m, verification=v)

        # ---------------------------------------------------------------- figure
        fig, ax = plt.subplots(2, 3, figsize=(18, 18 * 2 * H / (3 * W) + 1.6), dpi=200)
        gray = (0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]) / 255
        ax[0, 0].imshow(im)
        ax[0, 0].set_title(f"1. Input image ({r['generator']}, {r['mask_type']}, {W}×{H})")
        ax[0, 1].imshow(g, cmap=ListedColormap(["#000000", "#ffffff"]), vmin=0, vmax=1, interpolation="nearest")
        ax[0, 1].set_title(f"2. Ground truth (mask ≥ {GT_T}; white = manipulated)")
        h = ax[0, 2].imshow(m, cmap=BLUE, vmin=0, vmax=1)
        ax[0, 2].set_title("3. TruFor localization map (probability, saved float16)")
        cb = fig.colorbar(h, ax=ax[0, 2], fraction=0.035, pad=0.02)
        cb.ax.axhline(PR_T, color="#0b0b0b", lw=1.2)
        cb.set_label(f"P(manipulated); line = pixel threshold {PR_T}")
        ax[1, 0].imshow(p, cmap=ListedColormap(["#000000", "#ffffff"]), vmin=0, vmax=1, interpolation="nearest")
        ax[1, 0].set_title(f"4. Binary prediction (map > {PR_T}; white = manipulated)")
        ov = im / 255.0
        ov = np.where(p[..., None], 0.5 * ov + 0.5 * rgb(PRED_C), ov)
        ax[1, 1].imshow(ov)
        ax[1, 1].contour(g, levels=[0.5], colors=["#ffffff"], linewidths=1.6)
        ax[1, 1].contour(g, levels=[0.5], colors=["#0b0b0b"], linewidths=0.6)
        ax[1, 1].set_title("5. Prediction overlay (tint) with GT outline (white)")
        err = np.repeat((0.35 + 0.4 * gray)[..., None], 3, 2)  # TN: dimmed grayscale input
        err[p & g], err[p & ~g], err[~p & g] = rgb(TP_C), rgb(FP_C), rgb(FN_C)
        ax[1, 2].imshow(err, interpolation="nearest")
        ax[1, 2].set_title("6. Errors vs GT")
        ax[1, 2].legend(handles=[Patch(color=TP_C, label=f"TP  {tp:,} px"), Patch(color=FP_C, label=f"FP  {fp:,} px"),
                                 Patch(color=FN_C, label=f"FN  {fn:,} px"),
                                 Patch(facecolor="#9a9a9a", label=f"TN  {tn:,} px (dimmed image)")],
                        loc="lower left", fontsize=9, framealpha=0.92)
        for a in ax.flat:
            a.set_xticks([]); a.set_yticks([])
            a.title.set_fontsize(11)
        fig.suptitle(f"{iid}   |   IoU {csv_m['iou']:.4f}   F1 {csv_m['pixel_f1']:.4f}   "
                     f"Precision {csv_m['pixel_precision']:.4f}   Recall {csv_m['pixel_recall']:.4f}   "
                     f"(pixel threshold: map > {PR_T})\n"
                     f"Image-level detection score {float(r['det_score']):.4f} → "
                     f"{'manipulated' if r['det_pred'] == '1' else 'authentic'} (separate threshold: score > 0.5)",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fn_out = os.path.join(out_dir, f"{iid}.png")
        fig.savefig(fn_out, dpi=200)
        plt.close(fig)
        report[iid]["figure"] = os.path.relpath(fn_out, root)
        print(f"  saved {fn_out}")
    json.dump(report, open(os.path.join(out_dir, "verification.json"), "w"), indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
