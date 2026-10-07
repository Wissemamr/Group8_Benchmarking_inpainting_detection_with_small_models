"""Detection and localization metrics, threshold calibration and the aggregated result tables."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .lpips_map import upsample

HEADLINE = [
    "ID", "Mask Type Ratio (BBox & Segmentation)", "Number", "Inpainting Generator (SD/PS)",
    "Category", "Aggregated F1 Score", "Aggregated Recall", "Aggregated Precision",
]


# ----------------------------------------------------------------------------- localization

def normalize(err: np.ndarray, method: str = "percentile", percentiles=(1.0, 99.0)) -> np.ndarray:
    """Per-image scaling of the error map to [0, 1]: min-max, or percentiles with clipping."""
    if method == "minmax":
        lo, hi = float(err.min()), float(err.max())
    elif method == "percentile":
        lo, hi = (float(v) for v in np.percentile(err, percentiles))
    else:
        raise ValueError(f"unknown normalization {method!r}")
    if hi <= lo:
        return np.zeros_like(err, dtype=np.float32)
    return np.clip((err - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def otsu_threshold(score: np.ndarray) -> float:
    lo, hi = float(score.min()), float(score.max())
    return lo if lo == hi else float(threshold_otsu(score))


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def pixel_counts_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Foreground metrics from a confusion matrix; undefined ratios are NaN (not silently 0)."""
    return {
        "iou": _ratio(tp, tp + fp + fn),
        "dice": _ratio(2 * tp, 2 * tp + fp + fn),
        "pixel_precision": _ratio(tp, tp + fp),
        "pixel_recall": _ratio(tp, tp + fn),
        "pixel_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "pixel_specificity": _ratio(tn, tn + fp),
        "pixel_accuracy": _ratio(tp + tn, tp + fp + fn + tn),
    }


def _polarity_scores(err: np.ndarray, norm: np.ndarray, polarity: str) -> tuple[np.ndarray, np.ndarray]:
    """(thresholdable score in [0, 1], unclipped ranking score), higher = more likely inpainted.
    inverted = AEROBLADE's premise (generated content reconstructs better), raw = the opposite."""
    if polarity == "inverted":
        return 1.0 - norm, -err
    if polarity == "raw":
        return norm, err
    raise ValueError(polarity)


def localization_one(
    layer_map: np.ndarray, gt: np.ndarray, polarity: str = "inverted",
    method: str = "percentile", percentiles=(1.0, 99.0),
) -> dict:
    """All localization numbers of one image at the ground truth's (native) resolution.

    The H/2 x W/2 LPIPS map is bilinearly upsampled to the GT size, normalised per image, turned
    into a score for the chosen polarity and binarised at its Otsu threshold. AP and ROC-AUC sweep
    every threshold (pixel ranking only). The opposite polarity is scored too (`alt_*`).
    """
    h, w = gt.shape
    err = upsample(layer_map, (h, w))
    norm = normalize(err, method, percentiles)
    alt = "raw" if polarity == "inverted" else "inverted"
    out: dict = {"polarity": polarity, "alt_polarity": alt}
    y = gt.ravel()
    scoreable = 0 < np.count_nonzero(y) < y.size
    for pol, prefix in ((polarity, ""), (alt, "alt_")):
        score, rank = _polarity_scores(err, norm, pol)
        t = otsu_threshold(score)
        pred = score > t
        tp = int(np.count_nonzero(pred & gt))
        fp = int(np.count_nonzero(pred & ~gt))
        fn = int(np.count_nonzero(~pred & gt))
        tn = int(gt.size - tp - fp - fn)
        res = pixel_counts_metrics(tp, fp, fn, tn)
        res |= {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "otsu_threshold": t,
                "pred_fraction": float(pred.mean())}
        if scoreable:
            res["pixel_ap"] = float(average_precision_score(y, rank.ravel()))
            res["pixel_roc_auc"] = float(roc_auc_score(y, rank.ravel())) if prefix == "" else None
        else:
            res["pixel_ap"] = res["pixel_roc_auc"] = float("nan")
        if prefix == "alt_":  # AUC(raw) = 1 - AUC(inverted) exactly (ties count 1/2 both ways)
            res["pixel_roc_auc"] = 1.0 - out["pixel_roc_auc"] if scoreable else float("nan")
            res = {k: v for k, v in res.items() if k in (
                "iou", "dice", "pixel_precision", "pixel_recall", "pixel_ap", "pixel_roc_auc")}
        out |= {prefix + k: v for k, v in res.items()}
    inside, outside = err[gt], err[~gt]
    out |= {
        "gt_fraction": float(gt.mean()),
        "err_inside_gt": float(inside.mean()) if inside.size else float("nan"),
        "err_outside_gt": float(outside.mean()) if outside.size else float("nan"),
    }
    out["err_ratio_in_out"] = _ratio(out["err_inside_gt"], out["err_outside_gt"])
    ys, xs = np.nonzero(gt)
    out |= {"gt_x0": int(xs.min()), "gt_y0": int(ys.min()), "gt_x1": int(xs.max()) + 1,
            "gt_y1": int(ys.max()) + 1} if xs.size else {}
    return out


def localization_job(job: dict) -> dict:
    """Process-pool worker: load one map (min over AEs) and its mask, score it."""
    t = time.perf_counter()
    layer_map = np.minimum.reduce([np.load(p) for p in job["map_files"]])
    with Image.open(job["mask_file"]) as im:
        gt = np.asarray(im.convert("L")) > job["mask_threshold"]
    res = localization_one(layer_map, gt, job["polarity"], job["normalize"], tuple(job["percentiles"]))
    res["image_id"] = job["image_id"]
    res["eval_seconds"] = time.perf_counter() - t
    return res


# ----------------------------------------------------------------------------- detection

def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Score threshold maximising TPR - FPR (predict inpainted when score >= threshold)."""
    fpr, tpr, thr = roc_curve(y, score)
    thr = thr.copy()
    thr[0] = np.nextafter(thr[1], np.inf)  # roc_curve's first threshold is +inf: "predict nothing"
    return float(thr[int(np.argmax(tpr - fpr))])


def crossfit_folds(sources: pd.DataFrame, k: int, seed: int) -> dict[int, int]:
    """source_id -> fold, balanced within each category (so every fold sees every category)."""
    rng = np.random.default_rng(seed)
    fold: dict[int, int] = {}
    for _, grp in sources.groupby("category"):
        ids = np.sort(grp["source_id"].to_numpy())
        rng.shuffle(ids)
        for i, s in enumerate(ids):
            fold[int(s)] = i % k
    return fold


def detection_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, fp = int(np.sum(pred & y)), int(np.sum(pred & ~y))
    fn, tn = int(np.sum(~pred & y)), int(np.sum(~pred & ~y))
    prec, rec, spec = _ratio(tp, tp + fp), _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    both = 0 < y.sum() < y.size
    return {
        "accuracy": _ratio(tp + tn, y.size),
        "precision": prec, "recall": rec,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([rec, spec])),
        "roc_auc": float(roc_auc_score(y, score)) if both else float("nan"),
        "ap": float(average_precision_score(y, score)) if both else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ----------------------------------------------------------------------------- tables

def mask_ratio_text(mask_types: Iterable[str]) -> str:
    s = pd.Series(list(mask_types))
    if s.empty:
        return "n/a"
    return f"{100 * (s == 'bbox').mean():.1f}% bbox / {100 * (s == 'segm').mean():.1f}% segm"


def _group_specs(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(generator, category) rows: every pair, a per-generator total, and the grand total."""
    pairs = sorted(df.loc[df["label"] == 1, ["generator", "category"]].drop_duplicates()
                   .itertuples(index=False, name=None), key=lambda t: (t[0] != "SD", t[0], t[1]))
    gens = sorted({g for g, _ in pairs}, key=lambda g: (g != "SD", g))
    return pairs + [(g, "ALL") for g in gens] + [("ALL", "ALL")]


def _select(df: pd.DataFrame, gen: str, cat: str, mask_type: str | None, positives: bool) -> pd.DataFrame:
    sel = df[df["label"] == (1 if positives else 0)]
    if positives and gen != "ALL":
        sel = sel[sel["generator"] == gen]
    if cat != "ALL":
        sel = sel[sel["category"] == cat]
    if positives and mask_type:
        sel = sel[sel["mask_type"] == mask_type]
    return sel


def _finish(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    table.insert(0, "ID", range(1, len(table) + 1))
    cols = HEADLINE + [c for c in table.columns if c not in HEADLINE]
    return table[cols]


def detection_table(det: pd.DataFrame, by_mask_type: bool = False) -> pd.DataFrame:
    """Aggregated detection metrics per (generator, category [, mask type]).

    A group's positives are its inpainted images; its negatives are the authentic images of the
    same category (they have no generator or mask type, so all groups of a category share them).
    The confusion matrix is pooled over the group. Number = inpainted images in the group.
    """
    rows = []
    for mask_type in (["bbox", "segm"] if by_mask_type else [None]):
        for gen, cat in _group_specs(det):
            pos = _select(det, gen, cat, mask_type, True)
            neg = _select(det, gen, cat, mask_type, False)
            if pos.empty or neg.empty:
                continue
            grp = pd.concat([pos, neg])
            m = detection_metrics(grp["label"].to_numpy(), grp["det_pred"].to_numpy(), grp["det_score"].to_numpy())
            row = {
                "Mask Type Ratio (BBox & Segmentation)": mask_ratio_text(pos["mask_type"]),
                "Number": len(pos),
                "Inpainting Generator (SD/PS)": "ALL (SD+PS)" if gen == "ALL" else gen,
                "Category": cat,
                "Aggregated F1 Score": m["f1"], "Aggregated Recall": m["recall"],
                "Aggregated Precision": m["precision"],
                "accuracy": m["accuracy"], "roc_auc": m["roc_auc"], "ap": m["ap"],
                "balanced_accuracy": m["balanced_accuracy"], "specificity": m["specificity"],
                "n_authentic": len(neg), "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
            }
            if by_mask_type:
                row["Mask Type"] = mask_type
            rows.append(row)
    return _finish(rows)


def localization_table(loc: pd.DataFrame, by_mask_type: bool = False) -> pd.DataFrame:
    """Aggregated localization metrics per (generator, category [, mask type]).

    Headline F1 / recall / precision are macro averages (the mean of the per-image scores, so
    every image counts once); `micro_*` pool the pixel counts of the group instead. Precision is
    undefined (NaN) for an image whose predicted mask is empty and is skipped in the mean.
    """
    rows = []
    for mask_type in (["bbox", "segm"] if by_mask_type else [None]):
        for gen, cat in _group_specs(loc):
            grp = _select(loc, gen, cat, mask_type, True)
            if grp.empty:
                continue
            tp, fp, fn = (int(grp[c].sum()) for c in ("tp", "fp", "fn"))
            micro = pixel_counts_metrics(tp, fp, fn, int(grp["tn"].sum()))
            row = {
                "Mask Type Ratio (BBox & Segmentation)": mask_ratio_text(grp["mask_type"]),
                "Number": len(grp),
                "Inpainting Generator (SD/PS)": "ALL (SD+PS)" if gen == "ALL" else gen,
                "Category": cat,
                "Aggregated F1 Score": grp["pixel_f1"].mean(),
                "Aggregated Recall": grp["pixel_recall"].mean(),
                "Aggregated Precision": grp["pixel_precision"].mean(),
                "iou": grp["iou"].mean(), "dice": grp["dice"].mean(),
                "pixel_ap": grp["pixel_ap"].mean(), "pixel_roc_auc": grp["pixel_roc_auc"].mean(),
                "mean_gt_fraction": grp["gt_fraction"].mean(),
                "micro_f1": micro["pixel_f1"], "micro_recall": micro["pixel_recall"],
                "micro_precision": micro["pixel_precision"], "micro_iou": micro["iou"],
                "n_precision_defined": int(grp["pixel_precision"].notna().sum()),
            }
            if by_mask_type:
                row["Mask Type"] = mask_type
            rows.append(row)
    return _finish(rows)


def to_markdown(table: pd.DataFrame, digits: int = 4) -> str:
    cols = list(table.columns)
    counts = {c for c in cols if c.startswith("n_") or c in ("Number", "tp", "fp", "fn", "tn")}

    def cell(v, col: str) -> str:
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "nan"
            return f"{v:.0f}" if col in counts else f"{v:.{digits}f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(v, c) for v, c in zip(row, cols)) + " |"
              for row in table.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def save_table(table: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(to_markdown(table))
