"""Detection and localization metrics in plain numpy (no scikit-learn), so they run in every model's env."""

from __future__ import annotations

import numpy as np


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with average ranks for ties; nan when only one class is present."""
    y = np.asarray(y).astype(bool).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    # average rank per group of tied scores
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_s)) + 1, len(s)]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[a:b]] = (a + b + 1) / 2.0
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """AP as in scikit-learn (step-wise, one point per distinct score); nan without positives."""
    y = np.asarray(y).astype(bool).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last index of each distinct score
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = np.asarray(pred).astype(bool).ravel()
    gt = np.asarray(gt).astype(bool).ravel()
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(np.count_nonzero(~pred & ~gt))

    def ratio(a: float, b: float) -> float:
        return float(a / b) if b else float("nan")

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "iou": ratio(tp, tp + fp + fn),
            "accuracy": ratio(tp + tn, tp + fp + fn + tn)}


def localization(score_map: np.ndarray, pred_mask: np.ndarray, gt: np.ndarray) -> dict:
    """Pixel metrics at the model's own binarization, plus threshold-free AUC and AP of its score map."""
    out = binary_metrics(pred_mask, gt)
    out["pixel_auc"] = roc_auc(gt, score_map)
    out["pixel_ap"] = average_precision(gt, score_map)
    out["gt_fraction"] = float(np.asarray(gt).mean())
    out["pred_fraction"] = float(np.asarray(pred_mask).mean())
    return out
