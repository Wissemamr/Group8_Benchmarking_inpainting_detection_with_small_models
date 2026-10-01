"""Binary pixel metrics; foreground means manipulated.

Masks must be HxW bool, 0/1 or 0/255. Undefined ratios are None (JSON
null), including empty/empty IoU and Dice. Average defined per-image
scores for macro results; sum counts first for micro results.
"""
import numpy as np


def binary_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size:
        raise ValueError('Expected a nonempty HxW mask.')
    if not (np.isin(mask, [0, 1]).all() or np.isin(mask, [0, 255]).all()):
        raise ValueError('Mask values must be 0/1 or 0/255.')
    return mask != 0


def metrics_from_counts(tp, fp, fn, tn):
    """Foreground metrics; also accepts pooled dataset confusion counts."""
    counts = (tp, fp, fn, tn)
    if any(not np.isfinite(v) or v < 0 or int(v) != v for v in counts):
        raise ValueError('Confusion counts must be nonnegative integers.')
    tp, fp, fn, tn = map(int, counts)
    def ratio(n, d):
        return n / d if d else None
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                iou=ratio(tp, tp + fp + fn), dice=f1, pixel_f1=f1,
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                specificity=ratio(tn, tn + fp),
                pixel_accuracy=ratio(tp + tn, tp + fp + fn + tn))


def localization_metrics(prediction, target):
    """Evaluate binary masks at the ground-truth resolution."""
    prediction, target = binary_mask(prediction), binary_mask(target)
    if prediction.shape != target.shape:
        raise ValueError('Prediction and ground-truth shapes must match.')
    return metrics_from_counts(np.count_nonzero(prediction & target),
        np.count_nonzero(prediction & ~target),
        np.count_nonzero(~prediction & target), np.count_nonzero(~prediction & ~target))
