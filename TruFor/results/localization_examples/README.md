# TruFor localization: two visual examples on TGIF

Two manipulated images were chosen from `outputs/tables/trufor_per_image.csv`: one with high and one with low
localization quality. Both are drawn from the **saved** TruFor maps; inference was not rerun. Both images are in all
three resamples, and both come from the same source photo (`7784`, kite) with the **same bbox GT mask**. Only the
generator differs (SD vs PS), so the comparison is controlled.

| | image_id | Generator / mask | Detection score | IoU | Pixel F1 | Pixel precision | Pixel recall | Figure |
|---|---|---|---|---|---|---|---|---|
| **A (high)** | `sd_7784_bbox_2` | SD / bbox | 0.8867 | 0.9893 | 0.9946 | 0.9973 | 0.9919 | [sd_7784_bbox_2.png](sd_7784_bbox_2.png) |
| **B (low)** | `ps_7784_bbox_1` | PS / bbox | 0.9370 | 0.2083 | 0.3448 | 0.9988 | 0.2084 | [ps_7784_bbox_1.png](ps_7784_bbox_1.png) |

**How they were selected.** Candidates were manipulated images present in all 3 resamples whose GT covers 8–40% of
the image, so the region is clearly visible (666 images). A has the highest IoU among them. B's IoU of 0.208 is below
the 5th percentile of all 4,068 manipulated images (0.263). The very lowest candidates (IoU 0.09–0.20) show the same
failure pattern as B (precision ≈ 1, low recall). B was chosen because it pairs with A.

## Thresholds (two different ones)

- **Pixel level, used in the figures and metrics:** a pixel is predicted manipulated if `map > 0.5`; a pixel is
  manipulated in the GT if the mask (converted to grayscale) is `>= 128`.
- **Image level, detection only:** the image is predicted manipulated if `score > 0.5`. Both images are detected
  correctly. Detection says nothing about where the manipulation is.

## What TruFor got right and wrong

**A, SD inpainting: almost perfect.** The probability map is close to 1 over the whole box (mean 0.95 inside the
GT, 0.01 outside). The binary prediction matches the rectangle, with only 620 FP and 1,857 FN pixels, all along the
box border. Some FP pixels sit where the kite's left wing tip crosses the box edge, where the SD content blends into
the authentic sky. Compared with the authentic original (`real_7784`), 92% of the missed pixels are essentially
unchanged (difference ≤ 10/255). The few errors are therefore boundary effects, not missed content.

**B, Photoshop generative fill: correct where it fires, but incomplete.** Precision is almost perfect: only 59 FP
pixels. TruFor flags the new kite (sail, tail and the two small kites) with high confidence. However, it leaves most
of the box unflagged: the mean probability inside the GT is only 0.31, and 182,026 pixels are FN. Two different
things make up this miss:

- **About 56% of the missed pixels are sky that Photoshop left almost unchanged** (difference ≤ 10/255 from the
  original). The bbox GT marks the whole rectangle as manipulated, but those pixels carry little or no manipulation
  signal. This shows how a bbox GT penalises a detector that tracks the actual changes.
- **About 44% of the missed pixels (79,282 px) were genuinely altered.** These are pixels Photoshop changed substantially,
  mainly smooth background between the generated kite parts, and TruFor still misses them. Counting only GT pixels
  that really changed, its recall is 0.37 for B, against 0.999 for A.

**Takeaway.** On this source image, TruFor localizes SD-generated content in a bbox almost perfectly. For
Photoshop's fill it detects the generated *objects* but not the regenerated *background*. This fits the aggregate
results: macro F1 is 0.837 for SD vs 0.795 for PS, and recall is the weak metric (macro 0.77, against 0.92 for
precision).

## Verification (before plotting)

The code is in `code/phaseD/localization_examples.py`, and its output is in `verification.json`.

- **Saved predictions:** `outputs/inference/maps/<file>.npz`, one per image (4,407 files). Each holds the float16
  probability map `map` at the full image resolution, plus `conf` and `imgsize`.
- **No binary masks were saved.** Binary masks are derived as `map > 0.5`. No spatial prediction was reconstructed
  from the TP/FP/FN/TN counts.
- **File links:** `image_id` → `file` and `mask_file` (from the CSV; paths relative to `data/tgif/`) → map at
  `outputs/inference/maps/<file>.npz`.
- **Alignment:** image, mask and map are all 1024×768 (W×H), and `imgsize` is stored as (H, W). There is no EXIF
  rotation. The mask has 3 identical channels and is converted to grayscale.
- **Metrics reproduced:** recomputing from the saved map and the mask gives TP/FP/FN/TN counts **identical** to the
  CSV. IoU, F1, precision and recall agree to ≤ 5e-11; the CSV stores metrics with 10 significant digits.
- **Comparison with the original:** the pixel differences against `real_7784` are an extra diagnostic for this
  report only. They are not part of any evaluation metric.
