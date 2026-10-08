# Per-image results file: column reference and template

Every model in the benchmark ships one file named `<model>_per_image.csv` with **one row per unique TGIF test image**
(4,407 rows) and **exactly the columns below, in this order**. Because the columns are identical, files from different
models can be stacked (concatenated) into a long table, or joined on `image_id` into a wide table.

`image_id` is the key. It is the same across the three resamples, so a single file serves all of them. Use the
`in_resample_k` flags to select the images of resample k.

## Template header (copy this line exactly)

```
model,image_id,source_id,category,generator,mask_type,variant,label,width,height,file,mask_file,in_resample_1,in_resample_2,in_resample_3,det_score,det_pred,det_correct,loc_iou,loc_dice,loc_pixel_precision,loc_pixel_recall,loc_pixel_f1,loc_tp,loc_fp,loc_fn,loc_tn
```

## Columns

| # | Column | Type | Meaning |
|---|---|---|---|
| 1 | `model` | string | Model name, the same on every row (for example `TruFor`). |
| 2 | `image_id` | string | Unique image key: `real_<source_id>`, `sd_<source_id>_<mask_type>_<variant>` or `ps_<source_id>_<mask_type>_<variant>`. |
| 3 | `source_id` | int | ID of the original image. All images derived from the same original share it. |
| 4 | `category` | string | Object category (13 categories, such as `airplane` or `stop sign`). |
| 5 | `generator` | string | `real` (authentic), `SD` (Stable Diffusion 2 inpainting, spliced) or `PS` (Photoshop generative fill, spliced). |
| 6 | `mask_type` | string | `bbox` or `segm` for manipulated images; `none` for real images. |
| 7 | `variant` | int | Generation variant, 0–2; `-1` for real images. |
| 8 | `label` | int | Ground truth for detection: `1` = manipulated, `0` = real. |
| 9 | `width` | int | Image width in pixels. |
| 10 | `height` | int | Image height in pixels. |
| 11 | `file` | string | Image path relative to the TGIF root. |
| 12 | `mask_file` | string | GT mask path relative to the TGIF root; empty for real images. |
| 13–15 | `in_resample_1`, `in_resample_2`, `in_resample_3` | 0/1 | 1 if the image is in that resample (3,575 images each). |
| 16 | `det_score` | float | The model's continuous image-level score; higher means more likely manipulated. Written at full float64 precision. Used for ROC-AUC. |
| 17 | `det_pred` | 0/1 | Binary detection decision: `1` if `det_score > 0.5`. |
| 18 | `det_correct` | 0/1 | `1` if `det_pred == label`. |
| 19 | `loc_iou` | float | Per-image IoU = TP / (TP + FP + FN). |
| 20 | `loc_dice` | float | Per-image Dice = 2TP / (2TP + FP + FN). |
| 21 | `loc_pixel_precision` | float | Per-image TP / (TP + FP). |
| 22 | `loc_pixel_recall` | float | Per-image TP / (TP + FN). |
| 23 | `loc_pixel_f1` | float | Per-image pixel F1. For binary masks this is identical to Dice; both are kept on purpose. |
| 24–27 | `loc_tp`, `loc_fp`, `loc_fn`, `loc_tn` | int | Pixel counts at the image's full resolution; they sum to `width × height`. |

## Rules (identical for every model)

- **Detection:** the positive class is *manipulated*, and `det_pred = det_score > 0.5` (strict).
- **Localization (neutral protocol):**
  - Prediction: a pixel is manipulated if `map > 0.5` (strict).
  - GT: a pixel is manipulated if the mask, converted to grayscale (L), is `>= 128`.
  - Metrics are computed at the original image resolution.
  - No "best of map and inverted map" and no exclusion of GT-boundary pixels.
- **Real images:** all `loc_*` columns are **empty (NaN), never 0**. Localization is evaluated on manipulated images
  only.
- **Per-image 0/0 convention:** if a metric's denominator is 0, the metric is 1 when the prediction and the GT are
  both empty, and 0 otherwise. For TruFor this never occurred: no GT mask is empty and no prediction is empty.
- **Aggregation**, which reproduces the published tables:
  - Macro: the mean of a `loc_*` metric column over the manipulated images of a resample.
  - Pooled (micro): sum `loc_tp`, `loc_fp` and `loc_fn` over those images, then apply the formulas above.
  - Detection metrics: computed from `label`, `det_pred` and `det_score` over the 3,575 images of a resample.
  - Mean ± std over the 3 resamples uses the sample std (ddof = 1).

## Using the files

```python
import pandas as pd
long = pd.concat([pd.read_csv(f) for f in ["trufor_per_image.csv", "othermodel_per_image.csv"]])  # stack
wide = long.pivot(index="image_id", columns="model", values="loc_iou")                             # compare per image
r1 = long[long.in_resample_1 == 1]                                                                 # one resample
```

When reading with pandas, empty `loc_*` cells become NaN, and `pandas.mean()` skips them. The metadata columns
(2–15) must be identical across models for the same `image_id`. Check this after merging.

## TruFor file notes

- `outputs/tables/trufor_per_image.csv`
- Built from the saved inference outputs; inference was not rerun.
- `det_score` is bit-identical to the inference `results.csv`.
- `loc_*` metrics are the exact per-image values the published tables were computed from (10 significant digits).
- Verified: all 15 numbers of each per-resample table and the mean ± std tables are reproduced from this file to
  4 decimals.
- A parquet copy was not written because `pyarrow` is not installed in the evaluation environment.
