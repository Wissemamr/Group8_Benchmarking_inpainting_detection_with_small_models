# ForMa evaluation: TGIF sheep sample

Six Photoshop-spliced variants of source scene `154947`, selected before evaluating. This is a one-scene pilot, not a full-dataset benchmark.

Input: 512x512 RGB. Predictions evaluated at original 1024x1024 resolution. Prediction threshold >0.5; Photoshop-adapted ground truth >127. CPU, four threads, one warmup.

## Scores

| Metric | Macro (mean per image) | Micro (pooled pixels) |
| --- | ---: | ---: |
| iou | 0.3025 | 0.2648 |
| dice | 0.4216 | 0.4188 |
| precision | 0.8418 | 0.6810 |
| recall | 0.3526 | 0.3024 |
| specificity | 0.9919 | 0.9918 |
| pixel_accuracy | 0.9539 | 0.9539 |

Dice equals pixel F1. High background accuracy should not be mistaken for strong localization: mean recall is only 35.26%.

## Per-image results

| Mask | Variant | IoU | Pixel F1 | Precision | Recall | Forward seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bbox | 0 | 0.4444 | 0.6153 | 0.9981 | 0.4448 | 5.064 |
| bbox | 1 | 0.0422 | 0.0810 | 1.0000 | 0.0422 | 5.010 |
| bbox | 2 | 0.0509 | 0.0968 | 1.0000 | 0.0509 | 5.581 |
| segm | 0 | 0.5213 | 0.6853 | 0.8005 | 0.5991 | 6.225 |
| segm | 1 | 0.5507 | 0.7103 | 0.9633 | 0.5625 | 5.801 |
| segm | 2 | 0.2056 | 0.3410 | 0.2889 | 0.4161 | 4.824 |

## Timing

- Model loading (including model import/initialization): 17.78s.
- Warmup, excluded from measured forwards: 5.27s.
- Six forward passes: 32.50s; mean 5.42s/image.
- Image processing, evaluation and output saving: 33.81s total, including forward passes.
- Total measured run: 57.80s, including loading/warmup; excludes interpreter/import startup and final report writing.

## Data structure

- Local download: 2,058 PS-sp images and 3,428 mask files in 13 categories.
- Sheep: 28 source scenes x 2 mask types x 3 variants = 168 images.
- Every local PS-sp image has a corresponding Photoshop-adapted mask with matching dimensions.
- Pair `<id>_mask_<type>.png_ps_<variant>.png` with `<id>_mask_<type>.png_ps_mask.png` under the same category.
- `bbox` and `segm` are distinct edits/targets. Numbered generated variants share a mask; they are not independent source scenes.

[Dataset filename conventions](https://github.com/IDLabMedia/tgif-dataset). Border thresholding here is an explicit local protocol choice.

See [visual comparison](report.html), [full JSON](report.json), [CSV](samples.csv), and [inventory](inventory.json).
