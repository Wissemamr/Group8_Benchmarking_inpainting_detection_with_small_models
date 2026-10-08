# TruFor on TGIF (testing split): detection and localization

Zero-shot evaluation of the **official pretrained [TruFor](https://github.com/grip-unina/TruFor)** (Guillaro et al.,
CVPR 2023) on the TGIF testing split, for both benchmark tasks. There is **no TGIF fine-tuning**: the official weights
and inference code are used unchanged. One forward pass per image gives both the image-level score (detection) and the
pixel-level anomaly map (localization).

> TruFor has **68.7 M parameters**, so it is not a "small model" under our < 10 M criterion. It is included as a
> strong **reference row**.

## Results (mean ± std over the 3 shared resamples)

Row label: *TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning)*. Each resample has 3,575 images
(275 real, 1,650 SD, 1,650 PS); localization uses its 3,300 manipulated images.

**Table 1: Detection**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TruFor | 0.9549 ± 0.0047 | 0.9888 ± 0.0008 | 0.9620 ± 0.0045 | 0.9752 ± 0.0026 | 0.9770 ± 0.0028 |

For comparison, the "always predict manipulated" baseline scores Accuracy 0.9231 and F1 0.9600, and TruFor's
balanced accuracy is 0.9156 ± 0.0065. TruFor misclassifies 34–39 of the 275 real images per resample.

**Table 2a: Localization, macro (mean of per-image metrics)**

| Model | Pixel Precision | Pixel Recall | Pixel F1 | Dice | IoU |
|---|---:|---:|---:|---:|---:|
| TruFor | 0.9178 ± 0.0013 | 0.7678 ± 0.0018 | 0.8162 ± 0.0019 | 0.8162 ± 0.0019 | 0.7215 ± 0.0017 |

**Table 2b: Localization, pooled (TP/FP/FN summed over all pixels)**

| Model | Pixel Precision | Pixel Recall | Pixel F1 | Dice | IoU |
|---|---:|---:|---:|---:|---:|
| TruFor | 0.9329 ± 0.0021 | 0.8214 ± 0.0007 | 0.8736 ± 0.0008 | 0.8736 ± 0.0008 | 0.7756 ± 0.0012 |

**Table 3: Efficiency**: not measured yet (pending).

Notes on Tables 1–2:

- **Dice and F1:** pixel F1 and Dice are identical for binary masks; both are reported because the group format
  lists both.
- **Std:** std is the sample std (ddof = 1) over the 3 resamples. The resamples share about 80% of their images, so
  the std understates the true sampling uncertainty.
- **Empty masks:** for the macro average, a per-image metric with a 0/0 denominator counts as 1 if prediction and GT
  are both empty, and 0 otherwise. This case never occurred: no GT mask is empty and no prediction is empty.
- **Per-resample tables:** in [`outputs/tables/`](outputs/tables/) (`tables.md` plus one CSV per table).
- **Splits:** SD vs PS, bbox vs segm and per category are in the `diag_*.csv` files. Examples: macro F1 is 0.837 for
  SD and 0.795 for PS, and detection AUC is 0.981 for SD vs real and 0.973 for PS vs real.

**Diagnostic only, not comparable with other models: official TruFor protocol.** The official TruFor `metrics.py`
excludes GT-boundary pixels and keeps the better of the map and the inverted map. Under that protocol, F1 at 0.5 is
0.8651 ± 0.0004, and F1 at the per-image oracle threshold is 0.9104 ± 0.0019. Never put these numbers in the main
tables.

## Per-image file for cross-model comparison

[`outputs/tables/trufor_per_image.csv`](outputs/tables/trufor_per_image.csv) has one row per image (4,407 rows), keyed by
`image_id`. It holds the detection score and decision, the per-image localization metrics, and the TP/FP/FN/TN pixel
counts. All the tables above can be recomputed from it to 4 decimals.
[`outputs/tables/README_columns.md`](outputs/tables/README_columns.md) documents each column and gives a **template
header** so every model's file has identical columns and can be concatenated or joined on `image_id`.

## Protocol

| | Rule |
|---|---|
| Data | TGIF testing split; images listed in the 3 shared resamples in [`../AEROBLADE/resamples/`](../AEROBLADE/resamples/) (SHA-256 checked). Their union is 4,407 unique images: 339 real, 2,034 SD, 2,034 PS. 4 of the 343 source images never fall into any resample. |
| Input | Full-resolution RGB, no resizing, exactly as the official `test.py` (the runner reproduces it with max abs diff 0). |
| Detection | Positive = manipulated. Predicted manipulated if `score > 0.5` (official rule). ROC-AUC from the continuous score. |
| Localization | Prediction: `map > 0.5` (strict, as in the official `metrics.py`). GT: `mask_L >= 128` (the masks have soft edges; same as `> 127` in AEROBLADE/ForMa). Original resolution. No best-of-inverted map and no boundary exclusion. |
| Aggregation | Macro and pooled, per resample, then mean ± std over the 3 resamples. |

The stored maps are float16 (`np.savez_compressed`); scalars are stored at full float64 precision. In the smoke
test, the float16 cast changed at most 6 pixels per image at the 0.5 cut.

## Layout

```
TruFor/
├── code/
│   ├── build_manifest.py          # unique images across resamples + validation
│   ├── phaseA/                    # archive inspection, mask value analysis
│   ├── phaseA2/check_checkpoint.py   # safe checkpoint load (pickle globals allowlist), strict key match
│   ├── phaseB/run_trufor.py       # inference runner (resumable), used for the smoke test and the full run
│   ├── phaseB/smoke_eval.py       # 6-image smoke test vs official test.py
│   ├── phaseC/run_full_inference.sh  # full run (nohup + caffeinate, resumable)
│   └── phaseD/                    # integrity checks, metrics + tables, per-image file, example figures
├── config/                        # config.json, requirements lock files (inference env, evaluation env)
├── outputs/
│   ├── manifest/                  # manifest_unique.csv (4,407 images) + validation
│   ├── inference/results.csv      # per image: score, detection logit, sizes, timings (maps are not committed)
│   ├── phaseA2/, phaseB_smoke/    # checkpoint check, smoke-test report and figure
│   ├── phaseD/                    # integrity.json, per_image_localization.csv, metric checks
│   └── tables/                    # all tables (Markdown + CSV), per-image master file, column docs
└── results/localization_examples/ # 2 annotated example figures + short report
```

## Reproduce

1. **Data.** Put the TGIF testing files under `TruFor/data/tgif/` (`orig/`, `sd2-sp/`, `ps-sp/`, `masks/`, each
   with `testing/<category>/`). See the TGIF section of the main README for the download.
2. **Official code.** Clone the official code at the pinned commit. It is not redistributed here (GRIP-UNINA
   license, nonprofit use).
   ```
   git clone https://github.com/grip-unina/TruFor TruFor/TruFor
   git -C TruFor/TruFor checkout ae54475df6f41a491d7615100feb19263dec13f7
   ```
3. **Weights.** Download https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip and check its MD5
   (`7bee48f3476c75616c3c5721ab256ff8`). Unzip it to `TruFor/weights/extracted/`, which gives
   `weights/trufor.pth.tar` with SHA-256 `ac1d90e3…7844`.
4. **Environments.**
   - `.venv-trufor` (Python 3.10, CPU): `config/requirements-trufor.lock.txt`, with torch 2.2.2, numpy 1.26.4,
     timm 0.5.4 and yacs 0.1.8. The official conda yaml targets Linux, CUDA and Python 3.7.
   - `.venv-eval` (Python 3.13): `config/requirements-eval.lock.txt`, with scipy and scikit-learn, for the
     metrics.
5. **Run.** From `TruFor/`:
   ```
   .venv-data/bin/python -I code/build_manifest.py           # manifest (any Python 3 with stdlib + Pillow)
   zsh code/phaseC/run_full_inference.sh                     # ~8 h on a 10-core Apple-silicon CPU, resumable
   .venv-eval/bin/python code/phaseD/integrity.py .          # 40 integrity checks
   .venv-eval/bin/python code/phaseD/metrics.py .            # Tables 1-2, diagnostics, sanity checks
   .venv-eval/bin/python code/phaseD/per_image_master.py .   # per-image file + check against the tables
   ```

## Known deviations and caveats

- **Prediction rule:** the earlier phases used `map >= 0.5` for the binary prediction. The final evaluation uses
  `map > 0.5` (official). The two differ only on pixels exactly equal to 0.5, which changes pooled F1 by at most
  1e-4.
- **Detection class balance:** the classes are 92% manipulated. Read F1 and accuracy against the always-manipulated
  baseline and balanced accuracy.
- **Official-protocol NaNs:** 11 thin segm masks leave no GT core after the 15 px erosion. The official code returns
  NaN for them and drops them with `nanmean`; this affects only the diagnostic table.
- **Efficiency:** Table 3 (efficiency) is still to do.
