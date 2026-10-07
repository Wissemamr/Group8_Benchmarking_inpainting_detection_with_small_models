# AEROBLADE on TGIF (testing split): detection and localization

Zero-shot evaluation of [AEROBLADE](https://github.com/jonasricker/aeroblade) (training-free, autoencoder
reconstruction error) on the TGIF **testing** split, for both benchmark tasks:

* **Detection** – is the image inpainted? (image level; authentic vs SD2 / Photoshop inpainting)
* **Localization** – which pixels were inpainted? (pixel level, against TGIF's masks)

One pass of AEROBLADE over an image serves both tasks, so every image is processed once.
The whole benchmark is also run on **3 shareable resamples** of the testing set (6 extra dataframes:
3 detection + 3 localization), and every stage is timed.

## What is evaluated

Only TGIF's testing files are used (no training/validation data, no fitting on anything else):

| set | files | label |
|---|---|---|
| `orig`   | 343 authentic MS-COCO images (`*_orig.png`) | 0 |
| `sd2-sp` | 2 058 inpainted: 343 sources × {bbox, segm} × 3 variants, Stable Diffusion 2 inpainting | 1 (`generator = SD`) |
| `ps-sp`  | 2 058 inpainted: same structure, Adobe Photoshop generative fill | 1 (`generator = PS`) |

= **4 459 images** from 343 source images in 13 COCO categories. The ground truth of both inpainted sets is
`masks/testing/<category>/<id>_mask_<bbox|segm>.png_ps_mask.png` (binarised at `> 127`, as in `ForMa/`).
The `*_orig_512.png`, `*_orig_1024.png`, `*_mask_512.png`... files belong to TGIF's other subsets and are not used.
`results/index/all_images.csv` lists every image with its mask, size and checks that all 4 116 masks exist and
match their image size.

`variant` is the generation index (0–2); `-1` / `mask_type = none` / `generator = real` mean "authentic".

## Method (what AEROBLADE does here)

1. Resize the image to 512×512 (bicubic) – the same input size as ForMa, applied to authentic and inpainted images alike.
2. Reconstruct it with the **SD2 autoencoder** (encode → sample the latent with seed 1 → decode → 8-bit PNG), as in AEROBLADE.
3. LPIPS (VGG, layer 2 = `relu2_2`) between image and reconstruction, kept as a map of 256×256.
4. **Detection score** = AEROBLADE's distance, the spatial mean of that map (computed exactly as AEROBLADE does,
   no padding); an image is more likely generated the *lower* the distance, so `det_score = -distance`.
5. **Localization map** = the same map computed on reflect-padded inputs (removes VGG's zero-padding border artifact),
   bilinearly upsampled to the original image size, normalised per image (1st–99th percentile, clipped) and turned into
   `1 - error` (AEROBLADE's premise: generated content reconstructs better), then binarised with a per-image **Otsu**
   threshold. The opposite polarity (`raw`) is also scored (`alt_*` columns) so the premise can be checked.

Single autoencoder (SD2) because a full pass costs ~17 s/image on the CPU used (see Timing); AEROBLADE proper takes the
minimum over SD1, SD2 and Kandinsky 2.1. The code supports it (`--aes SD1 SD2 KD2.1`, results are cached per AE),
but it was **not** run here, so the SD2 autoencoder is the only one matched to the SD images (an advantage for the SD
rows; PS images come from an unknown generator).

### Decisions that are conventions, not facts

* **Detection threshold.** AEROBLADE has no threshold, so accuracy / precision / recall / F1 need one. It is the Youden
  point (max TPR − FPR) of a calibration set that never contains the images it is applied to: 5-fold cross-fitting over
  source images for the full run, and for resample *k* the source images **outside** resample *k*. ROC-AUC and AP
  need no threshold. The testing split is the only data used, so there is no separate validation set.
* **Class imbalance.** Each source image has 1 authentic and 12 inpainted (6 SD + 6 PS) images, so per generator it is 1 : 6
  and overall 1 : 12. A classifier that flags everything gets precision 0.92 / recall 1 / F1 0.96 overall; read F1 together
  with `balanced_accuracy`, `specificity` and `roc_auc`.
* **Aggregation.** Detection: one pooled confusion matrix per row. Localization: the headline F1 / recall / precision are the
  *mean of the per-image scores* (macro); `micro_*` columns pool the pixels instead. `Number` = number of inpainted images in the row.
  Detection rows use as negatives the authentic images of the same category.
* **Localization** is scored on inpainted images only (authentic images have an empty mask) at the original resolution.
* **Resamples** (see below) are our choice of scheme; AEROBLADE has no trainable part, so a resample only changes which
  images are scored (and, for detection, which images calibrate the threshold).

## Results folder

```
results/
  index/all_images.csv              every image (+ index_summary.json)
  full/                             all 4 459 images
  resample_1/ resample_2/ resample_3/
  comparison_detection.{csv,md}     full vs resamples (+ mean / std over the resamples)
  comparison_localization.{csv,md}
  figures/overlays/                 one qualitative overlay per (generator, category)
```

Every run folder (`full/`, `resample_k/`) has the same files:

| file | content |
|---|---|
| `detection_per_image.{csv,parquet}` | one row per image: `image_id, source_id, category, generator, mask_type, variant, label`, `det_score`, `aeroblade_distance`, `det_threshold`, `det_pred`, `correct`, `outcome` (TP/FP/FN/TN), `grp_{accuracy,precision,recall,f1,roc_auc}` (the metrics of the image's *generator × category* group, repeated on its row – undefined for a single image), statistics of the LPIPS map (`map_mean … map_p99`), and per-stage timings |
| `detection_table.{csv,md}` | **the requested table** `ID · Mask Type Ratio (BBox & Segmentation) · Number · Inpainting Generator (SD/PS) · Category · Aggregated F1 Score · Aggregated Recall · Aggregated Precision`, then accuracy, ROC-AUC, AP, balanced accuracy, specificity, n_authentic, TP/FP/FN/TN. Rows: every generator × category, a per-generator `ALL` row and the grand total |
| `detection_table_by_mask_type.{csv,md}` | same, split into bbox / segm |
| `localization_per_image.{csv,parquet}` | one row per inpainted image: IoU, `dice`, `pixel_precision`, `pixel_recall`, `pixel_f1`, `pixel_ap`, `pixel_roc_auc`, `tp/fp/fn/tn`, `gt_fraction`, `pred_fraction`, Otsu threshold, the opposite polarity (`alt_*`), mean error inside vs outside the GT (`err_inside_gt`, `err_outside_gt`, `err_ratio_in_out`), GT bounding box, timings |
| `localization_table{,_by_mask_type}.{csv,md}` | same table layout as detection (macro F1 / recall / precision; IoU, dice, AP, AUC, `micro_*` as extra columns) |
| `run_summary.json` | sizes, thresholds used, overall metrics (ALL / SD / PS), timing, config + hash, environment, versions, git commit |

## Resamples (to share)

`resamples/` holds the three resamples as plain CSV manifests (same columns as the index plus `resample_id`):
`resample_1.csv`, `resample_2.csv`, `resample_3.csv`, `resample_sources.csv` (which source image is in which resample)
and `resamples.json` (parameters, seeds, sizes, overlaps, checksums).

* Scheme: **stratified by category, 80 % of the 343 source images, drawn without replacement**, seeds 2026 / 2027 / 2028
  (`seed = 2025 + k`). A drawn source image keeps its whole family (1 authentic + 6 SD + 6 PS), so each resample is a
  complete detection *and* localization benchmark (275 sources, 3 575 images each). Two resamples share ≈ 220 source images
  (≈ 80 % of each), so the three are overlapping draws, not independent ones; 4 source images are in none of them.
* Use the same CSVs for every model so all of us score the same images. Another model only needs `file` / `mask_file`
  (paths relative to the TGIF folder layout above). `python scripts/make_resamples.py --materialize DIR` also
  creates the folders with the files (hard links when possible) if you want to zip them.
* Regenerating with other settings (`--n`, `--fraction`, `--seed`) takes seconds, and re-running
  `scripts/evaluate.py` then rescores them from the cache in minutes (no new inference).

## Timing

AEROBLADE's per-image inference time (reconstruction + LPIPS) is recorded for every image (`t_*` columns, per stage) and
the wall-clock of each compute invocation, the model load times and the hardware are in `cache/<…>/runs/*.json` and
`results/*/run_summary.json`. A resample reuses the full run's per-image inference, so its inference time is the sum over
its own images; `localization_pixel_metrics_seconds_sum` and the `*_aggregation_seconds` fields give the evaluation cost.

## Resume and backup

The full pass takes ~21 h on a laptop CPU, so nothing is lost if it stops. Every image is saved as it finishes (a
record line in `cache/<…>/records/*.jsonl`, and the LPIPS map and reconstruction written through a temporary file), so
closing the terminal, killing the process, a reboot or a WSL restart costs at most the image in flight.

* **Continue anytime**: `powershell -ExecutionPolicy Bypass -File scripts\resume.ps1` (Windows; runs inside WSL, keeps the PC
  awake while it works) or `bash scripts/run_all.sh` (Linux / inside WSL). Finished images are skipped, and when every image
  is done the evaluation runs by itself. Running it again after completion just redoes the evaluation.
* **Pause**: Ctrl+C in the terminal, or simply stop the process; continue later with the same command.
* **Backup**: `powershell -File scripts\backup_cache.ps1 -Dest <folder> [-EveryMinutes 30]` copies `cache/` incrementally and
  never deletes anything from the backup. Restore with `robocopy <folder> <this folder>\cache /E`. A backup on the same disk
  guards against deleting or corrupting the cache, not against a dead disk.
* **Several machines**: `--shard i/n` splits the source images; copy the machines' `cache/<…>/` folders into one (file names
  are unique per image, record files are named per host) and run `scripts/evaluate.py`.
* The results themselves (`results/`) are small CSV / JSON files meant to be committed.

## Reproduce

```bash
conda create -n aeroblade python=3.10 && conda activate aeroblade && pip install -r requirements.txt
# data: data/tgif/{orig,sd2-sp,ps-sp,masks}/testing/...   (TGIF downloads, see ../README.md)
python scripts/build_index.py              # index + integrity checks
python scripts/make_resamples.py           # the 3 shareable resamples (already in resamples/)
python scripts/run_aeroblade.py            # AEROBLADE over all images (resumable; --max-sources N, --shard i/n)
python scripts/evaluate.py                 # dataframes + tables for full and resample_1..3
# or all at once: bash scripts/run_all.sh
```

Weights are downloaded on first use: the SD2 VAE (from `sd2-community/stable-diffusion-2-base`, sha256-checked; the
original `stabilityai` repo is no longer public) and VGG16 (LPIPS). The cache (`cache/`, ~3 GB: reconstructions,
LPIPS maps `.npy`, per-image records `.jsonl`) is git-ignored; `--shard i/n` lets several machines share the work and
their `cache/` folders can be merged.
