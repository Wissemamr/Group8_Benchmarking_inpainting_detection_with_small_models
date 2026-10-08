# ObjectFormer on TGIF (testing split): detection and localization

[ObjectFormer](https://github.com/wdrink/ObjectFormer) (Wang et al., *ObjectFormer for Image Manipulation Detection
and Localization*, CVPR 2022) trained by us and evaluated on the TGIF testing split, for both benchmark tasks:


## Why we trained it ourselves, and on what

The authors released no usable weights, and there was no disk space for TGIF's training split. So the model is
trained on part of TGIF **testing** and evaluated on the rest, split **by source image** so that no source image
(and none of its 13 derived images) appears on both sides:

| group | source images | images | authentic / forged |
|---|---:|---:|---:|
| train | 201 | 2 600 | 200 / 2 400 |
| val (monitoring only) | 22 | 273 | 21 / 252 |
| held out (evaluation) | 120 | 1 534 | 118 / 1 416 |

`HELDOUT_FRACTION = 0.35`, `VAL_FRACTION = 0.10`, seed 2026. The images come from the union of the three shared
resamples (4 407 unique images), so the 4 source images that are in no resample are not used. The exact split is in
`results/benchmark/source_split.json`.

| | resample 1 | resample 2 | resample 3 |
|---|---:|---:|---:|
| evaluated images | 1 287 | 1 248 | 1 235 |
| authentic / forged | 99 / 1 188 | 96 / 1 152 | 95 / 1 140 |
| source images | 99 | 96 | 95 |

**So ObjectFormer is scored on a subset of each resample (~35 %), not on the full 3 575 images the other models use.**
Say this whenever its numbers sit next to another model's: *evaluated on the subset of each resample whose source
images were excluded from training.*

## Training

| setting | ours | upstream config (`configs/objectformer_bs24_lr2.5e-4.yaml`) |
|---|---|---|
| initialisation | random (`PRETRAINED=""`) | ImageNet-pretrained backbone |
| input size | 256 × 256 | train 434 resize, test 380 × 380 |
| batch | 8 | 24 |
| optimiser | AdamW, lr 1e-4, weight decay 0.05, cosine schedule, grad clip 5 | lr 2.5e-4 |
| epochs | 3 (975 steps) | `default.yaml`: `MAX_EPOCH: 40` |
| augmentation | horizontal flip | see config |

* Architecture as upstream: `ObjectFormer(MODEL_CFG)` with the repo's layer sizes, **34 413 204 parameters**
  (asserted in the notebook).
* Loss: `L = L_cls + 1.0 · L_seg`. `L_cls` is BCE-with-logits with `pos_weight = 0.0833` (≈ 1 authentic per 12
  forged, to stop the head collapsing to "always forged"). `L_seg` is BCE on each of the three mask heads (already
  sigmoid-activated), averaged.
* Masks binarised at `> 127`; images normalised with ImageNet mean/std.
* No CUDA: the repo's `build_helper.build_model()` asserts CUDA and its BCE loss calls `.cuda()`, so the notebook builds
  `ObjectFormer(MODEL_CFG)` directly and uses PyTorch's own BCE. Nothing in the upstream code is modified.

Validation during training:

| epoch | val ROC-AUC | val IoU |
|---:|---:|---:|
| 1 | 0.4868 | 0.0000 |
| 2 | 0.5045 | 0.0201 |
| 3 | 0.5058 | 0.0326 |

Degeneracy check after training: validation scores span 0.5192–0.5247 (std 0.0013). This passes the notebook's
`std < 1e-3` test, but only just — in practice the detection output is nearly constant and always above 0.5.

**Likely causes**, untested: random initialisation instead of the ImageNet backbone, 3 epochs instead of tens,
2 600 training images, and 256 px input. These are the first things to change for a rerun.

## Evaluation

* **Detection score** = `sigmoid(logit)`; predicted forged if `> 0.5`. ROC-AUC needs no threshold. The summary also gives
  every metric at the best-F1 threshold (`*_bestf1` columns), which is **optimistic** because it is chosen on the same
  images it is scored on.
* **Localization**: the last mask head (256 × 256) is upsampled **bilinearly** to the image's native width × height,
  binarised at 0.5 and compared with the native-resolution ground truth (nearest-neighbour resize only if the sizes
  differ). Scored on forged images only. Headline numbers are the **mean of per-image scores** (macro).
  Per-image pixel F1 equals Dice (`2TP / (2TP + FP + FN)`), so it is not stored separately.
* **Class imbalance**: 1 authentic per 12 forged, so predicting "forged" for everything gives accuracy 0.923, precision
  0.923, recall 1, F1 0.960. Every summary has an `always_forged` row with exactly this baseline. Read ROC-AUC first.

## Results

### Detection (held-out subset, threshold 0.5)

| | resample 1 | resample 2 | resample 3 |
|---|---:|---:|---:|
| **ROC-AUC** | **0.5051** | **0.5046** | **0.5033** |
| accuracy | 0.9231 | 0.9231 | 0.9231 |
| precision | 0.9231 | 0.9231 | 0.9231 |
| recall | 1.0000 | 1.0000 | 1.0000 |
| F1 | 0.9600 | 0.9600 | 0.9600 |
| always-forged baseline F1 | 0.9600 | 0.9600 | 0.9600 |
| beats the baseline? | **no** | **no** | **no** |

TN = 0 and FN = 0 on every resample: no image is predicted authentic.

### Localization (forged images of the held-out subset, threshold 0.5)

| | resample 1 | resample 2 | resample 3 |
|---|---:|---:|---:|
| IoU | 0.0337 | 0.0287 | 0.0304 |
| Dice (= pixel F1) | 0.0548 | 0.0464 | 0.0489 |
| pixel precision | 0.1738 | 0.1409 | 0.1626 |
| pixel recall | 0.0366 | 0.0315 | 0.0336 |

Resample 1 mean IoU, 95 % bootstrap interval over **source images** (2 000 draws, 99 sources): [0.0209, 0.0482].

Ground-truth mask coverage differs by mask type (mean fraction of the image): bbox ≈ 0.075–0.077, segm ≈ 0.039–0.041,
so bbox and segm scores are not directly comparable.

### Across the three resamples

The resamples are overlapping 80 % draws from the same 343 source images, not independent replicates, so we show them
side by side and do not quote a standard deviation as a confidence interval.

* **SD vs PS (IoU)** – the ordering does **not** hold: PS higher on resamples 1 and 2, SD higher on 3.
* **bbox vs segm (IoU)** – the ordering does **not** hold: it flips between resamples.
* All these gaps are ≤ 0.006 IoU, well inside the bootstrap interval above.
* **SD vs PS (detection AUC) cannot be computed from these files.** The `generator=SD` / `generator=PS` slices contain
  forged images only, so their ROC-AUC is `NaN`. The line in `stability.md` reporting this ordering as "HOLDS" is an
  artifact of comparing `NaN` values and should be ignored.

### Efficiency

| parameters | checkpoint (fp32 state dict) | training | inference |
|---:|---:|---:|---:|
| 34 413 204 | 138 MB | 1 210 s for 975 steps (~1.2 s/step) | 92 s for 1 534 images, including native-resolution scoring |

Hardware: Apple M2 MacBook, PyTorch 2.14 on MPS, `NUM_WORKERS = 0`. FLOPs and peak memory were not measured.


## Reproduce

```bash
# 1. upstream code (not modified; tested at commit a967ba9)
git clone https://github.com/wdrink/ObjectFormer ObjectFormer/ObjectFormer
git -C ObjectFormer/ObjectFormer checkout a967ba91a77e79f6d64e5a02e15f4214197ceb61

# 2. environment: Python 3.11. The repo's requirements.txt pins 2022 versions that no longer install,
#    so install only what is needed, unpinned:
pip install torch torchvision timm einops fvcore opencv-python scikit-learn pandas numpy jupyter

# 3. data (not in git): TGIF testing files, same layout as AEROBLADE
#    ObjectFormer/tgif/{orig,sd2-sp,ps-sp,masks}/testing/<category>/...
```

4. Open `results/benchmark/objectformer_tgif_heldout.ipynb`, set `BASE` in the **0 · Config** cell to the absolute
   path of this `ObjectFormer/` folder, and run all cells. It runs on CUDA, MPS or CPU. Keep `NUM_WORKERS = 0` on
   macOS and Windows: their DataLoader workers cannot load the `TGIF` dataset class defined inside the notebook.
5. To shorten a run, set `MAX_TRAIN_IMAGES`.
