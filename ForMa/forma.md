# Documentation of the ForMa model testing on the TGIF Dataset


## Original Paper
@article{guo2025light,
  title={A Lightweight and Effective Image Tampering Localization Network with Vision Mamba},
  author={Guo, Kun and Cao Gang and Lou, Zijie and Huang, Xianglin and Liu, Jiaoyun},
  journal={IEEE Signal Processing Letters},
  year={2025}
}


## Methodology

                  OFFICIAL FORMA
                pretrained weights
                       │
                       ▼
              ┌─────────────────┐
              │   TGIF image    │
              └────────┬────────┘
                       │
                ForMa preprocessing
                       │
                       ▼
                ForMa inference
                       │
                       ▼
              Predicted pixel mask
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Ground-truth TGIF      Prediction
             mask               mask
             │                   │
             └─────────┬─────────┘
                       ▼
                 IoU / Dice /
                 Pixel F1 etc.  


## How
We need our images to be pairs of an image and a mask, the mask is a bianry iamge localsiiing the inpainted area


In our case, for ForMa, our experiment is : **Zero-shot / out-of-distribution evaluation of ForMa on TGIF**


We want to answer the question
How well does the publicly released ForMa model, using the authors' pretrained checkpoint, generalize to the TGIF text-guided inpainting dataset?



- We downlaod the model weihgts : https://drive.google.com/file/d/15NcJQr5kxhqC4V-WjyJhAqIsobI6u-yh/view
- We check the oriignal forma repo and identify the minimum files required for inference and add the necessary dependencies ( we manage dependepncies with `uv`)


```
(Group8_Benchmarking_inpainting_detection_with_small_models) PS C:\Users\PC CABA DZ\DRAIEVAL\Group8_Benchmarking_inpainting_detection_with_small_models> uv pip list
Package            Version
------------------ ------------
certifi            2022.12.7
charset-normalizer 2.1.1
idna               3.4
numpy              2.2.6
pillow             12.3.0
requests           2.28.1
torch              1.13.1+cu117
torchvision        0.14.1+cu117
typing-extensions  4.16.0
urllib3            1.26.13


```
## Single-image inference and localization metrics

From the `ForMa` directory with the project virtual environment active:

```powershell
python inference.py --image "data_tgif/image.png" --mask "data_tgif/mask.png" --output-dir "results/image"
```

Replace both paths with an actual corresponding image/mask pair. Omit `--mask`
for prediction only; scores will be `null`, not fabricated. The ground-truth
mask must have the original image dimensions and use 0/1 or 0/255, with white
meaning manipulated. Soft masks are rejected. Do not invert mask labels based
on whichever orientation gives a better score.

The checkpoint defaults to `ForMa_weights/ForMa_weights.pth`. Use `--weights`
to override it. `--device auto` selects CUDA when available, otherwise CPU.
Use `--device cpu` explicitly if needed. `--input-size 512` and
`--threshold 0.5` are the defaults. Smaller sizes (multiples of 32) are useful
for debugging but change the evaluation protocol.

To instantiate and run from Python while in `ForMa`:

```python
from inference import load_model, preprocess_image, run_inference, prediction_maps

model = load_model(device="cpu")  # eval mode, strict checkpoint loading
image_tensor, original = preprocess_image("data_tgif/image.png")
logits = run_inference(model, image_tensor)
probability, binary_prediction = prediction_maps(logits, original.size)
```

The full checkpoint already contains Noiseprint weights. The inference loader
passes `noiseprint_path=None` to avoid requiring a separate `np++.pth`, then
loads all parameters with `strict=True`. Missing weights cause an error rather
than silently benchmarking a partly random model. Use trusted checkpoint files:
the released training checkpoint needs the legacy full-checkpoint loader in
this PyTorch environment.

Preprocessing follows the authors' [dataset loader](https://github.com/multimediaFor/ForMa/blob/main/mydatasets.py):
OpenCV RGB, linear square resize, float32 scaled to [0, 1], no ImageNet normalization.
Binary output follows their [test script](https://github.com/multimediaFor/ForMa/blob/main/test.py):
class 1 is foreground; at the default threshold take the two-class decision
(ties are background) and nearest-resize to the original image resolution.
Scores here are calculated on that saved full-resolution binary mask; this is
an explicit benchmark protocol, not a claim of reproducing every paper score.

The implementation uses OpenCV for matching image resizing. If setting up
another environment with the existing PyTorch 1.13 installation:

```powershell
uv pip install "numpy<2" "opencv-python-headless<4.12" einops
```

Each output directory contains:

- `prediction.png`: binary localization, 0 background / 255 manipulated.
- `overlay.png`: predicted pixels highlighted in red.
- `probabilities.npy`: float32 foreground probabilities at original resolution.
- `probabilities.png`: 8-bit probability visualization; use the NPY for analysis.
- `metrics.json`: scores, confusion counts, paths and SHA-256 hashes, sizes,
  threshold, device, available acceleration backends, versions and forward time.

`metrics.py` defines foreground metrics with TP, FP, FN and TN counted per pixel:

| Metric | Definition |
| --- | --- |
| IoU | TP / (TP + FP + FN) |
| Dice / pixel F1 | 2 TP / (2 TP + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| Pixel accuracy | (TP + TN) / all pixels |

Dice and pixel F1 are the same metric for binary masks. A zero denominator
returns `None` / JSON `null`, including IoU and Dice when both masks are empty.
An empty prediction against a nonempty target scores 0 for IoU and Dice.
A false positive on an authentic image likewise scores 0. Report the number
of defined values when averaging; do not silently replace null scores with 1.
Pixel accuracy can be high for a poor localization model when background dominates.

For a dataset benchmark, retain one report per image in a distinct output
folder. Macro scores average defined per-image values; micro scores sum TP,
FP, FN and TN across images and call `metrics_from_counts`. Keep resolution,
threshold, mask conventions and preprocessing identical across models. Choose
thresholds using validation data, not the TGIF test masks. A single cold forward
time is a smoke-test measurement, not a stable throughput benchmark; use repeated
warmed runs on the same hardware/backend for speed comparisons.

Validation from the project root:

```powershell
python -m unittest discover -s ForMa/tests -v
```

The initial local end-to-end check used the released checkpoint on a synthetic
image/mask at 512x512 input, on CPU. Outputs are under `results/synthetic_smoke`.
This verifies execution and output generation only; its localization scores
are not TGIF results. The `data_tgif` directory was empty at implementation time.

## Exploring the downloaded TGIF Photoshop-spliced test subset

The local download now has this layout:

```text
data_tgif/
  ps-sp_testing/testing/<category>/<id>_mask_<type>.png_ps_<variant>.png
  masks_testing/testing/<category>/<id>_mask_<type>.png_ps_mask.png
```

Inventory: 2,058 manipulated images and 3,428 mask files across 13 categories.
The mask count is larger because several mask/crop representations are provided.
The sheep category contains 168 images: 28 source IDs x 2 mask types x 3 variants.
All 2,058 PS-sp image files have an adapted mask with matching dimensions.
This validates local pairing; it does not independently certify archive completeness.

According to the [TGIF dataset documentation](https://github.com/IDLabMedia/tgif-dataset),
`PS-sp` identifies Photoshop/Firefly inpainting followed by splicing, `bbox` is
a bounding-box mask, `segm` is an object-segmentation mask, and variants `0`, `1`,
`2` are separate generated versions. The category and source ID identify the
original object/scene; variants of the same scene are correlated samples.

For example, pair:

```text
ps-sp_testing/testing/sheep/154947_mask_bbox.png_ps_0.png
masks_testing/testing/sheep/154947_mask_bbox.png_ps_mask.png
```

Use the Photoshop-adapted mask, not just `154947_mask_bbox.png`: the adapted
mask includes a blended border. Local adapted masks contain intermediate gray
values. Our sample evaluator binarizes them with `value > 127`, explicitly
recorded in the report. White is manipulated. This is a declared evaluation
choice for the soft border, not a claim of exact equivalence to the TGIF paper's
complete evaluation protocol. Keep this choice fixed in subsequent comparisons.
The generic `_mask_512.png` / `_mask_1024.png` and crop-specific files are not
interchangeable with the adapted splicing mask; for this source, `_mask_1024.png`
is entirely white and would give an incorrect localization target.

Run the initial six-image sample from `ForMa`:

```powershell
python evaluate_tgif_sample.py --category sheep --source-id 154947
```

It loads the model once, discards one warmup, then evaluates all six variants
(two mask types x three generations) at 512x512 input. Original-resolution masks
are scored after nearest resizing of the probability map, at fixed prediction
threshold 0.5. It saves `results/sheep_sample/report.html` with side-by-side
input, ground truth and prediction, plus `report.json`, `samples.csv`, and
`inventory.json`. Each image also has its own scores and float probability map.

The JSON separates model-loading time, warmup, forward-pass time, per-image
processing time, and total elapsed time before final report writing. Macro
scores average images; micro scores pool confusion counts. This is a one-scene
pilot, not a representative sheep-category or full TGIF benchmark. Do not split
generated siblings of the same source ID between training and evaluation.

The current environment was recreated as Python 3.12 and required installing
its inference dependencies again. Runtime versions are recorded in the report;
its timings are not directly comparable with the earlier Python 3.10 / PyTorch
1.13 synthetic check.
