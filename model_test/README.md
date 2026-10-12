# Four-model GPU test: AEROBLADE, TruFor, ForMa, PSCC-Net

A small, self-contained test of the group's four models on **65 TGIF images**, run on one GPU machine
with detailed compute profiling: latency per stage, throughput, FLOPs, parameters, process RAM,
PyTorch GPU memory, whole-GPU VRAM / utilization / power / energy (`nvidia-smi`), and a per-operator
kernel profile (`torch.profiler`). Detection and localization quality are measured too, but on 65
images they are only indicative.

The test images are **committed in `testset/`** (63 MB) so nothing else has to be downloaded for
the data. Remove that folder once the test is done.

## What is in the test set

One source photo per TGIF category (13), drawn with a fixed seed. From each: the authentic original,
the SD2-sp bbox and segm inpaintings, the PS-sp bbox and segm inpaintings (generation variant 0) and
their ground-truth masks, in TGIF's folder layout, listed in `testset/manifest.csv`.
Regenerate with `python model_test/make_testset.py --tgif-root <folder with orig/ sd2-sp/ ps-sp/ masks/>`.

## Run it on the GPU machine (Windows, Anaconda Prompt)

```bat
git clone https://github.com/Wissemamr/Group8_Benchmarking_inpainting_detection_with_small_models
cd Group8_Benchmarking_inpainting_detection_with_small_models
git checkout model-test-gpu

:: 1. code and weights of the third-party models (TruFor 261 MB, ForMa ~0.4 GB; needs git)
python model_test\setup_models.py

:: 2. pick / prepare one conda env per model (prints the plan; --apply carries it out)
python model_test\check_envs.py
python model_test\check_envs.py --apply

:: 3. run the 4 models, then the report (a few minutes on a GPU)
python model_test\run_all.py
```

The report is `model_test/outputs/report/REPORT.md` (with figures); per model,
`model_test/outputs/<model>/` holds `per_image.csv`, `profile.json`, `torch_ops.txt`, `gpu_samples.csv`
and map previews. Commit and push `model_test/outputs/` to bring the results back
(the chrome traces `trace.json.gz` are git-ignored because of their size).

Quick check before the full run: `python model_test\run_all.py --limit 5`.
One model only: `python model_test\run_all.py --models trufor`.

### Environments

`check_envs.py` probes every conda env (Python, PyTorch, CUDA working, packages) and, per model:

1. uses the preferred env if it already has everything: `Aeroblade` for AEROBLADE (created from
   `AEROBLADE/requirements-gpu.txt`), `pcss` for PSCC-Net;
2. otherwise uses any env that already has everything, with a working CUDA PyTorch;
3. otherwise **adds** the few missing small packages to an env that has a working CUDA PyTorch
   (TruFor needs `timm yacs opencv`; ForMa needs `timm fvcore pyyaml opencv einops`). Only *missing* packages are
   installed (pinned: `timm==0.5.4`, `yacs==0.1.8`, `fvcore==0.1.5.post20221221`,
   `opencv-python-headless==4.10.0.84`, `einops==0.8.0`), never upgraded or downgraded, so existing projects keep working.
   The `Aeroblade` env is preferred for this, so friends' envs are left alone;
4. otherwise **creates** `<model>-gpu` (Python 3.10, PyTorch 2.1.2 + CUDA 12.1): ~2.5 GB, 10–20 min.

The choice is saved in `model_test/envs.json`. An RTX 50-series card needs a newer PyTorch than 2.1.2;
`check_envs.py` warns, see `AEROBLADE/README.md`, "Run on a GPU".

### PSCC-Net

The runner uses an existing PSCC-Net folder with its code and checkpoints: the `PSCC_ROOT` environment
variable, `--pscc-root`, a `PSCC-TGIF` / `PSCC-Net` folder inside or next to the repo, or
`~/projects/PSCC-TGIF`. On the GPU laptop that is `C:\Users\Asus\projects\PSCC-TGIF`, found
automatically. If none exists, `setup_models.py` clones the official repository (its checkpoints are in
git). Two compatibility shims, both without changing results: `np.int` is restored when a recent numpy
removed it (the official HRNet code uses it), and HRNet's ImageNet initialization weights (overwritten by
the PSCC checkpoints) are read into CPU memory.

## How each model is run

| model | input | detection score (predicted inpainted if >) | localization (binary) |
|---|---|---|---|
| AEROBLADE | resized to 512x512 | −AEROBLADE distance (> −0.0521, the full TGIF run's cross-fitted Youden point) | 1 − normalized LPIPS error, Otsu |
| TruFor | full resolution | official integrity score (> 0.5) | softmax map > 0.5 |
| ForMa | resized to 512x512 | **derived**: mean of the top 1 % of its map (> 0.5); ForMa has no detection head | softmax map > 0.5 |
| PSCC-Net | full resolution | detection head P(forged) (> 0.5, = official argmax) | sigmoid map > 0.5 |

Each follows the group's own benchmark code for that model (`AEROBLADE/`, `TruFor/`, `ForMa/`) or the
authors' `test.py` (PSCC-Net). ForMa's selective scan uses the fastest backend importable in its env
(CUDA kernels, Triton, or plain PyTorch); the one used is in the report, as it changes the GPU time a lot.

## What is measured, and how

| quantity | how |
|---|---|
| latency | per image, batch 1, after `--warmup` images (default 3); `torch.cuda.synchronize()` around every stage (preprocess / forward / postprocess); median, p95, min, max |
| throughput | images / total measured latency |
| parameters | sum over the model's modules |
| FLOPs | `torch.utils.flop_counter` on the forward stages of every image: 2 FLOPs per multiply-add of matmuls, convolutions and attention (element-wise ops and custom kernels are not counted); per stage and per operator in `profile.json` |
| process RAM | resident memory sampled every 50 ms, peak per phase (load, measure) and peak of the process |
| PyTorch GPU memory | `torch.cuda.max_memory_allocated / reserved`, reset before every image |
| whole-GPU VRAM, utilization, power, SM clock, temperature | `nvidia-smi` every 100 ms (`gpu_samples.csv`); energy = power integrated over the measured pass |
| operator profile | `torch.profiler` (CPU + CUDA activities, shapes, memory, FLOPs) over `--profile-images` images (default 3): `torch_ops.txt` / `.csv`, chrome trace `trace.json.gz` (open in `chrome://tracing` or ui.perfetto.dev) |
| quality | detection ROC-AUC / AP / accuracy (13 authentic vs 52 inpainted); localization F1 / IoU at each model's own binarization, and pixel ROC-AUC / AP |

Tested before the GPU run: AEROBLADE, TruFor and PSCC-Net ran end to end on a CPU (WSL, PyTorch
2.1.2, 3 images each, all outputs written). ForMa was **not** run to completion (too slow on a CPU),
so try `python model_test\run_all.py --models forma --limit 2` first. The GPU-only parts (CUDA timings,
`nvidia-smi` sampling, cuDNN settings) only activate on a GPU and were not tested.
