# Group 8 — Benchmarking Small Models for Inpainting Detection and Localization

**Data Readiness and AI Evaluation — Course Project**


``Pr. Adrian Popescu and Dr. Eva Feillet``

## Collaborators

Wissem AMAR · Hasti HOSSEINPOUR · Dominika CHOJNACKA · Youssef EL OTMANI · Sathya GNANOU

---

## 1. Motivation

Recent research proposes a growing number of models for detecting and localizing AI-generated image manipulations, including inpainting. Several studies report strong performance for relatively lightweight architectures and describe their proposed model as an effective or efficient solution.

However, direct comparison across these methods is difficult because existing studies often use **different datasets, data splits, preprocessing pipelines, training protocols, and evaluation metrics**.

As a result, performance reported in separate papers cannot necessarily be compared directly.

This project addresses this issue through a **controlled benchmark of small AI models for synthetic image inpainting detection and localization**.

Our objective is to evaluate several architectures under a common experimental protocol and study their trade-offs in terms of:

* Detection performance
* Localization performance
* Performance across the two inpainting generators
* Computational efficiency

### Research Question

> **Which small AI model provides the most favorable trade-off between detection and localization performance, generalization, and computational efficiency for synthetic images generated through inpainting?**

---

## 2. What is Image Inpainting?

**Image inpainting** is the process of filling or reconstructing a missing, removed, or intentionally masked region of an image.

In the context of generative AI, inpainting models can modify an existing image by replacing a selected region with newly generated content while attempting to preserve visual consistency with the surrounding image.

For example:

```text
Original image
      │
      ▼
Selected region / mask
      │
      ▼
Generative inpainting model
      │
      ▼
Inpainted image
```

The resulting image may appear visually realistic while containing subtle traces of manipulation.

Our benchmark focuses on identifying these synthetic inpainting manipulations.

---

## 3. Tasks

Our benchmark considers two complementary tasks.

### 3.1 Inpainting Detection

**Detection** is an image-level binary classification task:

> **Has this image been manipulated using inpainting?**

The model produces an image-level prediction indicating whether the image is:

* **Authentic**
* **Inpainted / manipulated**

Detection therefore evaluates whether a model can identify the presence of an inpainting manipulation.

### 3.2 Inpainting Localization

**Localization** is a pixel-level prediction task:

> **Which region of the image has been modified through inpainting?**

The model produces a predicted manipulation mask, which is compared with the ground-truth inpainting mask.

```text
Input image
     │
     ├──────────────► Detection
     │                Authentic / Inpainted
     │
     └──────────────► Localization
                      Predicted manipulation mask
```

Localization is therefore more fine-grained than detection: a model may correctly identify that an image has been manipulated while still failing to accurately identify the manipulated region.

---

## 4. Evaluation Metrics

We evaluate the models separately for **detection** and **localization**.

### 4.1 Detection

The following standard classification metrics are used:

1. **Accuracy**
2. **Precision**
3. **Recall**
4. **F1-score**
5. **ROC-AUC**

These metrics evaluate the ability of each model to distinguish authentic images from inpainted images.

### 4.2 Localization

Localization performance is evaluated at the pixel level using:

1.  **Intersection over Union (IoU)**
2. **Pixel-wise Precision**
3. **Pixel-wise Recall**
4. **Pixel-wise F1-score**
5. **Dice Score**

The predicted localization mask is compared against the ground-truth manipulation mask.

---

## 5. Dataset: TGIF testing split

The benchmark evaluates **TGIF (Text-Guided Inpainting Forgery)** testing images: authentic MS-COCO originals and their inpainted versions from **Photoshop / Firefly (PS-sp)** and **Stable Diffusion 2 (SD2-sp)**. The SP images preserve the original content outside the selected inpainting region. Ground-truth masks identify the manipulated pixels.

* [Official TGIF repository](https://github.com/IDLabMedia/tgif-dataset)
* [Dataset download](https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o)
* [TGIF paper](https://arxiv.org/abs/2407.11566)

### Evaluated subset

The full evaluation inventory contains **4,459 images across 13 categories**, derived from **343 source images**:

| Subset | Images | Detection label |
| --- | ---: | --- |
| Authentic originals | 343 | Authentic (0) |
| Photoshop / Firefly (PS-sp) | 2,058 | Inpainted (1) |
| Stable Diffusion 2 (SD2-sp) | 2,058 | Inpainted (1) |
| **Total** | **4,459** | **343 authentic + 4,116 inpainted** |

Each generator contributes two mask types (bounding box and segmentation) and three variants per source image. Authentic originals are counted once in the combined inventory. Localization is evaluated on manipulated images with ground-truth masks; the full manipulated subset contains **4,116 images**.

Categories: airplane, bear, elephant, hair-drier, kite, scissors, sheep, skis, snowboard, sports-ball, stop-sign, tennis-racket, and zebra.

The tables below reproduce the supplied project summary. Model-specific resampling, authentic-image selection, thresholds, and metric aggregation differ; see the results notes before interpreting them as a controlled comparison.

---

## 6. Selected Models

The final comparison includes **four models: TruFor, PSCC-Net, ForMa, and AEROBLADE**.

| Model | Approach and role | Implementation and evaluation details |
| --- | --- | --- |
| **TruFor** | Pretrained image-forensics model producing an image-level detection score and a pixel-level anomaly map. A larger reference model (68.7 M parameters), outside the project's <10 M small-model criterion. | [TruFor documentation](TruFor/README.md) |
| **PSCC-Net** | CNN-based manipulation detection and localization; 3.67 M parameters. Official pretrained weights evaluated without TGIF fine-tuning. | [PSCC-Net documentation](PSCC-Net/README.md) |
| **ForMa** | Mamba/VMamba state-space model for manipulation localization. Binary detection is derived from the fraction of pixels predicted as manipulated. | [ForMa methodology](ForMa/forma.md) and [usage](ForMa/README.md) |
| **AEROBLADE** | Training-free detection using autoencoder reconstruction error. This benchmark uses the SD2 autoencoder and a spatial error map for localization. | [AEROBLADE documentation](AEROBLADE/README.md) |

---

## 7. Experimental Setup

The central principle of our benchmark is to evaluate the selected models under a **controlled and as-consistent-as-possible experimental protocol**.

The general pipeline is:

```text
TGIF testing images (authentic + PS-sp + SD2-sp)
                        |
             Model-specific preprocessing
                        |
       TruFor / PSCC-Net / ForMa / AEROBLADE
                        |
          Detection scores + localization maps
                        |
          Classification + pixel-level metrics
                        |
                 Model comparison
```

Whenever possible, models are evaluated using:

* The same datasets
* The same data splits
* The same preprocessing
* The same input resolution
* The same evaluation metrics
* The same test sets
* The same computational environment

Model-specific requirements are documented when they cannot be standardized.

---

## 8. Experimental Dimensions

The benchmark evaluates models along three main dimensions.

### 8.1 Detection and Localization Performance

We compare:

* Detection: F1, ROC-AUC, Precision, Recall, Accuracy
* Localization: IoU, Dice, Pixel-wise F1, Precision, Recall

### 8.2 Generator, Category, and Mask-Type Analysis

The repository reports performance across Photoshop / Firefly and Stable Diffusion 2, the 13 object categories, and bounding-box versus segmentation masks. Model-specific documentation and result files describe these breakdowns and any resampling used.

### 8.3 Computational Efficiency

Because the project focuses on **small models**, computational efficiency is also considered.

We record, where applicable:

* Number of parameters
* FLOPs / computational cost
* Throughput
* Inference latency
* Peak memory usage

This allows us to study the relationship between model performance and computational requirements rather than considering predictive performance alone.

---

## 9. Reproducibility

To ensure reproducibility, the repository documents:

* Dataset preparation
* Data splits
* Preprocessing
* Hyperparameters
* Model checkpoints
* Evaluation procedures
* Hardware/software environment
* Random seeds

All reported benchmark results should be generated using the scripts provided in this repository.

---


## 10. Results

The following values reproduce the supplied project summary on a 0-1 scale, combining categories and both generators. They are descriptive point estimates; confidence intervals are not supplied here. The full test inventory has 4,459 images, but individual reported runs and aggregations differ as detailed below.

### Detection

Positive class: **inpainted**. Detection requires both authentic and manipulated images.

| Metric | TruFor | PSCC-Net | ForMa | AEROBLADE | Always "fake" baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 0.95 | 0.66 | 0.77 | 0.57 | 0.92 |
| Precision | 0.98 | 0.86 | 0.93 | 0.92 | 0.92 |
| Recall | 0.96 | 0.69 | 0.82 | 0.59 | 1.00 |
| F1 | 0.97 | 0.76 | 0.87 | 0.70 | 0.96 |
| ROC-AUC | 0.97 | 0.61 | 0.48 | 0.51 | 0.50 |

The full inventory is **92.31% manipulated**. Predicting "fake" for every image therefore achieves approximately 0.92 accuracy and 0.96 F1 without distinguishing authentic images. Interpret precision and F1 alongside ROC-AUC and, where available, specificity and balanced accuracy.

### Localization

Pixel-level scores compare predicted manipulation masks with the ground truth. Authentic images are excluded from the localization evaluations described in the model documentation.

| Metric | TruFor | PSCC-Net | ForMa | AEROBLADE | Coin-flip baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| IoU | 0.72 | 0.33 | 0.21 | 0.07 | 0.06 |
| Dice | 0.87 | 0.41 | 0.29 | 0.13 | 0.10 |
| Pixel precision | 0.93 | 0.46 | 0.41 | 0.08 | 0.06 |
| Pixel recall | 0.82 | 0.48 | 0.30 | 0.70 | 0.50 |

The coin-flip figures are the supplied comparison baseline, not a fifth model. Its sampling procedure and random seed are not specified in the supplied summary.

### Traceability and comparison limits

These tables preserve the supplied presentation values. They should not be described as identical-protocol, full-dataset macro results:

* **TruFor:** its [documented results](TruFor/README.md) are means over three shared resamples, each with 3,575 detection images and 3,300 manipulated localization images. The supplied localization row combines macro IoU (0.7215) with pooled Dice (0.8736), precision (0.9329), and recall (0.8214). The consistent macro alternative is IoU **0.7215**, Dice **0.8162**, precision **0.9178**, and recall **0.7678**.
* **PSCC-Net:** the supplied detection precision, F1, and AUC correspond approximately to the documented protocol including original, 512-pixel, and 1024-pixel authentic versions. With original authentic images only, its three-resample means are accuracy **0.6566**, precision **0.9243**, recall **0.6840**, F1 **0.7862**, and ROC-AUC **0.4976**. See the [protocol comparison](PSCC-Net/README.md). Localization uses a 0.2 mask threshold and macro averaging.
* **ForMa:** detection uses all **4,459 images**, a pixel probability threshold strictly greater than 0.5, and an image manipulated-area threshold strictly greater than 0.01. The stored detection precision is **0.9227** (0.92 rounded to two decimals), whereas the supplied summary states **0.93**. Localization averages per-image metrics over **4,116 manipulated images**. See [ForMa methodology and results](ForMa/forma.md).
* **AEROBLADE:** the benchmark uses the SD2 autoencoder only, with detection threshold calibration by source-level cross-fitting for the full run. Localization headline metrics are per-image macro averages. See [AEROBLADE protocol](AEROBLADE/README.md).

A final standardized ranking requires the same image IDs, authentic-image selection, aggregation, and threshold-selection policy for every model. The supplied summary values above have not been recomputed to impose that protocol.

### Efficiency

The following measurements reproduce the supplied GPU efficiency summary. Latency is reported per image, with both median and mean values. Full-test durations are estimates for **4,459 images**, not measured end-to-end completion times.

| Model | Parameters | GFLOPs / image | GPU latency / image, median (mean) | Throughput | Peak RAM | Peak GPU memory | Estimated full TGIF test (4,459 images) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AEROBLADE | 98.4 M | 3,841 | 891 ms (833 ms) | 1.20 images/s | 1,160 MiB | 1,351 MiB | ~62 min |
| TruFor | 68.7 M | 1,255 | 872 ms (836 ms) | 1.20 images/s | 992 MiB | 2,395 MiB | ~62 min |
| ForMa | 37.3 M | Not reported | 11,628 ms (11,659 ms) | 0.09 images/s | 1,679 MiB | 446 MiB | ~14.4 h |
| PSCC-Net | 3.7 M | 542 | 856 ms (897 ms) | 1.12 images/s | 2,796 MiB | 2,527 MiB | ~67 min |

Estimated total time is `4,459 * mean latency per image`. Throughput is rounded, so dividing the image count by the displayed throughput can produce a different estimate. ForMa's GFLOPs were not supplied. The summary does not specify GPU model, batch size, warm-up, or timing boundaries; consult the model-specific profiling records before treating these measurements as a hardware-controlled speed ranking.

---

## 11. Project Scope

This project focuses on **benchmarking and comparative evaluation**, rather than proposing a new model architecture.

The main objective is to establish a controlled experimental framework in which different small architectures can be compared under common conditions.

The reported comparison covers:

* Detection performance
* Localization accuracy
* Performance across Photoshop / Firefly and Stable Diffusion 2
* Computational efficiency
* Performance–efficiency trade-offs

---

## 12. References

The implementation and evaluation of each model are based on the corresponding original publications and official implementations.

Model-specific documentation links to the original publications, implementations, and evaluation procedures:

* **TruFor** - [model documentation](TruFor/README.md)
* **PSCC-Net** - [model documentation](PSCC-Net/README.md)
* **ForMa** - [model documentation](ForMa/README.md), [evaluation methodology](ForMa/forma.md)
* **AEROBLADE** - [model documentation](AEROBLADE/README.md)
