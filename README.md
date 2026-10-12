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
* Generalization across datasets
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

## 5. Datasets

The benchmark uses datasets specifically designed for the detection and localization of **AI-generated image inpainting**. The datasets are selected based on their scale, diversity of inpainting methods, availability of manipulation masks, and relevance to recent image-forensics research.

### 5.1 TGIF — Text-Guided Inpainting Forgery Dataset

**Official repository:**
https://github.com/IDLabMedia/tgif-dataset

**Download:**
https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o

**Original paper:**
[TGIF: Text-Guided Inpainting Forgery Dataset](https://arxiv.org/abs/2407.11566)

TGIF (*Text-Guided Inpainting Forgery*) is a large-scale dataset specifically designed for studying the detection and localization of **text-guided image inpainting forgeries**.

The dataset contains authentic images originating from **MS-COCO** and corresponding manipulated images generated using several text-guided inpainting methods, including:

* **Stable Diffusion 2 (SD2)**
* **Stable Diffusion XL (SDXL)**
* **Adobe Firefly**

The authentic images are sourced from **MS-COCO** and are distributed under the **CC BY 4.0** license. Images have resolutions of up to **1024 × 1024 pixels**.

For each manipulation, TGIF can provide information about the manipulated region, making it suitable for both:

1. **Image-level inpainting detection**
2. **Pixel-level inpainting localization**

#### Manipulation Types

TGIF provides two representations of the generated content, when available:

* **SP (Spliced/Partial):** the original image in which only the selected region has been replaced through inpainting.
* **FR (Fully Regenerated):** a fully regenerated version of the image produced through the corresponding generative process.

The **SP** version is particularly relevant to our benchmark because it directly represents the typical inpainting scenario: a localized region of an otherwise authentic image has been synthetically modified.

---

### TGIF2 Extension

The original TGIF dataset was subsequently extended through **TGIF2: Extended Text-Guided Inpainting Forgery Dataset & Benchmark**.

TGIF2 adds approximately **196K new manipulated images**, substantially increasing the scale and diversity of the benchmark.

This extension allows us to investigate whether models trained on a large and diverse collection of inpainting manipulations can generalize across different image-generation methods and manipulation distributions.

---

### Dataset Statistics

The following table summarizes the splits used in our benchmark.

| Dataset                             | Training | Validation | Testing |       Total |
| ----------------------------------- | -------: | ---------: | ------: | ----------: |
| **Authentic images**                |    2,440 |        341 |     343 |   **3,124** |
| **TGIF manipulated images**         |   58,560 |      8,184 |   8,232 |  **74,976** |
| **TGIF2 manipulated images**        |  153,720 |     21,483 |  21,609 | **196,812** |
| **TGIF + TGIF2 manipulated images** |  212,280 |     29,667 |  29,841 | **271,788** |

The resulting benchmark therefore contains a substantial number of manipulated samples, with TGIF and TGIF2 providing complementary manipulation data.

---

### Why TGIF Was Selected

TGIF was selected as one of the benchmark datasets for several reasons:

#### 1. Large-scale dataset

TGIF provides nearly **75K manipulated images**, while the TGIF2 extension adds approximately **197K additional manipulated images**. This makes the combined dataset sufficiently large for training and comparing multiple models.

#### 2. Diversity of generative methods

The dataset contains manipulations generated using different text-guided inpainting systems, including **SD2, SDXL, and Adobe Firefly**. This diversity is important for evaluating whether models learn generalizable traces of inpainting rather than artifacts specific to a single generation method.

#### 3. Relevance to our research question

Unlike datasets focused only on generic image manipulation, TGIF specifically targets **text-guided inpainting forgeries**, which directly matches the scope of our research question.

#### 4. Detection and localization

The dataset provides information about the manipulated regions, allowing us to evaluate both:

* **Detection:** whether an image contains an inpainting manipulation.
* **Localization:** whether the model can identify the manipulated pixels.

This makes TGIF particularly suitable for our two-task benchmark.

#### 5. Recent and established benchmark

TGIF was introduced in a peer-reviewed information-forensics context and subsequently extended through **TGIF2**, providing a recent and substantially larger benchmark for text-guided inpainting forgery detection and localization.



> **Note:** The exact subset, split strategy, preprocessing, and whether TGIF2 is included in the training or evaluation set will be specified in the experimental protocol to ensure a fair comparison between all models.


### SAGI-D

[Dataset description to be added]

For each dataset, we document:

* Number of images
* Authentic/manipulated image distribution
* Availability of ground-truth masks
* Image resolution
* Inpainting generation process
* Train/validation/test splits

The datasets are used under a common evaluation protocol to enable a controlled comparison between the selected models.

---

## 6. Selected Models

We selected models representing different architectural families used for image manipulation detection and localization.

### 6.1 ForMa

**Architecture:** State Space Model (SSM) / Mamba-based architecture

ForMa is based on the Mamba/VMamba family of state-space models and is designed for image manipulation localization.

**Category:** SSM / Mamba-based

---

### 6.2 PSCC-Net

**Architecture:** Convolutional Neural Network (CNN)

PSCC-Net is a CNN-based approach for image manipulation detection and localization.

**Category:** CNN-based

---

### 6.3 SegFormer-B1

**Architecture:** Transformer

SegFormer is a lightweight Transformer-based architecture originally developed for semantic segmentation. The B1 variant is considered a relatively compact configuration.

**Category:** Transformer-based

---

### 6.4 AeroBlade

**Architecture:** Diffusion-based

AeroBlade uses diffusion-model-related representations for detecting synthetic image manipulations.

**Category:** Diffusion-based

---

### 6.5 ObjectFormer

**Architecture:** Frequency-domain / Transformer-based

ObjectFormer exploits frequency-related information for image manipulation analysis.

**Category:** Frequency-based

---

## 7. Experimental Setup

The central principle of our benchmark is to evaluate the selected models under a **controlled and as-consistent-as-possible experimental protocol**.

The general pipeline is:

```text
                 Dataset
                    │
                    ▼
             Preprocessing
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
        ForMa              PSCC-Net
          │                   │
          ├───────┬───────────┤
          │       │           │
          ▼       ▼           ▼
       Detection Localization
          │       │
          ▼       ▼
       Classification  Segmentation
         Metrics         Metrics
          │       │
          └───────┴──────────┐
                             ▼
                    Model Comparison
                             │
                             ▼
                  Performance + Efficiency
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

### 8.2 Generalization

Where possible, models are evaluated both **in-domain** and **across datasets**.

For example:

```text
Training Dataset A
       │
       ├──────────► Test Dataset A
       │
       ├──────────► Test Dataset B
       │
       └──────────► Test Dataset C
```

Cross-dataset evaluation allows us to investigate whether models generalize beyond the specific distribution on which they were trained.

### 8.3 Computational Efficiency

Because the project focuses on **small models**, computational efficiency is also considered.

We record, where applicable:

* Number of parameters
* FLOPs / computational cost
* Model size
* Inference latency
* Peak memory usage

This allows us to study the relationship between model performance and computational requirements rather than considering predictive performance alone.

---

## 9. Reproducibility

To ensure reproducibility, the repository documents:

* Dataset preparation
* Data splits
* Preprocessing
* Training configuration
* Hyperparameters
* Model checkpoints
* Evaluation procedures
* Hardware/software environment
* Random seeds

All reported benchmark results should be generated using the scripts provided in this repository.

---


## 13. Results

The final benchmark will report results using a common format.

### Detection

| Model        | Accuracy | Precision | Recall | F1 | ROC-AUC |
| ------------ | -------: | --------: | -----: | -: | ------: |
| ForMa        |        — |         — |      — |  — |       — |
| PSCC-Net     |        — |         — |      — |  — |       — |
| SegFormer-B1 |        — |         — |      — |  — |       — |
| AeroBlade    |        — |         — |      — |  — |       — |
| ObjectFormer |        — |         — |      — |  — |       — |

### Localization

| Model        | Pixel Precision | Pixel Recall | Pixel F1 | Dice | IoU |
| ------------ | --------------: | -----------: | -------: | ---: | --: |
| ForMa        |               — |            — |        — |    — |   — |
| PSCC-Net     |               — |            — |        — |    — |   — |
| SegFormer-B1 |               — |            — |        — |    — |   — |
| AeroBlade    |               — |            — |        — |    — |   — |
| ObjectFormer |               — |            — |        — |    — |   — |

### Efficiency

| Model        | Parameters | FLOPs | Model Size | Latency | Peak Memory |
| ------------ | ---------: | ----: | ---------: | ------: | ----------: |
| ForMa        |          — |     — |          — |       — |           — |
| PSCC-Net     |          — |     — |          — |       — |           — |
| SegFormer-B1 |          — |     — |          — |       — |           — |
| AeroBlade    |          — |     — |          — |       — |           — |
| ObjectFormer |          — |     — |          — |       — |           — |

---

## 14. Project Scope

This project focuses on **benchmarking and comparative evaluation**, rather than proposing a new model architecture.

The main objective is to establish a controlled experimental framework in which different small architectures can be compared under common conditions.

The final analysis will consider:

* Detection performance
* Localization accuracy
* Cross-dataset generalization
* Computational efficiency
* Performance–efficiency trade-offs

---

## 15. References

The implementation and evaluation of each model are based on the corresponding original publications and official implementations.

References and links to the original papers and repositories will be provided here:

* **ForMa** — [Paper / Official Repository]
* **PSCC-Net** — [Paper / Official Repository]
* **SegFormer** — [Paper / Official Repository]
* **AeroBlade** — [Paper / Official Repository]
* **ObjectFormer** — [Paper / Official Repository]

