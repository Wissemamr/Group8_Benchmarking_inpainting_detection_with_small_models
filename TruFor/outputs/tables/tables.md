### Table 1 — Detection, resample_1

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning) | 0.9603 | 0.9895 | 0.9673 | 0.9782 | 0.9801 |

### Table 2a — Localization (macro), resample_1

| Model | Pixel Precision | Pixel Recall | Pixel F1 | Dice | IoU |
|---|---|---|---|---|---|
| TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning) | 0.9175 | 0.7697 | 0.8181 | 0.8181 | 0.7232 |

### Table 2b — Localization (pooled), resample_1

| Model | Pixel Precision | Pixel Recall | Pixel F1 | Dice | IoU |
|---|---|---|---|---|---|
| TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning) | 0.9326 | 0.8221 | 0.8739 | 0.8739 | 0.7760 |

### Table 1 — Detection, resample_2

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning) | 0.9527 | 0.9891 | 0.9594 | 0.9740 | 0.9759 |

### Table 2a — Localization (macro), resample_2

| Model | Pixel Precision | Pixel Recall | Pixel F1 | Dice | IoU |
|---|---|---|---|---|---|
| TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning) | 0.9167 | 0.7675 | 0.8143 | 0.8143 | 0.7198 |

### Table 2b — Localization (pooled), resample_2

| Model | Pixel Precision | Pixel Recall | Pixel F1 | Dice | IoU |
|---|---|---|---|---|---|
| TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning) | 0.9310 | 0.8213 | 0.8727 | 0.8727 | 0.7742 |

### Table 1 — Detection, resample_3

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning) | 0.9516 | 0.9878 | 0.9594 | 0.9734 | 0.9749 |

### Table 2a — Localization (macro), resample_3

| Model | Pixel Precision | Pixel Recall | Pixel F1 | Dice | IoU |
|---|---|---|---|---|---|
| TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning) | 0.9191 | 0.7662 | 0.8163 | 0.8163 | 0.7215 |

### Table 2b — Localization (pooled), resample_3

| Model | Pixel Precision | Pixel Recall | Pixel F1 | Dice | IoU |
|---|---|---|---|---|---|
| TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning) | 0.9352 | 0.8207 | 0.8742 | 0.8742 | 0.7765 |

### Table 1 — Detection, mean ± std over 3 resamples

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning) | 0.9549 ± 0.0047 | 0.9888 ± 0.0008 | 0.9620 ± 0.0045 | 0.9752 ± 0.0026 | 0.9770 ± 0.0028 |

### Table 2a — Localization (macro), mean ± std over 3 resamples

| Model | Pixel Precision | Pixel Recall | Pixel F1 | Dice | IoU |
|---|---|---|---|---|---|
| TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning) | 0.9178 ± 0.0013 | 0.7678 ± 0.0018 | 0.8162 ± 0.0019 | 0.8162 ± 0.0019 | 0.7215 ± 0.0017 |

### Table 2b — Localization (pooled), mean ± std over 3 resamples

| Model | Pixel Precision | Pixel Recall | Pixel F1 | Dice | IoU |
|---|---|---|---|---|---|
| TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning) | 0.9329 ± 0.0021 | 0.8214 ± 0.0007 | 0.8736 ± 0.0008 | 0.8736 ± 0.0008 | 0.7756 ± 0.0012 |

### DIAGNOSTIC — official TruFor protocol (not comparable with other models)

| Resample | F1 @0.5, best of map/inverted, boundary excluded (macro) | F1 @ oracle per-image threshold, best of map/inverted, boundary excluded (macro) | NaN images skipped (th / best) |
|---|---|---|---|
| resample_1 | 0.8652 | 0.9085 | 10 / 0 |
| resample_2 | 0.8654 | 0.9104 | 11 / 0 |
| resample_3 | 0.8646 | 0.9122 | 1 / 0 |
| mean ± std | 0.8651 ± 0.0004 | 0.9104 ± 0.0019 |  |
