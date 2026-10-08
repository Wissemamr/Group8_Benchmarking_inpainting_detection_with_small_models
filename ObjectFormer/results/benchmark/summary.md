# ObjectFormer on TGIF — summary

Trained by us on 2600 images from 201 source images; evaluated on the subset of each resample whose source images were excluded from training (120 held-out sources).
Input 256px, 3 epochs, AdamW lr 0.0001, pos_weight 0.083, 34,413,204 parameters.

**Resample 1** (1287 images, 1188 forged / 99 authentic)
- detection ROC-AUC 0.5051; accuracy 0.9231, F1 0.9600 at 0.5
- always-forged baseline: accuracy 0.9231, F1 0.9600 -> **does not beat it — not a result**
- localization mean IoU 0.0337, Dice 0.0548, pixel precision 0.1738, recall 0.0366

**Resample 2** (1248 images, 1152 forged / 96 authentic)
- detection ROC-AUC 0.5046; accuracy 0.9231, F1 0.9600 at 0.5
- always-forged baseline: accuracy 0.9231, F1 0.9600 -> **does not beat it — not a result**
- localization mean IoU 0.0287, Dice 0.0464, pixel precision 0.1409, recall 0.0315

**Resample 3** (1235 images, 1140 forged / 95 authentic)
- detection ROC-AUC 0.5033; accuracy 0.9231, F1 0.9600 at 0.5
- always-forged baseline: accuracy 0.9231, F1 0.9600 -> **does not beat it — not a result**
- localization mean IoU 0.0304, Dice 0.0489, pixel precision 0.1626, recall 0.0336

ROC-AUC is the number to lead with: the classes are about 12 forged per authentic image, so accuracy and F1 are inflated by the imbalance.

Caveat to state: the authors released no usable weights, so this is ObjectFormer trained by us, not the published model. The training and evaluation images share no source image, so nothing leaks, but the evaluation subset is smaller than the full resample.