# Cross-resample stability

```
resample  det_auc  det_f1      iou     dice  auc_SD   iou_SD  auc_PS   iou_PS  iou_bbox  iou_segm
       1 0.505085    0.96 0.033723 0.054843     NaN 0.031062     NaN 0.036384  0.032644  0.034803
       2 0.504643    0.96 0.028694 0.046403     NaN 0.026328     NaN 0.031061  0.029578  0.027811
       3 0.503319    0.96 0.030357 0.048949     NaN 0.030791     NaN 0.029923  0.029988  0.030726
```

- IoU SD vs IoU PS: ordering DOES NOT HOLD in all three (0.0311 vs 0.0364, 0.0263 vs 0.0311, 0.0308 vs 0.0299)
- IoU bbox vs IoU segm: ordering DOES NOT HOLD in all three (0.0326 vs 0.0348, 0.0296 vs 0.0278, 0.0300 vs 0.0307)
- AUC SD vs AUC PS: ordering HOLDS in all three (nan vs nan, nan vs nan, nan vs nan)

- Resample 1 mean IoU 0.0337, 95% CI [0.0209, 0.0482] (2000 bootstraps resampled by source_id, n=99 sources)