"""Derive manipulated-only image detection from saved ForMa pixel decisions.

The score is predicted manipulated-area fraction, not probability of forgery.
No inference, mask loading or test-set threshold optimization is performed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def detection_scores(frame):
    required = {'subset', 'evaluation_id', 'category', 'source_id', 'mask_type', 'variant',
                'image_name', 'image_path', 'tp', 'fp', 'fn', 'tn', 'width', 'height',
                'checkpoint_sha256', 'input_size', 'prediction_threshold',
                'ground_truth_threshold', 'postprocessing'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing workbook columns: {sorted(missing)}')
    frame = frame.loc[frame.subset.isin(['PS-sp', 'SD2-sp'])].copy()
    if 'profiling' in frame:
        frame = frame.loc[frame.profiling.isna()]
    if frame.empty:
        raise ValueError('No ordinary PS-sp or SD2-sp evaluations found.')
    if not frame.groupby('subset').evaluation_id.nunique().eq(1).all():
        raise ValueError('Multiple evaluations per subset; provide a workbook with one selected configuration per subset.')
    key = ['subset', 'category', 'source_id', 'mask_type', 'variant']
    if frame[key].isna().any().any() or frame.duplicated(key).any():
        raise ValueError('Missing sample identifiers or duplicate logical images.')
    settings = ['checkpoint_sha256', 'input_size', 'prediction_threshold',
                'ground_truth_threshold', 'postprocessing']
    for setting in settings:
        if frame[setting].isna().any() or frame[setting].nunique() != 1:
            raise ValueError(f'Inconsistent or missing {setting}.')
    counts = frame[['tp', 'fp', 'fn', 'tn']].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError('Pixel counts must be finite nonnegative integers.')
    total = counts.sum(axis=1)
    if (total <= 0).any() or not np.array_equal(total, (frame.width * frame.height).to_numpy()):
        raise ValueError('Pixel counts do not match image dimensions.')
    result = frame[key + ['image_name', 'image_path', 'evaluation_id']].copy()
    result['y_true'] = 1  # These two TGIF subsets contain manipulated images only.
    result['predicted_positive_pixels'] = frame.tp + frame.fp
    result['total_pixels'] = total.astype('int64')
    result['area_score'] = (frame.tp + frame.fp) / total
    result['pixel_threshold'] = frame.prediction_threshold
    return result.reset_index(drop=True)


def recall_summary(scores):
    rows = []
    def add(label, subset, category, group):
        tp = int(group.y_pred.sum())
        rows.append(dict(scope=label, subset=subset, category=category, images=len(group),
                         sources=group.source_id.nunique(), detected_inpainted=tp,
                         missed_inpainted=len(group)-tp, recall=tp/len(group)))
    add('overall', 'all', 'all', scores)
    for subset, group in scores.groupby('subset'):
        add('generator', subset, 'all', group)
    for category, group in scores.groupby('category'):
        add('category', 'all', category, group)
    for (subset, category), group in scores.groupby(['subset', 'category']):
        add('generator_category', subset, category, group)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/results_excel.xlsx')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/detection')
    parser.add_argument('--image-threshold', type=float,
                        help='Explicit fraction in [0,1]; predict inpainted when area_score is strictly greater. Omit for scores only.')
    args = parser.parse_args(argv)
    if args.image_threshold is not None and not 0 <= args.image_threshold <= 1:
        parser.error('--image-threshold must be between 0 and 1.')
    try:
        frame = pd.read_excel(args.input, sheet_name='images', dtype={'source_id': str})
        scores = detection_scores(frame)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    # Separate score-only and thresholded exports to avoid stale summary files.
    name = 'scores_only' if args.image_threshold is None else f'threshold_{args.image_threshold}'
    output = args.output_dir / name
    output.mkdir(parents=True, exist_ok=True)
    if args.image_threshold is not None:
        scores['image_threshold'] = args.image_threshold
        scores['y_pred'] = (scores.area_score > args.image_threshold).astype(int)
        scores['prediction'] = scores.y_pred.map({1: 'inpainted', 0: 'authentic'})
        summary = recall_summary(scores)
        summary.to_csv(output / 'recall_summary.csv', index=False)
        print(summary.loc[summary.scope.isin(['overall', 'generator'])].to_string(index=False))
    scores.to_csv(output / 'image_scores.csv', index=False)
    report = dict(method='predicted manipulated-area fraction from saved binary masks',
                  formula='(pixel_tp + pixel_fp) / (pixel_tp + pixel_fp + pixel_fn + pixel_tn)',
                  image_threshold=args.image_threshold, comparison='strictly greater than',
                  pixel_threshold=float(scores.pixel_threshold.iloc[0]),
                  images=len(scores), sources=int(scores.source_id.nunique()),
                  categories=int(scores.category.nunique()), generator_counts=scores.subset.value_counts().to_dict(),
                  evaluation_ids=sorted(scores.evaluation_id.unique().tolist()),
                  workbook=str(args.input.resolve()),
                  limitation='Manipulated-only evaluation. Only sensitivity/recall is reported. Authentic images are required for false-positive rate, specificity, balanced accuracy, ROC-AUC and meaningful two-class precision/F1/accuracy.',
                  threshold_provenance='User-supplied; not calibrated by this script. Choose on independent validation data with both classes.',
                  score_interpretation='Fraction of predicted-positive pixels, not a calibrated image-level probability.',
                  coverage='Includes saved rows only; check generator/category counts against intended test inventory.')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Exported {len(scores)} images from {scores.category.nunique()} categories to {output}')
    print(report['limitation'])
    if args.image_threshold is None:
        print('Scores only: supply a validation-selected --image-threshold to obtain binary predictions.')


if __name__ == '__main__':
    main()
