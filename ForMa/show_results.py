"""Print category and overall macro localization metrics from an evaluator workbook."""
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parent / 'results' / 'results_excel.xlsx'
METRICS = ['pixel_f1', 'recall', 'precision', 'iou', 'dice', 'specificity', 'pixel_accuracy']
LABELS = dict(pixel_f1='F1', recall='Recall', precision='Precision', iou='IoU',
              dice='Dice', specificity='Specificity', pixel_accuracy='Pixel accuracy')


def summarize(frame):
    """Summarize one configuration; undefined values are excluded per metric."""
    grouped = frame.groupby('category', sort=True)
    means = grouped[METRICS].mean()
    categories = grouped.agg(Images=('source_id', 'size'), Sources=('source_id', 'nunique'))
    categories = categories.join(means).rename(columns=LABELS)
    overall = pd.DataFrame(
        [frame[METRICS].mean(), means.mean()],
        index=['All images (equal image weight)', 'All categories (equal category weight)'],
    ).rename(columns=LABELS)
    return categories, overall


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='Results workbook (.xlsx)')
    args = parser.parse_args(argv)
    try:
        frame = pd.read_excel(args.input, sheet_name='images', dtype={'source_id': str})
    except (OSError, ValueError) as error:
        parser.error(f'Cannot read results: {error}')
    required = {'evaluation_id', 'category', 'source_id', *METRICS}
    missing = required - set(frame.columns)
    if missing:
        parser.error(f'Missing columns in images sheet: {", ".join(sorted(missing))}')
    if frame.empty:
        print('No evaluated images in this workbook yet.')
        return
    if frame[['evaluation_id', 'category', 'source_id']].isna().any().any():
        parser.error('Some rows have missing evaluation_id, category, or source_id.')

    print(f'Results: {args.input.resolve()}')
    print('Macro metrics on a 0-1 scale; bbox and segm variants combined.')
    print('Only evaluated images/categories are included. Configurations are shown separately.')
    for evaluation_id, group in frame.groupby('evaluation_id', sort=True):
        categories, overall = summarize(group)
        print(f'\nEvaluation: {evaluation_id}')
        print(f'{len(group)} images | {group.source_id.nunique()} sources | '
              f'{group.category.nunique()} categories')
        print(categories.to_string(float_format=lambda v: f'{v:.4f}', na_rep='N/A'))
        print('\nOverall averages:')
        print(overall.to_string(float_format=lambda v: f'{v:.4f}', na_rep='N/A'))
        undefined = group.groupby('category')[METRICS].agg(lambda values: values.isna().sum())
        if undefined.to_numpy().any():
            print('\nUndefined scores excluded from averages (counts by category):')
            print(undefined.rename(columns=LABELS).to_string())
            print('Equal-category averages exclude categories with no defined score for that metric.')
    print('\nF1 and Dice are identical in this evaluator. Overall averages are macro, not pooled pixel metrics.')


if __name__ == '__main__':
    main()
