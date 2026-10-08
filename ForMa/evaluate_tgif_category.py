"""Evaluate TGIF PS-sp or SD2-sp categories and resume a shared Excel results dataframe."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

from inference import ROOT, DEFAULT_WEIGHTS, load_model, preprocess_image, run_inference, prediction_maps, sha256
from metrics import localization_metrics, metrics_from_counts
from run_stats import RunStats

METRICS = ['pixel_f1', 'recall', 'precision', 'iou', 'dice', 'specificity', 'pixel_accuracy']
PATTERNS = {
    'ps-sp': re.compile(r'(?P<source_id>\d+)_mask_(?P<mask_type>bbox|segm)\.png_ps_(?P<variant>[012])\.png'),
    'sd2-sp': re.compile(r'(?P<source_id>\d+)_mask_(?P<mask_type>bbox|segm)\.png_ps_mask\.png_sd2_(?P<variant>[012])\.png'),
}


def category_directory(root, category):
    exact = root / category
    if exact.is_dir():
        return exact
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.replace(' ', '-') == category.replace(' ', '-')] if root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    return exact


def discover(data_root, category, subset='ps-sp'):
    image_root = data_root / f'{subset}_testing' / 'testing'
    image_directory = category_directory(image_root, category)
    categories = sorted(p.name for p in image_root.iterdir() if p.is_dir()) if image_root.is_dir() else []
    if not image_directory.is_dir():
        raise ValueError(f'Unknown category {category!r}. Available: {", ".join(categories) or "none (check --data-root)"}')
    mask_root = data_root / 'masks_testing' / 'testing'
    mask_directory = category_directory(mask_root, category)
    pairs = []
    for image in sorted(image_directory.glob('*.png')):
        match = PATTERNS[subset].fullmatch(image.name)
        if not match:
            raise ValueError(f'Unexpected {subset} filename: {image}')
        mask = mask_directory / f'{match["source_id"]}_mask_{match["mask_type"]}.png_ps_mask.png'
        if not mask.is_file():
            raise ValueError(f'Missing matching mask: {mask}')
        pairs.append((image, mask, match.groupdict()))
    if not pairs:
        raise ValueError(f'No images in category {category!r}')
    return pairs


def summary_frame(frame):
    rows = []
    for (evaluation_id, category), group in frame.groupby(['evaluation_id', 'category'], sort=True):
        for mask_type, subset in [('all', group), *list(group.groupby('mask_type', sort=True))]:
            counts = {k: int(subset[k].sum()) for k in ['tp', 'fp', 'fn', 'tn']}
            micro = metrics_from_counts(**counts)
            row = dict(evaluation_id=evaluation_id, model='ForMa', task='localization',
                       generator=group.generator.iloc[0] if 'generator' in group else 'PS',
                       subset=group.subset.iloc[0] if 'subset' in group else 'PS-sp', category=category, mask_type=mask_type,
                       number=len(subset), source_count=subset.source_id.nunique(),
                       bbox_count=int((subset.mask_type == 'bbox').sum()),
                       segm_count=int((subset.mask_type == 'segm').sum()),
                       mean_forward_seconds=subset.forward_seconds.mean())
            row['mask_type_ratio_bbox_segm'] = f'{row["bbox_count"]}:{row["segm_count"]}'
            for metric in METRICS:
                row[f'macro_{metric}'] = subset[metric].mean()
                row[f'defined_{metric}_count'] = int(subset[metric].notna().sum())
                row[f'micro_{metric}'] = micro[metric]
            rows.append(row)
    return pd.DataFrame(rows)


def aggregated_frame(summary):
    """One category row per configuration, combining bbox and segm samples."""
    columns = ['category', 'model', 'number', 'source_count', 'bbox_count', 'segm_count',
               *[f'macro_{metric}' for metric in METRICS],
               *[f'micro_{metric}' for metric in METRICS],
               *[f'defined_{metric}_count' for metric in METRICS],
               'mean_forward_seconds', 'generator', 'subset', 'evaluation_id']
    return (summary.loc[summary.mask_type == 'all', columns]
            .rename(columns={'number': 'number_of_samples'})
            .sort_values(['category', 'evaluation_id']).reset_index(drop=True))


def save_workbook(frame, output):
    """Replace only after all sheets are successfully written; retain previous file on failure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.xlsx', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with pd.ExcelWriter(temporary, engine='openpyxl') as writer:
            frame.to_excel(writer, sheet_name='images', index=False)
            summary = summary_frame(frame)
            summary.to_excel(writer, sheet_name='category_summary', index=False)
            aggregated_frame(summary).to_excel(writer, sheet_name='aggregated_results', index=False)
            for sheet in writer.book.worksheets:
                sheet.freeze_panes = 'A2'
                sheet.auto_filter.ref = sheet.dimensions
                for column in sheet.columns:
                    sheet.column_dimensions[column[0].column_letter].width = min(55, max(16, len(str(column[0].value)) + 2))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', required=True, help='Category name, or all to evaluate every available category')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data_tgif')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'results_excel.xlsx')
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--input-size', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Evaluate the first N images for a smoke check')
    parser.add_argument('--subset', choices=['ps-sp', 'sd2-sp'], default='ps-sp')
    parser.add_argument('--run-dir', type=Path, help='Run reports directory; default: next to output workbook in runs/')
    parser.add_argument('--profile', action='store_true', help='Enable cProfile diagnostic report (adds overhead; separate evaluation ID)')
    parser.add_argument('--check-only', action='store_true', help='Validate image/mask pairing and dimensions without inference')
    args = parser.parse_args(argv)
    if args.input_size < 32 or args.input_size % 32 or not 0 <= args.threshold <= 1 or args.threads < 1:
        parser.error('Input size must be a positive multiple of 32, threshold in [0,1], and threads positive.')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    if args.output.suffix.lower() != '.xlsx':
        parser.error('--output must end in .xlsx')
    image_root = args.data_root / f'{args.subset}_testing' / 'testing'
    if args.category == 'all':
        categories = sorted(p.name for p in image_root.iterdir() if p.is_dir()) if image_root.is_dir() else []
        if not categories:
            parser.error(f'No categories found in {image_root}')
    else:
        categories = [args.category]
    for category in categories:
        args.category = category.replace(' ', '-')
        with RunStats(args.run_dir or args.output.parent / 'runs',
                      dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                           subset=args.subset, category=args.category), profile=args.profile) as stats:
            evaluate(args, stats)


def evaluate(args, stats):
    stats.info['code_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for p in
                                 [ROOT / 'evaluate_tgif_category.py', ROOT / 'inference.py',
                                  ROOT / 'metrics.py', ROOT / 'run_stats.py', *sorted((ROOT / 'models').glob('*.py'))]}
    with stats.phase('discovery'):
        pairs = discover(args.data_root, args.category, args.subset)
    if args.check_only:
        with stats.phase('validate_dimensions'):
            for image_path, mask_path, _ in pairs:
                with Image.open(image_path) as image, Image.open(mask_path) as mask:
                    if image.size != mask.size:
                        raise ValueError(f'Mask dimensions do not match image: {mask_path}')
        stats.info.update(validated_images=len(pairs), inference_performed=False)
        print(f'{args.subset} / {args.category}: {len(pairs)} matching image/mask pairs validated.')
        return
    total = len(pairs)
    print(f'Category {args.category}: {total} images, {len({p[2]["source_id"] for p in pairs})} source IDs; '
          f'{sum(p[2]["mask_type"] == "bbox" for p in pairs)} bbox / {sum(p[2]["mask_type"] == "segm" for p in pairs)} segm.', flush=True)
    if args.limit:
        pairs = pairs[:args.limit]
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu.')
    config = dict(protocol=f'forma-{args.subset}-v1', checkpoint_sha256=sha256(args.weights),
                  input_size=args.input_size, prediction_threshold=args.threshold, ground_truth_threshold=127,
                  postprocessing='class-1 softmax; nearest resize; strict greater-than thresholds',
                  device=device, threads=args.threads, torch_version=str(torch.__version__))
    if args.profile:
        config['profiling'] = 'cProfile'
    evaluation_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    with stats.phase('read_workbook'):
        frame = pd.read_excel(args.output, sheet_name='images', dtype={'source_id': str}) if args.output.exists() else pd.DataFrame()
    required = {'sample_key', 'evaluation_id', 'category', 'source_id', 'mask_type', 'forward_seconds', 'tp', 'fp', 'fn', 'tn', *METRICS}
    if args.output.exists() and not required.issubset(frame.columns):
        raise ValueError('Existing workbook is not an evaluator workbook; choose a new --output.')
    completed = set(frame.sample_key) if not frame.empty else set()
    pending = []
    with stats.phase('hashing_and_resume'):
        for image, mask, info in tqdm(pairs, desc='Checking completed images', unit='image'):
            identity = dict(evaluation_id=evaluation_id, image_path=str(image.resolve()), mask_path=str(mask.resolve()),
                            image_sha256=sha256(image), mask_sha256=sha256(mask))
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if key not in completed:
                pending.append((image, mask, info, identity, key))
    print(f'Selected {len(pairs)}/{total}; completed {len(pairs)-len(pending)}; remaining {len(pending)}. Output: {args.output.resolve()}', flush=True)
    stats.info.update(evaluation_id=evaluation_id, configuration=config, selected_images=len(pairs),
                      skipped_images=len(pairs)-len(pending), discovered_images=total)
    if not pending:
        with stats.phase('workbook_write'):
            save_workbook(frame, args.output)
        return
    torch.set_num_threads(args.threads)
    print(f'Loading ForMa on {device} and running one discarded warmup...', flush=True)
    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()
    with stats.phase('model_load'):
        model = load_model(args.weights, device)
        sync()
    stats.info['parameter_count'] = sum(p.numel() for p in model.parameters())
    stats.info['available_backends'] = {
        name: {flag: bool(getattr(module, flag)) for flag in dir(module) if flag.startswith('WITH_')}
        for name in ['models.csms6s', 'models.csm_triton']
        if (module := sys.modules.get(name)) is not None
    }
    with stats.phase('warmup'):
        warm, _ = preprocess_image(pending[0][0], args.input_size)
        run_inference(model, warm.to(device))
        sync()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        stats.info['gpu_name'] = torch.cuda.get_device_name()
    with tqdm(pending, desc=f'ForMa / {args.category}', unit='image') as progress:
        for image_path, mask_path, info, identity, key in progress:
            started = time.perf_counter()
            with stats.phase('preprocessing'):
                tensor, image = preprocess_image(image_path, args.input_size)
                with Image.open(mask_path) as mask_image:
                    raw = np.asarray(mask_image.convert('L'))
                if raw.shape != (image.height, image.width):
                    raise ValueError(f'Mask dimensions do not match image: {mask_path}')
                target = raw > 127
            with stats.phase('device_transfer'):
                tensor = tensor.to(device)
                sync()
            with stats.phase('forward'):
                logits = run_inference(model, tensor)
                sync()
            forward_seconds = stats.stages['forward'][-1]
            with stats.phase('postprocessing_and_metrics'):
                _, prediction = prediction_maps(logits, image.size, args.threshold)
                scores = localization_metrics(prediction, target)
            if device == 'cuda':
                stats.info['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                stats.info['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            row = dict(sample_key=key, **identity, **config, model='ForMa', task='localization',
                       generator='PS' if args.subset == 'ps-sp' else 'SD2',
                       subset='PS-sp' if args.subset == 'ps-sp' else 'SD2-sp', split='testing', run_id=stats.directory.name, category=args.category,
                       **info, image_name=image_path.name, mask_name=mask_path.name,
                       checkpoint_path=str(args.weights.resolve()), width=image.width, height=image.height,
                       category_total_images=total, foreground_fraction=float(target.mean()),
                       soft_border_pixels=int(((raw > 0) & (raw < 255)).sum()),
                       evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
                       forward_seconds=forward_seconds, **scores,
                       processing_seconds=time.perf_counter()-started)
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True) if not frame.empty else pd.DataFrame([row])
            try:
                with stats.phase('workbook_write'):
                    save_workbook(frame, args.output)
            except PermissionError as error:
                raise PermissionError(f'Cannot update {args.output}. Close it in Excel and rerun; previously saved images will be skipped.') from error
            stats.record_image(dict(sample_key=key, image_name=image_path.name, mask_type=info['mask_type'],
                                    **{name + '_seconds': stats.stages[name][-1] for name in
                                       ['preprocessing', 'device_transfer', 'forward', 'postprocessing_and_metrics', 'workbook_write']},
                                    wall_seconds=time.perf_counter()-started))
            progress.set_postfix(iou=scores['iou'], f1=scores['pixel_f1'])
    print(f'Saved {len(frame)} total image rows to {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
