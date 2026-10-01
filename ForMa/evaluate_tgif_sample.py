"""Inspect local TGIF PS-sp pairs and evaluate all six variants of one source ID."""
import argparse
from collections import Counter
import csv
import html
import json
import platform
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from inference import ROOT, DEFAULT_WEIGHTS, load_model, preprocess_image, run_inference, prediction_maps, sha256
from metrics import localization_metrics, metrics_from_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data_tgif')
    parser.add_argument('--category', default='sheep')
    parser.add_argument('--source-id', default='154947')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'sheep_sample')
    args = parser.parse_args()
    started = time.perf_counter()
    image_root = args.data_root / 'ps-sp_testing' / 'testing'
    mask_root = args.data_root / 'masks_testing' / 'testing'
    images = sorted(image_root.rglob('*.png'))
    masks = sorted(mask_root.rglob('*.png'))
    missing = []
    for image in images:
        name = image.name.rsplit('_ps_', 1)[0] + '_ps_mask.png'
        if not (mask_root / image.parent.name / name).is_file():
            missing.append(str(image))
    selected = [p for p in images if p.parent.name == args.category and p.name.startswith(args.source_id + '_mask_')]
    if not selected:
        parser.error('No images matched the requested category/source ID.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = dict(images=len(images), masks=len(masks),
        images_per_category=dict(sorted(Counter(p.parent.name for p in images).items())),
        masks_per_category=dict(sorted(Counter(p.parent.name for p in masks).items())),
        source_ids_per_category={c: len({p.name.split('_')[0] for p in images if p.parent.name == c}) for c in sorted({p.parent.name for p in images})},
        missing_adapted_masks=missing)
    (args.output_dir / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    load_start = time.perf_counter()
    model = load_model(DEFAULT_WEIGHTS, device)
    sync()
    load_seconds = time.perf_counter() - load_start
    warm, _ = preprocess_image(selected[0])
    sync()
    warm_start = time.perf_counter()
    run_inference(model, warm.to(device))
    sync()
    warm_seconds = time.perf_counter() - warm_start
    rows, panels = [], []
    for index, path in enumerate(selected):
        item_start = time.perf_counter()
        mask_path = mask_root / path.parent.name / (path.name.rsplit('_ps_', 1)[0] + '_ps_mask.png')
        tensor, image = preprocess_image(path)
        raw_mask = np.asarray(Image.open(mask_path).convert('L'))
        if raw_mask.shape != (image.height, image.width):
            raise ValueError(f'Mask size mismatch: {path.name}')
        # Explicit benchmark convention for Photoshop's feathered border.
        target = raw_mask > 127
        tensor = tensor.to(device)
        sync()
        forward_start = time.perf_counter()
        logits = run_inference(model, tensor)
        sync()
        forward_seconds = time.perf_counter() - forward_start
        probability, prediction = prediction_maps(logits, image.size)
        scores = localization_metrics(prediction, target)
        folder = args.output_dir / path.stem
        folder.mkdir(exist_ok=True)
        image.save(folder / 'input.png')
        Image.fromarray(target.astype(np.uint8) * 255).save(folder / 'ground_truth.png')
        Image.fromarray(prediction.astype(np.uint8) * 255).save(folder / 'prediction.png')
        np.save(folder / 'probabilities.npy', probability)
        row = dict(image=path.name, mask=mask_path.name, width=image.width, height=image.height,
            mask_type='bbox' if '_bbox' in path.name else 'segm',
            variant=int(path.stem.rsplit('_', 1)[1]),
            foreground_fraction=float(target.mean()), soft_border_pixels=int(((raw_mask>0)&(raw_mask<255)).sum()),
            forward_seconds=forward_seconds, **scores)
        detail = dict(**row, image_path=str(path.resolve()), mask_path=str(mask_path.resolve()),
            image_sha256=sha256(path), mask_sha256=sha256(mask_path))
        (folder / 'metrics.json').write_text(json.dumps(detail, indent=2, allow_nan=False), encoding='utf-8')
        row['processing_seconds'] = time.perf_counter() - item_start
        rows.append(row)
        panels.append(f'<h3>{html.escape(path.name)}</h3><p>IoU {scores["iou"]:.4f}; F1 {scores["pixel_f1"]:.4f}; forward {forward_seconds:.3f}s</p><div class="panels">' + ''.join(f'<figure><img src="{path.stem}/{filename}"><figcaption>{label}</figcaption></figure>' for filename,label in [('input.png','Manipulated input'),('ground_truth.png','Adapted ground truth >127'),('prediction.png','ForMa prediction >0.5')]) + '</div>')
        print(f'{index+1}/{len(selected)} {path.name}: IoU={scores["iou"]:.4f}, F1={scores["pixel_f1"]:.4f}, forward={forward_seconds:.3f}s', flush=True)
    metric_names = ['iou','dice','pixel_f1','precision','recall','specificity','pixel_accuracy']
    macro = {k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in metric_names}
    micro = metrics_from_counts(**{k:sum(r[k] for r in rows) for k in ['tp','fp','fn','tn']})
    report = dict(category=args.category, source_id=args.source_id, sample_size=len(rows),
        selection='All available bbox/segm and generation variants for the specified source ID; selected before inference.',
        limitations='One source scene with correlated variants; not a representative category or full-dataset estimate.',
        input_size=512, prediction_threshold=0.5, ground_truth_threshold='adapted Photoshop mask >127 (0..255)',
        postprocessing='Class-1 softmax; nearest resize to original resolution; foreground >0.5.',
        device=str(device), cpu=platform.processor(), threads=4,
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        checkpoint=str(DEFAULT_WEIGHTS), checkpoint_sha256=sha256(DEFAULT_WEIGHTS),
        model_load_seconds=load_seconds, warmup_seconds=warm_seconds,
        mean_forward_seconds=float(np.mean([r['forward_seconds'] for r in rows])),
        median_forward_seconds=float(np.median([r['forward_seconds'] for r in rows])),
        total_forward_seconds=sum(r['forward_seconds'] for r in rows),
        total_processing_seconds=sum(r['processing_seconds'] for r in rows),
        elapsed_before_report_seconds=time.perf_counter()-started,
        timing='Model loaded once, one discarded warmup, one measured forward per image; processing includes reads, evaluation and per-image writes. Total includes inventory, loading and warmup; excludes interpreter/import startup and final report writing.',
        macro=macro, macro_defined_counts={k:sum(r[k] is not None for r in rows) for k in metric_names}, micro=micro, samples=rows)
    (args.output_dir / 'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    with (args.output_dir / 'samples.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary=f'Mean IoU {macro["iou"]:.4f}; mean F1 {macro["pixel_f1"]:.4f}; mean forward {report["mean_forward_seconds"]:.3f}s on {device}.'
    (args.output_dir / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>ForMa sheep sample</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px;background:#fafafa} .panels{display:flex;gap:12px} figure{margin:0;flex:1;min-width:0}img{width:100%;height:260px;object-fit:contain;background:#ddd}figcaption{padding:8px}h3{margin-top:40px}</style><h1>ForMa: TGIF Photoshop-spliced sheep sample</h1><p>'+summary+'</p><p>One source scene, six correlated variants. Ground truth: adapted Photoshop mask >127. Prediction: class-1 probability >0.5. Input: 512x512. Not a full benchmark.</p><p><a href="report.json">Full report</a> | <a href="samples.csv">CSV scores</a> | <a href="inventory.json">Dataset inventory</a></p>'+''.join(panels),encoding='utf-8')
    print(summary)
    print('Report:',args.output_dir / 'report.html')


if __name__ == '__main__':
    main()
