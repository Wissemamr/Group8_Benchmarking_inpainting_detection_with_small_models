"""Evaluate authentic TGIF originals plus saved manipulated-image detection scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from evaluate_detection import detection_scores
from inference import ROOT, DEFAULT_WEIGHTS, load_model, preprocess_image, run_inference, prediction_maps, sha256


def classification_metrics(frame, threshold):
    y = frame.y_true.to_numpy(dtype=int)
    score = frame.area_score.to_numpy(dtype=float)
    predicted = score > threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    def ratio(a, b):
        return a / b if b else None
    sensitivity, specificity = ratio(tp, tp+fn), ratio(tn, tn+fp)
    positives, negatives = int(y.sum()), int((y == 0).sum())
    auc = None
    ap = None
    if positives and negatives:
        ranks = pd.Series(score).rank(method='average').to_numpy()
        auc = float((ranks[y == 1].sum() - positives*(positives+1)/2)/(positives*negatives))
        grouped = pd.DataFrame({'score': score, 'positive': y}).groupby('score').positive.agg(['sum','size']).sort_index(ascending=False)
        ap = float(((grouped['sum']/positives)*(grouped['sum'].cumsum()/grouped['size'].cumsum())).sum())
    return dict(images=len(frame), sources=int(frame.source_id.nunique()), authentic=negatives,
                inpainted=positives, tp=tp, fp=fp, fn=fn, tn=tn,
                accuracy=ratio(tp+tn,len(frame)), precision=ratio(tp,tp+fp), recall=sensitivity,
                f1=ratio(2*tp,2*tp+fp+fn), specificity=specificity,
                false_positive_rate=ratio(fp,fp+tn),
                balanced_accuracy=(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
                roc_auc=auc, average_precision=ap)


def summary_table(scores, threshold):
    real = scores[scores.y_true == 0]
    fake = scores[scores.y_true == 1]
    rows = []
    for generator in ['all', 'PS-sp', 'SD2-sp']:
        selected = fake if generator == 'all' else fake[fake.subset == generator]
        group = pd.concat([selected, real], ignore_index=True)
        rows.append(dict(subset=generator, category='all', **classification_metrics(group,threshold)))
        for category, part in group.groupby('category'):
            rows.append(dict(subset=generator, category=category, **classification_metrics(part,threshold)))
    return pd.DataFrame(rows)


def authentic_manifest(frame, root):
    rows = []
    for (category, source_id), group in frame.groupby(['category','source_id']):
        candidates = [root / name / f'{source_id}_orig.png' for name in [category, category.replace('-', ' ')]]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Missing authentic original for {category}/{source_id}')
        with Image.open(path) as image:
            if set(zip(group.width,group.height)) != {image.size}:
                raise ValueError(f'Authentic dimensions mismatch: {path}')
        rows.append(dict(category=category,source_id=source_id,image_path=str(path.resolve()),
                         image_name=path.name,image_sha256=sha256(path)))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/results_excel.xlsx')
    parser.add_argument('--original-root',type=Path,default=ROOT/'data_tgif/original/testing')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/full_detection')
    parser.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS)
    parser.add_argument('--image-threshold',type=float,default=.01)
    parser.add_argument('--limit',type=int,help='Smoke check only; partial outputs are explicitly labeled')
    args=parser.parse_args(argv)
    if not 0<=args.image_threshold<=1 or (args.limit is not None and args.limit<1):
        parser.error('Invalid threshold or limit.')
    started=time.perf_counter()
    raw=pd.read_excel(args.input,sheet_name='images',dtype={'source_id':str})
    fake=detection_scores(raw)
    if set(fake.subset)!={'PS-sp','SD2-sp'}:
        raise ValueError('Both generator evaluations are required.')
    selected=raw[raw.evaluation_id.isin(fake.evaluation_id.unique())].copy()
    keys=['category','source_id','mask_type','variant']
    if set(map(tuple,fake[fake.subset=='PS-sp'][keys].to_numpy())) != set(map(tuple,fake[fake.subset=='SD2-sp'][keys].to_numpy())):
        raise ValueError('Generators have different evaluated sample sets.')
    for (subset,category),group in selected.groupby(['subset','category']):
        if len(group)!=int(group.category_total_images.iloc[0]):
            raise ValueError(f'Incomplete saved category: {subset}/{category}')
    if selected.torch_version.nunique()!=1 or str(torch.__version__)!=selected.torch_version.iloc[0]:
        raise ValueError('Use the same PyTorch version as the saved evaluations: '+str(selected.torch_version.unique()))
    if selected.device.nunique()!=1 or selected.threads.nunique()!=1:
        raise ValueError('Saved device/thread settings differ.')
    config=dict(checkpoint_sha256=sha256(args.weights), input_size=int(selected.input_size.iloc[0]),
                pixel_threshold=float(selected.prediction_threshold.iloc[0]),device=selected.device.iloc[0],
                threads=int(selected.threads.iloc[0]),torch_version=str(torch.__version__),
                method='original-full-resolution-area-v1',
                inference_sha256=sha256(ROOT/'inference.py'),
                model_code_sha256={p.name:sha256(p) for p in sorted((ROOT/'models').glob('*.py'))})
    if selected.checkpoint_sha256.nunique()!=1 or config['checkpoint_sha256']!=selected.checkpoint_sha256.iloc[0]:
        raise ValueError('Checkpoint differs from saved evaluation.')
    manifest=authentic_manifest(selected,args.original_root)
    all_sources=len(manifest)
    if args.limit: manifest=manifest[:args.limit]
    config_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cache=args.output_dir/'authentic_cache'/config_id[:16]
    cache.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config['threads'])
    model=None
    real_rows=[]
    for item in tqdm(manifest,desc='Authentic detection',unit='image'):
        key=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
        path=cache/(key[:24]+'.json')
        if path.is_file():
            row=json.loads(path.read_text())
        else:
            if model is None:
                print('Loading matching checkpoint on '+config['device'],flush=True)
                model=load_model(args.weights,config['device'])
            tensor,image=preprocess_image(item['image_path'],config['input_size'])
            t=time.perf_counter()
            logits=run_inference(model,tensor.to(config['device']))
            _,prediction=prediction_maps(logits,image.size,config['pixel_threshold'])
            row=dict(item,subset='authentic',mask_type='none',variant=-1,y_true=0,
                     predicted_positive_pixels=int(prediction.sum()),total_pixels=int(prediction.size),
                     area_score=float(prediction.mean()),pixel_threshold=config['pixel_threshold'],
                     evaluation_id=config_id,inference_and_postprocessing_seconds=time.perf_counter()-t,
                     evaluated_at_utc=datetime.now(timezone.utc).isoformat())
            temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(row,indent=2),encoding='utf-8')
            temporary.replace(path)
        real_rows.append(row)
    real=pd.DataFrame(real_rows)
    partial=len(real)<all_sources
    if partial:
        included=set(zip(real.category,real.source_id))
        fake=fake[[tuple(v) in included for v in fake[['category','source_id']].to_numpy()]]
    scores=pd.concat([fake,real],ignore_index=True)
    scores['y_pred']=(scores.area_score>args.image_threshold).astype(int)
    scores['prediction']=scores.y_pred.map({0:'authentic',1:'inpainted'})
    output=args.output_dir/('partial' if partial else 'complete')/f'threshold_{args.image_threshold}'
    output.mkdir(parents=True,exist_ok=True)
    scores.to_csv(output/'image_predictions.csv',index=False)
    summary=summary_table(scores,args.image_threshold)
    summary.to_csv(output/'detection_summary.csv',index=False)
    with pd.ExcelWriter(output/'detection_results.xlsx',engine='openpyxl') as writer:
        scores.to_excel(writer,sheet_name='images',index=False)
        summary.to_excel(writer,sheet_name='summary',index=False)
    report=dict(status='partial' if partial else 'complete',authentic_expected=all_sources,authentic_evaluated=len(real),
                inpainted_evaluated=len(fake),unique_test_images=len(scores),image_threshold=args.image_threshold,
                config=config,original_choice='One uncropped <source>_orig.png per source; dimensions matched to both manipulated subsets.',
                negatives='Authentic images counted once overall; the same negatives reused for each separate generator comparison.',
                score='Fraction of pixels whose class-1 probability exceeds saved pixel threshold; not calibrated probability.',
                threshold_provenance='Fixed previous user operating point; not tuned on these test data.',
                runtime_seconds=time.perf_counter()-started,
                source_workbook_sha256=sha256(args.input),
                manipulated_evaluation_ids=sorted(fake.evaluation_id.unique().tolist()),
                limitations='Imbalanced test prevalence; compare balanced accuracy and ROC-AUC alongside accuracy. Source-related variants are correlated. No confidence intervals. Authentic inference uses matching checkpoint, torch version and settings; historical full environment provenance is unavailable.')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    cols=['subset','images','authentic','inpainted','tp','fp','fn','tn','accuracy','precision','recall','f1','specificity','balanced_accuracy','roc_auc','average_precision']
    print(summary[summary.category=='all'][cols].to_string(index=False),flush=True)
    print('Results:',output,flush=True)


if __name__=='__main__':
    main()
