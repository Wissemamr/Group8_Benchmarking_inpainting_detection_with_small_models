"""Render completed binary detection tables and notebook outputs; optionally wait for inference."""
import argparse
from pathlib import Path
import time
import json
import pandas as pd
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def markdown_table(frame):
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---']*len(frame.columns)) + ' |']
    for _, row in frame.iterrows():
        values = [f'{value:.4f}' if isinstance(value, float) else str(value) for value in row]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    result = ROOT / 'results/full_detection/complete/threshold_0.01'
    deadline = time.monotonic() + 7200
    while not (result/'report.json').is_file():
        if not args.wait:
            raise FileNotFoundError('Full detection is still running. Retry after it finishes or use --wait.')
        if time.monotonic() > deadline:
            raise TimeoutError('No complete results after two hours.')
        time.sleep(10)
    report = json.loads((result/'report.json').read_text())
    assert report['status'] == 'complete'
    summary = pd.read_csv(result/'detection_summary.csv')
    images = pd.read_csv(result/'image_predictions.csv')
    assert len(images)==4459 and (images.y_true==0).sum()==343
    for _,row in summary[summary.category=='all'].iterrows():
        g=images if row.subset=='all' else images[(images.subset==row.subset)|(images.y_true==0)]
        assert row.images==len(g)
        for key,actual,predicted in [('tp',1,1),('fp',0,1),('fn',1,0),('tn',0,0)]:
            assert row[key]==int(((g.y_true==actual)&(g.y_pred==predicted)).sum())
    columns=['subset','images','authentic','inpainted','accuracy','precision','recall','f1','specificity',
             'false_positive_rate','balanced_accuracy','roc_auc','average_precision']
    overall=summary[summary.category=='all'][columns]
    text = '# ForMa binary detection results\n\nFixed image area threshold >0.01; pixel probability threshold >0.5. This operating point was not calibrated on independent validation data.\n\n'
    text += markdown_table(overall)
    text += '\n\n## Per-category results\n\n' + markdown_table(summary[summary.category!='all'][['category',*columns]])
    text += '\n\nAuthentic originals are counted once overall and reused in each separate generator comparison. These are image-level detection metrics, not localization metrics. Source-related variants are correlated; confidence intervals are not included.\n'
    (result/'summary.md').write_text(text,encoding='utf-8')
    notebook=ROOT/'final_visualizatio.ipynb'
    saved=nbformat.read(notebook,as_version=4)
    cells=[c for c in saved.cells if c.cell_type=='code' and c.metadata.get('forma_detection_section')]
    if len(cells)!=4:
        raise ValueError('Expected four detection code cells in final notebook.')
    run=nbformat.v4.new_notebook(cells=cells,metadata=saved.metadata)
    NotebookClient(run,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    # Re-read before saving so edits to unrelated cells during inference are preserved.
    current=nbformat.read(notebook,as_version=4)
    by_id={c.id:c for c in run.cells}
    for c in current.cells:
        if c.id in by_id:
            rendered=by_id[c.id]
            if c.source!=rendered.source:
                raise ValueError('A detection cell was edited while rendering; rerun to avoid stale output.')
            c.outputs=rendered.outputs
            c.execution_count=None
    nbformat.validate(current)
    nbformat.write(current,notebook)
    print(overall.to_string(index=False),flush=True)
    print('Detection tables and notebook charts saved.',flush=True)


if __name__=='__main__':
    main()
