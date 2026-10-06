"""Workbook append/resume and mask pairing tests with deterministic model output."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_tgif_category as evaluator


class CategoryEvaluationTests(unittest.TestCase):
    def test_append_resume_and_changed_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'weights.pth'
            weights.write_bytes(b'test fixture')
            output = root / 'results.xlsx'
            for category, mask_type in [('bear', 'bbox'), ('airplane', 'segm')]:
                images = root / 'ps-sp_testing/testing' / category
                masks = root / 'masks_testing/testing' / category
                images.mkdir(parents=True)
                masks.mkdir(parents=True)
                Image.new('RGB', (32, 32)).save(images / f'123_mask_{mask_type}.png_ps_0.png')
                Image.new('L', (32, 32), 255).save(masks / f'123_mask_{mask_type}.png_ps_mask.png')
            logits = torch.zeros(1, 2, 32, 32)
            logits[:, 1] = 2
            common = ['--data-root', str(root), '--output', str(output), '--weights', str(weights), '--device', 'cpu', '--input-size', '32']
            with patch.object(evaluator, 'load_model') as loader, patch.object(evaluator, 'run_inference', return_value=logits):
                evaluator.main(common + ['--category', 'bear'])
                evaluator.main(common + ['--category', 'airplane'])
                evaluator.main(common + ['--category', 'bear'])
                self.assertEqual(loader.call_count, 2)
                frame = pd.read_excel(output, sheet_name='images')
                self.assertEqual(len(frame), 2)
                self.assertEqual(set(frame.category), {'bear', 'airplane'})
                self.assertTrue((frame.pixel_f1 == 1).all())
                self.assertTrue(all(Path(p).is_file() for p in frame.mask_path))
                summary = pd.read_excel(output, sheet_name='category_summary')
                self.assertEqual(len(summary), 4)
                self.assertTrue((summary.micro_iou == 1).all())
                evaluator.main(common + ['--category', 'bear', '--threshold', '0.99'])
                frame = pd.read_excel(output, sheet_name='images')
                self.assertEqual(len(frame), 3)
                self.assertEqual(frame.iloc[-1].pixel_f1, 0)
                self.assertEqual(frame.evaluation_id.nunique(), 2)
                aggregated = pd.read_excel(output, sheet_name='aggregated_results')
                self.assertEqual(len(aggregated), 3)
                self.assertEqual(aggregated.number_of_samples.sum(), 3)
                self.assertEqual(set(aggregated.category), {'bear', 'airplane'})

    def test_category_aggregation_combines_masks_and_distinguishes_macro_micro(self):
        rows = []
        for category, mask_type, tp, fn in [('bear', 'bbox', 1, 0), ('bear', 'segm', 1, 3), ('airplane', 'bbox', 2, 0)]:
            rows.append(dict(evaluation_id='config', category=category, mask_type=mask_type,
                             source_id='123', forward_seconds=1,
                             **evaluator.metrics_from_counts(tp, 0, fn, 0)))
        result = evaluator.aggregated_frame(evaluator.summary_frame(pd.DataFrame(rows)))
        bear = result.set_index('category').loc['bear']
        self.assertEqual(len(result), 2)
        self.assertEqual(bear.number_of_samples, 2)
        self.assertEqual(bear.source_count, 1)
        self.assertEqual((bear.bbox_count, bear.segm_count), (1, 1))
        self.assertAlmostEqual(bear.macro_iou, 0.625)
        self.assertAlmostEqual(bear.micro_iou, 0.4)

    def test_missing_mask_and_unknown_category(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'ps-sp_testing/testing/bear'
            images.mkdir(parents=True)
            (images / '123_mask_bbox.png_ps_0.png').touch()
            with self.assertRaisesRegex(ValueError, 'Missing matching mask'):
                evaluator.discover(root, 'bear')
            with self.assertRaisesRegex(ValueError, 'Available: bear'):
                evaluator.discover(root, 'unknown')


if __name__ == '__main__':
    unittest.main()
