"""Workbook append/resume and mask pairing tests with deterministic model output."""
import sys
import json
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
    def test_sd2_pairing_shared_workbook_and_profile_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / 'masks_testing/testing/hair drier'
            masks.mkdir(parents=True)
            Image.new('L', (32, 32), 255).save(masks / '123_mask_bbox.png_ps_mask.png')
            for subset, category, filename in [
                ('ps-sp', 'hair-drier', '123_mask_bbox.png_ps_0.png'),
                ('sd2-sp', 'hair drier', '123_mask_bbox.png_ps_mask.png_sd2_0.png'),
            ]:
                folder = root / f'{subset}_testing/testing' / category
                folder.mkdir(parents=True)
                Image.new('RGB', (32, 32)).save(folder / filename)
                pairs = evaluator.discover(root, 'hair-drier', subset)
                self.assertEqual(len(pairs), 1)
                self.assertEqual(pairs[0][1].parent, masks)
            weights = root / 'weights.pth'
            weights.write_bytes(b'fixture')
            output = root / 'results.xlsx'
            common = ['--data-root', str(root), '--output', str(output), '--weights', str(weights),
                      '--device', 'cpu', '--input-size', '32', '--category', 'hair-drier']
            logits = torch.zeros(1, 2, 32, 32)
            logits[:, 1] = 2
            with patch.object(evaluator, 'load_model'), patch.object(evaluator, 'run_inference', return_value=logits):
                evaluator.main(common)
                evaluator.main(common + ['--subset', 'sd2-sp'])
                evaluator.main(common + ['--subset', 'sd2-sp'])
                evaluator.main(common + ['--subset', 'sd2-sp', '--profile'])
            frame = pd.read_excel(output, sheet_name='images')
            self.assertEqual(len(frame), 3)
            self.assertEqual(frame.evaluation_id.nunique(), 3)
            summary = pd.read_excel(output, sheet_name='category_summary')
            self.assertEqual(set(zip(summary.generator, summary.subset)), {('PS', 'PS-sp'), ('SD2', 'SD2-sp')})
            reports = [json.loads(p.read_text()) for p in (root / 'runs').glob('*/run.json')]
            self.assertEqual(len(reports), 4)
            resumed = [r for r in reports if r['newly_saved_images'] == 0]
            self.assertEqual(len(resumed), 1)
            self.assertEqual(resumed[0]['skipped_images'], 1)
            self.assertIsNone(resumed[0]['forward_only_images_per_second'])
            profiled = next(r for r in reports if r['profiled'])
            folder = root / 'runs' / profiled['run_id']
            self.assertTrue((folder / 'profile.pstats').is_file())
            self.assertTrue((folder / 'images.csv').is_file())
            self.assertEqual(profiled['stages']['forward']['count'], 1)
            self.assertEqual(profiled['status'], 'completed')

    def test_check_only_and_failed_run_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / 'runs'
            with patch.object(evaluator, 'load_model') as loader:
                with self.assertRaisesRegex(ValueError, 'Unknown category'):
                    evaluator.main(['--data-root', str(root), '--category', 'bear',
                                    '--check-only', '--run-dir', str(runs)])
                loader.assert_not_called()
            report = json.loads(next(runs.glob('*/run.json')).read_text())
            self.assertEqual(report['status'], 'failed')

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
