"""Run with: python -m unittest discover -s ForMa/tests -v"""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import localization_metrics, metrics_from_counts
from inference import prediction_maps, preprocess_image


class LocalizationTests(unittest.TestCase):
    def test_hand_calculated_confusion(self):
        result = localization_metrics([[1, 1], [0, 0]], [[1, 0], [1, 0]])
        self.assertEqual([result[k] for k in ('tp', 'fp', 'fn', 'tn')], [1, 1, 1, 1])
        self.assertAlmostEqual(result['iou'], 1 / 3)
        for name in ('dice', 'pixel_f1', 'precision', 'recall', 'specificity', 'pixel_accuracy'):
            self.assertEqual(result[name], 0.5)

    def test_perfect_and_disjoint(self):
        self.assertEqual(localization_metrics([[255, 0]], [[1, 0]])['iou'], 1)
        self.assertEqual(localization_metrics([[0, 1]], [[1, 0]])['dice'], 0)

    def test_empty_conventions(self):
        result = localization_metrics([[0, 0]], [[0, 0]])
        self.assertIsNone(result['iou'])
        self.assertIsNone(result['dice'])
        self.assertIsNone(result['recall'])
        self.assertEqual(result['pixel_accuracy'], 1)
        result = localization_metrics([[1, 0]], [[0, 0]])
        self.assertEqual(result['iou'], 0)

    def test_invalid_inputs(self):
        for pred, target in (([[0.5]], [[1]]), ([[0]], [[0, 1]]), ([[float('nan')]], [[0]])):
            with self.assertRaises(ValueError):
                localization_metrics(pred, target)
        with self.assertRaises(ValueError):
            metrics_from_counts(-1, 0, 0, 1)

    def test_postprocessing_matches_argmax_and_preserves_geometry(self):
        logits = torch.tensor([[[[0., 2.], [1., 0.]], [[1., 0.], [1., 2.]]]])
        probability, mask = prediction_maps(logits, (6, 4))
        expected = Image.fromarray(logits.argmax(1)[0].numpy().astype(np.uint8)).resize((6, 4), Image.Resampling.NEAREST)
        np.testing.assert_array_equal(mask, np.asarray(expected).astype(bool))
        np.testing.assert_array_equal(mask, probability > 0.5)

    def test_rgb_preprocessing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'image.png'
            Image.new('RGB', (40, 20), (255, 0, 128)).save(path)
            tensor, original = preprocess_image(path, 32)
        self.assertEqual(original.size, (40, 20))
        self.assertEqual(tuple(tensor.shape), (1, 3, 32, 32))
        torch.testing.assert_close(tensor[0, :, 0, 0], torch.tensor([1., 0., 128 / 255]))

    def test_cpu_scan_does_not_use_installed_cuda_extension(self):
        from models import csms6s
        u = torch.ones(1, 1, 2)
        with patch.object(csms6s, 'WITH_SELECTIVESCAN_OFLEX', True):
            result = csms6s.selective_scan_fn(u, u, -torch.ones(1, 1),
                torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2),
                delta_softplus=False, backend='oflex')
        torch.testing.assert_close(result, torch.tensor([[[1., 1. + np.exp(-1)]]], dtype=torch.float32))


if __name__ == '__main__':
    unittest.main()
