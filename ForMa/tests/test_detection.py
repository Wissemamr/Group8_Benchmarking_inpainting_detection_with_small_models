import sys
from pathlib import Path
import unittest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_detection import detection_scores, recall_summary


class DetectionTests(unittest.TestCase):
    def frame(self):
        base = dict(subset='PS-sp', evaluation_id='config', category='bear', source_id='1',
                    mask_type='bbox', variant=0, image_name='a.png', image_path='a.png',
                    width=10, height=10, checkpoint_sha256='hash', input_size=512,
                    prediction_threshold=.5, ground_truth_threshold=127, postprocessing='nearest')
        return pd.DataFrame([dict(base, tp=10, fp=10, fn=30, tn=50),
                             dict(base, image_name='b.png', variant=1, tp=0, fp=20, fn=40, tn=40)])

    def test_score_uses_only_predicted_area_and_ties_are_background(self):
        scores = detection_scores(self.frame())
        self.assertEqual(scores.area_score.tolist(), [.2, .2])
        scores['y_pred'] = (scores.area_score > .2).astype(int)
        summary = recall_summary(scores)
        overall = summary.iloc[0]
        self.assertEqual(overall.missed_inpainted, 2)
        self.assertEqual(overall.recall, 0)
        scores['y_pred'] = (scores.area_score > .1).astype(int)
        self.assertEqual(recall_summary(scores).iloc[0].recall, 1)

    def test_invalid_counts_and_duplicate_samples_rejected(self):
        frame = self.frame()
        frame.loc[0, 'tn'] = 49
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            detection_scores(frame)
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            detection_scores(pd.concat([frame, frame]))

    def test_multiple_configurations_rejected(self):
        frame = self.frame()
        frame.loc[1, 'evaluation_id'] = 'another'
        with self.assertRaisesRegex(ValueError, 'Multiple evaluations'):
            detection_scores(frame)


if __name__ == '__main__':
    unittest.main()
