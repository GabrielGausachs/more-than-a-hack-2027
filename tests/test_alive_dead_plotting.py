"""Synthetic tests for ALIVE/DEAD prediction timelines."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from modeling.alive_dead.evaluate import PredictionRun, score_prediction_runs
from modeling.alive_dead.plotting import _group_runs_by_half, save_prediction_timeline


def prediction_run(start, labels, predictions=None, half_id=1):
    labels = np.asarray(labels, dtype=np.int8)
    if predictions is None:
        predictions = labels
    predictions = np.asarray(predictions, dtype=np.int8)
    return PredictionRun(start, start + len(labels), half_id, labels, predictions)


class PredictionTimelineTests(unittest.TestCase):
    def test_halves_and_disconnected_runs_are_ordered_without_joining(self):
        second_half = prediction_run(20, [1, 0], half_id=2)
        right_first_half = prediction_run(8, [0, 1], half_id=1)
        left_first_half = prediction_run(0, [1, 1], half_id=1)

        grouped = _group_runs_by_half(
            [second_half, right_first_half, left_first_half]
        )

        self.assertEqual([half for half, _ in grouped], [1, 2])
        self.assertEqual([run.start for run in grouped[0][1]], [0, 8])
        self.assertEqual(len(grouped[0][1]), 2)

    def test_invalid_runs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            save_prediction_timeline([], 10.0, "unused.png", "Unused")
        with self.assertRaisesRegex(ValueError, "binary"):
            save_prediction_timeline(
                [prediction_run(0, [0, 2])], 10.0, "unused.png", "Unused"
            )
        with self.assertRaisesRegex(ValueError, "aligned"):
            run = PredictionRun(
                0,
                2,
                1,
                np.array([0, 1], dtype=np.int8),
                np.array([0], dtype=np.int8),
            )
            save_prediction_timeline([run], 10.0, "unused.png", "Unused")

    def test_png_is_saved_for_two_halves(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Matplotlib is not installed")

        runs = [
            prediction_run(0, [0, 0, 1, 1], [0, 1, 1, 1], half_id=1),
            prediction_run(10, [1, 1, 0], [1, 0, 0], half_id=2),
        ]
        metrics_before = score_prediction_runs(runs)
        values_before = [
            (run.labels.copy(), run.predictions.copy()) for run in runs
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.png"
            result = save_prediction_timeline(runs, 10.0, path, "Synthetic")
            self.assertEqual(result, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
        self.assertEqual(score_prediction_runs(runs), metrics_before)
        for run, (labels, predictions) in zip(runs, values_before):
            np.testing.assert_array_equal(run.labels, labels)
            np.testing.assert_array_equal(run.predictions, predictions)


if __name__ == "__main__":
    unittest.main()
