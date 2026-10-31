"""Synthetic tests for direct Home/Away/Dead evaluation."""

import unittest

import numpy as np

from modeling.alive_dead.evaluate_had import HADRun, compute_had_metrics


class HADEvaluationTests(unittest.TestCase):
    def test_exact_frame_metrics_and_confusion_orientation(self):
        run = HADRun(
            0,
            6,
            1,
            np.array([0, 0, 1, 1, 2, 2], dtype=np.int8),
            np.array([0, 1, 1, 1, 2, 0], dtype=np.int8),
        )
        metrics = compute_had_metrics([run])
        self.assertEqual(
            metrics["frame_metrics"]["confusion_matrix"],
            [[1, 1, 0], [0, 2, 0], [1, 0, 1]],
        )
        self.assertAlmostEqual(metrics["frame_metrics"]["accuracy"], 4 / 6)
        self.assertEqual(metrics["frame_metrics"]["per_class"]["A"]["support"], 2)

    def test_disconnected_valid_runs_break_spells_and_transitions(self):
        runs = [
            HADRun(0, 3, 1, np.zeros(3, dtype=np.int8), np.zeros(3, dtype=np.int8)),
            HADRun(4, 7, 1, np.zeros(3, dtype=np.int8), np.zeros(3, dtype=np.int8)),
            HADRun(10, 13, 2, np.zeros(3, dtype=np.int8), np.zeros(3, dtype=np.int8)),
        ]
        metrics = compute_had_metrics(runs, source_fps=10.0)
        self.assertEqual(metrics["spell_metrics"]["n_ground_truth"], 3)
        self.assertEqual(metrics["spell_metrics"]["n_predicted"], 3)
        self.assertEqual(metrics["transition_metrics"]["5"]["n_gt"], 0)
        self.assertAlmostEqual(
            metrics["possession_summary"]["ground_truth_mean_duration_seconds"], 0.3
        )

    def test_non_had_values_are_rejected(self):
        run = HADRun(
            0, 2, 1, np.array([0, -1], dtype=np.int8), np.array([0, 1], dtype=np.int8)
        )
        with self.assertRaisesRegex(ValueError, "invalid class"):
            compute_had_metrics([run])


if __name__ == "__main__":
    unittest.main()
