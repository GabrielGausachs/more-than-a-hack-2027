"""Synthetic tests for full-timeline Metrica H/A/D transfer helpers."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from modeling.alive_dead.metrica_had import (
    average_categorical_probabilities,
    decode_probabilities_by_half,
    evaluate_labels,
    load_comparison_config,
    load_global_decoder,
    reconstruct_source_labels,
    reconstruct_source_probabilities,
    write_submission,
)
from modeling.alive_dead.postprocess_had import Decoder


class MetricaHADTransferTests(unittest.TestCase):
    def test_single_model_transfer_config_is_supported(self):
        config = {
            "schema_version": "metrica_had_model_comparison_v1",
            "ensemble_weights": {"fold1": 0.5, "fold2": 0.5},
            "models": [{
                "name": "one_model",
                "run_dir": "run",
                "decoder_selection": "selection.json",
                "expected_transition_loss_weight": 0.0,
                "expected_decoder": {"method": "viterbi", "transition_penalty": 1.0},
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            models, weights = load_comparison_config(path)
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "one_model")
        self.assertEqual(weights, {1: 0.5, 2: 0.5})

    def test_weighted_categorical_ensemble(self):
        first = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]], dtype=np.float32)
        second = np.array([[0.2, 0.7, 0.1], [0.3, 0.2, 0.5]], dtype=np.float32)
        result = average_categorical_probabilities(
            {1: first, 2: second}, {1: 0.5, 2: 0.5}
        )
        np.testing.assert_allclose(result, [[0.5, 0.4, 0.1], [0.2, 0.45, 0.35]])
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_reconstructs_source_labels_and_x_outside_scored_intervals(self):
        labels, valid = reconstruct_source_labels(
            np.array([pm.H, pm.A, pm.D], dtype=np.int8),
            np.array([-1, 0, 0, 1, 2, -1], dtype=np.int64),
        )
        np.testing.assert_array_equal(labels, [pm.X, pm.H, pm.H, pm.A, pm.D, pm.X])
        np.testing.assert_array_equal(valid, [False, True, True, True, True, False])

    def test_reconstructs_categorical_probabilities(self):
        source, valid = reconstruct_source_probabilities(
            np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]], dtype=np.float32),
            np.array([0, 0, -1, 1], dtype=np.int64),
        )
        np.testing.assert_allclose(source[0], [0.8, 0.1, 0.1])
        np.testing.assert_allclose(source[1], source[0])
        np.testing.assert_allclose(source[2], 0.0)
        np.testing.assert_array_equal(valid, [True, True, False, True])

    def test_raw_decoding_is_argmax_and_restarts_by_half(self):
        values = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.9, 0.05, 0.05],
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.9, 0.05],
            ],
            dtype=np.float32,
        )
        halves = np.array([1, 1, 1, 2, 2, 2], dtype=np.int8)
        raw = decode_probabilities_by_half(values, halves, Decoder("raw"))
        decoded = decode_probabilities_by_half(values, halves, Decoder("viterbi", 100.0))
        np.testing.assert_array_equal(raw, values.argmax(axis=1))
        np.testing.assert_array_equal(decoded, [pm.H, pm.H, pm.H, pm.A, pm.A, pm.A])

    def test_rejects_obsolete_fold_specific_decoder_artifact(self):
        report = {
            "schema_version": "wcsf_had_temporal_decoding_v1",
            "best": {"selected": {"method": "viterbi", "transition_penalty": 6.0}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "v2 or v3"):
                load_global_decoder(
                    path, {"method": "viterbi", "transition_penalty": 6.0}
                )

    def test_submission_round_trip_passes_official_validation(self):
        labels = np.array([pm.X, pm.H, pm.A, pm.D, pm.X], dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            path = write_submission(Path(directory) / "submission.csv", labels)
            frame = pm.read_label_csv(path)
            self.assertIsNone(pm.validate_submission(frame, len(labels)))
            np.testing.assert_array_equal(pm.label_codes(frame), labels)

    def test_metric_report_matches_authoritative_evaluator(self):
        gt = np.array([pm.H, pm.H, pm.A, pm.A, pm.D, pm.D], dtype=np.int8)
        prediction = np.array([pm.H, pm.A, pm.A, pm.A, pm.D, pm.H], dtype=np.int8)
        direct = pm.evaluate(gt, prediction, bootstrap_n=0)
        report = evaluate_labels(gt, prediction, bootstrap_n=0)
        self.assertEqual(report["macro_f1"], direct["macro_f1"])
        self.assertEqual(report["spell_f1_mean"], direct["spell_f1_mean"])
        self.assertEqual(report["confusion"], direct["confusion"])
        self.assertEqual(report["transitions"]["12"]["f1"], direct["transitions"][12]["f1"])


if __name__ == "__main__":
    unittest.main()
