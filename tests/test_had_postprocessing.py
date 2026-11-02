"""Synthetic tests for leakage-safe H/A/D temporal decoding."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from modeling.alive_dead.evaluate_had import HADRun, compute_had_metrics
from modeling.alive_dead.postprocess_had import (
    Decoder,
    ProbabilityRun,
    _combined_oof_report,
    decode_probability_runs,
    load_probability_cache,
    save_probability_cache,
    select_global_decoders,
    smooth_probabilities,
    viterbi_decode,
)


def probability_run(start, half, labels, predicted=None, confidence=0.9):
    labels = np.asarray(labels, dtype=np.int8)
    predicted = labels if predicted is None else np.asarray(predicted, dtype=np.int8)
    probabilities = np.full((len(labels), 3), (1.0 - confidence) / 2.0, dtype=np.float32)
    probabilities[np.arange(len(labels)), predicted] = confidence
    return ProbabilityRun(start, start + len(labels), half, labels, probabilities)


class HADPostprocessingTests(unittest.TestCase):
    def test_centered_smoothing_renormalizes_sequence_edges(self):
        probabilities = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        smoothed = smooth_probabilities(probabilities, 3)
        np.testing.assert_allclose(smoothed[0], [0.5, 0.5, 0.0])
        np.testing.assert_allclose(smoothed[1], [1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(smoothed.sum(axis=1), 1.0)
        with self.assertRaisesRegex(ValueError, "odd"):
            smooth_probabilities(probabilities, 4)

    def test_viterbi_zero_penalty_equals_raw_argmax(self):
        probabilities = np.array(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            viterbi_decode(probabilities, 0.0), probabilities.argmax(axis=1)
        )

    def test_viterbi_penalty_removes_short_fragment(self):
        probabilities = np.array(
            [[0.9, 0.05, 0.05], [0.45, 0.50, 0.05], [0.9, 0.05, 0.05]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(viterbi_decode(probabilities, 0.0), [0, 1, 0])
        np.testing.assert_array_equal(viterbi_decode(probabilities, 1.0), [0, 0, 0])

    def test_decoding_restarts_at_disconnected_and_unknown_label_gaps(self):
        runs = (
            probability_run(0, 1, [0, 0]),
            probability_run(3, 1, [1, 1]),
            probability_run(10, 2, [2, 2]),
        )
        decoded = decode_probability_runs(runs, Decoder("smoothing", 41))
        self.assertEqual([(run.start, run.end) for run in decoded], [(0, 2), (3, 5), (10, 12)])
        np.testing.assert_array_equal(decoded[0].predictions, [0, 0])
        np.testing.assert_array_equal(decoded[1].predictions, [1, 1])
        np.testing.assert_array_equal(decoded[2].predictions, [2, 2])

    def test_raw_decoder_matches_existing_had_metrics(self):
        source = probability_run(0, 1, [0, 0, 1, 2], [0, 1, 1, 2])
        decoded = decode_probability_runs([source], Decoder("raw"))
        expected = HADRun(0, 4, 1, source.labels, np.array([0, 1, 1, 2], dtype=np.int8))
        self.assertEqual(compute_had_metrics(decoded), compute_had_metrics([expected]))

    def test_global_selection_uses_all_six_inner_blocks(self):
        blocks = {
            f"fold{fold}_block{block}": (
                probability_run(fold * 100 + block * 10, fold, [0, 0, 1, 1, 2, 2]),
            )
            for fold in (1, 2)
            for block in (1, 2, 3)
        }
        report, selected = select_global_decoders(
            blocks, 10.0, smoothing_windows=(3, 5), viterbi_penalties=(0.0, 1.0)
        )
        self.assertEqual(set(report["raw"]["blocks"]), set(blocks))
        self.assertIn("six inner-validation blocks", report["selection_scope"])
        self.assertEqual(set(selected), {"raw", "smoothing", "viterbi", "selected"})
        self.assertEqual(selected["raw"], Decoder("raw"))

        incomplete = dict(blocks)
        incomplete.pop("fold2_block3")
        with self.assertRaisesRegex(ValueError, "six inner blocks"):
            select_global_decoders(
                incomplete,
                10.0,
                smoothing_windows=(3,),
                viterbi_penalties=(0.0,),
            )

    def test_probability_cache_round_trip_and_identity_rejection(self):
        identity = {
            "schema_version": "test",
            "entries": ["inner_f1_b1"],
            "files": {"checkpoint": "digest"},
        }
        entries = {"inner_f1_b1": (probability_run(4, 1, [0, 1, 2]),)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probabilities.npz"
            save_probability_cache(path, identity, entries)
            loaded = load_probability_cache(path, identity)
            self.assertEqual(loaded["inner_f1_b1"][0].start, 4)
            np.testing.assert_allclose(
                loaded["inner_f1_b1"][0].probabilities,
                entries["inner_f1_b1"][0].probabilities,
            )
            changed = {**identity, "files": {"checkpoint": "different"}}
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_probability_cache(path, changed)

    def test_oof_mapping_is_enforced(self):
        selections = {
            name: Decoder("raw")
            for name in ("raw", "smoothing", "viterbi", "selected")
        }
        outer = {
            1: (probability_run(10, 2, [0, 1, 2]),),
            2: (probability_run(0, 1, [0, 1, 2]),),
        }
        report = _combined_oof_report(outer, selections, 10.0)
        self.assertEqual(report["oof_mapping"], {"half_1": "fold_2", "half_2": "fold_1"})
        self.assertEqual(
            report["approaches"]["selected"]["folds"]["1"]["decoder"],
            report["approaches"]["selected"]["folds"]["2"]["decoder"],
        )
        with self.assertRaisesRegex(ValueError, "Fold 2 for H1"):
            _combined_oof_report({1: outer[2], 2: outer[1]}, selections, 10.0)


if __name__ == "__main__":
    unittest.main()
