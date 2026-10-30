"""Non-training tests for WCSF ALIVE/DEAD evaluation."""

import unittest
from pathlib import Path

import numpy as np
import torch

from modeling.alive_dead.config import ExperimentConfig
from modeling.alive_dead.evaluate import (
    CheckpointEvaluation,
    PredictionRun,
    combine_oof_runs,
    checkpoint_path,
    compute_frame_metrics,
    compute_segment_metrics,
    extract_segments,
    normalizer_from_state,
    inner_checkpoint_path,
    validate_checkpoint,
    validate_run_coverage,
)
from modeling.alive_dead.splits import build_fold_split


def prediction_run(start, values, predictions=None, half_id=1):
    labels = np.asarray(values, dtype=np.int8)
    predicted = labels.copy() if predictions is None else np.asarray(predictions, dtype=np.int8)
    return PredictionRun(start, start + len(labels), half_id, labels, predicted)


class FrameMetricTests(unittest.TestCase):
    def test_exact_metrics_and_confusion_orientation(self):
        result = compute_frame_metrics(
            np.array([0, 0, 1, 1], dtype=np.int8),
            np.array([0, 1, 1, 1], dtype=np.int8),
        )
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["per_class"]["DEAD"]["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(result["per_class"]["ALIVE"]["f1"], 0.8)
        self.assertAlmostEqual(result["macro_f1"], (2.0 / 3.0 + 0.8) / 2.0)

    def test_non_binary_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            compute_frame_metrics(np.array([0, 2]), np.array([0, 1]))


class SegmentMetricTests(unittest.TestCase):
    def test_metrica_style_iou_mean_and_headline(self):
        run = prediction_run(
            0,
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        )
        result = compute_segment_metrics([run])
        self.assertEqual(tuple(result["per_iou"]), ("0.3", "0.4", "0.5", "0.6", "0.7"))
        self.assertAlmostEqual(result["headline_f1"], 1.0)
        self.assertAlmostEqual(result["per_iou"]["0.7"]["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(result["mean_f1"], (4.0 + 2.0 / 3.0) / 5.0)
        self.assertEqual(result["n_ground_truth"], 3)
        self.assertEqual(result["n_predicted"], 3)

    def test_disconnected_runs_are_not_merged(self):
        first = prediction_run(0, [1, 1, 1], half_id=1)
        second = prediction_run(10, [1, 1], half_id=2)
        self.assertEqual(len(extract_segments(first.labels)), 1)
        result = compute_segment_metrics([first, second])
        self.assertEqual(result["n_ground_truth"], 2)
        self.assertEqual(result["n_predicted"], 2)
        self.assertAlmostEqual(result["mean_f1"], 1.0)

    def test_one_prediction_can_match_only_one_ground_truth_segment(self):
        run = prediction_run(0, [1, 1, 0, 1, 1], [1, 1, 1, 1, 1])
        result = compute_segment_metrics([run])["per_iou"]["0.3"]
        self.assertEqual(result["n_matched"], 1)
        self.assertEqual(result["n_missed"], 2)
        self.assertEqual(result["n_spurious"], 0)


class CheckpointAndCoverageTests(unittest.TestCase):
    def _normalizer_state(self):
        return {
            "player_speed_mean": 2.0,
            "player_speed_std": 1.5,
            "team_speed_mean": [2.0, 1.5, 0.5],
            "team_speed_std": [1.0, 1.0, 0.25],
            "epsilon": 1e-6,
        }

    def _checkpoint(self, fold=1, model_type="team_tcn", seed=42):
        config = ExperimentConfig()
        config.split.fold = fold
        config.model.model_type = model_type
        config.training.seed = seed
        return {
            "model_state": {},
            "config": config.to_dict(),
            "fold": fold,
            "normalizer": self._normalizer_state(),
            "seed": seed,
            "epoch": 1,
            "source_fps": 10.0,
        }

    def test_checkpoint_normalizer_is_reconstructed_without_fitting(self):
        state = self._normalizer_state()
        normalizer = normalizer_from_state(state)
        self.assertEqual(normalizer.state_dict(), state)

    def test_checkpoint_identity_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fold"):
            validate_checkpoint(self._checkpoint(fold=1), 2, "team_tcn", 42)
        with self.assertRaisesRegex(ValueError, "model type"):
            validate_checkpoint(self._checkpoint(), 1, "deepsets_tcn", 42)

    def test_new_paths_never_select_legacy_root_checkpoint(self):
        root = Path("runs")
        self.assertEqual(
            checkpoint_path(root, 1, "team_tcn", 42),
            root / "fold1_team_tcn_seed42/refit/final_checkpoint.pt",
        )
        self.assertEqual(
            inner_checkpoint_path(root, 1, "team_tcn", 42, 3),
            root / "fold1_team_tcn_seed42/inner_block3/best_checkpoint.pt",
        )

    def test_training_mode_and_block_mismatch_are_rejected(self):
        checkpoint = self._checkpoint()
        checkpoint["training_mode"] = "inner_cv"
        checkpoint["validation_block"] = 1
        checkpoint["config"]["split"]["validation_block"] = 1
        validate_checkpoint(checkpoint, 1, "team_tcn", 42, "inner_cv", 1)
        with self.assertRaisesRegex(ValueError, "validation block"):
            validate_checkpoint(checkpoint, 1, "team_tcn", 42, "inner_cv", 2)
        with self.assertRaisesRegex(ValueError, "training mode"):
            validate_checkpoint(checkpoint, 1, "team_tcn", 42, "full_half_refit")

    def test_incomplete_partition_coverage_is_rejected(self):
        half_id = np.r_[np.ones(20, dtype=np.int8), np.full(20, 2, dtype=np.int8)]
        split = build_fold_split(half_id, fold=1, validation_fraction=0.2, buffer_frames=2)
        test_segment = split.segments["test"][0]
        incomplete = prediction_run(
            test_segment.start,
            np.zeros(test_segment.length - 1, dtype=np.int8),
            half_id=2,
        )
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            validate_run_coverage([incomplete], split, "test", len(half_id))

    def test_oof_mapping_uses_fold2_for_h1_and_fold1_for_h2(self):
        fold1_run = prediction_run(4, [0, 1, 0, 1], half_id=2)
        fold2_run = prediction_run(0, [0, 1, 0, 1], half_id=1)
        base = {
            "model_type": "team_tcn",
            "seed": 42,
            "checkpoint_path": Path("unused.pt"),
            "n_frames_total": 8,
            "source_fps": 10.0,
        }
        fold1 = CheckpointEvaluation(
            fold=1,
            runs=(fold1_run,),
            report={"partition": "test"},
            **base,
        )
        fold2 = CheckpointEvaluation(
            fold=2,
            runs=(fold2_run,),
            report={"partition": "test"},
            **base,
        )
        combined = combine_oof_runs(fold1, fold2)
        self.assertEqual([run.half_id for run in combined], [1, 2])
        self.assertEqual([run.start for run in combined], [0, 4])


class RealCheckpointTests(unittest.TestCase):
    def test_existing_checkpoints_have_consistent_metadata_and_finite_weights(self):
        root = Path(__file__).resolve().parents[1]
        runs_dir = root / "data/open_data_groningen/model_runs"
        found = 0
        for fold in (1, 2):
            for model_type in ("team_tcn", "deepsets_tcn"):
                path = runs_dir / f"fold{fold}_{model_type}_seed42/best_checkpoint.pt"
                if not path.exists():
                    continue
                found += 1
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                validate_checkpoint(checkpoint, fold, model_type, 42)
                self.assertTrue(
                    all(
                        torch.isfinite(value).all()
                        for value in checkpoint["model_state"].values()
                        if torch.is_floating_point(value)
                    )
                )
        if not found:
            self.skipTest("Local trained checkpoints are unavailable")


if __name__ == "__main__":
    unittest.main()
