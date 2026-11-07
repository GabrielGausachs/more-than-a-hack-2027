import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_engineering.wcsf_features import build_team_features
from modeling.alive_dead.augmentation import MaskAugmentedWindowTransform
from modeling.alive_dead.config import ExperimentConfig
from modeling.alive_dead.dataset import TemporalWindowDataset, WCSFData
from modeling.alive_dead.mask_augmented_pipeline import (
    _record_stage,
    _stage_complete,
    build_comparison_rows,
    load_pipeline_config,
)
from modeling.alive_dead.preprocessing import FeatureNormalizer
from modeling.alive_dead.robustness import METRIC_PATHS
from modeling.alive_dead.splits import Segment


def raw_data(n=30):
    players = np.zeros((n, 2, 24, 3), dtype=np.float32)
    location = np.zeros((n, 2, 24), dtype=bool)
    speed = np.zeros_like(location)
    location[:, :, :4] = True
    speed[:, :, :4] = True
    players[:, :, :4, 0] = 0.4
    players[:, :, :4, 1] = 0.6
    players[:, :, :4, 2] = 2.0
    team, team_valid = build_team_features(players, location, speed)
    ball = np.zeros((n, 3), dtype=np.float32)
    ball[:, 0] = np.arange(n) / 100.0
    ball[:, 1] = 0.5
    ball[:, 2] = 10.5
    return WCSFData(
        players, location, speed, team, team_valid,
        (np.arange(n) % 3).astype(np.int8),
        np.arange(n, dtype=np.int64) * 100,
        np.ones(n, dtype=np.int8), np.arange(n, dtype=np.int64), 10.0,
        ball, np.ones(n, dtype=bool), np.ones(n, dtype=bool),
    )


def augmented_config(scenario="contiguous_ball_gaps"):
    config = ExperimentConfig()
    config.model.model_type = "deepsets_tcn"
    config.model.classification_mode = "had"
    config.model.use_ball = True
    config.model.player_encoder_type = "ball_conditioned"
    config.data.num_workers = 0
    config.augmentation.enabled = True
    config.augmentation.probabilities = {
        name: float(name == scenario)
        for name in (
            "clean", "iid_ball_dropout", "contiguous_ball_gaps",
            "player_observation_dropout", "player_speed_masking",
            "ball_speed_masking", "combined",
        )
    }
    config.augmentation.severity_factors = (1.0,)
    return config


class MaskAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "ball_location_missing_by_half": {"1": 0.3},
            "ball_gap_lengths_by_half": {"1": [3, 5]},
            "player_observation_dropout_rate": 0.25,
            "player_speed_masking_rate": 0.3,
            "ball_speed_masking_rate": 0.4,
        }
        self.normalizer = FeatureNormalizer(
            2.0, 1.0, np.array([2.0, 2.0, 0.0]),
            np.array([1.0, 1.0, 1.0]), ball_speed_mean=10.0,
            ball_speed_std=1.0,
        )

    def test_disabled_config_is_backward_compatible(self):
        config = ExperimentConfig.from_dict({})
        self.assertFalse(config.augmentation.enabled)

    def test_selection_is_reproducible_and_epoch_aware(self):
        transform = MaskAugmentedWindowTransform(
            augmented_config(), self.normalizer, self.profile, "block2"
        )
        transform.set_epoch(1)
        first = transform.selection(7)
        self.assertEqual(first, transform.selection(7))
        transform.set_epoch(2)
        self.assertNotEqual(first[2], transform.selection(7)[2])

    def test_transform_changes_masks_without_changing_labels(self):
        data = raw_data()
        original_ball = data.ball.copy()
        original_valid = data.ball_location_valid.copy()
        transform = MaskAugmentedWindowTransform(
            augmented_config(), self.normalizer, self.profile, "block1"
        )
        dataset = TemporalWindowDataset(
            data, (Segment(0, len(data.labels), "train", 1),), 30, 30, 0,
            window_transform=transform,
        )
        dataset.set_epoch(1)
        batch = dataset[0]
        np.testing.assert_array_equal(batch["labels"].numpy(), data.labels.astype(np.float32))
        self.assertLess(int(batch["ball_location_valid"].sum()), len(data.labels))
        self.assertTrue(np.isfinite(batch["ball"].numpy()).all())
        self.assertTrue(np.all(batch["ball"].numpy()[~batch["ball_location_valid"].numpy()] == 0.0))
        np.testing.assert_array_equal(data.ball, original_ball)
        np.testing.assert_array_equal(data.ball_location_valid, original_valid)

    def test_full_player_dropout_masks_location_and_speed(self):
        transform = MaskAugmentedWindowTransform(
            augmented_config("player_observation_dropout"),
            self.normalizer, self.profile, "refit",
        )
        transform.set_epoch(1)
        output = transform(raw_data(), type("W", (), {"start": 0, "valid_length": 30})(), 0, 30)
        removed = ~output["player_location_valid"]
        self.assertTrue(np.all(~output["player_speed_valid"][removed]))
        self.assertTrue(np.all(output["players"][..., :2][removed] == 0.0))


def result_row(model, scenario, severity, seed, output, value):
    row = {
        "model": model, "scenario": scenario, "severity_factor": severity,
        "seed": seed, "output": output,
    }
    for metric in METRIC_PATHS:
        row[metric] = value
    return row


class PipelineReportTests(unittest.TestCase):
    def test_completed_stage_is_reused_only_while_hash_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = directory / "artifact.json"
            artifact.write_text("{}", encoding="utf-8")
            manifest = {"stages": {}}
            _record_stage(
                directory, manifest, "example", "completed",
                required_artifacts=[str(artifact)],
            )
            self.assertTrue(_stage_complete(manifest, "example"))
            artifact.write_text('{"changed": true}', encoding="utf-8")
            self.assertFalse(_stage_complete(manifest, "example"))

    def test_paired_formulas_make_less_loss_positive(self):
        baseline = [
            result_row("base", "clean", 0.0, "", "raw", 0.8),
            result_row("base", "iid_ball_dropout", 1.0, "0", "raw", 0.5),
        ]
        augmented = [
            result_row("aug", "clean", 0.0, "", "raw", 0.75),
            result_row("aug", "iid_ball_dropout", 1.0, "0", "raw", 0.6),
        ]
        rows = build_comparison_rows(baseline, augmented, {"base": "aug"})
        degraded = next(row for row in rows if row["scenario"] != "clean")
        self.assertAlmostEqual(degraded["baseline_degradation_spell_f1"], -0.3)
        self.assertAlmostEqual(degraded["augmented_degradation_spell_f1"], -0.15)
        self.assertAlmostEqual(degraded["robustness_gain_spell_f1"], 0.15)

    def test_pipeline_config_rejects_metrica_ground_truth(self):
        value = {
            "schema_version": "wcsf_mask_augmented_pipeline_v1",
            "baseline_robustness_config": "evaluation/gt/possession_gt.csv",
            "output_root": "out",
            "training_runs": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ground truth"):
                load_pipeline_config(path)


if __name__ == "__main__":
    unittest.main()
