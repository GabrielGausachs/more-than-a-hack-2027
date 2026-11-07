import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_engineering.robustness import (
    CorruptionSpec,
    audit_feature_arrays,
    corrupt_feature_arrays,
    derive_corruption_profile,
    missing_run_lengths,
    recompute_ball_speed,
)
from feature_engineering.wcsf_features import build_team_features, validate_features
try:
    import torch  # noqa: F401
    from modeling.alive_dead.dataset import load_wcsf_data
    from modeling.alive_dead.evaluate import select_device
    from modeling.alive_dead.postprocess_had import Decoder
    from modeling.alive_dead.robustness import (
        _evaluate_probability_runs,
        _infer_oof,
        _load_config,
        _load_frozen_models,
        _validate_clean_baseline,
        enrich_metrics,
    )
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False
    _load_config = enrich_metrics = None


def feature_archive(n_frames=20, players_per_team=4):
    players = np.zeros((n_frames, 2, 24, 3), dtype=np.float32)
    location = np.zeros((n_frames, 2, 24), dtype=bool)
    speed = np.zeros_like(location)
    location[:, :, :players_per_team] = True
    speed[:, :, :players_per_team] = True
    for frame in range(n_frames):
        for team in range(2):
            for slot in range(players_per_team):
                players[frame, team, slot] = (
                    0.1 + 0.01 * frame + 0.001 * slot,
                    0.2 + 0.01 * team + 0.001 * slot,
                    1.0 + slot,
                )
    team_features, team_valid = build_team_features(players, location, speed)
    half = np.ones(n_frames, dtype=np.int8)
    half[n_frames // 2 :] = 2
    timestamp = np.r_[
        np.arange(n_frames // 2) * 100,
        np.arange(n_frames - n_frames // 2) * 100 + 100000,
    ].astype(np.int64)
    ball = np.zeros((n_frames, 3), dtype=np.float32)
    ball[:, 0] = np.linspace(0.1, 0.9, n_frames)
    ball[:, 1] = 0.5
    ball_location = np.ones(n_frames, dtype=bool)
    ball_speed = recompute_ball_speed(ball, ball_location, half, timestamp)
    return {
        "players": players,
        "player_location_valid": location,
        "player_speed_valid": speed,
        "team_features": team_features,
        "team_feature_valid": team_valid,
        "ball": ball,
        "ball_location_valid": ball_location,
        "ball_speed_valid": ball_speed,
        "timestamp_ms": timestamp,
        "half_id": half,
        "sample_index": np.arange(n_frames, dtype=np.int64),
        "source_fps": np.array(10.0, dtype=np.float32),
        "player_feature_names": np.array(["x", "y", "speed"]),
        "team_feature_names": np.array([
            "mean_speed", "median_speed", "speed_std", "tracked_player_count",
            "valid_speed_count", "speed_coverage",
        ]),
        "ball_feature_names": np.array(["x", "y", "speed"]),
        "team_order": np.array(["Home", "Away"]),
        "schema_version": np.array("1.1"),
    }


class FeatureAuditTests(unittest.TestCase):
    def test_missing_runs_are_split_by_half(self):
        valid = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
        half = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int8)
        self.assertEqual(missing_run_lengths(valid, half), {"1": [2], "2": [3]})

    def test_audit_detects_exact_coordinate_duplicate(self):
        features = feature_archive()
        features["players"][0, 0, 1, :2] = features["players"][0, 0, 0, :2]
        audit = audit_feature_arrays(features)
        duplicate = audit["duplicates"]["player_coordinates"]
        self.assertEqual(duplicate["frame_team_cases"], 1)
        self.assertEqual(duplicate["excess_player_slots"], 1)

    def test_profile_uses_mean_player_count_not_archive_length(self):
        wcsf = feature_archive(20, 4)
        metrica = feature_archive(30, 3)
        metrica["ball_location_valid"][[2, 3, 18]] = False
        metrica["ball"][[2, 3, 18], :2] = 0.0
        profile = derive_corruption_profile(wcsf, metrica)
        self.assertAlmostEqual(profile["player_observation_dropout_rate"], 0.25)


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        self.source = feature_archive(40, 4)
        reference = feature_archive(40, 3)
        reference["ball_location_valid"][[2, 3, 4, 22, 23, 24]] = False
        reference["ball"][[2, 3, 4, 22, 23, 24], :2] = 0.0
        reference["player_speed_valid"][:, :, 2] = False
        reference["players"][:, :, 2, 2] = 0.0
        reference["ball_speed_valid"][::2] = False
        reference["ball"][::2, 2] = 0.0
        self.profile = derive_corruption_profile(self.source, reference)

    def test_iid_corruption_is_deterministic_and_does_not_mutate_source(self):
        before = {key: value.copy() for key, value in self.source.items()}
        spec = CorruptionSpec("iid_ball_dropout", 1.0, 7)
        first, first_info, first_masks = corrupt_feature_arrays(self.source, spec, self.profile)
        second, second_info, second_masks = corrupt_feature_arrays(self.source, spec, self.profile)
        self.assertEqual(first_info, second_info)
        self.assertTrue(np.array_equal(
            first_masks["ball_location_dropped"], second_masks["ball_location_dropped"]
        ))
        for key in before:
            self.assertTrue(np.array_equal(self.source[key], before[key]), key)
        validate_features(first)

    def test_contiguous_gap_budget_is_exact_and_half_local(self):
        result, info, masks = corrupt_feature_arrays(
            self.source, CorruptionSpec("contiguous_ball_gaps", 1.0, 4), self.profile
        )
        dropped = masks["ball_location_dropped"]
        for half in (1, 2):
            in_half = self.source["half_id"] == half
            expected = round(
                self.profile["ball_location_missing_by_half"][str(half)] * in_half.sum()
            )
            self.assertEqual(int(dropped[in_half].sum()), expected)
        self.assertEqual(info["mask_counts"]["ball_location_dropped"], int(dropped.sum()))
        validate_features(result)

    def test_ball_dropout_invalidates_speed_at_both_gap_edges(self):
        ball = self.source["ball"].copy()
        valid = self.source["ball_location_valid"].copy()
        valid[5] = False
        ball[5, :2] = 0.0
        speed_valid = recompute_ball_speed(
            ball, valid, self.source["half_id"], self.source["timestamp_ms"]
        )
        self.assertFalse(speed_valid[5])
        self.assertFalse(speed_valid[6])
        self.assertEqual(ball[5, 2], 0.0)
        self.assertEqual(ball[6, 2], 0.0)

    def test_player_observation_and_speed_only_semantics(self):
        full, _, full_masks = corrupt_feature_arrays(
            self.source,
            CorruptionSpec("player_observation_dropout", 1.0, 1),
            self.profile,
        )
        dropped = full_masks["player_observation_dropped"]
        self.assertTrue((~full["player_location_valid"][dropped]).all())
        self.assertTrue((~full["player_speed_valid"][dropped]).all())
        speed_only, _, speed_masks = corrupt_feature_arrays(
            self.source,
            CorruptionSpec("player_speed_masking", 1.0, 1),
            self.profile,
        )
        masked = speed_masks["player_speed_masked"]
        self.assertTrue(speed_only["player_location_valid"][masked].all())
        self.assertTrue((~speed_only["player_speed_valid"][masked]).all())
        expected, expected_valid = build_team_features(
            speed_only["players"],
            speed_only["player_location_valid"],
            speed_only["player_speed_valid"],
        )
        self.assertTrue(np.array_equal(expected_valid, speed_only["team_feature_valid"]))
        self.assertTrue(np.allclose(expected, speed_only["team_features"]))


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for frozen-checkpoint CLI tests")
class RobustnessCliTests(unittest.TestCase):
    def test_config_rejects_competition_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "schema_version": "wcsf_had_robustness_v1",
                "wcsf_features": "w.npz",
                "wcsf_labels": "labels.npz",
                "metrica_features": "m.npz",
                "ground_truth": "evaluation/gt/file.csv",
                "output_root": "out",
                "models": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ground truth"):
                _load_config(path)

    def test_enriched_metrics_report_both_team_accuracies(self):
        metrics = {
            "frame_metrics": {"confusion_matrix": [[8, 1, 1], [2, 7, 1], [1, 1, 8]]}
        }
        enriched = enrich_metrics(metrics)
        self.assertAlmostEqual(enriched["ownership"]["strict_team_accuracy"], 0.75)
        self.assertAlmostEqual(enriched["ownership"]["both_alive_team_accuracy"], 15 / 18)
        self.assertTrue(enriched["ownership"]["predicted_share_defined"])


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for frozen-checkpoint acceptance")
class FullRobustnessAcceptanceTests(unittest.TestCase):
    def test_clean_oof_matches_both_saved_baselines(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs" / "alive_dead" / "wcsf_robustness.json"
        config = _load_config(config_path)
        required = [
            Path(config["wcsf_features"]),
            Path(config["wcsf_labels"]),
            *(Path(model["run_dir"]) for model in config["models"]),
        ]
        if not all(path.exists() for path in required):
            self.skipTest("Local gated WCSF checkpoints are not available")
        device = select_device("cpu")
        feature_path = Path(config["wcsf_features"])
        features = np.load(feature_path, allow_pickle=False)
        feature_arrays = {key: features[key].copy() for key in features.files}
        features.close()
        raw = load_wcsf_data(feature_path, config["wcsf_labels"], "had", True, "direct")
        models = _load_frozen_models(config["models"], device, feature_path)
        for model in models:
            runs = _infer_oof(model, raw, feature_arrays, device)
            raw_metrics = _evaluate_probability_runs(runs, Decoder("raw"), raw.source_fps)
            decoded_metrics = _evaluate_probability_runs(
                runs, model.decoder, raw.source_fps
            )
            _validate_clean_baseline(model, raw_metrics, decoded_metrics)


if __name__ == "__main__":
    unittest.main()
