"""Lightweight integrity tests for the ALIVE/DEAD modeling package."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from modeling.alive_dead.config import ExperimentConfig, ModelConfig
from modeling.alive_dead.dataset import (
    TemporalWindowDataset,
    WCSFData,
    build_dataloaders,
    load_wcsf_data,
)
from modeling.alive_dead.player_encoder import DeepSetPlayerEncoder
from modeling.alive_dead.preprocessing import FeatureNormalizer
from modeling.alive_dead.splits import Segment, build_fold_split, build_refit_split
from modeling.alive_dead.tcn import build_model
from modeling.alive_dead.train import masked_bce_components
from modeling.alive_dead.train import run_cross_validation


def synthetic_data(n_frames=24):
    players = np.zeros((n_frames, 2, 24, 3), dtype=np.float32)
    location_valid = np.zeros((n_frames, 2, 24), dtype=bool)
    speed_valid = np.zeros_like(location_valid)
    location_valid[:, :, :2] = True
    speed_valid[:, :, :2] = True
    players[:, :, :2, 0] = 0.5
    players[:, :, :2, 1] = 0.5
    players[:, :, 0, 2] = 1.0
    players[:, :, 1, 2] = 3.0
    team_features = np.zeros((n_frames, 2, 6), dtype=np.float32)
    time = np.arange(n_frames, dtype=np.float32)[:, None]
    team_features[..., 0] = 1.0 + 0.10 * time
    team_features[..., 1] = 1.5 + 0.08 * time
    team_features[..., 2] = 0.5 + 0.03 * time
    team_features[..., 3:] = [2.0, 2.0, 1.0]
    return WCSFData(
        players=players,
        player_location_valid=location_valid,
        player_speed_valid=speed_valid,
        team_features=team_features,
        team_feature_valid=np.ones((n_frames, 2, 6), dtype=bool),
        labels=(np.arange(n_frames) % 2).astype(np.int8),
        timestamp_ms=np.arange(n_frames, dtype=np.int64) * 100,
        half_id=np.where(np.arange(n_frames) < n_frames // 2, 1, 2).astype(np.int8),
        sample_index=np.arange(n_frames, dtype=np.int64),
        source_fps=10.0,
    )


class SplitTests(unittest.TestCase):
    def test_exact_three_block_fold_boundaries(self):
        half_id = np.r_[np.ones(28014, dtype=np.int8), np.full(29360, 2, dtype=np.int8)]
        expected = {
            1: [(4902, 9104), (11906, 16108), (18909, 23111)],
            2: [(33152, 37556), (40492, 44896), (47832, 52236)],
        }
        for fold in (1, 2):
            for block, interval in enumerate(expected[fold], 1):
                split = build_fold_split(half_id, fold, 0.15, 100, block)
                validation = split.segments["validation"][0]
                self.assertEqual((validation.start, validation.end), interval)
                self.assertEqual(split.validation_block, block)
                left_buffer, right_buffer = split.segments["buffer"]
                self.assertEqual(
                    (left_buffer.start, left_buffer.end),
                    (validation.start - 100, validation.start),
                )
                self.assertEqual(
                    (right_buffer.start, right_buffer.end),
                    (validation.end, validation.end + 100),
                )
                left_train, right_train = split.segments["train"]
                self.assertEqual(left_train.end, left_buffer.start)
                self.assertEqual(right_train.start, right_buffer.end)
                self.assertEqual(np.count_nonzero(split.buffer_mask), 200)
                self.assertEqual(
                    np.count_nonzero(split.validation_mask), 4202 if fold == 1 else 4404
                )
                split.assert_valid()

    def test_full_half_refit_has_no_validation_or_buffer(self):
        half_id = np.r_[np.ones(28014, dtype=np.int8), np.full(29360, 2, dtype=np.int8)]
        fold1 = build_refit_split(half_id, 1)
        fold2 = build_refit_split(half_id, 2)
        self.assertEqual(
            tuple(map(np.count_nonzero, (fold1.train_mask, fold1.validation_mask, fold1.buffer_mask, fold1.test_mask))),
            (28014, 0, 0, 29360),
        )
        self.assertEqual(
            tuple(map(np.count_nonzero, (fold2.train_mask, fold2.validation_mask, fold2.buffer_mask, fold2.test_mask))),
            (29360, 0, 0, 28014),
        )
        fold1.assert_valid()
        fold2.assert_valid()


class DatasetAndNormalizationTests(unittest.TestCase):
    def test_windows_do_not_cross_segments_and_weights_sum_to_one(self):
        data = synthetic_data(24)
        segments = (Segment(0, 7, "train", 1), Segment(12, 24, "train", 2))
        dataset = TemporalWindowDataset(data, segments, sequence_frames=6, stride_frames=3)
        reconstructed_weight = np.zeros(24, dtype=np.float32)
        for item in dataset:
            indices = item["global_index"].numpy()
            mask = item["output_mask"].numpy()
            valid_indices = indices[mask]
            self.assertTrue(np.all(np.diff(valid_indices) == 1))
            self.assertFalse(np.any((valid_indices >= 7) & (valid_indices < 12)))
            reconstructed_weight[valid_indices] += item["loss_weight"].numpy()[mask]
        np.testing.assert_allclose(reconstructed_weight[:7], 1.0)
        np.testing.assert_allclose(reconstructed_weight[12:], 1.0)
        np.testing.assert_allclose(reconstructed_weight[7:12], 0.0)

    def test_normalizer_uses_training_frames_and_preserves_discrete_features(self):
        data = synthetic_data(24)
        data.players[12:, ..., 2] = 1000.0
        train_mask = np.zeros(24, dtype=bool)
        train_mask[:12] = True
        normalizer = FeatureNormalizer.fit(
            data.players,
            data.player_speed_valid,
            data.team_features,
            data.team_feature_valid,
            train_mask,
        )
        self.assertAlmostEqual(normalizer.player_speed_mean, 2.0)
        players, teams = normalizer.transform(
            data.players,
            data.player_location_valid,
            data.player_speed_valid,
            data.team_features,
            data.team_feature_valid,
        )
        self.assertTrue(np.isfinite(players).all())
        self.assertTrue(np.isfinite(teams).all())
        np.testing.assert_array_equal(teams[..., 3:], data.team_features[..., 3:])
        self.assertTrue(np.all(players[..., 2][~data.player_speed_valid] == 0.0))


class ModelTests(unittest.TestCase):
    def _batch(self, batch_size=2, length=20):
        return {
            "players": torch.randn(batch_size, length, 2, 24, 3),
            "player_location_valid": torch.ones(batch_size, length, 2, 24, dtype=torch.bool),
            "player_speed_valid": torch.ones(batch_size, length, 2, 24, dtype=torch.bool),
            "team_features": torch.randn(batch_size, length, 2, 6),
            "team_feature_valid": torch.ones(batch_size, length, 2, 6, dtype=torch.bool),
            "output_mask": torch.ones(batch_size, length, dtype=torch.bool),
            "labels": torch.randint(0, 2, (batch_size, length), dtype=torch.float32),
            "loss_weight": torch.ones(batch_size, length),
        }

    def test_deep_sets_is_permutation_invariant(self):
        torch.manual_seed(1)
        encoder = DeepSetPlayerEncoder().eval()
        players = torch.randn(2, 5, 2, 24, 3)
        location_valid = torch.rand(2, 5, 2, 24) > 0.2
        speed_valid = location_valid & (torch.rand(2, 5, 2, 24) > 0.2)
        permutation = torch.randperm(24)
        original = encoder(players, location_valid, speed_valid)
        permuted = encoder(
            players[..., permutation, :],
            location_valid[..., permutation],
            speed_valid[..., permutation],
        )
        torch.testing.assert_close(original, permuted)

    def test_both_models_produce_finite_sequence_logits_and_loss(self):
        batch = self._batch()
        for model_type in ("team_tcn", "deepsets_tcn"):
            with self.subTest(model_type=model_type):
                config = ModelConfig(model_type=model_type)
                model = build_model(config).eval()
                with torch.no_grad():
                    logits = model(batch)
                self.assertEqual(logits.shape, batch["labels"].shape)
                self.assertTrue(torch.isfinite(logits).all())
                numerator, denominator = masked_bce_components(
                    logits, batch["labels"], batch["output_mask"], batch["loss_weight"]
                )
                self.assertTrue(torch.isfinite(numerator / denominator))

    def test_default_receptive_field_and_unsafe_configuration(self):
        config = ExperimentConfig()
        self.assertEqual(config.model.receptive_field_frames, 61)
        config.validate(10.0)
        config.model.dilations = (1, 2, 4, 8, 16, 32)
        with self.assertRaisesRegex(ValueError, "receptive field"):
            config.validate(10.0)

    def test_padding_values_cannot_change_valid_logits(self):
        torch.manual_seed(2)
        model = build_model(ModelConfig(model_type="team_tcn", dropout=0.0)).eval()
        batch = self._batch(batch_size=1, length=100)
        batch["output_mask"][:, 60:] = False
        changed = {key: value.clone() for key, value in batch.items()}
        changed["team_features"][:, 60:] = 1e6
        with torch.no_grad():
            first = model(batch)[:, :60]
            second = model(changed)[:, :60]
        torch.testing.assert_close(first, second)


class RealArchiveLoadingTests(unittest.TestCase):
    def test_existing_archives_load_without_label_provenance(self):
        root = Path(__file__).resolve().parents[1]
        feature_path = root / "data/open_data_groningen/processed/wcsf_player_team_features.npz"
        label_path = root / "data/open_data_groningen/processed/wcsf_alive_dead_labels.npz"
        if not feature_path.exists() or not label_path.exists():
            self.skipTest("Local processed WCSF archives are unavailable")
        data = load_wcsf_data(feature_path, label_path)
        self.assertEqual(data.players.shape, (57374, 2, 24, 3))
        self.assertEqual(data.labels.shape, (57374,))
        self.assertFalse(hasattr(data, "alive_vote_fraction"))
        self.assertFalse(hasattr(data, "dead_vote_count"))

    def test_refit_normalizer_uses_complete_training_half_only(self):
        root = Path(__file__).resolve().parents[1]
        feature_path = root / "data/open_data_groningen/processed/wcsf_player_team_features.npz"
        label_path = root / "data/open_data_groningen/processed/wcsf_alive_dead_labels.npz"
        if not feature_path.exists() or not label_path.exists():
            self.skipTest("Local processed WCSF archives are unavailable")
        config = ExperimentConfig()
        config.data.feature_path = str(feature_path)
        config.data.label_path = str(label_path)
        refit_loaders = build_dataloaders(config, full_half_refit=True)
        self.assertEqual(np.count_nonzero(refit_loaders.split.train_mask), 28014)
        self.assertEqual(len(refit_loaders.validation.dataset), 0)
        self.assertFalse(
            refit_loaders.split.test_mask[refit_loaders.split.train_mask].any()
        )
        raw = load_wcsf_data(feature_path, label_path)
        expected_refit = FeatureNormalizer.fit(
            raw.players,
            raw.player_speed_valid,
            raw.team_features,
            raw.team_feature_valid,
            refit_loaders.split.train_mask,
        )
        np.testing.assert_allclose(
            refit_loaders.normalizer.team_speed_mean, expected_refit.team_speed_mean
        )
        self.assertAlmostEqual(
            refit_loaders.normalizer.player_speed_mean,
            expected_refit.player_speed_mean,
        )

        config.split.validation_block = 1
        inner_loaders = build_dataloaders(config)
        excluded = (
            inner_loaders.split.validation_mask
            | inner_loaders.split.buffer_mask
            | inner_loaders.split.test_mask
        )
        self.assertFalse(inner_loaders.split.train_mask[excluded].any())
        expected_inner = FeatureNormalizer.fit(
            raw.players,
            raw.player_speed_valid,
            raw.team_features,
            raw.team_feature_valid,
            inner_loaders.split.train_mask,
        )
        np.testing.assert_allclose(
            inner_loaders.normalizer.team_speed_mean,
            expected_inner.team_speed_mean,
        )
        self.assertAlmostEqual(
            inner_loaders.normalizer.player_speed_mean,
            expected_inner.player_speed_mean,
        )


class CrossValidationWorkflowTests(unittest.TestCase):
    def test_three_blocks_choose_median_epoch_before_refit(self):
        config = ExperimentConfig()
        config.model.model_type = "team_tcn"
        best_epochs = {1: 5, 2: 9, 3: 7}

        def fake_train(inner_config, run_dir):
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            block = inner_config.split.validation_block
            return {
                "checkpoint_path": str(run_dir / "best_checkpoint.pt"),
                "best_validation_loss": 0.4 - block * 0.01,
                "best_epoch": best_epochs[block],
            }

        def fake_evaluate(*args, validation_block, **kwargs):
            return SimpleNamespace(
                report={
                    "runs": [{"start": validation_block * 10, "end": validation_block * 10 + 5}],
                    "frame_metrics": {"macro_f1": 0.8 + validation_block * 0.01},
                    "segment_metrics": {"mean_f1": 0.5 + validation_block * 0.01},
                }
            )

        with tempfile.TemporaryDirectory() as temporary:
            config.training.output_dir = temporary
            with mock.patch(
                "modeling.alive_dead.train.train_experiment", side_effect=fake_train
            ) as inner_train, mock.patch(
                "modeling.alive_dead.train.evaluate_checkpoint", side_effect=fake_evaluate
            ), mock.patch(
                "modeling.alive_dead.train.train_full_half_refit",
                return_value={"checkpoint_path": "final_checkpoint.pt"},
            ) as refit:
                summary = run_cross_validation(config)

            self.assertEqual(inner_train.call_count, 3)
            self.assertEqual(summary["inner_best_epochs"], [5, 9, 7])
            self.assertEqual(summary["refit_epochs"], 7)
            self.assertEqual(refit.call_args.args[1], 7)
            self.assertTrue(
                (Path(temporary) / "fold1_team_tcn_seed42/cv_summary.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
