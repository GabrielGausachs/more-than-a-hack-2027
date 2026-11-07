"""Deterministic raw-window mask augmentation for WCSF H/A/D training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from feature_engineering.robustness import (
    CorruptionSpec,
    corrupt_feature_arrays,
    derive_corruption_profile,
    file_sha256,
    load_feature_archive,
)

from .config import ExperimentConfig
from .preprocessing import FeatureNormalizer


RNG_RULE = (
    "SHA-256(training seed, fold, validation block/refit, epoch, window index); "
    "scenario and severity are sampled from independent digest-derived RNG streams"
)


def _seed(*parts: object) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def augmentation_provenance(config: ExperimentConfig) -> dict[str, Any] | None:
    """Return the complete label-free augmentation provenance for a run."""
    if not config.augmentation.enabled:
        return None
    wcsf_path = config.data.feature_path
    reference_path = config.augmentation.reference_feature_path
    profile = derive_corruption_profile(
        load_feature_archive(wcsf_path), load_feature_archive(reference_path)
    )
    return {
        "enabled": True,
        "policy": config.augmentation.probabilities,
        "severity_factors": list(config.augmentation.severity_factors),
        "reference_feature_path": reference_path,
        "reference_feature_sha256": file_sha256(reference_path),
        "wcsf_feature_sha256": file_sha256(wcsf_path),
        "derived_profile": profile,
        "rng_rule": RNG_RULE,
        "normalization": "clean training-fold normalizer applied after corruption",
        "validation_and_test": "clean",
    }


@dataclass
class MaskAugmentedWindowTransform:
    """Corrupt one raw temporal window, then apply its clean-fold normalizer."""

    config: ExperimentConfig
    normalizer: FeatureNormalizer
    profile: Mapping[str, Any]
    mode_token: str
    epoch: int = 0

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        normalizer: FeatureNormalizer,
        *,
        full_half_refit: bool,
    ) -> "MaskAugmentedWindowTransform":
        profile = derive_corruption_profile(
            load_feature_archive(config.data.feature_path),
            load_feature_archive(config.augmentation.reference_feature_path),
        )
        mode = "refit" if full_half_refit else f"block{config.split.validation_block}"
        return cls(config, normalizer, profile, mode)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("Augmentation epoch must be positive")
        self.epoch = int(epoch)

    def selection(self, window_index: int) -> tuple[str, float, int]:
        """Return deterministic scenario, severity, and corruption seed."""
        if self.epoch < 1:
            raise RuntimeError("set_epoch must be called before augmented sampling")
        base = (
            self.config.training.seed,
            self.config.split.fold,
            self.mode_token,
            self.epoch,
            int(window_index),
        )
        rng = np.random.default_rng(_seed(*base, "selection"))
        names = tuple(self.config.augmentation.probabilities)
        probabilities = np.asarray(
            [self.config.augmentation.probabilities[name] for name in names],
            dtype=np.float64,
        )
        scenario = str(rng.choice(names, p=probabilities))
        severity = float(rng.choice(self.config.augmentation.severity_factors))
        return scenario, severity, _seed(*base, "corruption")

    @staticmethod
    def _raw_window(data: Any, window: Any) -> dict[str, np.ndarray]:
        start = window.start
        stop = start + window.valid_length
        return {
            "players": data.players[start:stop].copy(),
            "player_location_valid": data.player_location_valid[start:stop].copy(),
            "player_speed_valid": data.player_speed_valid[start:stop].copy(),
            "team_features": data.team_features[start:stop].copy(),
            "team_feature_valid": data.team_feature_valid[start:stop].copy(),
            "ball": data.ball[start:stop].copy(),
            "ball_location_valid": data.ball_location_valid[start:stop].copy(),
            "ball_speed_valid": data.ball_speed_valid[start:stop].copy(),
            "timestamp_ms": data.timestamp_ms[start:stop].copy(),
            "half_id": data.half_id[start:stop].copy(),
            "sample_index": np.arange(window.valid_length, dtype=np.int64),
        }

    @staticmethod
    def _restore_boundary_speed(
        before: Mapping[str, np.ndarray], after: dict[str, np.ndarray]
    ) -> None:
        # The first speed may legitimately depend on the clean predecessor just
        # outside this window. Preserve it only if this frame's location survived.
        if (
            bool(before["ball_speed_valid"][0])
            and bool(after["ball_location_valid"][0])
        ):
            after["ball"][0, 2] = before["ball"][0, 2]
            after["ball_speed_valid"][0] = True

    def _corrupt(
        self, source: dict[str, np.ndarray], scenario: str, severity: float, seed: int
    ) -> dict[str, np.ndarray]:
        if scenario == "clean":
            return source
        scenarios = (
            ("contiguous_ball_gaps", "player_observation_dropout", "player_speed_masking", "ball_speed_masking")
            if scenario == "combined"
            else (scenario,)
        )
        result = source
        for offset, item in enumerate(scenarios):
            before = result
            result, _, _ = corrupt_feature_arrays(
                result,
                CorruptionSpec(item, severity, _seed(seed, item, offset)),
                self.profile,
            )
            if item in ("iid_ball_dropout", "contiguous_ball_gaps"):
                self._restore_boundary_speed(before, result)
        return result

    @staticmethod
    def _pad(values: np.ndarray, length: int) -> np.ndarray:
        output = np.zeros((length, *values.shape[1:]), dtype=values.dtype)
        output[: len(values)] = values
        return output

    def __call__(
        self, data: Any, window: Any, window_index: int, sequence_frames: int
    ) -> dict[str, np.ndarray]:
        scenario, severity, seed = self.selection(window_index)
        raw = self._raw_window(data, window)
        augmented = self._corrupt(raw, scenario, severity, seed)
        players, team = self.normalizer.transform(
            augmented["players"],
            augmented["player_location_valid"],
            augmented["player_speed_valid"],
            augmented["team_features"],
            augmented["team_feature_valid"],
        )
        ball = self.normalizer.transform_ball(
            augmented["ball"],
            augmented["ball_location_valid"],
            augmented["ball_speed_valid"],
        )
        names = {
            "players": players,
            "player_location_valid": augmented["player_location_valid"],
            "player_speed_valid": augmented["player_speed_valid"],
            "team_features": team,
            "team_feature_valid": augmented["team_feature_valid"],
            "ball": ball,
            "ball_location_valid": augmented["ball_location_valid"],
            "ball_speed_valid": augmented["ball_speed_valid"],
        }
        return {name: self._pad(values, sequence_frames) for name, values in names.items()}
