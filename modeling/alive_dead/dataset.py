"""Aligned WCSF archive loading and leakage-safe temporal window datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .preprocessing import FeatureNormalizer
from .splits import FoldSplit, Segment, build_fold_split, build_refit_split


@dataclass(frozen=True)
class WCSFData:
    players: np.ndarray
    player_location_valid: np.ndarray
    player_speed_valid: np.ndarray
    team_features: np.ndarray
    team_feature_valid: np.ndarray
    labels: np.ndarray
    timestamp_ms: np.ndarray
    half_id: np.ndarray
    sample_index: np.ndarray
    source_fps: float
    ball: np.ndarray | None = None
    ball_location_valid: np.ndarray | None = None
    ball_speed_valid: np.ndarray | None = None
    label_valid: np.ndarray | None = None
    alive_labels: np.ndarray | None = None
    alive_label_valid: np.ndarray | None = None
    possession_labels: np.ndarray | None = None
    possession_label_valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.label_valid is None:
            object.__setattr__(self, "label_valid", np.ones(len(self.labels), dtype=bool))
        elif self.label_valid.shape != (len(self.labels),) or self.label_valid.dtype != np.bool_:
            raise ValueError("label_valid must be Boolean with one value per frame")
        ball_arrays = (self.ball, self.ball_location_valid, self.ball_speed_valid)
        if any(value is not None for value in ball_arrays) and not all(
            value is not None for value in ball_arrays
        ):
            raise ValueError("Ball values and both validity masks must be provided together")
        if self.ball is not None:
            if self.ball.shape != (len(self.labels), 3):
                raise ValueError("ball must have shape (N, 3)")
            if (
                self.ball_location_valid.shape != (len(self.labels),)
                or self.ball_speed_valid.shape != (len(self.labels),)
            ):
                raise ValueError("Ball validity masks must have shape (N,)")
        hierarchical = (
            self.alive_labels,
            self.alive_label_valid,
            self.possession_labels,
            self.possession_label_valid,
        )
        if any(value is not None for value in hierarchical) and not all(
            value is not None for value in hierarchical
        ):
            raise ValueError("Hierarchical targets and validity masks must be provided together")
        if self.alive_labels is not None:
            for name, values in (
                ("alive_labels", self.alive_labels),
                ("possession_labels", self.possession_labels),
            ):
                if values.shape != (len(self.labels),) or values.dtype != np.int8:
                    raise ValueError(f"{name} must be int8 with shape (N,)")
            for name, values in (
                ("alive_label_valid", self.alive_label_valid),
                ("possession_label_valid", self.possession_label_valid),
            ):
                if values.shape != (len(self.labels),) or values.dtype != np.bool_:
                    raise ValueError(f"{name} must be Boolean with shape (N,)")
            if not np.isin(self.alive_labels[self.alive_label_valid], (0, 1)).all():
                raise ValueError("Valid ALIVE targets must be binary")
            if not np.isin(
                self.possession_labels[self.possession_label_valid], (0, 1)
            ).all():
                raise ValueError("Valid possession targets must be H=0 or A=1")
            if np.any(self.alive_labels[~self.alive_label_valid] != -1) or np.any(
                self.possession_labels[~self.possession_label_valid] != -1
            ):
                raise ValueError("Invalid hierarchical targets must use the -1 sentinel")

    @property
    def n_frames(self) -> int:
        return len(self.labels)


def load_wcsf_data(
    feature_path: str | Path,
    label_path: str | Path,
    classification_mode: str = "binary",
    use_ball: bool = False,
    had_head_type: str = "direct",
) -> WCSFData:
    """Load approved model arrays and verify exact feature/label alignment."""
    feature_keys = {
        "players",
        "player_location_valid",
        "player_speed_valid",
        "team_features",
        "team_feature_valid",
        "timestamp_ms",
        "half_id",
        "sample_index",
        "source_fps",
    }
    with np.load(feature_path, allow_pickle=False) as archive:
        missing = feature_keys.difference(archive.files)
        if missing:
            raise ValueError(f"Feature archive is missing arrays: {sorted(missing)}")
        features = {key: archive[key].copy() for key in feature_keys}
        ball_keys = {"ball", "ball_location_valid", "ball_speed_valid"}
        present = ball_keys.intersection(archive.files)
        if present and present != ball_keys:
            raise ValueError(f"Feature archive has incomplete ball arrays: {sorted(present)}")
        if use_ball and present != ball_keys:
            raise ValueError("use_ball=True requires a ball-enabled feature archive")
        if use_ball:
            features.update({key: archive[key].copy() for key in ball_keys})

    hierarchical_targets = None
    with np.load(label_path, allow_pickle=False) as archive:
        required = {"labels", "timestamp_ms", "half_id", "sample_index"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Label archive is missing arrays: {sorted(missing)}")
        labels = archive["labels"].copy()
        if "label_valid" in archive.files:
            label_valid = archive["label_valid"].copy()
        elif classification_mode == "binary":
            label_valid = np.ones(len(labels), dtype=bool)
        else:
            raise ValueError("H/A/D label archives must contain label_valid")
        if classification_mode == "had":
            if "label_names" not in archive.files or not np.array_equal(
                archive["label_names"], np.array(["H", "A", "D"])
            ):
                raise ValueError("H/A/D archives must declare label_names [H, A, D]")
        if had_head_type == "hierarchical":
            if classification_mode != "had":
                raise ValueError("Hierarchical targets require classification_mode='had'")
            provenance = {
                "alive_vote_count",
                "dead_vote_count",
                "home_possession_vote_count",
                "away_possession_vote_count",
            }
            missing = provenance.difference(archive.files)
            if missing:
                raise ValueError(
                    f"Hierarchical H/A/D labels are missing provenance: {sorted(missing)}"
                )
            alive_votes = archive["alive_vote_count"].astype(np.int64)
            dead_votes = archive["dead_vote_count"].astype(np.int64)
            home_votes = archive["home_possession_vote_count"].astype(np.int64)
            away_votes = archive["away_possession_vote_count"].astype(np.int64)
            if any(
                values.shape != labels.shape
                for values in (alive_votes, dead_votes, home_votes, away_votes)
            ):
                raise ValueError("Hierarchical vote provenance has unexpected shapes")
            if np.any(alive_votes + dead_votes == 0) or not np.array_equal(
                alive_votes + dead_votes, home_votes + away_votes
            ):
                raise ValueError("Hierarchical vote provenance is empty or inconsistent")
            unanimous_alive = (alive_votes > 0) & (dead_votes == 0)
            unanimous_dead = (dead_votes > 0) & (alive_votes == 0)
            alive_valid = unanimous_alive | unanimous_dead
            alive_targets = np.full(len(labels), -1, dtype=np.int8)
            alive_targets[unanimous_dead] = 0
            alive_targets[unanimous_alive] = 1
            valid_home = unanimous_alive & (home_votes > 0) & (away_votes == 0)
            valid_away = unanimous_alive & (away_votes > 0) & (home_votes == 0)
            possession_valid = valid_home | valid_away
            possession_targets = np.full(len(labels), -1, dtype=np.int8)
            possession_targets[valid_home] = 0
            possession_targets[valid_away] = 1
            hierarchical_targets = (
                alive_targets,
                alive_valid.astype(bool),
                possession_targets,
                possession_valid.astype(bool),
            )
            expected_had_valid = unanimous_dead | valid_home | valid_away
            expected_had = np.full(len(labels), -1, dtype=np.int8)
            expected_had[valid_home] = 0
            expected_had[valid_away] = 1
            expected_had[unanimous_dead] = 2
            if not np.array_equal(label_valid, expected_had_valid) or not np.array_equal(
                labels, expected_had
            ):
                raise ValueError("Hierarchical provenance disagrees with H/A/D labels")
        label_timeline = {key: archive[key].copy() for key in required if key != "labels"}

    for key in ("timestamp_ms", "half_id", "sample_index"):
        if not np.array_equal(features[key], label_timeline[key]):
            raise ValueError(f"Feature and label archives disagree on {key}")
    n_frames = len(labels)
    if features["players"].shape != (n_frames, 2, 24, 3):
        raise ValueError("players must have shape (N, 2, 24, 3)")
    if features["player_location_valid"].shape != (n_frames, 2, 24):
        raise ValueError("player_location_valid has an unexpected shape")
    if features["player_speed_valid"].shape != (n_frames, 2, 24):
        raise ValueError("player_speed_valid has an unexpected shape")
    if features["team_features"].shape != (n_frames, 2, 6):
        raise ValueError("team_features must have shape (N, 2, 6)")
    if features["team_feature_valid"].shape != (n_frames, 2, 6):
        raise ValueError("team_feature_valid has an unexpected shape")
    if classification_mode not in ("binary", "had"):
        raise ValueError("classification_mode must be 'binary' or 'had'")
    if had_head_type not in ("direct", "hierarchical"):
        raise ValueError("had_head_type must be 'direct' or 'hierarchical'")
    allowed = (0, 1) if classification_mode == "binary" else (0, 1, 2)
    if labels.dtype != np.int8:
        raise ValueError("labels must use int8")
    if label_valid.dtype != np.bool_ or label_valid.shape != (n_frames,):
        raise ValueError("label_valid must be Boolean with one value per frame")
    if not np.isin(labels[label_valid], allowed).all():
        raise ValueError(f"Valid {classification_mode} labels have unexpected values")
    if np.any(labels[~label_valid] != -1):
        raise ValueError("Invalid labels must use the -1 sentinel")
    if features["players"].dtype != np.float32 or features["team_features"].dtype != np.float32:
        raise ValueError("Model feature arrays must use float32")
    for key in ("player_location_valid", "player_speed_valid", "team_feature_valid"):
        if features[key].dtype != np.bool_:
            raise ValueError(f"{key} must be Boolean")
    if not np.isfinite(features["players"]).all() or not np.isfinite(features["team_features"]).all():
        raise ValueError("Model inputs contain non-finite values")
    if not np.array_equal(features["sample_index"], np.arange(n_frames)):
        raise ValueError("sample_index must be contiguous and zero-based")
    if use_ball:
        if features["ball"].shape != (n_frames, 3) or features["ball"].dtype != np.float32:
            raise ValueError("ball must have shape (N, 3) and dtype float32")
        for key in ("ball_location_valid", "ball_speed_valid"):
            if features[key].shape != (n_frames,) or features[key].dtype != np.bool_:
                raise ValueError(f"{key} must be Boolean with shape (N,)")
        if not np.isfinite(features["ball"]).all():
            raise ValueError("Ball inputs contain non-finite values")
        if np.any(features["ball"][:, :2][~features["ball_location_valid"]] != 0.0):
            raise ValueError("Invalid ball locations must be zero-filled")
        if np.any(features["ball"][:, 2][~features["ball_speed_valid"]] != 0.0):
            raise ValueError("Invalid ball speeds must be zero-filled")

    source_fps = float(features["source_fps"])
    for half in np.unique(features["half_id"]):
        times = features["timestamp_ms"][features["half_id"] == half]
        expected_step = int(round(1000.0 / source_fps))
        if len(times) > 1 and not np.all(np.diff(times) == expected_step):
            raise ValueError(f"Half {half} is not contiguous at {source_fps:g} Hz")

    return WCSFData(
        players=features["players"],
        player_location_valid=features["player_location_valid"],
        player_speed_valid=features["player_speed_valid"],
        team_features=features["team_features"],
        team_feature_valid=features["team_feature_valid"],
        labels=labels,
        timestamp_ms=features["timestamp_ms"],
        half_id=features["half_id"],
        sample_index=features["sample_index"],
        source_fps=source_fps,
        ball=features.get("ball"),
        ball_location_valid=features.get("ball_location_valid"),
        ball_speed_valid=features.get("ball_speed_valid"),
        label_valid=label_valid,
        alive_labels=None if hierarchical_targets is None else hierarchical_targets[0],
        alive_label_valid=None if hierarchical_targets is None else hierarchical_targets[1],
        possession_labels=None if hierarchical_targets is None else hierarchical_targets[2],
        possession_label_valid=None if hierarchical_targets is None else hierarchical_targets[3],
    )


@dataclass(frozen=True)
class Window:
    start: int
    valid_length: int
    supervision_start: int
    supervision_end: int


class TemporalWindowDataset(Dataset):
    """Sequence-to-sequence windows that never cross a supplied segment boundary."""

    def __init__(
        self,
        data: WCSFData,
        segments: Sequence[Segment],
        sequence_frames: int,
        stride_frames: int,
        context_radius: int,
        require_transition_coverage: bool = False,
        window_transform: Callable[[WCSFData, Window, int, int], Dict[str, np.ndarray]]
        | None = None,
    ) -> None:
        if sequence_frames < 1 or stride_frames < 1 or stride_frames > sequence_frames:
            raise ValueError("Require 1 <= stride_frames <= sequence_frames")
        self.data = data
        self.sequence_frames = sequence_frames
        self.window_transform = window_transform
        central_width = sequence_frames - 2 * context_radius
        if context_radius < 0 or central_width < 1:
            raise ValueError("Context radius leaves no supervised center")
        if stride_frames > central_width:
            raise ValueError("stride_frames cannot exceed the supervised central width")
        self.windows = []
        coverage = np.zeros(data.n_frames, dtype=np.int32)
        transition_coverage = np.zeros(data.n_frames, dtype=np.int32)
        segments = tuple(segments)
        for segment in segments:
            if segment.start < 0 or segment.end > data.n_frames or segment.start >= segment.end:
                raise ValueError(f"Invalid temporal segment: {segment}")
            if segment.length <= sequence_frames:
                starts = [segment.start]
            else:
                last_start = segment.end - sequence_frames
                starts = list(range(segment.start, last_start + 1, stride_frames))
                if starts[-1] != last_start:
                    starts.append(last_start)
            for start in starts:
                valid_length = min(sequence_frames, segment.end - start)
                supervision_start = 0 if start == segment.start else context_radius
                supervision_end = (
                    valid_length
                    if start + valid_length == segment.end
                    else valid_length - context_radius
                )
                if supervision_start >= supervision_end:
                    raise ValueError("Window has no eligible supervised outputs")
                window = Window(
                    start,
                    valid_length,
                    supervision_start,
                    supervision_end,
                )
                self.windows.append(window)
                coverage[
                    start + supervision_start : start + supervision_end
                ] += 1
                transition_coverage[
                    start + supervision_start + 1 : start + supervision_end
                ] += 1
        expected = np.zeros(data.n_frames, dtype=bool)
        for segment in segments:
            expected[segment.start : segment.end] = True
        if np.any(coverage[expected] == 0) or np.any(coverage[~expected] != 0):
            raise ValueError("Window construction does not exactly cover the supplied segments")
        expected_pairs = np.zeros(data.n_frames, dtype=bool)
        for segment in segments:
            expected_pairs[segment.start + 1 : segment.end] = True
        if require_transition_coverage and (
            np.any(transition_coverage[expected_pairs] == 0)
            or np.any(transition_coverage[~expected_pairs] != 0)
        ):
            raise ValueError(
                "Window construction does not cover every adjacent transition pair"
            )
        self.coverage = coverage
        self.transition_coverage = transition_coverage

    def __len__(self) -> int:
        return len(self.windows)

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic augmentation epoch, when a transform is active."""
        if self.window_transform is not None:
            setter = getattr(self.window_transform, "set_epoch", None)
            if setter is None:
                raise TypeError("window_transform does not implement set_epoch")
            setter(epoch)

    def _padded(self, array: np.ndarray, window: Window, fill_value: float | int = 0) -> np.ndarray:
        output = np.full(
            (self.sequence_frames, *array.shape[1:]), fill_value, dtype=array.dtype
        )
        output[: window.valid_length] = array[
            window.start : window.start + window.valid_length
        ]
        return output

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        window = self.windows[index]
        transformed = (
            self.window_transform(self.data, window, index, self.sequence_frames)
            if self.window_transform is not None
            else None
        )

        def feature(name: str) -> np.ndarray:
            if transformed is not None:
                return transformed[name]
            return self._padded(getattr(self.data, name), window)

        output_mask = np.zeros(self.sequence_frames, dtype=bool)
        output_mask[: window.valid_length] = True
        supervision_mask = np.zeros(self.sequence_frames, dtype=bool)
        supervision_mask[window.supervision_start : window.supervision_end] = True
        loss_weight = np.zeros(self.sequence_frames, dtype=np.float32)
        supervised_global = slice(
            window.start + window.supervision_start,
            window.start + window.supervision_end,
        )
        loss_weight[window.supervision_start : window.supervision_end] = (
            1.0 / self.coverage[supervised_global]
        )
        transition_supervision_mask = np.zeros(self.sequence_frames, dtype=bool)
        transition_loss_weight = np.zeros(self.sequence_frames, dtype=np.float32)
        pair_start = window.supervision_start + 1
        pair_end = window.supervision_end
        transition_supervision_mask[pair_start:pair_end] = True
        pair_global = slice(window.start + pair_start, window.start + pair_end)
        transition_loss_weight[pair_start:pair_end] = (
            1.0 / self.transition_coverage[pair_global]
        )
        batch = {
            "players": torch.from_numpy(feature("players")),
            "player_location_valid": torch.from_numpy(feature("player_location_valid")),
            "player_speed_valid": torch.from_numpy(feature("player_speed_valid")),
            "team_features": torch.from_numpy(feature("team_features")),
            "team_feature_valid": torch.from_numpy(feature("team_feature_valid")),
            "labels": torch.from_numpy(
                self._padded(self.data.labels, window, fill_value=-1).astype(np.float32)
            ),
            "label_valid": torch.from_numpy(
                self._padded(self.data.label_valid, window, fill_value=False)
            ),
            "output_mask": torch.from_numpy(output_mask),
            "supervision_mask": torch.from_numpy(supervision_mask),
            "loss_weight": torch.from_numpy(loss_weight),
            "transition_supervision_mask": torch.from_numpy(
                transition_supervision_mask
            ),
            "transition_loss_weight": torch.from_numpy(transition_loss_weight),
        }
        if self.data.ball is not None:
            batch.update(
                {
                    "ball": torch.from_numpy(feature("ball")),
                    "ball_location_valid": torch.from_numpy(feature("ball_location_valid")),
                    "ball_speed_valid": torch.from_numpy(feature("ball_speed_valid")),
                }
            )
        if self.data.alive_labels is not None:
            batch.update(
                {
                    "alive_labels": torch.from_numpy(
                        self._padded(self.data.alive_labels, window, fill_value=-1).astype(
                            np.float32
                        )
                    ),
                    "alive_label_valid": torch.from_numpy(
                        self._padded(
                            self.data.alive_label_valid, window, fill_value=False
                        )
                    ),
                    "possession_labels": torch.from_numpy(
                        self._padded(
                            self.data.possession_labels, window, fill_value=-1
                        ).astype(np.float32)
                    ),
                    "possession_label_valid": torch.from_numpy(
                        self._padded(
                            self.data.possession_label_valid, window, fill_value=False
                        )
                    ),
                }
            )
        return batch


def normalize_wcsf_data(data: WCSFData, normalizer: FeatureNormalizer) -> WCSFData:
    """Apply one fitted normalizer while preserving timelines, labels and masks."""
    players, team_features = normalizer.transform(
        data.players,
        data.player_location_valid,
        data.player_speed_valid,
        data.team_features,
        data.team_feature_valid,
    )
    ball = None
    if data.ball is not None:
        ball = normalizer.transform_ball(
            data.ball, data.ball_location_valid, data.ball_speed_valid
        )
    return WCSFData(
        players=players,
        player_location_valid=data.player_location_valid,
        player_speed_valid=data.player_speed_valid,
        team_features=team_features,
        team_feature_valid=data.team_feature_valid,
        labels=data.labels,
        timestamp_ms=data.timestamp_ms,
        half_id=data.half_id,
        sample_index=data.sample_index,
        source_fps=data.source_fps,
        ball=ball,
        ball_location_valid=data.ball_location_valid,
        ball_speed_valid=data.ball_speed_valid,
        label_valid=data.label_valid,
        alive_labels=data.alive_labels,
        alive_label_valid=data.alive_label_valid,
        possession_labels=data.possession_labels,
        possession_label_valid=data.possession_label_valid,
    )


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    split: FoldSplit
    normalizer: FeatureNormalizer
    data: WCSFData


def build_dataloaders(
    config: ExperimentConfig, full_half_refit: bool = False
) -> LoaderBundle:
    """Build aligned normalized loaders for an inner split or full-half refit."""
    raw = load_wcsf_data(
        config.data.feature_path,
        config.data.label_path,
        config.model.classification_mode,
        config.model.use_ball,
        config.model.had_head_type,
    )
    config.validate(raw.source_fps)
    if full_half_refit:
        split = build_refit_split(raw.half_id, config.split.fold)
    else:
        buffer_frames = config.seconds_to_frames(config.split.buffer_seconds, raw.source_fps)
        split = build_fold_split(
            raw.half_id,
            config.split.fold,
            config.split.validation_fraction,
            buffer_frames,
            config.split.validation_block,
        )
    normalizer = FeatureNormalizer.fit(
        raw.players,
        raw.player_speed_valid,
        raw.team_features,
        raw.team_feature_valid,
        split.train_mask,
        ball=raw.ball,
        ball_speed_valid=raw.ball_speed_valid,
    )
    data = normalize_wcsf_data(raw, normalizer)
    sequence_frames = config.seconds_to_frames(config.data.sequence_seconds, raw.source_fps)
    train_stride = config.seconds_to_frames(config.data.train_stride_seconds, raw.source_fps)
    window_transform = None
    train_data = data
    if config.augmentation.enabled:
        from .augmentation import MaskAugmentedWindowTransform

        window_transform = MaskAugmentedWindowTransform.create(
            config, normalizer, full_half_refit=full_half_refit
        )
        train_data = raw
    train_dataset = TemporalWindowDataset(
        train_data,
        split.segments["train"],
        sequence_frames,
        train_stride,
        config.model.context_radius_frames,
        require_transition_coverage=config.training.use_transition_loss,
        window_transform=window_transform,
    )
    generator = torch.Generator().manual_seed(config.training.seed)
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "persistent_workers": config.data.num_workers > 0,
    }
    return LoaderBundle(
        train=DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        split=split,
        normalizer=normalizer,
        data=data,
    )
