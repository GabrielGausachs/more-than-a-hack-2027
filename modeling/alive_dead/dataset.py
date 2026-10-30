"""Aligned WCSF archive loading and leakage-safe temporal window datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

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

    @property
    def n_frames(self) -> int:
        return len(self.labels)


def load_wcsf_data(feature_path: str | Path, label_path: str | Path) -> WCSFData:
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

    with np.load(label_path, allow_pickle=False) as archive:
        required = {"labels", "timestamp_ms", "half_id", "sample_index"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Label archive is missing arrays: {sorted(missing)}")
        labels = archive["labels"].copy()
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
    if labels.dtype != np.int8 or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary int8 values")
    if features["players"].dtype != np.float32 or features["team_features"].dtype != np.float32:
        raise ValueError("Model feature arrays must use float32")
    for key in ("player_location_valid", "player_speed_valid", "team_feature_valid"):
        if features[key].dtype != np.bool_:
            raise ValueError(f"{key} must be Boolean")
    if not np.isfinite(features["players"]).all() or not np.isfinite(features["team_features"]).all():
        raise ValueError("Model inputs contain non-finite values")
    if not np.array_equal(features["sample_index"], np.arange(n_frames)):
        raise ValueError("sample_index must be contiguous and zero-based")

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
    )


@dataclass(frozen=True)
class Window:
    start: int
    valid_length: int
    segment: Segment


class TemporalWindowDataset(Dataset):
    """Sequence-to-sequence windows that never cross a supplied segment boundary."""

    def __init__(
        self,
        data: WCSFData,
        segments: Sequence[Segment],
        sequence_frames: int,
        stride_frames: int,
    ) -> None:
        if sequence_frames < 1 or stride_frames < 1 or stride_frames > sequence_frames:
            raise ValueError("Require 1 <= stride_frames <= sequence_frames")
        self.data = data
        self.segments = tuple(segments)
        self.sequence_frames = sequence_frames
        self.stride_frames = stride_frames
        self.windows = []
        coverage = np.zeros(data.n_frames, dtype=np.int32)
        for segment in self.segments:
            if segment.start < 0 or segment.end > data.n_frames or segment.start >= segment.end:
                raise ValueError(f"Invalid temporal segment: {segment}")
            for offset in range(0, segment.length, stride_frames):
                start = segment.start + offset
                valid_length = min(sequence_frames, segment.end - start)
                window = Window(start, valid_length, segment)
                self.windows.append(window)
                coverage[start : start + valid_length] += 1
        expected = np.zeros(data.n_frames, dtype=bool)
        for segment in self.segments:
            expected[segment.start : segment.end] = True
        if np.any(coverage[expected] == 0) or np.any(coverage[~expected] != 0):
            raise ValueError("Window construction does not exactly cover the supplied segments")
        self.coverage = coverage

    def __len__(self) -> int:
        return len(self.windows)

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
        output_mask = np.zeros(self.sequence_frames, dtype=bool)
        output_mask[: window.valid_length] = True
        loss_weight = np.zeros(self.sequence_frames, dtype=np.float32)
        frame_slice = slice(window.start, window.start + window.valid_length)
        loss_weight[: window.valid_length] = 1.0 / self.coverage[frame_slice]
        global_index = np.full(self.sequence_frames, -1, dtype=np.int64)
        global_index[: window.valid_length] = np.arange(
            window.start, window.start + window.valid_length
        )
        return {
            "players": torch.from_numpy(self._padded(self.data.players, window)),
            "player_location_valid": torch.from_numpy(
                self._padded(self.data.player_location_valid, window)
            ),
            "player_speed_valid": torch.from_numpy(
                self._padded(self.data.player_speed_valid, window)
            ),
            "team_features": torch.from_numpy(self._padded(self.data.team_features, window)),
            "team_feature_valid": torch.from_numpy(
                self._padded(self.data.team_feature_valid, window)
            ),
            "labels": torch.from_numpy(
                self._padded(self.data.labels, window).astype(np.float32)
            ),
            "output_mask": torch.from_numpy(output_mask),
            "loss_weight": torch.from_numpy(loss_weight),
            "global_index": torch.from_numpy(global_index),
        }


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    split: FoldSplit
    normalizer: FeatureNormalizer
    data: WCSFData


def build_dataloaders(
    config: ExperimentConfig, full_half_refit: bool = False
) -> LoaderBundle:
    """Build aligned normalized loaders for an inner split or full-half refit."""
    raw = load_wcsf_data(config.data.feature_path, config.data.label_path)
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
    )
    normalized_players, normalized_team = normalizer.transform(
        raw.players,
        raw.player_location_valid,
        raw.player_speed_valid,
        raw.team_features,
        raw.team_feature_valid,
    )
    data = WCSFData(
        players=normalized_players,
        player_location_valid=raw.player_location_valid,
        player_speed_valid=raw.player_speed_valid,
        team_features=normalized_team,
        team_feature_valid=raw.team_feature_valid,
        labels=raw.labels,
        timestamp_ms=raw.timestamp_ms,
        half_id=raw.half_id,
        sample_index=raw.sample_index,
        source_fps=raw.source_fps,
    )
    sequence_frames = config.seconds_to_frames(config.data.sequence_seconds, raw.source_fps)
    train_stride = config.seconds_to_frames(config.data.train_stride_seconds, raw.source_fps)
    eval_stride = config.seconds_to_frames(config.data.eval_stride_seconds, raw.source_fps)
    datasets = {
        "train": TemporalWindowDataset(data, split.segments["train"], sequence_frames, train_stride),
        "validation": TemporalWindowDataset(
            data, split.segments["validation"], sequence_frames, eval_stride
        ),
        "test": TemporalWindowDataset(data, split.segments["test"], sequence_frames, eval_stride),
    }
    generator = torch.Generator().manual_seed(config.training.seed)
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "persistent_workers": config.data.num_workers > 0,
    }
    return LoaderBundle(
        train=DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
        validation=DataLoader(datasets["validation"], shuffle=False, **common),
        test=DataLoader(datasets["test"], shuffle=False, **common),
        split=split,
        normalizer=normalizer,
        data=data,
    )
