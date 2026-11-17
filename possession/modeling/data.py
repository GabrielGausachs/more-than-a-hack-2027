"""Feature archive loading, normalization and period-isolated chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..preprocessing.common_features import LABEL_POLICY, SCHEMA_VERSION


LABELS = ("H", "A", "D")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
IGNORE_INDEX = -100


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_ROOT = ROOT / "data" / "derived" / "feature_extracted" / "skillcorner"

# (array, validity array, normalize). A validity array is broadcast to the
# numeric array. Invalid normalized entries remain neutral zero.
ENGINEERED_SPECS = (
    ("team_centroid_xy_m", "team_geometry_valid", True),
    ("team_longitudinal_std_m", "team_geometry_valid", True),
    ("team_lateral_std_m", "team_geometry_valid", True),
    ("team_compactness_rms_m", "team_geometry_valid", True),
    ("team_valid_player_count", None, True),
    ("team_mean_speed_mps", "team_speed_stats_valid", True),
    ("team_median_speed_mps", "team_speed_stats_valid", True),
    ("team_speed_std_mps", "team_speed_stats_valid", True),
    ("team_valid_speed_count", None, True),
    ("team_centroid_displacement_1s_m", "team_temporal_valid", True),
    ("team_compactness_change_1s_m", "team_temporal_valid", True),
    ("team_geometry_valid", None, False),
    ("team_speed_stats_valid", None, False),
    ("team_temporal_valid", None, False),
    ("relative_centroid_home_minus_away_xy_m", "relative_team_valid", True),
    ("relative_longitudinal_std_home_minus_away_m", "relative_team_valid", True),
    ("relative_lateral_std_home_minus_away_m", "relative_team_valid", True),
    ("relative_compactness_home_minus_away_m", "relative_team_valid", True),
    ("relative_team_valid", None, False),
    ("ball_xy_m", "ball_position_valid", True),
    ("ball_position_valid", None, False),
    ("ball_displacement_0_5s_m", "ball_displacement_valid", True),
    ("ball_displacement_valid", None, False),
    ("ball_time_since_valid_s", "ball_time_since_valid_valid", True),
    ("ball_time_since_valid_valid", None, False),
    ("ball_nearest_player_distance_m", "ball_team_geometry_valid", True),
    ("ball_players_within_5m", "ball_team_geometry_valid", True),
    ("ball_team_geometry_valid", None, False),
    ("ball_nearest_home_minus_away_m", "ball_nearest_difference_valid", True),
    ("ball_nearest_difference_valid", None, False),
)


@dataclass(frozen=True)
class Normalization:
    player_xy_mean: np.ndarray
    player_xy_std: np.ndarray
    player_speed_mean: np.ndarray
    player_speed_std: np.ndarray
    engineered_mean: np.ndarray
    engineered_std: np.ndarray

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.__dict__)

    @classmethod
    def load(cls, path: Path) -> "Normalization":
        with np.load(path, allow_pickle=False) as data:
            return cls(**{name: data[name] for name in cls.__annotations__})


class RunningStats:
    def __init__(self, width: int):
        self.count = np.zeros(width, dtype=np.int64)
        self.total = np.zeros(width, dtype=np.float64)
        self.squares = np.zeros(width, dtype=np.float64)

    def update(self, values: np.ndarray, valid: np.ndarray) -> None:
        values = values.reshape(-1, values.shape[-1]).astype(np.float64, copy=False)
        valid = np.broadcast_to(valid, values.shape).reshape(values.shape)
        self.count += valid.sum(axis=0)
        self.total += np.where(valid, values, 0).sum(axis=0)
        self.squares += np.where(valid, values * values, 0).sum(axis=0)

    def finish(self) -> Tuple[np.ndarray, np.ndarray]:
        denominator = np.maximum(self.count, 1)
        mean = self.total / denominator
        variance = np.maximum(self.squares / denominator - mean * mean, 1e-8)
        std = np.sqrt(variance)
        mean[self.count == 0] = 0
        std[self.count == 0] = 1
        return mean.astype(np.float32), std.astype(np.float32)


def load_json(path: Path) -> Dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_archive(match_id: int, archive_root: Path = DEFAULT_ARCHIVE_ROOT) \
        -> Dict[str, np.ndarray]:
    path = Path(archive_root) / (str(int(match_id)) + ".npz")
    if not path.is_file():
        raise IOError("SkillCorner feature archive not found: {}".format(path))
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if str(arrays["schema_version"].item()) != SCHEMA_VERSION:
        raise ValueError("{} has schema {!r}, expected {}".format(
            path, arrays["schema_version"].item(), SCHEMA_VERSION))
    if str(arrays["label_policy"].item()) != LABEL_POLICY:
        raise ValueError("{} has label policy {!r}, expected {}".format(
            path, arrays["label_policy"].item(), LABEL_POLICY))
    return arrays


def target_codes(labels: np.ndarray) -> np.ndarray:
    result = np.full(len(labels), IGNORE_INDEX, dtype=np.int64)
    for label, code in LABEL_TO_INDEX.items():
        result[labels == label] = code
    return result


def possession_duration_weights(
        labels: np.ndarray, period: np.ndarray, frame_rate_hz: float,
        reference_duration_s: float = 8.0, minimum: float = 0.25,
        maximum: float = 4.0) -> np.ndarray:
    """Return full-timeline loss weights for complete H/A possession spells."""

    labels = np.asarray(labels, dtype="U1")
    period = np.asarray(period)
    if labels.shape != period.shape:
        raise ValueError("labels and period must have the same one-dimensional shape")
    if frame_rate_hz <= 0 or reference_duration_s <= 0 or minimum <= 0 or maximum < minimum:
        raise ValueError("invalid possession-duration weighting parameters")

    weights = np.zeros(len(labels), dtype=np.float32)
    for _, period_start, period_stop in period_slices(period):
        period_labels = labels[period_start:period_stop]
        period_weights = weights[period_start:period_stop]
        period_weights[period_labels == "D"] = 1.0
        frame = period_start
        while frame < period_stop:
            owner = labels[frame]
            if owner not in ("H", "A"):
                frame += 1
                continue
            spell_start = frame
            while frame < period_stop and labels[frame] == owner:
                frame += 1
            spell_stop = frame
            touches_boundary = spell_start == period_start or spell_stop == period_stop
            touches_unknown = (
                (spell_start > period_start and labels[spell_start - 1] == "U") or
                (spell_stop < period_stop and labels[spell_stop] == "U"))
            if touches_boundary or touches_unknown:
                weight = 1.0
            else:
                duration_s = (spell_stop - spell_start) / float(frame_rate_hz)
                weight = float(np.clip(
                    reference_duration_s / duration_s, minimum, maximum))
            weights[spell_start:spell_stop] = weight
    return weights


def _flat(array: np.ndarray) -> np.ndarray:
    return array.reshape(len(array), -1).astype(np.float32, copy=False)


def _broadcast_valid(valid: np.ndarray, array: np.ndarray) -> np.ndarray:
    while valid.ndim < array.ndim:
        valid = valid[..., None]
    return np.broadcast_to(valid, array.shape).reshape(len(array), -1)


def _engineered_parts(arrays: Mapping[str, np.ndarray],
                      frame_mask: Optional[np.ndarray] = None) \
        -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Flatten engineered values and build their normalization masks."""

    values, validities = [], []
    for name, valid_name, normalize in ENGINEERED_SPECS:
        raw = arrays[name]
        flattened = _flat(raw).copy()
        if not normalize:
            valid = np.zeros_like(flattened, dtype=bool)
        elif valid_name is None:
            valid = np.ones_like(flattened, dtype=bool)
        else:
            valid = _broadcast_valid(arrays[valid_name], raw)
        if frame_mask is not None:
            valid = valid & np.broadcast_to(frame_mask[:, None], flattened.shape)
        values.append(flattened)
        validities.append(valid)
    return values, validities


def engineered_matrix(arrays: Mapping[str, np.ndarray],
                      normalization: Optional[Normalization] = None) -> np.ndarray:
    columns, validities = _engineered_parts(arrays)
    if normalization is None:
        return np.concatenate(columns, axis=1)

    offset = 0
    for values, valid, (_, _, normalize) in zip(
            columns, validities, ENGINEERED_SPECS):
        width = values.shape[1]
        if normalize:
            mean = normalization.engineered_mean[offset:offset + width]
            std = normalization.engineered_std[offset:offset + width]
            values[valid] = ((values - mean) / std)[valid]
            values[~valid] = 0
        offset += width
    return np.concatenate(columns, axis=1)


def normalized_player_inputs(arrays: Mapping[str, np.ndarray], normalization: Normalization) \
        -> np.ndarray:
    xy = arrays["player_xy_m"].astype(np.float32, copy=True)
    speed = arrays["player_speed_mps"].astype(np.float32, copy=True)
    position_valid = arrays["player_position_valid"]
    speed_valid = arrays["player_speed_valid"]
    normalized_xy = (xy - normalization.player_xy_mean) / normalization.player_xy_std
    xy[position_valid] = normalized_xy[position_valid]
    xy[~position_valid] = 0
    normalized_speed = ((speed - normalization.player_speed_mean) /
                        normalization.player_speed_std)
    speed[speed_valid] = normalized_speed[speed_valid]
    speed[~speed_valid] = 0
    return np.concatenate((xy, speed[..., None], position_valid[..., None].astype(np.float32),
                           speed_valid[..., None].astype(np.float32)), axis=-1)


def iter_sequence_chunks(arrays: Mapping[str, np.ndarray],
                         normalization: Normalization,
                         core_size: int, halo: int,
                         frame_weights: Optional[np.ndarray] = None) \
        -> Iterator[Dict[str, np.ndarray]]:
    """Yield period-isolated model inputs with zero-padded temporal context."""

    players = normalized_player_inputs(arrays, normalization)
    engineered = engineered_matrix(arrays, normalization)
    targets = (target_codes(arrays["target_label"])
               if "target_label" in arrays else
               np.full(len(arrays["period"]), IGNORE_INDEX, dtype=np.int64))
    if frame_weights is not None:
        frame_weights = np.asarray(frame_weights, dtype=np.float32)
        if frame_weights.shape != targets.shape:
            raise ValueError("frame weights must align with the full target timeline")
    input_size = core_size + 2 * halo

    for _, period_start, period_stop in period_slices(arrays["period"]):
        for core_start in range(period_start, period_stop, core_size):
            core_stop = min(period_stop, core_start + core_size)
            source_start = max(period_start, core_start - halo)
            source_stop = min(period_stop, core_start + core_size + halo)
            destination_start = source_start - (core_start - halo)
            destination_stop = destination_start + source_stop - source_start

            player_chunk = np.zeros(
                (input_size,) + players.shape[1:], dtype=np.float32)
            engineered_chunk = np.zeros(
                (input_size, engineered.shape[1]), dtype=np.float32)
            player_chunk[destination_start:destination_stop] = \
                players[source_start:source_stop]
            engineered_chunk[destination_start:destination_stop] = \
                engineered[source_start:source_stop]

            frames = np.arange(core_start, core_stop, dtype=np.int32)
            target_chunk = np.full(core_size, IGNORE_INDEX, dtype=np.int64)
            target_chunk[:len(frames)] = targets[frames]
            weight_chunk = np.zeros(core_size, dtype=np.float32)
            if frame_weights is not None:
                weight_chunk[:len(frames)] = frame_weights[frames]
            yield {
                "players": player_chunk,
                "engineered": engineered_chunk,
                "target": target_chunk,
                "loss_weight": weight_chunk,
                "frame": frames,
            }


def fit_normalization(match_ids: Sequence[int], archive_root: Path) -> Normalization:
    xy_stats = RunningStats(2)
    speed_stats = RunningStats(1)
    engineered_stats: Optional[RunningStats] = None
    for match_id in match_ids:
        arrays = load_archive(match_id, archive_root)
        in_period = np.isin(arrays["period"], (1, 2))
        xy = arrays["player_xy_m"]
        xy_valid = (arrays["player_position_valid"] & in_period[:, None, None])[..., None]
        xy_stats.update(xy.reshape(-1, 2), np.broadcast_to(xy_valid, xy.shape).reshape(-1, 2))
        speed_stats.update(arrays["player_speed_mps"].reshape(-1, 1),
                           (arrays["player_speed_valid"] &
                            in_period[:, None, None]).reshape(-1, 1))
        # Preserve the selected model's original normalization semantics. Engineered
        # features use their own validity masks; count channels intentionally include
        # the complete provider timeline because they have no separate validity mask.
        matrix_parts, valid_parts = _engineered_parts(arrays)
        matrix = np.concatenate(matrix_parts, axis=1)
        valid = np.concatenate(valid_parts, axis=1)
        if engineered_stats is None:
            engineered_stats = RunningStats(matrix.shape[1])
        engineered_stats.update(matrix, valid)
    if engineered_stats is None:
        raise ValueError("cannot fit normalization without training matches")
    xy_mean, xy_std = xy_stats.finish()
    speed_mean, speed_std = speed_stats.finish()
    engineered_mean, engineered_std = engineered_stats.finish()
    return Normalization(xy_mean, xy_std, speed_mean, speed_std,
                         engineered_mean, engineered_std)


def period_slices(period: np.ndarray) -> Iterator[Tuple[int, int, int]]:
    for value in (1, 2):
        indices = np.flatnonzero(period == value)
        if not len(indices):
            continue
        start, stop = int(indices[0]), int(indices[-1]) + 1
        if not np.all(period[start:stop] == value):
            raise ValueError("period {} is not one contiguous interval".format(value))
        yield value, start, stop
