"""Shared feature schema and calculations for both tracking providers."""

from __future__ import annotations

import math
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "common_tracking_v1"
LABEL_POLICY = "skillcorner_final_v1"
MAX_PLAYERS = 24
MIN_TEAM_PLAYERS = 8
TEAM_SIDES = ("H", "A")
QUALITY_INVALID = 0
QUALITY_DETECTED = 1
QUALITY_EXTRAPOLATED = 2
QUALITY_PROVIDER_UNKNOWN = 3

MODEL_INPUT_ARRAYS = (
    "player_xy_m", "player_speed_mps", "player_position_valid", "player_speed_valid",
    "team_centroid_xy_m", "team_longitudinal_std_m", "team_lateral_std_m",
    "team_compactness_rms_m", "team_valid_player_count", "team_mean_speed_mps",
    "team_median_speed_mps", "team_speed_std_mps", "team_valid_speed_count",
    "team_centroid_displacement_1s_m", "team_compactness_change_1s_m",
    "team_geometry_valid", "team_speed_stats_valid", "team_temporal_valid",
    "relative_centroid_home_minus_away_xy_m",
    "relative_longitudinal_std_home_minus_away_m",
    "relative_lateral_std_home_minus_away_m",
    "relative_compactness_home_minus_away_m", "relative_team_valid",
    "ball_xy_m", "ball_position_valid", "ball_displacement_0_5s_m",
    "ball_displacement_valid", "ball_time_since_valid_s", "ball_time_since_valid_valid",
    "ball_nearest_player_distance_m", "ball_players_within_5m",
    "ball_team_geometry_valid", "ball_nearest_home_minus_away_m",
    "ball_nearest_difference_valid",
)


def orient_xy(x: float, y: float, home_direction: Optional[str]) -> Tuple[float, float]:
    """Rotate coordinates so Home always attacks toward positive x."""

    if home_direction == "left_to_right":
        return float(x), float(y)
    if home_direction == "right_to_left":
        return -float(x), -float(y)
    raise ValueError("unsupported/invalid attack direction {!r}".format(home_direction))


def allocate_arrays(n_frames: int) -> Dict[str, np.ndarray]:
    """Allocate the complete frozen archive schema."""

    t = int(n_frames)
    return {
        "frame_index": np.zeros(t, dtype=np.int32),
        "timestamp_s": np.zeros(t, dtype=np.float32),
        "period": np.zeros(t, dtype=np.int8),
        "orientation_valid": np.zeros(t, dtype=bool),
        "player_xy_m": np.zeros((t, 2, MAX_PLAYERS, 2), dtype=np.float32),
        "player_speed_mps": np.zeros((t, 2, MAX_PLAYERS), dtype=np.float32),
        "player_position_valid": np.zeros((t, 2, MAX_PLAYERS), dtype=bool),
        "player_speed_valid": np.zeros((t, 2, MAX_PLAYERS), dtype=bool),
        "audit_player_quality": np.zeros((t, 2, MAX_PLAYERS), dtype=np.int8),
        "team_centroid_xy_m": np.zeros((t, 2, 2), dtype=np.float32),
        "team_longitudinal_std_m": np.zeros((t, 2), dtype=np.float32),
        "team_lateral_std_m": np.zeros((t, 2), dtype=np.float32),
        "team_compactness_rms_m": np.zeros((t, 2), dtype=np.float32),
        "team_valid_player_count": np.zeros((t, 2), dtype=np.int8),
        "team_mean_speed_mps": np.zeros((t, 2), dtype=np.float32),
        "team_median_speed_mps": np.zeros((t, 2), dtype=np.float32),
        "team_speed_std_mps": np.zeros((t, 2), dtype=np.float32),
        "team_valid_speed_count": np.zeros((t, 2), dtype=np.int8),
        "team_centroid_displacement_1s_m": np.zeros((t, 2), dtype=np.float32),
        "team_compactness_change_1s_m": np.zeros((t, 2), dtype=np.float32),
        "team_geometry_valid": np.zeros((t, 2), dtype=bool),
        "team_speed_stats_valid": np.zeros((t, 2), dtype=bool),
        "team_temporal_valid": np.zeros((t, 2), dtype=bool),
        "relative_centroid_home_minus_away_xy_m": np.zeros((t, 2), dtype=np.float32),
        "relative_longitudinal_std_home_minus_away_m": np.zeros(t, dtype=np.float32),
        "relative_lateral_std_home_minus_away_m": np.zeros(t, dtype=np.float32),
        "relative_compactness_home_minus_away_m": np.zeros(t, dtype=np.float32),
        "relative_team_valid": np.zeros(t, dtype=bool),
        "ball_xy_m": np.zeros((t, 2), dtype=np.float32),
        "ball_position_valid": np.zeros(t, dtype=bool),
        "audit_ball_quality": np.zeros(t, dtype=np.int8),
        "ball_displacement_0_5s_m": np.zeros(t, dtype=np.float32),
        "ball_displacement_valid": np.zeros(t, dtype=bool),
        "ball_time_since_valid_s": np.zeros(t, dtype=np.float32),
        "ball_time_since_valid_valid": np.zeros(t, dtype=bool),
        "ball_nearest_player_distance_m": np.zeros((t, 2), dtype=np.float32),
        "ball_players_within_5m": np.zeros((t, 2), dtype=np.int8),
        "ball_team_geometry_valid": np.zeros((t, 2), dtype=bool),
        "ball_nearest_home_minus_away_m": np.zeros(t, dtype=np.float32),
        "ball_nearest_difference_valid": np.zeros(t, dtype=bool),
    }


def store_team(arrays: MutableMapping[str, np.ndarray], frame: int, team: int,
               observations: Sequence[Tuple[float, float, float, bool, int]]) -> None:
    if len(observations) > MAX_PLAYERS:
        raise ValueError("frame {} team {} has {} observations; capacity is {}".format(
            frame, TEAM_SIDES[team], len(observations), MAX_PLAYERS))
    ordered = sorted(observations, key=lambda item: (item[0], item[1]))
    for slot, (x, y, speed, speed_valid, quality) in enumerate(ordered):
        arrays["player_xy_m"][frame, team, slot] = (x, y)
        arrays["player_position_valid"][frame, team, slot] = True
        arrays["audit_player_quality"][frame, team, slot] = quality
        if speed_valid and math.isfinite(speed):
            arrays["player_speed_mps"][frame, team, slot] = speed
            arrays["player_speed_valid"][frame, team, slot] = True


def _continuous(mask: np.ndarray, period: np.ndarray, lag: int) -> np.ndarray:
    """True when every frame from t-lag through t is valid and in one period."""

    result = np.zeros_like(mask, dtype=bool)
    if lag <= 0 or len(mask) <= lag:
        return result
    bad = (~mask).astype(np.int32)
    prefix = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
    valid_window = (prefix[lag + 1:] - prefix[:-(lag + 1)]) == 0
    same_period = (period[lag:] != 0) & (period[lag:] == period[:-lag])
    result[lag:] = valid_window & same_period
    return result


def compute_common_features(arrays: MutableMapping[str, np.ndarray], frame_rate: float,
                            team_lag_frames: int, ball_lag_frames: int) -> None:
    """Populate the shared geometry, movement, relative-team and ball features."""

    pos_valid = arrays["player_position_valid"]
    speed_valid = arrays["player_speed_valid"]
    xy = arrays["player_xy_m"]
    speed = arrays["player_speed_mps"]
    periods = arrays["period"]
    counts = pos_valid.sum(axis=2)
    speed_counts = speed_valid.sum(axis=2)
    arrays["team_valid_player_count"][:] = counts.astype(np.int8)
    arrays["team_valid_speed_count"][:] = speed_counts.astype(np.int8)
    geometry_valid = counts >= MIN_TEAM_PLAYERS
    speed_stats_valid = speed_counts >= MIN_TEAM_PLAYERS
    arrays["team_geometry_valid"][:] = geometry_valid
    arrays["team_speed_stats_valid"][:] = speed_stats_valid

    for team in range(2):
        denominator = np.maximum(counts[:, team], 1)[:, None]
        centroid = (xy[:, team] * pos_valid[:, team, :, None]).sum(axis=1) / denominator
        centroid[~geometry_valid[:, team]] = 0
        arrays["team_centroid_xy_m"][:, team] = centroid.astype(np.float32)
        delta = xy[:, team] - centroid[:, None, :]
        variance = ((delta ** 2) * pos_valid[:, team, :, None]).sum(axis=1) / denominator
        long_std = np.sqrt(variance[:, 0])
        lat_std = np.sqrt(variance[:, 1])
        compactness = np.sqrt(variance.sum(axis=1))
        for values, name in ((long_std, "team_longitudinal_std_m"),
                             (lat_std, "team_lateral_std_m"),
                             (compactness, "team_compactness_rms_m")):
            values[~geometry_valid[:, team]] = 0
            arrays[name][:, team] = values.astype(np.float32)

        speed_denominator = np.maximum(speed_counts[:, team], 1)
        means = (speed[:, team] * speed_valid[:, team]).sum(axis=1) / speed_denominator
        speed_delta = speed[:, team] - means[:, None]
        stds = np.sqrt(
            ((speed_delta ** 2) * speed_valid[:, team]).sum(axis=1) / speed_denominator)
        medians = np.zeros(len(periods), dtype=np.float32)
        for frame in np.flatnonzero(speed_stats_valid[:, team]):
            medians[frame] = np.median(speed[frame, team, speed_valid[frame, team]])
        means[~speed_stats_valid[:, team]] = 0
        stds[~speed_stats_valid[:, team]] = 0
        arrays["team_mean_speed_mps"][:, team] = means.astype(np.float32)
        arrays["team_median_speed_mps"][:, team] = medians
        arrays["team_speed_std_mps"][:, team] = stds.astype(np.float32)

        temporal_valid = _continuous(geometry_valid[:, team], periods, team_lag_frames)
        arrays["team_temporal_valid"][:, team] = temporal_valid
        current = arrays["team_centroid_xy_m"][team_lag_frames:, team]
        previous = arrays["team_centroid_xy_m"][:-team_lag_frames, team]
        displacement = np.sqrt(((current - previous) ** 2).sum(axis=1))
        compact_change = (arrays["team_compactness_rms_m"][team_lag_frames:, team] -
                          arrays["team_compactness_rms_m"][:-team_lag_frames, team])
        good = temporal_valid[team_lag_frames:]
        arrays["team_centroid_displacement_1s_m"][team_lag_frames:, team] = np.where(
            good, displacement, 0).astype(np.float32)
        arrays["team_compactness_change_1s_m"][team_lag_frames:, team] = np.where(
            good, compact_change, 0).astype(np.float32)

    relative_valid = geometry_valid.all(axis=1)
    arrays["relative_team_valid"][:] = relative_valid
    relative_specs = (
        ("team_centroid_xy_m", "relative_centroid_home_minus_away_xy_m"),
        ("team_longitudinal_std_m", "relative_longitudinal_std_home_minus_away_m"),
        ("team_lateral_std_m", "relative_lateral_std_home_minus_away_m"),
        ("team_compactness_rms_m", "relative_compactness_home_minus_away_m"),
    )
    for source, target in relative_specs:
        difference = arrays[source][:, 0] - arrays[source][:, 1]
        if difference.ndim == 1:
            difference[~relative_valid] = 0
        else:
            difference[~relative_valid, :] = 0
        arrays[target][:] = difference.astype(np.float32)

    ball_valid = arrays["ball_position_valid"]
    ball_continuous = _continuous(ball_valid, periods, ball_lag_frames)
    arrays["ball_displacement_valid"][:] = ball_continuous
    ball_delta = (arrays["ball_xy_m"][ball_lag_frames:] -
                  arrays["ball_xy_m"][:-ball_lag_frames])
    displacement = np.sqrt((ball_delta ** 2).sum(axis=1))
    arrays["ball_displacement_0_5s_m"][ball_lag_frames:] = np.where(
        ball_continuous[ball_lag_frames:], displacement, 0).astype(np.float32)

    last_valid = None
    last_period = 0
    for frame in range(len(periods)):
        period = int(periods[frame])
        if period == 0 or period != last_period:
            last_valid = None
        if period != 0 and ball_valid[frame]:
            last_valid = frame
            arrays["ball_time_since_valid_valid"][frame] = True
        elif period != 0 and last_valid is not None:
            arrays["ball_time_since_valid_s"][frame] = (frame - last_valid) / frame_rate
            arrays["ball_time_since_valid_valid"][frame] = True
        last_period = period

    for team in range(2):
        valid = ball_valid & geometry_valid[:, team]
        arrays["ball_team_geometry_valid"][:, team] = valid
        delta = xy[:, team] - arrays["ball_xy_m"][:, None, :]
        distance = np.sqrt((delta ** 2).sum(axis=2))
        distance[~pos_valid[:, team]] = np.inf
        nearest = distance.min(axis=1)
        nearest[~valid] = 0
        within = ((distance <= 5.0) & pos_valid[:, team]).sum(axis=1)
        within[~valid] = 0
        arrays["ball_nearest_player_distance_m"][:, team] = nearest.astype(np.float32)
        arrays["ball_players_within_5m"][:, team] = within.astype(np.int8)
    difference_valid = arrays["ball_team_geometry_valid"].all(axis=1)
    arrays["ball_nearest_difference_valid"][:] = difference_valid
    nearest_difference = (arrays["ball_nearest_player_distance_m"][:, 0] -
                          arrays["ball_nearest_player_distance_m"][:, 1])
    nearest_difference[~difference_valid] = 0
    arrays["ball_nearest_home_minus_away_m"][:] = nearest_difference.astype(np.float32)


def metadata_arrays(provider: str, match_id: str, frame_rate: float,
                    pitch_length: float, pitch_width: float,
                    ball_lag_s: float, speed_window_s: float,
                    label_policy: str = "") -> Dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "provider": np.asarray(provider),
        "match_id": np.asarray(str(match_id)),
        "frame_rate_hz": np.asarray(frame_rate, dtype=np.float32),
        "pitch_length_m": np.asarray(pitch_length, dtype=np.float32),
        "pitch_width_m": np.asarray(pitch_width, dtype=np.float32),
        "player_speed_window_s": np.asarray(speed_window_s, dtype=np.float32),
        "team_temporal_lag_s": np.asarray(1.0, dtype=np.float32),
        "ball_displacement_actual_lag_s": np.asarray(ball_lag_s, dtype=np.float32),
        "label_policy": np.asarray(label_policy),
        "model_input_arrays": np.asarray(MODEL_INPUT_ARRAYS, dtype="U64"),
    }


def validate_archive(arrays: Mapping[str, np.ndarray], require_target: bool) -> None:
    n_frames = len(arrays["frame_index"])
    for name in MODEL_INPUT_ARRAYS:
        if np.issubdtype(arrays[name].dtype, np.floating) and not np.isfinite(arrays[name]).all():
            raise ValueError("{} contains NaN or infinity".format(name))
    if require_target:
        if len(arrays["target_label"]) != n_frames:
            raise ValueError("SkillCorner archive target labels are missing or misaligned")
        if not set(np.unique(arrays["target_label"])).issubset({"H", "A", "D", "U"}):
            raise ValueError("target_label contains an invalid value")


def schema_document() -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "providers": ["skillcorner", "metrica"],
        "coordinate_system": "pitch-centred metres; Home attacks toward positive x",
        "frame_axis": "original provider timeline",
        "team_axis": ["Home", "Away"],
        "player_capacity_per_team": MAX_PLAYERS,
        "player_order": "lexicographic canonical x then y; identities are excluded",
        "minimum_players_for_team_features": MIN_TEAM_PLAYERS,
        "missing_values": "numeric zero accompanied by an explicit validity mask",
        "temporal_rules": {
            "player_speed": "SkillCorner uses 0.5 s stable-ID displacement; Metrica uses provider speed",
            "team_change": "1.0 s with continuous valid geometry",
            "ball_displacement": "0.5 s at 10 Hz and 0.52 s at 25 Hz",
            "period_boundaries": "never crossed",
        },
        "model_input_arrays": list(MODEL_INPUT_ARRAYS),
        "skillcorner_target": {
            "array": "target_label",
            "values": ["H", "A", "D", "U"],
            "label_policy": LABEL_POLICY,
        },
        "metrica_target": None,
    }
