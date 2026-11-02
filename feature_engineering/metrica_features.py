"""Build WCSF-compatible 10 Hz player and team features from Metrica ATD."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from games import asset_paths, half_video_blocks, source_frames
from feature_engineering.wcsf_features import (
    BALL_FEATURE_NAMES,
    MAX_PLAYERS_PER_TEAM,
    PLAYER_FEATURE_NAMES,
    TEAM_FEATURE_NAMES,
    TEAM_ORDER,
    build_team_features,
    save_features,
    validate_features,
)


MODEL_FPS = 10.0
EXPECTED_SOURCE_FPS = 25.0
RAW_ATTACK_DIRECTION = np.array(["left_to_right", "right_to_left"])
ROTATION_APPLIED = np.array([False, True], dtype=bool)


def _provider_parameters(element: ET.Element) -> Dict[str, str]:
    return {
        parameter.findtext("Name", default=""): parameter.findtext("Value", default="")
        for parameter in element.iter("ProviderParameter")
    }


def load_metrica_metadata(path: str | Path) -> Dict[str, object]:
    """Load frame rate, pitch, teams and source-slot team membership from XML."""
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"Expected two teams in {path}, found {len(teams)}")
    team_ids = [team.get("id") for team in teams]
    team_names = np.array([team.findtext("Name", default="") for team in teams])
    if "barcelona" not in team_names[0].lower():
        raise ValueError("The first XML team must be home team Barcelona")

    players = list(root.iter("Player"))
    if not players:
        raise ValueError(f"No player slots found in {path}")
    slot_team_index = np.empty(len(players), dtype=np.int8)
    slot_within_team = np.empty(len(players), dtype=np.int8)
    counts = [0, 0]
    for slot, player in enumerate(players):
        team_id = player.get("teamId")
        if team_id not in team_ids:
            raise ValueError(f"Player slot {slot + 1} has unknown teamId {team_id!r}")
        team_index = team_ids.index(team_id)
        slot_team_index[slot] = team_index
        slot_within_team[slot] = counts[team_index]
        counts[team_index] += 1
    if counts != [24, 22] or max(counts) > MAX_PLAYERS_PER_TEAM:
        raise ValueError(f"Expected 24 Home and 22 Away slots, found {counts}")

    frame_rate_text = root.findtext(".//FrameRate")
    field = root.find(".//FieldSize")
    if frame_rate_text is None or field is None:
        raise ValueError("Metrica XML is missing frame-rate or pitch metadata")
    source_fps = float(frame_rate_text)
    pitch_length = float(field.findtext("Width", default="nan"))
    pitch_width = float(field.findtext("Height", default="nan"))
    if source_fps != EXPECTED_SOURCE_FPS or not np.isfinite([pitch_length, pitch_width]).all():
        raise ValueError("Unexpected Metrica frame rate or pitch dimensions")

    home_direction = _provider_parameters(teams[0]).get("attack_direction_first_half")
    if home_direction != "left_to_right":
        raise ValueError(
            "Barcelona first-half direction is not the expected XML value left_to_right"
        )
    return {
        "source_fps": source_fps,
        "pitch_length_m": pitch_length,
        "pitch_width_m": pitch_width,
        "team_names": team_names,
        "slot_team_index": slot_team_index,
        "slot_within_team": slot_within_team,
    }


def _nearest_indices(reference_ms: np.ndarray, query_ms: np.ndarray) -> np.ndarray:
    """Indices of nearest sorted reference times, preferring the earlier time on ties."""
    right = np.searchsorted(reference_ms, query_ms, side="left")
    right_clipped = np.minimum(right, len(reference_ms) - 1)
    left = np.maximum(right - 1, 0)
    left_distance = np.abs(query_ms - reference_ms[left])
    right_distance = np.abs(reference_ms[right_clipped] - query_ms)
    choose_left = (right == len(reference_ms)) | (left_distance <= right_distance)
    return np.where(choose_left, left, right_clipped).astype(np.int64)


def build_resampling_index(
    n_source_frames: int,
    scored_intervals: Sequence[tuple[int, int]],
    source_fps: float = EXPECTED_SOURCE_FPS,
    target_fps: float = MODEL_FPS,
) -> Dict[str, np.ndarray]:
    """Build independent half-level 10 Hz grids and bidirectional frame alignment."""
    if n_source_frames < 1 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("Frame count and frequencies must be positive")
    source_period_ms = 1000.0 / source_fps
    target_period_ms = 1000.0 / target_fps
    if not source_period_ms.is_integer() or not target_period_ms.is_integer():
        raise ValueError("This builder requires integer-millisecond source and target periods")
    source_period_ms = int(source_period_ms)
    target_period_ms = int(target_period_ms)

    intervals = np.asarray(scored_intervals, dtype=np.int64)
    if intervals.shape != (2, 2):
        raise ValueError("Exactly two inclusive scored intervals are required")
    if (
        np.any(intervals < 0)
        or np.any(intervals[:, 0] > intervals[:, 1])
        or np.any(intervals[:, 1] >= n_source_frames)
        or intervals[0, 1] >= intervals[1, 0]
    ):
        raise ValueError("Scored intervals are invalid or overlapping")

    source_to_sample = np.full(n_source_frames, -1, dtype=np.int64)
    source_half_id = np.zeros(n_source_frames, dtype=np.int8)
    source_scored = np.zeros(n_source_frames, dtype=bool)
    target_times = []
    selected_frames = []
    half_ids = []
    sample_offset = 0

    for half_id, (start, end) in enumerate(intervals, start=1):
        source_frames_in_half = np.arange(start, end + 1, dtype=np.int64)
        source_times = source_frames_in_half * source_period_ms
        half_target_times = np.arange(
            source_times[0], source_times[-1] + 1, target_period_ms, dtype=np.int64
        )
        selected_local = _nearest_indices(source_times, half_target_times)
        selected = source_frames_in_half[selected_local]

        reverse_local = _nearest_indices(half_target_times, source_times)
        source_to_sample[start : end + 1] = sample_offset + reverse_local
        source_half_id[start : end + 1] = half_id
        source_scored[start : end + 1] = True
        target_times.append(half_target_times)
        selected_frames.append(selected)
        half_ids.append(np.full(len(selected), half_id, dtype=np.int8))
        sample_offset += len(selected)

    source_frame_index = np.concatenate(selected_frames)
    return {
        "timestamp_ms": np.concatenate(target_times),
        "half_id": np.concatenate(half_ids),
        "source_frame_index": source_frame_index,
        "source_timestamp_ms": source_frame_index * source_period_ms,
        "source_to_sample_index": source_to_sample,
        "source_half_id": source_half_id,
        "source_scored": source_scored,
        "scored_frame_intervals": intervals,
    }


def load_selected_player_features(
    tracking_path: str | Path,
    metadata: Mapping[str, object],
    alignment: Mapping[str, np.ndarray],
    n_source_frames: int,
) -> Dict[str, np.ndarray]:
    """Stream selected ATD frames into fixed Home/Away unordered player slots."""
    selected_frames = alignment["source_frame_index"]
    half_id = alignment["half_id"]
    selected_lookup = np.full(n_source_frames, -1, dtype=np.int64)
    selected_lookup[selected_frames] = np.arange(len(selected_frames), dtype=np.int64)
    n_samples = len(selected_frames)
    players = np.zeros((n_samples, 2, MAX_PLAYERS_PER_TEAM, 3), dtype=np.float32)
    location_valid = np.zeros((n_samples, 2, MAX_PLAYERS_PER_TEAM), dtype=bool)
    speed_valid = np.zeros_like(location_valid)
    ball = np.zeros((n_samples, 3), dtype=np.float32)
    ball_location_valid = np.zeros(n_samples, dtype=bool)
    ball_speed_valid = np.zeros(n_samples, dtype=bool)
    slot_team = np.asarray(metadata["slot_team_index"])
    slot_within_team = np.asarray(metadata["slot_within_team"])
    seen = np.zeros(n_samples, dtype=bool)

    line_count = 0
    with Path(tracking_path).open(encoding="utf-8") as handle:
        for line_count, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split(":", 2)
            if len(parts) != 3:
                raise ValueError(f"Malformed ATD record at line {line_count}")
            frame_id = int(parts[0])
            if frame_id != line_count:
                raise ValueError(
                    f"ATD frames must be contiguous from 1; line {line_count} has {frame_id}"
                )
            source_frame = frame_id - 1
            sample = selected_lookup[source_frame]
            if sample < 0:
                continue
            groups = parts[1].split(";")
            if len(groups) != len(slot_team):
                raise ValueError(
                    f"Frame {frame_id} has {len(groups)} player slots; expected {len(slot_team)}"
                )
            rotate = bool(ROTATION_APPLIED[half_id[sample] - 1])
            for source_slot, group in enumerate(groups):
                values = group.split(",")
                if len(values) != 3:
                    raise ValueError(f"Malformed player slot at ATD frame {frame_id}")
                x, y, speed = (float(value) for value in values)
                team = int(slot_team[source_slot])
                slot = int(slot_within_team[source_slot])
                if np.isfinite(x) and np.isfinite(y):
                    if rotate:
                        x, y = 1.0 - x, 1.0 - y
                    players[sample, team, slot, :2] = (x, y)
                    location_valid[sample, team, slot] = True
                if np.isfinite(speed):
                    players[sample, team, slot, 2] = speed
                    speed_valid[sample, team, slot] = True
            ball_values = parts[2].split(",")
            if len(ball_values) != 3:
                raise ValueError(f"Malformed ball observation at ATD frame {frame_id}")
            ball_x, ball_y, ball_speed = (float(value) for value in ball_values)
            if np.isfinite(ball_x) and np.isfinite(ball_y):
                if rotate:
                    ball_x, ball_y = 1.0 - ball_x, 1.0 - ball_y
                ball[sample, :2] = (ball_x, ball_y)
                ball_location_valid[sample] = True
            if np.isfinite(ball_speed):
                ball[sample, 2] = ball_speed
                ball_speed_valid[sample] = True
            seen[sample] = True

    if line_count != n_source_frames:
        raise ValueError(
            f"Expected {n_source_frames} ATD records, found {line_count} in {tracking_path}"
        )
    if not seen.all():
        missing = selected_frames[~seen]
        raise ValueError(f"Selected source frames were not read: {missing[:10].tolist()}")
    return {
        "players": players,
        "player_location_valid": location_valid,
        "player_speed_valid": speed_valid,
        "ball": ball,
        "ball_location_valid": ball_location_valid,
        "ball_speed_valid": ball_speed_valid,
    }


def build_metrica_features(
    metadata_path: str | Path,
    tracking_path: str | Path,
) -> Dict[str, np.ndarray]:
    """Build the complete scored-half, model-ready Metrica feature archive."""
    metadata = load_metrica_metadata(metadata_path)
    n_frames = source_frames("tactical")
    intervals = [(start, end) for _, start, end in half_video_blocks("BAR-ATH")]
    alignment = build_resampling_index(
        n_frames, intervals, float(metadata["source_fps"]), MODEL_FPS
    )
    result = load_selected_player_features(
        tracking_path, metadata, alignment, n_frames
    )
    team_features, team_valid = build_team_features(
        result["players"],
        result["player_location_valid"],
        result["player_speed_valid"],
    )
    result.update(alignment)
    result.update(
        {
            "team_features": team_features,
            "team_feature_valid": team_valid,
            "sample_index": np.arange(len(alignment["half_id"]), dtype=np.int64),
            "player_feature_names": PLAYER_FEATURE_NAMES.copy(),
            "ball_feature_names": BALL_FEATURE_NAMES.copy(),
            "team_feature_names": TEAM_FEATURE_NAMES.copy(),
            "team_order": TEAM_ORDER.copy(),
            "team_names": np.asarray(metadata["team_names"]),
            "source_fps": np.array(MODEL_FPS, dtype=np.float32),
            "original_source_fps": np.array(
                metadata["source_fps"], dtype=np.float32
            ),
            "original_frame_count": np.array(n_frames, dtype=np.int64),
            "pitch_length_m": np.array(metadata["pitch_length_m"], dtype=np.float32),
            "pitch_width_m": np.array(metadata["pitch_width_m"], dtype=np.float32),
            "schema_version": np.array("1.1"),
            "ball_speed_provenance": np.array(
                "provider-reported speed from the selected nearest 25 Hz ATD frame"
            ),
            "orientation": np.array(
                "home attacks +x; away attacks -x; both teams rotated 180 degrees in half 2"
            ),
            "raw_attack_direction": RAW_ATTACK_DIRECTION.copy(),
            "rotation_applied": ROTATION_APPLIED.copy(),
            "resampling_description": np.array(
                "independent half grids at 10 Hz; nearest 25 Hz frame; earlier frame on ties"
            ),
        }
    )
    validate_metrica_features(result)
    return result


def validate_metrica_features(result: Mapping[str, np.ndarray]) -> None:
    """Validate the shared feature schema and Metrica-specific alignment contract."""
    validate_features(result)
    required = {
        "source_frame_index",
        "source_timestamp_ms",
        "original_source_fps",
        "original_frame_count",
        "source_to_sample_index",
        "source_half_id",
        "source_scored",
        "scored_frame_intervals",
        "team_names",
        "raw_attack_direction",
        "rotation_applied",
        "resampling_description",
    }
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Metrica feature result is missing: {sorted(missing)}")
    n_samples = len(result["sample_index"])
    n_source = int(result["original_frame_count"])
    expected_dtypes = {
        "timestamp_ms": np.dtype(np.int64),
        "half_id": np.dtype(np.int8),
        "sample_index": np.dtype(np.int64),
        "source_frame_index": np.dtype(np.int64),
        "source_timestamp_ms": np.dtype(np.int64),
        "source_to_sample_index": np.dtype(np.int64),
        "source_half_id": np.dtype(np.int8),
        "source_scored": np.dtype(bool),
        "scored_frame_intervals": np.dtype(np.int64),
        "rotation_applied": np.dtype(bool),
    }
    for name, dtype in expected_dtypes.items():
        if result[name].dtype != dtype:
            raise ValueError(f"{name} must use dtype {dtype}")
    if float(result["source_fps"]) != MODEL_FPS:
        raise ValueError("Model-ready Metrica features must be sampled at 10 Hz")
    if float(result["original_source_fps"]) != EXPECTED_SOURCE_FPS:
        raise ValueError("Metrica original_source_fps must be 25 Hz")
    if result["source_frame_index"].shape != (n_samples,):
        raise ValueError("source_frame_index must contain one frame per sample")
    if result["source_timestamp_ms"].shape != (n_samples,):
        raise ValueError("source_timestamp_ms must contain one time per sample")
    if result["source_to_sample_index"].shape != (n_source,):
        raise ValueError("source_to_sample_index has the wrong length")
    if result["source_half_id"].shape != (n_source,) or result["source_scored"].shape != (
        n_source,
    ):
        raise ValueError("Full-source half/scored metadata has the wrong shape")
    if not np.array_equal(result["source_scored"], result["source_half_id"] > 0):
        raise ValueError("source_scored and source_half_id disagree")
    reverse = result["source_to_sample_index"]
    if np.any(reverse[result["source_scored"]] < 0) or np.any(
        reverse[result["source_scored"]] >= n_samples
    ):
        raise ValueError("Scored source frames must map to valid samples")
    if np.any(reverse[~result["source_scored"]] != -1):
        raise ValueError("Unscored source frames must map to -1")
    if not np.all(np.diff(result["source_frame_index"]) > 0):
        raise ValueError("Selected source frames must be strictly increasing")
    if not np.array_equal(result["team_order"], TEAM_ORDER):
        raise ValueError("Metrica team order must match WCSF")
    if not np.array_equal(result["rotation_applied"], ROTATION_APPLIED):
        raise ValueError("Unexpected Metrica half rotations")

    expected = build_resampling_index(
        n_source,
        [tuple(interval) for interval in result["scored_frame_intervals"]],
        float(result["original_source_fps"]),
        float(result["source_fps"]),
    )
    for name in expected:
        if not np.array_equal(result[name], expected[name]):
            raise ValueError(f"Metrica alignment array {name} is inconsistent")


def summarize_team_features(
    team_features: np.ndarray, team_feature_valid: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Return compact descriptive statistics across timestamps and teams."""
    summary = {}
    for index, name in enumerate(TEAM_FEATURE_NAMES.tolist()):
        valid = team_feature_valid[..., index]
        values = team_features[..., index][valid].astype(np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=0)),
            "p10": float(np.quantile(values, 0.1)),
            "p90": float(np.quantile(values, 0.9)),
            "valid_fraction": float(valid.mean()),
        }
    return summary


def _default_paths() -> tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data" / "metrica"
    paths = asset_paths(data_dir=str(data_root), source="tactical")
    return (
        Path(paths["atd_meta"]),
        Path(paths["atd_raw"]),
        data_root / "processed" / "metrica_player_team_ball_features.npz",
        root
        / "data"
        / "open_data_groningen"
        / "processed"
        / "wcsf_player_team_ball_features.npz",
    )


def main() -> None:
    metadata_default, tracking_default, output_default, wcsf_default = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=metadata_default)
    parser.add_argument("--tracking", type=Path, default=tracking_default)
    parser.add_argument("--output", type=Path, default=output_default)
    parser.add_argument("--wcsf-features", type=Path, default=wcsf_default)
    args = parser.parse_args()

    result = build_metrica_features(args.metadata, args.tracking)
    save_features(result, args.output)
    comparison = {
        "metrica": summarize_team_features(
            result["team_features"], result["team_feature_valid"]
        )
    }
    if args.wcsf_features.is_file():
        with np.load(args.wcsf_features, allow_pickle=False) as archive:
            comparison["wcsf"] = summarize_team_features(
                archive["team_features"], archive["team_feature_valid"]
            )
    half_counts = {
        int(half): int(count)
        for half, count in zip(*np.unique(result["half_id"], return_counts=True))
    }
    print(f"Saved {len(result['sample_index']):,} samples to {args.output}")
    print(f"Half counts: {half_counts}")
    print(
        "Valid locations: "
        f"{100.0 * result['player_location_valid'].mean():.2f}%; "
        f"valid speeds: {100.0 * result['player_speed_valid'].mean():.2f}%"
    )
    print(
        "Valid ball locations/speeds: "
        f"{100.0 * result['ball_location_valid'].mean():.2f}%/"
        f"{100.0 * result['ball_speed_valid'].mean():.2f}%"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
