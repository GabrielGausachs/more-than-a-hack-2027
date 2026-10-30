"""Build model-ready WCSF player-set and team-activity features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
MAX_PLAYERS_PER_TEAM = 24
SOURCE_FPS = 10.0

PLAYER_FEATURE_NAMES = np.array(["x", "y", "speed"])
TEAM_FEATURE_NAMES = np.array(
    [
        "mean_speed",
        "median_speed",
        "speed_std",
        "tracked_player_count",
        "valid_speed_count",
        "speed_coverage",
    ]
)
TEAM_ORDER = np.array(["home", "away"])

_ROW_DTYPE = np.dtype(
    [
        ("half_id", "i1"),
        ("timestamp_ms", "i8"),
        ("group_id", "i1"),
        ("player_id", "i8"),
        ("x", "f8"),
        ("y", "f8"),
        ("speed", "f8"),
    ]
)


def _float_or_nan(value: str) -> float:
    value = value.strip()
    return float(value) if value else float("nan")


def _read_half(path: Path, half_id: int) -> np.ndarray:
    required = {"timestamp_ms", "group_id", "player_id", "x", "y", "speed"}

    def rows() -> Iterable[Tuple[int, int, int, int, float, float, float]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    yield (
                        half_id,
                        int(row["timestamp_ms"]),
                        int(row["group_id"]),
                        int(row["player_id"]),
                        _float_or_nan(row["x"]),
                        _float_or_nan(row["y"]),
                        _float_or_nan(row["speed"]),
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid value in {path} at CSV line {line_number}") from exc

    return np.fromiter(rows(), dtype=_ROW_DTYPE)


def load_wcsf_positions(paths: Sequence[Path | str]) -> np.ndarray:
    """Load the required columns and attach 1-based half identifiers."""
    if not paths:
        raise ValueError("At least one WCSF positions file is required")
    halves = [_read_half(Path(path), half_id) for half_id, path in enumerate(paths, start=1)]
    return np.concatenate(halves)


def build_player_features(rows: np.ndarray) -> Dict[str, np.ndarray]:
    """Orient, normalize and densely pack timestamp-level unordered player sets."""
    if rows.dtype != _ROW_DTYPE:
        raise TypeError("rows must be the structured array returned by load_wcsf_positions")
    if not len(rows):
        raise ValueError("No WCSF player rows were loaded")

    order = np.lexsort(
        (rows["player_id"], rows["group_id"], rows["timestamp_ms"], rows["half_id"])
    )
    rows = rows[order]

    if not np.isin(rows["group_id"], (1, 2)).all():
        invalid = np.unique(rows["group_id"][~np.isin(rows["group_id"], (1, 2))])
        raise ValueError(f"group_id must be 1 (home) or 2 (away); found {invalid.tolist()}")

    duplicate = (
        (rows["half_id"][1:] == rows["half_id"][:-1])
        & (rows["timestamp_ms"][1:] == rows["timestamp_ms"][:-1])
        & (rows["group_id"][1:] == rows["group_id"][:-1])
        & (rows["player_id"][1:] == rows["player_id"][:-1])
    )
    if duplicate.any():
        row = rows[1:][np.flatnonzero(duplicate)[0]]
        raise ValueError(
            "Duplicate player observation for "
            f"half={row['half_id']}, timestamp_ms={row['timestamp_ms']}, "
            f"group_id={row['group_id']}, player_id={row['player_id']}"
        )

    frame_start = np.r_[
        True,
        (rows["half_id"][1:] != rows["half_id"][:-1])
        | (rows["timestamp_ms"][1:] != rows["timestamp_ms"][:-1]),
    ]
    frame_index = np.cumsum(frame_start) - 1
    team_index = rows["group_id"].astype(np.int64) - 1

    set_start = np.r_[
        True,
        frame_start[1:] | (rows["group_id"][1:] != rows["group_id"][:-1]),
    ]
    set_starts = np.flatnonzero(set_start)
    set_lengths = np.diff(np.r_[set_starts, len(rows)])
    if set_lengths.max() > MAX_PLAYERS_PER_TEAM:
        start = set_starts[np.argmax(set_lengths)]
        row = rows[start]
        raise ValueError(
            f"More than {MAX_PLAYERS_PER_TEAM} player observations for "
            f"half={row['half_id']}, timestamp_ms={row['timestamp_ms']}, "
            f"group_id={row['group_id']}: found {set_lengths.max()}"
        )
    slot_index = np.arange(len(rows)) - np.repeat(set_starts, set_lengths)

    n_frames = int(frame_index[-1]) + 1
    players = np.zeros((n_frames, 2, MAX_PLAYERS_PER_TEAM, 3), dtype=np.float32)
    location_valid = np.zeros((n_frames, 2, MAX_PLAYERS_PER_TEAM), dtype=bool)
    speed_valid = np.zeros_like(location_valid)

    location_is_valid = np.isfinite(rows["x"]) & np.isfinite(rows["y"])
    speed_is_valid = np.isfinite(rows["speed"])
    orientation = np.where(team_index == 0, 1.0, -1.0)
    x_normalized = rows["x"] * orientation / PITCH_LENGTH_M + 0.5
    y_normalized = rows["y"] * orientation / PITCH_WIDTH_M + 0.5

    index = (frame_index, team_index, slot_index)
    location_valid[index] = location_is_valid
    speed_valid[index] = speed_is_valid
    valid_location_index = tuple(axis[location_is_valid] for axis in index)
    players[valid_location_index + (np.zeros(location_is_valid.sum(), dtype=np.int64),)] = (
        x_normalized[location_is_valid]
    )
    players[valid_location_index + (np.ones(location_is_valid.sum(), dtype=np.int64),)] = (
        y_normalized[location_is_valid]
    )
    valid_speed_index = tuple(axis[speed_is_valid] for axis in index)
    players[valid_speed_index + (np.full(speed_is_valid.sum(), 2, dtype=np.int64),)] = rows[
        "speed"
    ][speed_is_valid]

    frame_rows = np.flatnonzero(frame_start)
    return {
        "players": players,
        "player_location_valid": location_valid,
        "player_speed_valid": speed_valid,
        "timestamp_ms": rows["timestamp_ms"][frame_rows].astype(np.int64),
        "half_id": rows["half_id"][frame_rows].astype(np.int8),
        "sample_index": np.arange(n_frames, dtype=np.int64),
    }


def build_team_features(
    players: np.ndarray,
    player_location_valid: np.ndarray,
    player_speed_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate independent per-timestamp team activity summaries."""
    expected_mask_shape = players.shape[:-1]
    if players.ndim != 4 or players.shape[-1] != 3 or players.shape[1:3] != (2, 24):
        raise ValueError("players must have shape (N, 2, 24, 3)")
    if player_location_valid.shape != expected_mask_shape:
        raise ValueError("player_location_valid shape does not match players")
    if player_speed_valid.shape != expected_mask_shape:
        raise ValueError("player_speed_valid shape does not match players")

    speeds = players[..., 2].astype(np.float64)
    speed_count = player_speed_valid.sum(axis=2)
    tracked_count = player_location_valid.sum(axis=2)
    has_speed = speed_count > 0
    has_player = tracked_count > 0

    speed_sum = np.where(player_speed_valid, speeds, 0.0).sum(axis=2)
    mean = np.divide(speed_sum, speed_count, out=np.zeros_like(speed_sum), where=has_speed)

    sorted_speeds = np.sort(np.where(player_speed_valid, speeds, np.inf), axis=2)
    lower_index = np.maximum((speed_count - 1) // 2, 0)[..., None]
    upper_index = np.maximum(speed_count // 2, 0)[..., None]
    lower = np.take_along_axis(sorted_speeds, lower_index, axis=2)[..., 0]
    upper = np.take_along_axis(sorted_speeds, upper_index, axis=2)[..., 0]
    median = np.where(has_speed, (lower + upper) / 2.0, 0.0)

    centered = np.where(player_speed_valid, speeds - mean[..., None], 0.0)
    variance = np.divide(
        (centered * centered).sum(axis=2),
        speed_count,
        out=np.zeros_like(speed_sum),
        where=has_speed,
    )
    std = np.sqrt(variance)
    coverage = np.divide(
        speed_count,
        tracked_count,
        out=np.zeros_like(speed_sum),
        where=has_player,
    )

    features = np.stack(
        (mean, median, std, tracked_count, speed_count, coverage), axis=-1
    ).astype(np.float32)
    valid = np.stack(
        (has_speed, has_speed, has_speed, np.ones_like(has_player), np.ones_like(has_player), has_player),
        axis=-1,
    )
    return features, valid


def validate_features(result: Mapping[str, np.ndarray]) -> None:
    """Validate array shapes, masking invariants, chronology and derived values."""
    required = {
        "players",
        "player_location_valid",
        "player_speed_valid",
        "team_features",
        "team_feature_valid",
        "timestamp_ms",
        "half_id",
        "sample_index",
    }
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Feature result is missing arrays: {sorted(missing)}")

    players = result["players"]
    location_valid = result["player_location_valid"]
    speed_valid = result["player_speed_valid"]
    team_features = result["team_features"]
    team_valid = result["team_feature_valid"]
    n_frames = len(result["timestamp_ms"])

    if players.shape != (n_frames, 2, 24, 3):
        raise ValueError(f"Unexpected players shape: {players.shape}")
    if location_valid.shape != (n_frames, 2, 24) or speed_valid.shape != location_valid.shape:
        raise ValueError("Unexpected player mask shape")
    if team_features.shape != (n_frames, 2, 6) or team_valid.shape != team_features.shape:
        raise ValueError("Unexpected team feature or mask shape")
    if players.dtype != np.float32 or team_features.dtype != np.float32:
        raise ValueError("Numeric feature arrays must use float32")
    if location_valid.dtype != np.bool_ or speed_valid.dtype != np.bool_ or team_valid.dtype != np.bool_:
        raise ValueError("Validity masks must use bool")
    if not np.isfinite(players).all() or not np.isfinite(team_features).all():
        raise ValueError("Saved numeric feature arrays must contain only finite values")
    if np.any(players[..., :2][~location_valid] != 0.0):
        raise ValueError("Invalid player locations must be zero-filled")
    if np.any(players[..., 2][~speed_valid] != 0.0):
        raise ValueError("Invalid player speeds must be zero-filled")
    if np.any(team_features[~team_valid] != 0.0):
        raise ValueError("Invalid team features must be zero-filled")
    if not np.array_equal(result["sample_index"], np.arange(n_frames)):
        raise ValueError("sample_index must be contiguous and zero-based")

    half_id = result["half_id"]
    timestamp_ms = result["timestamp_ms"]
    if len(half_id) != n_frames or len(result["sample_index"]) != n_frames:
        raise ValueError("Metadata lengths do not match the feature arrays")
    if np.any(np.diff(half_id) < 0):
        raise ValueError("Halves must be ordered")
    for half in np.unique(half_id):
        times = timestamp_ms[half_id == half]
        if len(times) > 1 and not np.all(np.diff(times) == 100):
            raise ValueError(f"Half {half} is not contiguous at 10 Hz")

    expected_features, expected_valid = build_team_features(players, location_valid, speed_valid)
    if not np.array_equal(team_valid, expected_valid):
        raise ValueError("team_feature_valid is inconsistent with player masks")
    if not np.allclose(team_features, expected_features, rtol=1e-6, atol=1e-6):
        raise ValueError("team_features are inconsistent with player features and masks")


def save_features(result: Mapping[str, np.ndarray], output_path: Path | str) -> None:
    """Validate and write a compressed, self-describing NPZ archive."""
    validate_features(result)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)


def build_features(paths: Sequence[Path | str]) -> Dict[str, np.ndarray]:
    """Run the complete in-memory WCSF feature pipeline."""
    result = build_player_features(load_wcsf_positions(paths))
    team_features, team_valid = build_team_features(
        result["players"], result["player_location_valid"], result["player_speed_valid"]
    )
    result.update(
        {
            "team_features": team_features,
            "team_feature_valid": team_valid,
            "player_feature_names": PLAYER_FEATURE_NAMES,
            "team_feature_names": TEAM_FEATURE_NAMES,
            "team_order": TEAM_ORDER,
            "source_fps": np.array(SOURCE_FPS, dtype=np.float32),
            "pitch_length_m": np.array(PITCH_LENGTH_M, dtype=np.float32),
            "pitch_width_m": np.array(PITCH_WIDTH_M, dtype=np.float32),
            "schema_version": np.array("1.0"),
            "orientation": np.array(
                "home attacks +x; away attacks -x; away source coordinates rotated 180 degrees"
            ),
        }
    )
    validate_features(result)
    return result


def _default_paths() -> Tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "open_data_groningen"
    return (
        data_dir / "positions_FIRST_HALF.csv",
        data_dir / "positions_SECOND_HALF.csv",
        data_dir / "processed" / "wcsf_player_team_features.npz",
    )


def main() -> None:
    first_default, second_default, output_default = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-half", type=Path, default=first_default)
    parser.add_argument("--second-half", type=Path, default=second_default)
    parser.add_argument("--output", type=Path, default=output_default)
    args = parser.parse_args()

    result = build_features((args.first_half, args.second_half))
    save_features(result, args.output)
    half_counts = {
        int(half): int(count)
        for half, count in zip(*np.unique(result["half_id"], return_counts=True))
    }
    print(f"Saved {len(result['timestamp_ms']):,} timestamps to {args.output}")
    print(f"Half counts: {half_counts}")
    print(f"players: {result['players'].shape}; team_features: {result['team_features'].shape}")


if __name__ == "__main__":
    main()
