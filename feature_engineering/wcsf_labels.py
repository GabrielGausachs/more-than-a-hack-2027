"""Build binary WCSF ALIVE/DEAD labels aligned with the tracking features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


MAJORITY_THRESHOLD = 0.75
LABEL_NAMES = np.array(["DEAD", "ALIVE"])

_VOTE_DTYPE = np.dtype(
    [
        ("half_id", "i1"),
        ("timestamp_ms", "i8"),
        ("group_id", "i1"),
        ("player_id", "i8"),
        ("alive_flag", "i1"),
    ]
)


def _parse_alive_flag(value: str, path: Path, line_number: int) -> int:
    value = value.strip()
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(
            f"AliveFlag must be 0 or 1 in {path} at CSV line {line_number}; found {value!r}"
        ) from exc
    if not np.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(
            f"AliveFlag must be 0 or 1 in {path} at CSV line {line_number}; found {value!r}"
        )
    return int(numeric)


def _read_half_votes(path: Path, half_id: int) -> np.ndarray:
    required = {"timestamp_ms", "group_id", "player_id", "AliveFlag"}

    def votes() -> Iterable[Tuple[int, int, int, int, int]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    timestamp_ms = int(row["timestamp_ms"])
                    group_id = int(row["group_id"])
                    player_id = int(row["player_id"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid identifier in {path} at CSV line {line_number}") from exc
                yield (
                    half_id,
                    timestamp_ms,
                    group_id,
                    player_id,
                    _parse_alive_flag(row["AliveFlag"], path, line_number),
                )

    return np.fromiter(votes(), dtype=_VOTE_DTYPE)


def load_alive_votes(paths: Sequence[Path | str]) -> np.ndarray:
    """Load per-player AliveFlag votes and attach 1-based half identifiers."""
    if not paths:
        raise ValueError("At least one WCSF positions file is required")
    halves = [
        _read_half_votes(Path(path), half_id)
        for half_id, path in enumerate(paths, start=1)
    ]
    return np.concatenate(halves)


def _feature_timeline(feature_archive_path: Path | str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(feature_archive_path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"half_id", "timestamp_ms", "sample_index"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Feature archive is missing arrays: {sorted(missing)}")
        return (
            archive["half_id"].copy(),
            archive["timestamp_ms"].copy(),
            archive["sample_index"].copy(),
        )


def build_alive_dead_labels(
    paths: Sequence[Path | str],
    feature_archive_path: Path | str,
    majority_threshold: float = MAJORITY_THRESHOLD,
) -> Dict[str, np.ndarray]:
    """Aggregate player votes into binary timestamp labels and align them to features."""
    if not 0.5 < majority_threshold < 1.0:
        raise ValueError("majority_threshold must be strictly between 0.5 and 1.0")

    votes = load_alive_votes(paths)
    if not len(votes):
        raise ValueError("No AliveFlag votes were loaded")
    if not np.isin(votes["group_id"], (1, 2)).all():
        invalid = np.unique(votes["group_id"][~np.isin(votes["group_id"], (1, 2))])
        raise ValueError(f"group_id must be 1 (home) or 2 (away); found {invalid.tolist()}")

    order = np.lexsort(
        (votes["player_id"], votes["group_id"], votes["timestamp_ms"], votes["half_id"])
    )
    votes = votes[order]
    duplicate = (
        (votes["half_id"][1:] == votes["half_id"][:-1])
        & (votes["timestamp_ms"][1:] == votes["timestamp_ms"][:-1])
        & (votes["group_id"][1:] == votes["group_id"][:-1])
        & (votes["player_id"][1:] == votes["player_id"][:-1])
    )
    if duplicate.any():
        vote = votes[1:][np.flatnonzero(duplicate)[0]]
        raise ValueError(
            "Duplicate AliveFlag vote for "
            f"half={vote['half_id']}, timestamp_ms={vote['timestamp_ms']}, "
            f"group_id={vote['group_id']}, player_id={vote['player_id']}"
        )

    frame_start = np.r_[
        True,
        (votes["half_id"][1:] != votes["half_id"][:-1])
        | (votes["timestamp_ms"][1:] != votes["timestamp_ms"][:-1]),
    ]
    frame_index = np.cumsum(frame_start) - 1
    n_frames = int(frame_index[-1]) + 1
    alive_count_i64 = np.bincount(
        frame_index, weights=votes["alive_flag"] == 1, minlength=n_frames
    ).astype(np.int64)
    dead_count_i64 = np.bincount(
        frame_index, weights=votes["alive_flag"] == 0, minlength=n_frames
    ).astype(np.int64)
    vote_count = alive_count_i64 + dead_count_i64
    if vote_count.max() > np.iinfo(np.uint8).max:
        raise ValueError("Vote counts exceed uint8 storage capacity")

    alive_fraction = alive_count_i64 / vote_count
    dead_fraction = dead_count_i64 / vote_count
    majority_share = np.maximum(alive_fraction, dead_fraction)
    insufficient = majority_share <= majority_threshold
    if insufficient.any():
        frame_rows = np.flatnonzero(frame_start)
        index = np.flatnonzero(insufficient)[0]
        vote = votes[frame_rows[index]]
        raise ValueError(
            f"No strict >{majority_threshold:.0%} AliveFlag majority for "
            f"half={vote['half_id']}, timestamp_ms={vote['timestamp_ms']}: "
            f"ALIVE={alive_count_i64[index]}/{vote_count[index]}, "
            f"DEAD={dead_count_i64[index]}/{vote_count[index]}"
        )

    frame_rows = np.flatnonzero(frame_start)
    half_id = votes["half_id"][frame_rows].astype(np.int8)
    timestamp_ms = votes["timestamp_ms"][frame_rows].astype(np.int64)
    sample_index = np.arange(n_frames, dtype=np.int64)
    labels = (alive_count_i64 > dead_count_i64).astype(np.int8)

    feature_half, feature_time, feature_index = _feature_timeline(feature_archive_path)
    if not (
        np.array_equal(half_id, feature_half)
        and np.array_equal(timestamp_ms, feature_time)
        and np.array_equal(sample_index, feature_index)
    ):
        raise ValueError("Label timeline does not align exactly with the player/team feature archive")

    result = {
        "labels": labels,
        "alive_vote_count": alive_count_i64.astype(np.uint8),
        "dead_vote_count": dead_count_i64.astype(np.uint8),
        "alive_vote_fraction": alive_fraction.astype(np.float32),
        "dead_vote_fraction": dead_fraction.astype(np.float32),
        "timestamp_ms": timestamp_ms,
        "half_id": half_id,
        "sample_index": sample_index,
        "label_names": LABEL_NAMES,
        "majority_threshold": np.array(majority_threshold, dtype=np.float32),
        "schema_version": np.array("1.0"),
    }
    validate_labels(result, feature_archive_path)
    return result


def validate_labels(
    result: Mapping[str, np.ndarray], feature_archive_path: Path | str
) -> None:
    """Validate binary labels, vote provenance, chronology and feature alignment."""
    required = {
        "labels",
        "alive_vote_count",
        "dead_vote_count",
        "alive_vote_fraction",
        "dead_vote_fraction",
        "timestamp_ms",
        "half_id",
        "sample_index",
        "majority_threshold",
    }
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Label result is missing arrays: {sorted(missing)}")

    labels = result["labels"]
    n_frames = len(labels)
    for name in (
        "alive_vote_count",
        "dead_vote_count",
        "alive_vote_fraction",
        "dead_vote_fraction",
        "timestamp_ms",
        "half_id",
        "sample_index",
    ):
        if result[name].shape != (n_frames,):
            raise ValueError(f"{name} must have shape ({n_frames},)")
    if labels.dtype != np.int8 or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be int8 with values only 0 (DEAD) and 1 (ALIVE)")
    if result["alive_vote_count"].dtype != np.uint8 or result["dead_vote_count"].dtype != np.uint8:
        raise ValueError("Vote counts must use uint8")
    if result["alive_vote_fraction"].dtype != np.float32 or result["dead_vote_fraction"].dtype != np.float32:
        raise ValueError("Vote fractions must use float32")

    alive_count = result["alive_vote_count"].astype(np.int64)
    dead_count = result["dead_vote_count"].astype(np.int64)
    total = alive_count + dead_count
    if np.any(total == 0):
        raise ValueError("Every timestamp must contain at least one AliveFlag vote")
    expected_alive_fraction = alive_count / total
    expected_dead_fraction = dead_count / total
    if not np.allclose(result["alive_vote_fraction"], expected_alive_fraction, atol=1e-7):
        raise ValueError("alive_vote_fraction is inconsistent with vote counts")
    if not np.allclose(result["dead_vote_fraction"], expected_dead_fraction, atol=1e-7):
        raise ValueError("dead_vote_fraction is inconsistent with vote counts")

    threshold = float(result["majority_threshold"])
    majority_share = np.maximum(expected_alive_fraction, expected_dead_fraction)
    if np.any(majority_share <= threshold):
        raise ValueError("Every timestamp must have a majority strictly above the threshold")
    expected_labels = (alive_count > dead_count).astype(np.int8)
    if not np.array_equal(labels, expected_labels):
        raise ValueError("labels are inconsistent with the AliveFlag vote majority")

    if not np.array_equal(result["sample_index"], np.arange(n_frames)):
        raise ValueError("sample_index must be contiguous and zero-based")
    if np.any(np.diff(result["half_id"]) < 0):
        raise ValueError("Halves must be ordered")
    for half in np.unique(result["half_id"]):
        times = result["timestamp_ms"][result["half_id"] == half]
        if len(times) > 1 and not np.all(np.diff(times) == 100):
            raise ValueError(f"Half {half} is not contiguous at 10 Hz")

    feature_half, feature_time, feature_index = _feature_timeline(feature_archive_path)
    if not (
        np.array_equal(result["half_id"], feature_half)
        and np.array_equal(result["timestamp_ms"], feature_time)
        and np.array_equal(result["sample_index"], feature_index)
    ):
        raise ValueError("Labels do not align exactly with the player/team feature archive")


def save_labels(
    result: Mapping[str, np.ndarray],
    output_path: Path | str,
    feature_archive_path: Path | str,
) -> None:
    """Validate and save a compressed ALIVE/DEAD label archive."""
    validate_labels(result, feature_archive_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)


def _default_paths() -> Tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "open_data_groningen"
    processed_dir = data_dir / "processed"
    return (
        data_dir / "positions_FIRST_HALF.csv",
        data_dir / "positions_SECOND_HALF.csv",
        processed_dir / "wcsf_player_team_features.npz",
        processed_dir / "wcsf_alive_dead_labels.npz",
    )


def main() -> None:
    first_default, second_default, features_default, output_default = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-half", type=Path, default=first_default)
    parser.add_argument("--second-half", type=Path, default=second_default)
    parser.add_argument("--features", type=Path, default=features_default)
    parser.add_argument("--output", type=Path, default=output_default)
    args = parser.parse_args()

    result = build_alive_dead_labels(
        (args.first_half, args.second_half), args.features, MAJORITY_THRESHOLD
    )
    save_labels(result, args.output, args.features)
    labels, counts = np.unique(result["labels"], return_counts=True)
    label_counts = {str(LABEL_NAMES[label]): int(count) for label, count in zip(labels, counts)}
    conflicts = np.count_nonzero(
        (result["alive_vote_count"] > 0) & (result["dead_vote_count"] > 0)
    )
    print(f"Saved {len(result['labels']):,} labels to {args.output}")
    print(f"Label counts: {label_counts}; majority-resolved conflicts: {conflicts}")


if __name__ == "__main__":
    main()
