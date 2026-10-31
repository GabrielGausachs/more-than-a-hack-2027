"""Build strict WCSF Home/Away/Dead labels aligned with tracking features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .wcsf_labels import _feature_timeline, _parse_alive_flag


LABEL_NAMES = np.array(["H", "A", "D"])
AGGREGATION_POLICY = np.array("strict_unanimity_dead_precedence")
H, A, D, UNKNOWN = 0, 1, 2, -1

_VOTE_DTYPE = np.dtype(
    [
        ("half_id", "i1"),
        ("timestamp_ms", "i8"),
        ("group_id", "i1"),
        ("player_id", "i8"),
        ("alive_flag", "i1"),
        ("possession_flag", "i1"),
    ]
)


def _parse_possession_flag(value: str, path: Path, line_number: int) -> int:
    value = value.strip()
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(
            f"PossessionFlag must be 1 or 2 in {path} at CSV line "
            f"{line_number}; found {value!r}"
        ) from exc
    if not np.isfinite(numeric) or numeric not in (1.0, 2.0):
        raise ValueError(
            f"PossessionFlag must be 1 or 2 in {path} at CSV line "
            f"{line_number}; found {value!r}"
        )
    return int(numeric)


def _read_half_votes(path: Path, half_id: int) -> np.ndarray:
    required = {
        "timestamp_ms",
        "group_id",
        "player_id",
        "AliveFlag",
        "PossessionFlag",
    }

    def votes() -> Iterable[Tuple[int, int, int, int, int, int]]:
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
                    raise ValueError(
                        f"Invalid identifier in {path} at CSV line {line_number}"
                    ) from exc
                yield (
                    half_id,
                    timestamp_ms,
                    group_id,
                    player_id,
                    _parse_alive_flag(row["AliveFlag"], path, line_number),
                    _parse_possession_flag(
                        row["PossessionFlag"], path, line_number
                    ),
                )

    return np.fromiter(votes(), dtype=_VOTE_DTYPE)


def load_had_votes(paths: Sequence[Path | str]) -> np.ndarray:
    """Load per-player ALIVE and possession votes with 1-based half IDs."""
    if not paths:
        raise ValueError("At least one WCSF positions file is required")
    return np.concatenate(
        [
            _read_half_votes(Path(path), half_id)
            for half_id, path in enumerate(paths, start=1)
        ]
    )


def build_had_labels(
    paths: Sequence[Path | str], feature_archive_path: Path | str
) -> Dict[str, np.ndarray]:
    """Build strict H/A/D targets; ambiguous timestamps receive label ``-1``."""
    votes = load_had_votes(paths)
    if not len(votes):
        raise ValueError("No WCSF flag votes were loaded")
    if not np.isin(votes["group_id"], (1, 2)).all():
        invalid = np.unique(votes["group_id"][~np.isin(votes["group_id"], (1, 2))])
        raise ValueError(
            f"group_id must be 1 (home) or 2 (away); found {invalid.tolist()}"
        )

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
            "Duplicate flag vote for "
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

    def count(values: np.ndarray) -> np.ndarray:
        result = np.bincount(frame_index, weights=values, minlength=n_frames).astype(
            np.int64
        )
        if result.max() > np.iinfo(np.uint8).max:
            raise ValueError("Vote counts exceed uint8 storage capacity")
        return result

    alive_count = count(votes["alive_flag"] == 1)
    dead_count = count(votes["alive_flag"] == 0)
    home_count = count(votes["possession_flag"] == 1)
    away_count = count(votes["possession_flag"] == 2)
    alive_conflict = (alive_count > 0) & (dead_count > 0)
    possession_conflict = (home_count > 0) & (away_count > 0)

    unanimous_dead = (dead_count > 0) & (alive_count == 0)
    unanimous_alive = (alive_count > 0) & (dead_count == 0)
    valid_home = unanimous_alive & (home_count > 0) & (away_count == 0)
    valid_away = unanimous_alive & (away_count > 0) & (home_count == 0)
    label_valid = unanimous_dead | valid_home | valid_away
    labels = np.full(n_frames, UNKNOWN, dtype=np.int8)
    labels[valid_home] = H
    labels[valid_away] = A
    labels[unanimous_dead] = D

    frame_rows = np.flatnonzero(frame_start)
    half_id = votes["half_id"][frame_rows].astype(np.int8)
    timestamp_ms = votes["timestamp_ms"][frame_rows].astype(np.int64)
    sample_index = np.arange(n_frames, dtype=np.int64)
    feature_half, feature_time, feature_index = _feature_timeline(feature_archive_path)
    if not (
        np.array_equal(half_id, feature_half)
        and np.array_equal(timestamp_ms, feature_time)
        and np.array_equal(sample_index, feature_index)
    ):
        raise ValueError("H/A/D labels do not align exactly with the feature archive")

    result = {
        "labels": labels,
        "label_valid": label_valid,
        "alive_conflict": alive_conflict,
        "possession_conflict": possession_conflict,
        "alive_vote_count": alive_count.astype(np.uint8),
        "dead_vote_count": dead_count.astype(np.uint8),
        "home_possession_vote_count": home_count.astype(np.uint8),
        "away_possession_vote_count": away_count.astype(np.uint8),
        "timestamp_ms": timestamp_ms,
        "half_id": half_id,
        "sample_index": sample_index,
        "label_names": LABEL_NAMES,
        "aggregation_policy": AGGREGATION_POLICY,
        "schema_version": np.array("1.0"),
    }
    validate_had_labels(result, feature_archive_path)
    return result


def validate_had_labels(
    result: Mapping[str, np.ndarray], feature_archive_path: Path | str
) -> None:
    """Validate H/A/D targets, provenance, chronology and feature alignment."""
    vector_names = (
        "labels",
        "label_valid",
        "alive_conflict",
        "possession_conflict",
        "alive_vote_count",
        "dead_vote_count",
        "home_possession_vote_count",
        "away_possession_vote_count",
        "timestamp_ms",
        "half_id",
        "sample_index",
    )
    required = set(vector_names) | {"aggregation_policy"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"H/A/D label result is missing arrays: {sorted(missing)}")
    n_frames = len(result["labels"])
    for name in vector_names:
        if result[name].shape != (n_frames,):
            raise ValueError(f"{name} must have shape ({n_frames},)")
    labels = result["labels"]
    if labels.dtype != np.int8 or not np.isin(labels, (UNKNOWN, H, A, D)).all():
        raise ValueError("H/A/D labels must be int8 with values -1, 0, 1, or 2")
    for name in ("label_valid", "alive_conflict", "possession_conflict"):
        if result[name].dtype != np.bool_:
            raise ValueError(f"{name} must be Boolean")
    count_names = (
        "alive_vote_count",
        "dead_vote_count",
        "home_possession_vote_count",
        "away_possession_vote_count",
    )
    if any(result[name].dtype != np.uint8 for name in count_names):
        raise ValueError("H/A/D vote counts must use uint8")

    alive = result["alive_vote_count"].astype(np.int64)
    dead = result["dead_vote_count"].astype(np.int64)
    home = result["home_possession_vote_count"].astype(np.int64)
    away = result["away_possession_vote_count"].astype(np.int64)
    if np.any(alive + dead == 0) or not np.array_equal(alive + dead, home + away):
        raise ValueError("Flag vote totals are empty or inconsistent")
    alive_conflict = (alive > 0) & (dead > 0)
    possession_conflict = (home > 0) & (away > 0)
    if not np.array_equal(result["alive_conflict"], alive_conflict):
        raise ValueError("alive_conflict is inconsistent with vote counts")
    if not np.array_equal(result["possession_conflict"], possession_conflict):
        raise ValueError("possession_conflict is inconsistent with vote counts")

    unanimous_dead = (dead > 0) & (alive == 0)
    valid_home = (alive > 0) & (dead == 0) & (home > 0) & (away == 0)
    valid_away = (alive > 0) & (dead == 0) & (away > 0) & (home == 0)
    expected_valid = unanimous_dead | valid_home | valid_away
    expected_labels = np.full(n_frames, UNKNOWN, dtype=np.int8)
    expected_labels[valid_home] = H
    expected_labels[valid_away] = A
    expected_labels[unanimous_dead] = D
    if not np.array_equal(result["label_valid"], expected_valid):
        raise ValueError("label_valid is inconsistent with strict H/A/D rules")
    if not np.array_equal(labels, expected_labels):
        raise ValueError("labels are inconsistent with strict H/A/D rules")
    if str(result["aggregation_policy"]) != str(AGGREGATION_POLICY):
        raise ValueError("Unexpected H/A/D aggregation policy")

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
        raise ValueError("H/A/D labels do not align exactly with the feature archive")


def save_had_labels(
    result: Mapping[str, np.ndarray],
    output_path: Path | str,
    feature_archive_path: Path | str,
) -> None:
    validate_had_labels(result, feature_archive_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)


def _default_paths() -> Tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "open_data_groningen"
    processed = data_dir / "processed"
    return (
        data_dir / "positions_FIRST_HALF.csv",
        data_dir / "positions_SECOND_HALF.csv",
        processed / "wcsf_player_team_features.npz",
        processed / "wcsf_had_labels.npz",
    )


def main() -> None:
    first, second, features, output = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-half", type=Path, default=first)
    parser.add_argument("--second-half", type=Path, default=second)
    parser.add_argument("--features", type=Path, default=features)
    parser.add_argument("--output", type=Path, default=output)
    args = parser.parse_args()
    result = build_had_labels((args.first_half, args.second_half), args.features)
    save_had_labels(result, args.output, args.features)
    values, counts = np.unique(
        result["labels"][result["label_valid"]], return_counts=True
    )
    summary = {str(LABEL_NAMES[value]): int(count) for value, count in zip(values, counts)}
    print(f"Saved {len(result['labels']):,} labels to {args.output}")
    print(f"Valid label counts: {summary}; invalid: {np.count_nonzero(~result['label_valid'])}")


if __name__ == "__main__":
    main()
