"""Feature-only audits and reproducible synthetic tracking corruption.

This module deliberately has no dependency on labels, checkpoints, or competition
ground truth.  It operates on the self-describing 10 Hz feature archives used by
the WCSF models.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .wcsf_features import build_team_features, validate_features


SCENARIOS = (
    "iid_ball_dropout",
    "contiguous_ball_gaps",
    "player_observation_dropout",
    "player_speed_masking",
    "ball_speed_masking",
)
QUANTILES = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)


@dataclass(frozen=True)
class CorruptionSpec:
    scenario: str
    severity_factor: float
    seed: int

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown corruption scenario: {self.scenario}")
        if not np.isfinite(self.severity_factor) or self.severity_factor < 0.0:
            raise ValueError("severity_factor must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def missing_run_lengths(valid: np.ndarray, half_id: np.ndarray) -> dict[str, list[int]]:
    """Return invalid run lengths independently for every half."""
    if valid.dtype != np.bool_ or valid.shape != half_id.shape:
        raise ValueError("valid must be Boolean and aligned with half_id")
    result: dict[str, list[int]] = {}
    for half in np.unique(half_id):
        missing = ~valid[half_id == half]
        edges = np.diff(np.r_[False, missing, False].astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        result[str(int(half))] = (ends - starts).astype(int).tolist()
    return result


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "quantiles": {}}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "quantiles": {
            f"{quantile:g}": float(np.quantile(values, quantile))
            for quantile in QUANTILES
        },
    }


def _duplicate_player_coordinates(
    players: np.ndarray, location_valid: np.ndarray
) -> dict[str, int]:
    cases = 0
    excess = 0
    for frame in range(len(players)):
        for team in range(2):
            coordinates = players[frame, team, location_valid[frame, team], :2]
            if len(coordinates) < 2:
                continue
            _, counts = np.unique(coordinates, axis=0, return_counts=True)
            duplicated = counts[counts > 1]
            if len(duplicated):
                cases += 1
                excess += int((duplicated - 1).sum())
    return {"frame_team_cases": cases, "excess_player_slots": excess}


def audit_feature_arrays(features: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Audit one feature archive without consulting labels."""
    required = {
        "players", "player_location_valid", "player_speed_valid",
        "team_features", "team_feature_valid", "ball",
        "ball_location_valid", "ball_speed_valid", "half_id",
        "sample_index", "timestamp_ms", "source_fps",
    }
    missing = required.difference(features)
    if missing:
        raise ValueError(f"Feature archive is missing arrays: {sorted(missing)}")
    players = features["players"]
    location_valid = features["player_location_valid"]
    speed_valid = features["player_speed_valid"]
    ball = features["ball"]
    ball_location_valid = features["ball_location_valid"]
    ball_speed_valid = features["ball_speed_valid"]
    half_id = features["half_id"]
    n_frames = len(half_id)
    if players.shape != (n_frames, 2, 24, 3):
        raise ValueError("players must have shape (N, 2, 24, 3)")

    player_x = players[..., 0][location_valid]
    player_y = players[..., 1][location_valid]
    player_speed = players[..., 2][speed_valid]
    ball_x = ball[:, 0][ball_location_valid]
    ball_y = ball[:, 1][ball_location_valid]
    ball_speed = ball[:, 2][ball_speed_valid]
    ball_runs = missing_run_lengths(ball_location_valid, half_id)
    speed_runs = missing_run_lengths(ball_speed_valid, half_id)
    sample_index = features["sample_index"]
    timestamps = features["timestamp_ms"]

    invalid_zero = {
        "player_location_nonzero": int(np.count_nonzero(players[..., :2][~location_valid])),
        "player_speed_nonzero": int(np.count_nonzero(players[..., 2][~speed_valid])),
        "ball_location_nonzero": int(np.count_nonzero(ball[:, :2][~ball_location_valid])),
        "ball_speed_nonzero": int(np.count_nonzero(ball[:, 2][~ball_speed_valid])),
        "team_feature_nonzero": int(
            np.count_nonzero(features["team_features"][~features["team_feature_valid"]])
        ),
    }
    valid_nonfinite = {
        "player_location": int(np.count_nonzero(~np.isfinite(players[..., :2][location_valid]))),
        "player_speed": int(np.count_nonzero(~np.isfinite(player_speed))),
        "ball_location": int(np.count_nonzero(~np.isfinite(ball[:, :2][ball_location_valid]))),
        "ball_speed": int(np.count_nonzero(~np.isfinite(ball_speed))),
    }
    return {
        "n_frames": n_frames,
        "source_fps": float(features["source_fps"]),
        "half_counts": {
            str(int(half)): int(np.count_nonzero(half_id == half))
            for half in np.unique(half_id)
        },
        "validity": {
            "ball_location_fraction": float(ball_location_valid.mean()),
            "ball_speed_fraction": float(ball_speed_valid.mean()),
            "ball_speed_given_location": float(
                ball_speed_valid.sum() / max(1, ball_location_valid.sum())
            ),
            "player_location_slot_fraction": float(location_valid.mean()),
            "player_speed_slot_fraction": float(speed_valid.mean()),
            "player_speed_given_location": float(speed_valid.sum() / max(1, location_valid.sum())),
            "speed_valid_without_location": int(np.count_nonzero(speed_valid & ~location_valid)),
            "location_valid_without_speed": int(np.count_nonzero(location_valid & ~speed_valid)),
        },
        "missing_runs": {
            "ball_location_by_half": ball_runs,
            "ball_speed_by_half": speed_runs,
            "ball_location_all": _summary(
                np.asarray([value for values in ball_runs.values() for value in values])
            ),
            "ball_speed_all": _summary(
                np.asarray([value for values in speed_runs.values() for value in values])
            ),
        },
        "player_counts_per_team_frame": {
            "location": _summary(location_valid.sum(axis=2).ravel()),
            "speed": _summary(speed_valid.sum(axis=2).ravel()),
        },
        "distributions": {
            "player_x": _summary(player_x), "player_y": _summary(player_y),
            "player_speed": _summary(player_speed), "ball_x": _summary(ball_x),
            "ball_y": _summary(ball_y), "ball_speed": _summary(ball_speed),
            **{
                f"team_{str(name)}": _summary(
                    features["team_features"][..., index][
                        features["team_feature_valid"][..., index]
                    ]
                )
                for index, name in enumerate(features.get(
                    "team_feature_names",
                    np.asarray([f"feature_{i}" for i in range(6)]),
                ).astype(str))
            },
        },
        "bounds_and_speed": {
            "player_locations_outside_normalized_pitch": int(
                np.count_nonzero(
                    (player_x < 0) | (player_x > 1) | (player_y < 0) | (player_y > 1)
                )
            ),
            "ball_locations_outside_normalized_pitch": int(
                np.count_nonzero(
                    (ball_x < 0) | (ball_x > 1) | (ball_y < 0) | (ball_y > 1)
                )
            ),
            "negative_player_speeds": int(np.count_nonzero(player_speed < 0)),
            "player_speeds_over_12_mps": int(np.count_nonzero(player_speed > 12)),
            "negative_ball_speeds": int(np.count_nonzero(ball_speed < 0)),
            "ball_speeds_over_45_mps": int(np.count_nonzero(ball_speed > 45)),
        },
        "duplicates": {
            "sample_index": int(n_frames - len(np.unique(sample_index))),
            "timestamp_ms": int(n_frames - len(np.unique(timestamps))),
            "player_coordinates": _duplicate_player_coordinates(players, location_valid),
        },
        "chronology": {
            "sample_index_strictly_increasing": bool(np.all(np.diff(sample_index) > 0)),
            "timestamp_ms_strictly_increasing": bool(np.all(np.diff(timestamps) > 0)),
        },
        "invalid_zero_violations": invalid_zero,
        "valid_nonfinite_counts": valid_nonfinite,
        "definitions": {
            "normalized_pitch_bounds": "0 <= x,y <= 1; excursions are reported, not discarded",
            "player_extreme_speed_mps": 12.0,
            "ball_extreme_speed_mps": 45.0,
            "coordinate_duplicate": "exact equal valid (x,y) within one frame and team",
        },
    }


def _distribution_values(features: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    players = features["players"]
    location = features["player_location_valid"]
    speed = features["player_speed_valid"]
    ball = features["ball"]
    ball_location = features["ball_location_valid"]
    ball_speed = features["ball_speed_valid"]
    values = {
        "player_x": players[..., 0][location],
        "player_y": players[..., 1][location],
        "player_speed": players[..., 2][speed],
        "ball_x": ball[:, 0][ball_location],
        "ball_y": ball[:, 1][ball_location],
        "ball_speed": ball[:, 2][ball_speed],
        "located_players_per_team": location.sum(axis=2).ravel(),
        "speed_valid_players_per_team": speed.sum(axis=2).ravel(),
    }
    names = features.get(
        "team_feature_names", np.asarray([f"feature_{index}" for index in range(6)])
    ).astype(str)
    for index, name in enumerate(names):
        valid = features["team_feature_valid"][..., index]
        values[f"team_{name}"] = features["team_features"][..., index][valid]
    return values


def _distribution_shift(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if not len(left) or not len(right):
        return {"standardized_mean_difference": None, "jensen_shannon_divergence": None}
    pooled = np.sqrt((left.var() + right.var()) / 2.0)
    smd = 0.0 if pooled == 0.0 else (right.mean() - left.mean()) / pooled
    low = min(float(left.min()), float(right.min()))
    high = max(float(left.max()), float(right.max()))
    if low == high:
        js = 0.0
    else:
        first, edges = np.histogram(left, bins=100, range=(low, high))
        second, _ = np.histogram(right, bins=edges)
        p = first.astype(np.float64) / max(1, first.sum())
        q = second.astype(np.float64) / max(1, second.sum())
        midpoint = (p + q) / 2.0
        def kl(values: np.ndarray) -> float:
            valid = values > 0
            return float(np.sum(values[valid] * np.log(values[valid] / midpoint[valid])))
        js = (kl(p) + kl(q)) / 2.0
    return {
        "standardized_mean_difference": float(smd),
        "jensen_shannon_divergence": float(js),
    }


def compare_feature_archives(
    wcsf: Mapping[str, np.ndarray], metrica: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    left = _distribution_values(wcsf)
    right = _distribution_values(metrica)
    return {
        name: {
            "wcsf": _summary(left[name]),
            "metrica": _summary(right[name]),
            "shift": _distribution_shift(left[name], right[name]),
        }
        for name in left
    }


def derive_corruption_profile(
    wcsf: Mapping[str, np.ndarray], metrica: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Derive label-free, Metrica-matched missingness targets."""
    w_location = wcsf["player_location_valid"]
    m_location = metrica["player_location_valid"]
    m_speed = metrica["player_speed_valid"]
    profile = {
        "ball_location_missing_by_half": {},
        "ball_gap_lengths_by_half": missing_run_lengths(
            metrica["ball_location_valid"], metrica["half_id"]
        ),
        "player_observation_dropout_rate": float(
            np.clip(
                1.0
                - m_location.sum(axis=2).mean()
                / max(np.finfo(float).eps, w_location.sum(axis=2).mean()),
                0.0,
                1.0,
            )
        ),
        "player_speed_masking_rate": float(
            np.clip(1.0 - m_speed.sum() / max(1, m_location.sum()), 0.0, 1.0)
        ),
        "ball_speed_masking_rate": float(
            np.clip(
                1.0
                - metrica["ball_speed_valid"].sum()
                / max(1, metrica["ball_location_valid"].sum()),
                0.0, 1.0,
            )
        ),
    }
    for half in np.unique(metrica["half_id"]):
        in_half = metrica["half_id"] == half
        profile["ball_location_missing_by_half"][str(int(half))] = float(
            1.0 - metrica["ball_location_valid"][in_half].mean()
        )
    return profile


def _stable_rng(seed: int, scenario: str, severity: float, half: int = 0) -> np.random.Generator:
    token = f"{seed}|{scenario}|{severity:.12g}|{half}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _exact_random_mask(candidates: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    result = np.zeros(candidates.shape, dtype=bool)
    available = np.flatnonzero(candidates)
    count = min(max(int(count), 0), len(available))
    if count:
        result.flat[rng.choice(available, size=count, replace=False)] = True
    return result


def _gap_mask(length: int, target: int, gap_lengths: list[int], rng: np.random.Generator) -> np.ndarray:
    """Place non-overlapping empirical gaps until exactly target frames are masked."""
    target = min(max(int(target), 0), length)
    result = np.zeros(length, dtype=bool)
    positive = np.asarray([value for value in gap_lengths if value > 0], dtype=int)
    if target and not len(positive):
        raise ValueError("Contiguous gap corruption requires positive reference gaps")
    attempts = 0
    while int(result.sum()) < target:
        remaining = target - int(result.sum())
        run_length = min(int(rng.choice(positive)), remaining, length)
        edges = np.diff(np.r_[False, ~result, False].astype(np.int8))
        interval_starts = np.flatnonzero(edges == 1)
        interval_ends = np.flatnonzero(edges == -1)
        capacities = np.maximum(0, interval_ends - interval_starts - run_length + 1)
        if not capacities.sum():
            # Fragmentation can make the sampled run impossible; finish in the
            # largest available interval while retaining contiguity.
            available_lengths = interval_ends - interval_starts
            run_length = min(remaining, int(available_lengths.max(initial=0)))
            if run_length <= 0:
                break
            capacities = np.maximum(0, available_lengths - run_length + 1)
        draw = int(rng.integers(int(capacities.sum())))
        cumulative = np.cumsum(capacities)
        interval = int(np.searchsorted(cumulative, draw, side="right"))
        before = int(cumulative[interval - 1]) if interval else 0
        start = int(interval_starts[interval] + draw - before)
        result[start : start + run_length] = True
        attempts += 1
        if attempts > length + target:
            raise RuntimeError("Unable to construct contiguous gap mask")
    if int(result.sum()) != target:
        raise RuntimeError("Contiguous gap mask did not meet its exact budget")
    return result


def recompute_ball_speed(
    ball: np.ndarray,
    ball_location_valid: np.ndarray,
    half_id: np.ndarray,
    timestamp_ms: np.ndarray,
) -> np.ndarray:
    """Recompute backward speed using only consecutive surviving observations."""
    ball[:, 2] = 0.0
    speed_valid = np.zeros(len(ball), dtype=bool)
    consecutive = (
        (half_id[1:] == half_id[:-1])
        & ball_location_valid[1:]
        & ball_location_valid[:-1]
        & (timestamp_ms[1:] > timestamp_ms[:-1])
    )
    current = np.flatnonzero(consecutive) + 1
    if len(current):
        dt = (timestamp_ms[current] - timestamp_ms[current - 1]).astype(np.float64) / 1000.0
        delta = ball[current, :2] - ball[current - 1, :2]
        pitch_delta = delta * np.asarray([105.0, 68.0])
        ball[current, 2] = (np.linalg.norm(pitch_delta, axis=1) / dt).astype(np.float32)
        speed_valid[current] = True
    return speed_valid


def corrupt_feature_arrays(
    source: Mapping[str, np.ndarray],
    spec: CorruptionSpec,
    profile: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """Return an independently corrupted copy, realization metadata, and masks."""
    result = {key: np.asarray(value).copy() for key, value in source.items()}
    half_id = result["half_id"]
    masks: dict[str, np.ndarray] = {}
    target_rate = 0.0

    if spec.scenario in ("iid_ball_dropout", "contiguous_ball_gaps"):
        dropped = np.zeros(len(half_id), dtype=bool)
        for half in np.unique(half_id):
            indices = np.flatnonzero(half_id == half)
            base = float(profile["ball_location_missing_by_half"][str(int(half))])
            rate = min(1.0, base * spec.severity_factor)
            count = int(round(rate * len(indices)))
            rng = _stable_rng(spec.seed, spec.scenario, spec.severity_factor, int(half))
            if spec.scenario == "iid_ball_dropout":
                local = _exact_random_mask(np.ones(len(indices), dtype=bool), count, rng)
            else:
                gaps = profile["ball_gap_lengths_by_half"][str(int(half))]
                local = _gap_mask(len(indices), count, gaps, rng)
            dropped[indices[local]] = True
        result["ball_location_valid"][dropped] = False
        result["ball"][dropped, :2] = 0.0
        result["ball_speed_valid"] = recompute_ball_speed(
            result["ball"], result["ball_location_valid"], half_id, result["timestamp_ms"]
        )
        masks["ball_location_dropped"] = dropped
        target_rate = float(dropped.mean())

    elif spec.scenario == "player_observation_dropout":
        target_rate = min(
            1.0, float(profile["player_observation_dropout_rate"]) * spec.severity_factor
        )
        rng = _stable_rng(spec.seed, spec.scenario, spec.severity_factor)
        candidates = result["player_location_valid"].copy()
        dropped = _exact_random_mask(candidates, round(target_rate * candidates.sum()), rng)
        result["player_location_valid"][dropped] = False
        result["player_speed_valid"][dropped] = False
        result["players"][..., :2][dropped] = 0.0
        result["players"][..., 2][dropped] = 0.0
        masks["player_observation_dropped"] = dropped

    elif spec.scenario == "player_speed_masking":
        target_rate = min(
            1.0, float(profile["player_speed_masking_rate"]) * spec.severity_factor
        )
        rng = _stable_rng(spec.seed, spec.scenario, spec.severity_factor)
        candidates = result["player_speed_valid"].copy()
        dropped = _exact_random_mask(candidates, round(target_rate * candidates.sum()), rng)
        result["player_speed_valid"][dropped] = False
        result["players"][..., 2][dropped] = 0.0
        masks["player_speed_masked"] = dropped

    elif spec.scenario == "ball_speed_masking":
        target_rate = min(
            1.0, float(profile["ball_speed_masking_rate"]) * spec.severity_factor
        )
        rng = _stable_rng(spec.seed, spec.scenario, spec.severity_factor)
        candidates = result["ball_speed_valid"].copy()
        dropped = _exact_random_mask(candidates, round(target_rate * candidates.sum()), rng)
        result["ball_speed_valid"][dropped] = False
        result["ball"][..., 2][dropped] = 0.0
        masks["ball_speed_masked"] = dropped

    team_features, team_valid = build_team_features(
        result["players"], result["player_location_valid"], result["player_speed_valid"]
    )
    result["team_features"] = team_features
    result["team_feature_valid"] = team_valid
    validate_features(result)
    mask_counts = {name: int(mask.sum()) for name, mask in masks.items()}
    realization = {
        "scenario": spec.scenario,
        "severity_factor": spec.severity_factor,
        "seed": spec.seed,
        "target_rate": target_rate,
        "mask_counts": mask_counts,
        "ball_location_valid_fraction": float(result["ball_location_valid"].mean()),
        "ball_speed_valid_fraction": float(result["ball_speed_valid"].mean()),
        "player_location_valid_fraction": float(result["player_location_valid"].mean()),
        "player_speed_valid_fraction": float(result["player_speed_valid"].mean()),
    }
    return result, realization, masks
