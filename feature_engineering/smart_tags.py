"""Parse, audit and featurize Metrica Smart Tag intervals without using labels.

The Smart Tag XML is aligned to the original tactical-video clock.  This module
deliberately knows nothing about possession ground truth or model predictions so
it can be used safely by label-free preparation pipelines.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


HOME_TEAM = "Barcelona"
AWAY_TEAM = "Atletic Club"
NO_TEAM = "N/A"
VALID_TEAMS = {HOME_TEAM, AWAY_TEAM, NO_TEAM}

ATTACKING_OWNERSHIP_CODES = {
    "BUILD UP",
    "PROGRESSION",
    "ATTACKING TRANSITION",
}
DEFENDING_OWNERSHIP_CODES = {
    "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENSIVE TRANSITION",
}
KICKOFF_TYPES = {"KICK OFF (START)", "KICK OFF (AFTER GOAL)"}


@dataclass(frozen=True)
class SmartTagEvent:
    index: int
    code: str
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    tags: dict[str, str]
    duplicate_tag_groups: tuple[str, ...] = ()


def _clean(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def load_smart_tags(
    path: str | Path, *, fps: float = 25.0, n_frames: int | None = None
) -> tuple[SmartTagEvent, ...]:
    """Load and strictly validate half-open Smart Tag intervals."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Smart Tag XML not found: {path}")
    events: list[SmartTagEvent] = []
    for index, instance in enumerate(ET.parse(path).iter("instance")):
        code = _clean(instance.findtext("code")).upper()
        try:
            start_s = float(instance.findtext("start"))
            end_s = float(instance.findtext("end"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Smart Tag event {index} has invalid timestamps") from error
        if not code or not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
            raise ValueError(f"Smart Tag event {index} has an invalid code or interval")
        start_frame = int(round(start_s * fps))
        end_frame = int(round(end_s * fps))
        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError(f"Smart Tag event {index} has an invalid frame interval")
        if n_frames is not None and end_frame > n_frames:
            raise ValueError(f"Smart Tag event {index} extends beyond the source timeline")

        grouped: dict[str, list[str]] = {}
        for label in instance.findall("label"):
            group = _clean(label.findtext("group"))
            value = _clean(label.findtext("text"))
            if not group or not value:
                raise ValueError(f"Smart Tag event {index} has an empty tag group or value")
            grouped.setdefault(group, []).append(value)
        duplicate_groups = tuple(sorted(group for group, values in grouped.items() if len(values) > 1))
        conflicting = [group for group, values in grouped.items() if len(set(values)) > 1]
        if conflicting:
            raise ValueError(f"Smart Tag event {index} has conflicting repeated groups: {conflicting}")
        tags = {group: values[-1] for group, values in grouped.items()}
        team = tags.get("Team")
        if team is not None and team not in VALID_TEAMS:
            raise ValueError(f"Smart Tag event {index} has unknown team {team!r}")
        events.append(
            SmartTagEvent(
                index=index,
                code=code,
                start_s=start_s,
                end_s=end_s,
                start_frame=start_frame,
                end_frame=end_frame,
                tags=tags,
                duplicate_tag_groups=duplicate_groups,
            )
        )
    if not events:
        raise ValueError("Smart Tag XML contains no events")
    return tuple(events)


def _coverage(events: tuple[SmartTagEvent, ...], n_frames: int) -> np.ndarray:
    difference = np.zeros(n_frames + 1, dtype=np.int32)
    for event in events:
        difference[event.start_frame] += 1
        difference[event.end_frame] -= 1
    return np.cumsum(difference[:-1])


def audit_smart_tags(
    events: tuple[SmartTagEvent, ...], *, n_frames: int, fps: float = 25.0
) -> dict[str, object]:
    """Return deterministic, JSON-safe coverage and conflict diagnostics."""
    coverage = _coverage(events, n_frames)
    durations = np.array([event.end_s - event.start_s for event in events], dtype=np.float64)
    team_counts = Counter(event.tags.get("Team", "MISSING") for event in events)
    code_counts = Counter(event.code for event in events)
    half_counts = Counter(event.tags.get("Half", "MISSING") for event in events)
    tag_values: dict[str, Counter[str]] = {}
    for event in events:
        for group, value in event.tags.items():
            tag_values.setdefault(group, Counter())[value] += 1

    overlapping_pairs = 0
    opposing_team_pairs = 0
    for left_index, left in enumerate(events):
        for right in events[left_index + 1 :]:
            if max(left.start_frame, right.start_frame) >= min(left.end_frame, right.end_frame):
                continue
            overlapping_pairs += 1
            left_team = left.tags.get("Team")
            right_team = right.tags.get("Team")
            if {left_team, right_team} == {HOME_TEAM, AWAY_TEAM}:
                opposing_team_pairs += 1

    home_events = tuple(event for event in events if event.tags.get("Team") == HOME_TEAM)
    away_events = tuple(event for event in events if event.tags.get("Team") == AWAY_TEAM)
    home_active = _coverage(home_events, n_frames) > 0
    away_active = _coverage(away_events, n_frames) > 0
    quantiles = np.quantile(durations, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "n_events": len(events),
        "fps": float(fps),
        "n_frames": int(n_frames),
        "code_counts": dict(sorted(code_counts.items())),
        "team_counts": dict(sorted(team_counts.items())),
        "half_counts": dict(sorted(half_counts.items())),
        "tag_values": {
            group: dict(sorted(values.items())) for group, values in sorted(tag_values.items())
        },
        "duration_seconds": {
            "mean": float(durations.mean()),
            "min": float(quantiles[0]),
            "p25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "p75": float(quantiles[3]),
            "p90": float(quantiles[4]),
            "max": float(quantiles[5]),
        },
        "coverage": {
            "frames_any": int(np.count_nonzero(coverage)),
            "fraction_any": float(np.mean(coverage > 0)),
            "frames_two_or_more": int(np.count_nonzero(coverage >= 2)),
            "max_simultaneous": int(coverage.max(initial=0)),
            "frames_both_named_teams": int(np.count_nonzero(home_active & away_active)),
        },
        "overlaps": {
            "event_pairs": overlapping_pairs,
            "opposing_named_team_pairs": opposing_team_pairs,
        },
        "duplicate_tag_groups": {
            str(event.index): list(event.duplicate_tag_groups)
            for event in events
            if event.duplicate_tag_groups
        },
    }


def _opposite(team: str) -> str:
    if team == HOME_TEAM:
        return AWAY_TEAM
    if team == AWAY_TEAM:
        return HOME_TEAM
    raise ValueError(f"Cannot invert non-team Smart Tag value {team!r}")


def _interval_taper(event: SmartTagEvent, fps: float, taper_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    frames = np.arange(event.start_frame, event.end_frame, dtype=np.int64)
    centers = frames.astype(np.float64) + 0.5
    since_start = (centers - event.start_frame) / fps
    until_end = (event.end_frame - centers) / fps
    weights = np.minimum(1.0, np.minimum(since_start, until_end) / taper_seconds)
    return frames, np.clip(weights, 0.0, 1.0).astype(np.float32)


def build_smart_tag_evidence(
    events: tuple[SmartTagEvent, ...],
    *,
    n_frames: int,
    fps: float = 25.0,
    ownership_taper_seconds: float = 2.0,
    kickoff_live_seconds: float = 2.0,
) -> dict[str, np.ndarray]:
    """Build conservative H/A ownership and kickoff dead/live evidence at 25 Hz."""
    if ownership_taper_seconds <= 0 or kickoff_live_seconds <= 0:
        raise ValueError("Evidence windows must be positive")
    home = np.zeros(n_frames, dtype=np.float32)
    away = np.zeros(n_frames, dtype=np.float32)
    kickoff_dead = np.zeros(n_frames, dtype=np.float32)
    kickoff_live = np.zeros(n_frames, dtype=np.float32)
    kickoff_boundary = np.zeros(n_frames, dtype=bool)

    for event in events:
        team = event.tags.get("Team")
        owner: str | None = None
        if event.code in ATTACKING_OWNERSHIP_CODES and team in {HOME_TEAM, AWAY_TEAM}:
            owner = team
        elif event.code in DEFENDING_OWNERSHIP_CODES and team in {HOME_TEAM, AWAY_TEAM}:
            owner = _opposite(team)
        if owner is not None:
            frames, weights = _interval_taper(event, fps, ownership_taper_seconds)
            target = home if owner == HOME_TEAM else away
            target[frames] = np.maximum(target[frames], weights)

        event_type = _clean(event.tags.get("Type")).upper()
        if event.code == "SET PIECES" and event_type in KICKOFF_TYPES:
            kickoff_dead[event.start_frame : event.end_frame] = 1.0
            stop = min(n_frames, event.end_frame + int(round(kickoff_live_seconds * fps)))
            frames = np.arange(event.end_frame, stop, dtype=np.int64)
            elapsed = (frames.astype(np.float64) + 0.5 - event.end_frame) / fps
            weights = np.clip(1.0 - elapsed / kickoff_live_seconds, 0.0, 1.0).astype(np.float32)
            kickoff_live[frames] = np.maximum(kickoff_live[frames], weights)
            kickoff_boundary[event.start_frame:stop] = True

    return {
        "home_ownership": home,
        "away_ownership": away,
        "kickoff_dead": kickoff_dead,
        "kickoff_live": kickoff_live,
        "kickoff_boundary": kickoff_boundary,
    }

