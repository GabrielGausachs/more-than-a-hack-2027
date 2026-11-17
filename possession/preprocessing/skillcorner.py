"""Minimal readers for the SkillCorner inputs used by the selected pipeline."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .common_features import (
    LABEL_POLICY, QUALITY_DETECTED, QUALITY_EXTRAPOLATED, TEAM_SIDES,
    allocate_arrays, compute_common_features, metadata_arrays, orient_xy,
    store_team, validate_archive,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "data" / "external" / "skillcorner-opendata" / "data"
DEFAULT_LABEL_ROOT = ROOT / "data" / "derived" / "skillcorner_labels" / "matches"
REQUIRED_SUFFIXES = {
    "match": "_match.json",
    "tracking": "_tracking_extrapolated.jsonl",
    "dynamic_events": "_dynamic_events.csv",
    "phases_of_play": "_phases_of_play.csv",
}


@dataclass(frozen=True)
class MatchPaths:
    """Resolved files for one local SkillCorner match."""

    match_id: int
    data_root: Path
    match_dir: Path
    match: Path
    tracking: Path
    dynamic_events: Path
    phases_of_play: Path


@dataclass(frozen=True)
class SkillCornerMatch:
    """Match JSON plus the identity maps needed for tracking joins."""

    raw: Mapping[str, object]
    match_id: int
    pitch_length: float
    pitch_width: float
    periods: Tuple[Mapping[str, object], ...]
    player_to_side: Mapping[int, str]
    team_to_side: Mapping[int, str]


def _root(data_root: Optional[Union[os.PathLike, str]]) -> Path:
    return Path(data_root).expanduser().resolve() if data_root else DEFAULT_DATA_ROOT.resolve()


def discover_matches(data_root: Optional[Union[os.PathLike, str]] = None) -> Dict[str, object]:
    """Return catalogue entries and the IDs whose four files exist locally.

    Catalogue availability and local availability are deliberately separate:
    sparse SkillCorner checkouts can contain ``matches.json`` entries without
    materialising the corresponding large match files.
    """

    root = _root(data_root)
    catalogue_path = root / "matches.json"
    if not catalogue_path.is_file():
        raise IOError("SkillCorner catalogue not found: {}".format(catalogue_path))
    with catalogue_path.open(encoding="utf-8") as handle:
        catalogue = json.load(handle)
    if not isinstance(catalogue, list):
        raise ValueError("{}: expected a JSON list".format(catalogue_path))

    catalogue_ids = [int(item["id"]) for item in catalogue]
    local_ids = []
    matches_dir = root / "matches"
    if matches_dir.is_dir():
        for match_id in catalogue_ids:
            folder = matches_dir / str(match_id)
            if all((folder / (str(match_id) + suffix)).is_file()
                   for suffix in REQUIRED_SUFFIXES.values()):
                local_ids.append(match_id)
    return {
        "data_root": root,
        "catalogue_path": catalogue_path,
        "catalogue": catalogue,
        "catalogue_ids": catalogue_ids,
        "local_ids": local_ids,
    }


def resolve_match_paths(match_id: int,
                        data_root: Optional[Union[os.PathLike, str]] = None) -> MatchPaths:
    """Resolve and validate the four files required for ``match_id``."""

    root = _root(data_root)
    match_id = int(match_id)
    folder = root / "matches" / str(match_id)
    paths = {name: folder / (str(match_id) + suffix)
             for name, suffix in REQUIRED_SUFFIXES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise IOError(
            "SkillCorner match {} is not fully available under {}. Missing:\n  {}"
            .format(match_id, root, "\n  ".join(missing)))
    return MatchPaths(match_id, root, folder, paths["match"], paths["tracking"],
                      paths["dynamic_events"], paths["phases_of_play"])


def load_match_metadata(match: Union[MatchPaths, os.PathLike, str]) -> SkillCornerMatch:
    """Load metadata and construct stable player/team/H-A lookup tables."""

    path = match.match if isinstance(match, MatchPaths) else Path(match)
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    home_id = int(raw["home_team"]["id"])
    away_id = int(raw["away_team"]["id"])
    team_to_side = {home_id: "H", away_id: "A"}
    player_to_side = {
        int(player["id"]): team_to_side.get(int(player["team_id"]))
        for player in raw.get("players", ())
    }
    return SkillCornerMatch(
        raw=raw,
        match_id=int(raw["id"]),
        pitch_length=float(raw["pitch_length"]),
        pitch_width=float(raw["pitch_width"]),
        periods=tuple(raw.get("match_periods", ())),
        player_to_side=player_to_side,
        team_to_side=team_to_side,
    )


def iter_tracking(tracking: Union[MatchPaths, os.PathLike, str]) -> Iterator[Dict[str, object]]:
    """Yield one decoded tracking record at a time, with line-aware errors."""

    path = tracking.tracking if isinstance(tracking, MatchPaths) else Path(tracking)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("{}:{}: malformed JSON: {}".format(path, line_number, error))
            if not isinstance(record, dict) or "frame" not in record:
                raise ValueError("{}:{}: expected a frame dictionary".format(path, line_number))
            yield record


def _optional_int(value: str) -> Optional[int]:
    return None if value == "" or value is None else int(value)


def _load_selected_csv(path: Path, fields: Sequence[str],
                       integer_fields: Sequence[str],
                       event_type: Optional[str] = None) -> List[Dict[str, object]]:
    """Read only the columns consumed by label generation."""

    integer_fields = set(integer_fields)
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(fields)
        if event_type is not None:
            required.add("event_type")
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("{}: missing columns {}".format(path, sorted(missing)))
        for row_number, source in enumerate(reader, 2):
            if event_type is not None and source["event_type"] != event_type:
                continue
            try:
                row = {
                    field: (_optional_int(source[field]) if field in integer_fields
                            else source[field] or None)
                    for field in fields
                }
            except (TypeError, ValueError) as error:
                raise ValueError("{}:{}: {}".format(path, row_number, error))
            rows.append(row)
    return rows


def load_phases(phases: Union[MatchPaths, os.PathLike, str]) -> List[Dict[str, object]]:
    """Load the five phase columns used by the frozen label policy."""

    path = phases.phases_of_play if isinstance(phases, MatchPaths) else Path(phases)
    fields = ("index", "frame_start", "frame_end", "period", "team_in_possession_id")
    return _load_selected_csv(path, fields, fields)


def load_player_possessions(
        dynamic_events: Union[MatchPaths, os.PathLike, str]) -> List[Dict[str, object]]:
    """Load only player-possession rows and the fields used by label generation."""

    path = dynamic_events.dynamic_events if isinstance(dynamic_events, MatchPaths) else Path(dynamic_events)
    fields = (
        "frame_start", "frame_end", "period", "team_id", "player_id",
        "game_interruption_before", "game_interruption_after",
    )
    integer_fields = ("frame_start", "frame_end", "period", "team_id", "player_id")
    return _load_selected_csv(
        path, fields, integer_fields, event_type="player_possession")


def side_for_team(team_id: Optional[int], metadata: SkillCornerMatch) -> Optional[str]:
    """Map a SkillCorner team ID to ``H``/``A`` without guessing."""

    return None if team_id is None else metadata.team_to_side.get(int(team_id))


def _load_labels(match_id: int, label_root: Path) -> np.ndarray:
    path = Path(label_root) / str(match_id) / "labels.csv"
    if not path.is_file():
        raise IOError("frozen SkillCorner labels not found: {}".format(path))
    frames, labels = [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frames.append(int(row["frame"]))
            labels.append(row["label"])
    if frames != list(range(len(frames))):
        raise ValueError("{}: label frames are not contiguous from zero".format(path))
    if not set(labels).issubset({"H", "A", "D", "U"}):
        raise ValueError("{}: invalid target labels {}".format(path, sorted(set(labels))))
    return np.asarray(labels, dtype="U1")


def _official_periods(metadata: SkillCornerMatch, n_frames: int) -> np.ndarray:
    """Create the official P1/P2 frame channel stored in the feature archive."""

    periods = np.zeros(n_frames, dtype=np.int8)
    for item in metadata.periods:
        period = int(item["period"])
        if period in (1, 2):
            start = int(item["start_frame"])
            stop = min(int(item["end_frame"]) + 1, n_frames)
            periods[start:stop] = period
    return periods


def extract_skillcorner_match(match_id: int, data_root=None,
                              label_root: Path = DEFAULT_LABEL_ROOT) -> Dict[str, np.ndarray]:
    """Convert one SkillCorner match to the shared feature schema."""

    paths = resolve_match_paths(match_id, data_root or DEFAULT_DATA_ROOT)
    metadata = load_match_metadata(paths)
    labels = _load_labels(int(match_id), Path(label_root))
    arrays = allocate_arrays(len(labels))
    arrays["frame_index"][:] = np.arange(len(labels), dtype=np.int32)
    arrays["timestamp_s"][:] = np.arange(len(labels), dtype=np.float32) / 10.0
    arrays["period"][:] = _official_periods(metadata, len(labels))
    speed_lag = 5
    histories: Dict[int, deque] = {}
    seen = 0
    directions = tuple(metadata.raw.get("home_team_side", ()))

    for record in iter_tracking(paths):
        frame = int(record["frame"])
        if frame != seen or frame >= len(labels):
            raise ValueError("match {} tracking frame {}, expected {}".format(match_id, frame, seen))
        seen += 1
        period = int(arrays["period"][frame])
        if period == 0 or len(directions) < period:
            continue
        direction = directions[period - 1]
        arrays["orientation_valid"][frame] = True
        teams = [[], []]
        for player in record.get("player_data") or ():
            player_id = player.get("player_id")
            x, y = player.get("x"), player.get("y")
            if player_id is None or x is None or y is None:
                continue
            player_id = int(player_id)
            side = metadata.player_to_side.get(player_id)
            if side not in TEAM_SIDES:
                raise ValueError("match {} frame {}: unknown player {}".format(
                    match_id, frame, player_id))
            ox, oy = orient_xy(float(x), float(y), direction)
            history = histories.setdefault(player_id, deque(maxlen=speed_lag + 1))
            if history and (history[-1][0] != frame - 1 or history[-1][1] != period):
                history.clear()
            history.append((frame, period, ox, oy))
            speed_valid = (len(history) == speed_lag + 1 and
                           history[0][0] == frame - speed_lag)
            speed = (math.hypot(ox - history[0][2], oy - history[0][3]) / 0.5
                     if speed_valid else 0.0)
            quality = (QUALITY_DETECTED if player.get("is_detected") is True
                       else QUALITY_EXTRAPOLATED)
            teams[TEAM_SIDES.index(side)].append(
                (ox, oy, speed, speed_valid, quality))
        for team in range(2):
            store_team(arrays, frame, team, teams[team])

        ball = record.get("ball_data") or {}
        bx, by = ball.get("x"), ball.get("y")
        if bx is not None and by is not None:
            arrays["ball_xy_m"][frame] = orient_xy(float(bx), float(by), direction)
            arrays["ball_position_valid"][frame] = True
            arrays["audit_ball_quality"][frame] = (
                QUALITY_DETECTED if ball.get("is_detected") is True else QUALITY_EXTRAPOLATED)
    if seen != len(labels):
        raise ValueError("match {} has {} tracking frames but {} labels".format(
            match_id, seen, len(labels)))

    compute_common_features(arrays, 10.0, team_lag_frames=10, ball_lag_frames=5)
    arrays["target_label"] = labels
    arrays.update(metadata_arrays(
        "skillcorner", str(match_id), 10.0, metadata.pitch_length,
        metadata.pitch_width, 0.5, 0.5, LABEL_POLICY))
    validate_archive(arrays, require_target=True)
    return arrays



