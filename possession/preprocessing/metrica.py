"""Dependency-light reader for Metrica's tactical FIFA/EPTS tracking feed."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from games import GAME, half_video_blocks, require, source_frames

from .common_features import (
    QUALITY_PROVIDER_UNKNOWN, TEAM_SIDES, allocate_arrays, compute_common_features,
    metadata_arrays, orient_xy, store_team, validate_archive,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICA_DATA_DIR = ROOT / "data" / "metrica"


@dataclass(frozen=True)
class MetricaTrackingMetadata:
    game: str
    source: str
    match_id: str
    frame_rate: float
    n_frames: int
    pitch_length: float
    pitch_width: float
    home_team_id: str
    away_team_id: str
    slot_sides: Tuple[str, ...]
    home_attack_first_half: str
    period_blocks: Tuple[Tuple[int, int, int], ...]
    raw_path: Path
    xml_path: Path


@dataclass(frozen=True)
class MetricaFrame:
    frame_index: int
    period: int
    players: Tuple[Tuple[str, float, float, float], ...]
    ball: Tuple[float, float, float]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first(root: ET.Element, name: str) -> ET.Element:
    for element in root.iter():
        if _local(element.tag) == name:
            return element
    raise ValueError("EPTS XML is missing <{}>".format(name))


def _provider_parameter(element: ET.Element, name: str) -> Optional[str]:
    for parameter in element.iter():
        if _local(parameter.tag) != "ProviderParameter":
            continue
        values = {_local(child.tag): child.text for child in parameter}
        if values.get("Name") == name:
            return values.get("Value")
    return None


def load_metadata(game: str = GAME, data_dir=None,
                  source: str = "tactical") -> MetricaTrackingMetadata:
    """Read pitch, team/slot mapping and scored-half alignment metadata."""

    effective_data_dir = data_dir
    if effective_data_dir is None and DEFAULT_METRICA_DATA_DIR.is_dir():
        effective_data_dir = str(DEFAULT_METRICA_DATA_DIR)
    xml_name, raw_name = require(
        ["atd_meta", "atd_raw"], game=game, data_dir=effective_data_dir, source=source)
    xml_path, raw_path = Path(xml_name), Path(raw_name)
    root = ET.parse(xml_path).getroot()
    global_config = _first(root, "GlobalConfig")
    frame_rate = float(_first(global_config, "FrameRate").text)
    field = _first(root, "FieldSize")
    dims = {_local(child.tag): float(child.text) for child in field}
    score = _first(root, "Score")
    home_id = score.attrib["idLocalTeam"]
    away_id = score.attrib["idVisitingTeam"]

    players_parent = _first(root, "Players")
    slot_sides = []
    for player in players_parent:
        if _local(player.tag) != "Player":
            continue
        team_id = player.attrib.get("teamId")
        if team_id == home_id:
            slot_sides.append("H")
        elif team_id == away_id:
            slot_sides.append("A")
        else:
            raise ValueError("EPTS player slot references unknown team {!r}".format(team_id))

    teams = _first(root, "Teams")
    home_team = next((team for team in teams if team.attrib.get("id") == home_id), None)
    if home_team is None:
        raise ValueError("EPTS XML has no home-team definition")
    direction = _provider_parameter(home_team, "attack_direction_first_half")
    if direction not in ("left_to_right", "right_to_left"):
        raise ValueError("unsupported home attack direction {!r}".format(direction))
    blocks = tuple(half_video_blocks(game)) if source == "tactical" else tuple()
    if source == "tactical" and not blocks:
        raise ValueError("no scored-half blocks configured for {}".format(game))
    return MetricaTrackingMetadata(
        game=game, source=source,
        match_id="{}_{}".format(game, source),
        frame_rate=frame_rate, n_frames=source_frames(source, game),
        pitch_length=dims["Width"], pitch_width=dims["Height"],
        home_team_id=home_id, away_team_id=away_id,
        slot_sides=tuple(slot_sides), home_attack_first_half=direction,
        period_blocks=blocks, raw_path=raw_path, xml_path=xml_path,
    )


def period_for_frame(metadata: MetricaTrackingMetadata, frame_index: int) -> int:
    for period, start, end in metadata.period_blocks:
        if start <= frame_index <= end:
            return int(period)
    return 0


def home_attack_direction(metadata: MetricaTrackingMetadata, period: int) -> Optional[str]:
    if period not in (1, 2):
        return None
    if period == 1:
        return metadata.home_attack_first_half
    return ("right_to_left" if metadata.home_attack_first_half == "left_to_right"
            else "left_to_right")


def _triple(text: str) -> Tuple[float, float, float]:
    values = text.split(",")
    if len(values) != 3:
        raise ValueError("expected x,y,speed triple; got {!r}".format(text))
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def iter_tracking(metadata: MetricaTrackingMetadata) -> Iterator[MetricaFrame]:
    """Stream raw frames, preserving native normalized coordinates and speed."""

    expected_id = 1
    with metadata.raw_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                raise ValueError("{}:{}: malformed EPTS row".format(metadata.raw_path, line_number))
            frame_id = int(parts[0])
            if frame_id != expected_id:
                raise ValueError("{}:{}: expected frame {}, got {}".format(
                    metadata.raw_path, line_number, expected_id, frame_id))
            player_text = parts[1].split(";") if parts[1] else []
            if len(player_text) != len(metadata.slot_sides):
                raise ValueError("{}:{}: {} player slots, expected {}".format(
                    metadata.raw_path, line_number, len(player_text), len(metadata.slot_sides)))
            players = tuple((side,) + _triple(value)
                            for side, value in zip(metadata.slot_sides, player_text))
            yield MetricaFrame(
                frame_index=frame_id - 1,
                period=period_for_frame(metadata, frame_id - 1),
                players=players,
                ball=_triple(parts[2]),
            )
            expected_id += 1
    if expected_id - 1 != metadata.n_frames:
        raise ValueError("{}: {} frames, expected {}".format(
            metadata.raw_path, expected_id - 1, metadata.n_frames))


def extract_metrica_tactical(game: str = GAME, data_dir=None):
    """Convert Metrica tactical tracking to the shared feature schema."""

    metadata = load_metadata(game=game, data_dir=data_dir, source="tactical")
    arrays = allocate_arrays(metadata.n_frames)
    arrays["frame_index"][:] = np.arange(metadata.n_frames, dtype=np.int32)
    arrays["timestamp_s"][:] = (
        np.arange(metadata.n_frames, dtype=np.float32) / metadata.frame_rate)
    seen = 0
    for record in iter_tracking(metadata):
        frame, period = record.frame_index, record.period
        if frame != seen:
            raise ValueError("Metrica frame {}, expected {}".format(frame, seen))
        seen += 1
        arrays["period"][frame] = period
        if period == 0:
            continue
        direction = home_attack_direction(metadata, period)
        arrays["orientation_valid"][frame] = True
        teams = [[], []]
        for side, x_norm, y_norm, native_speed in record.players:
            if not (math.isfinite(x_norm) and math.isfinite(y_norm)):
                continue
            x = (x_norm - 0.5) * metadata.pitch_length
            y = (y_norm - 0.5) * metadata.pitch_width
            ox, oy = orient_xy(x, y, direction)
            speed_valid = math.isfinite(native_speed)
            teams[TEAM_SIDES.index(side)].append(
                (ox, oy, native_speed if speed_valid else 0.0, speed_valid,
                 QUALITY_PROVIDER_UNKNOWN))
        for team in range(2):
            store_team(arrays, frame, team, teams[team])
        bx, by, _ = record.ball
        if math.isfinite(bx) and math.isfinite(by):
            x = (bx - 0.5) * metadata.pitch_length
            y = (by - 0.5) * metadata.pitch_width
            arrays["ball_xy_m"][frame] = orient_xy(x, y, direction)
            arrays["ball_position_valid"][frame] = True
            arrays["audit_ball_quality"][frame] = QUALITY_PROVIDER_UNKNOWN
    if seen != metadata.n_frames:
        raise ValueError("Metrica yielded {} frames, expected {}".format(
            seen, metadata.n_frames))
    compute_common_features(
        arrays, metadata.frame_rate, team_lag_frames=25, ball_lag_frames=13)
    arrays.update(metadata_arrays(
        "metrica", "{}_tactical".format(game), metadata.frame_rate,
        metadata.pitch_length, metadata.pitch_width,
        13.0 / metadata.frame_rate, 0.0))
    validate_archive(arrays, require_target=False)
    return arrays


