"""Create the frozen SkillCorner H/A/D/U supervision labels."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .skillcorner import (
    DEFAULT_DATA_ROOT, SkillCornerMatch, MatchPaths, discover_matches, iter_tracking,
    load_match_metadata, load_phases, load_player_possessions, resolve_match_paths,
    side_for_team,
)


ROOT = Path(__file__).resolve().parents[2]
FPS = 10.0
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "skillcorner_labels"
LOCALITY_FRAMES = 30
VALID_LABELS = frozenset(("H", "A", "D", "U"))
LABEL_COLUMNS = (
    "match_id", "frame", "period", "timestamp", "label", "label_reason",
    "phase_index", "controller_player_id", "gap_id", "interruption_type",
)
U_REASONS = (
    "period_missing", "phase_period_conflict", "controller_phase_contradiction",
    "unsupported_phase_gap", "outside_phase_coverage",
)


def _tracking_columns(paths: MatchPaths) -> Dict[str, List[object]]:
    columns: Dict[str, List[object]] = {"frame": [], "timestamp": [], "period": []}
    for record in iter_tracking(paths):
        columns["frame"].append(int(record["frame"]))
        columns["timestamp"].append(record.get("timestamp"))
        columns["period"].append(record.get("period"))
    return columns


def _phase_channels(phases: Sequence[Mapping[str, object]],
                    tracking: Mapping[str, List[object]],
                    metadata: SkillCornerMatch) -> Dict[str, List[object]]:
    n_frames = len(tracking["frame"])
    channel = {
        "phase_team": [None] * n_frames,
        "phase_index": [None] * n_frames,
        "phase_period_conflict": [False] * n_frames,
    }
    period_metadata = {int(item["period"]): item for item in metadata.periods}
    for phase in phases:
        start = int(phase["frame_start"])
        end = int(phase["frame_end"])
        period = int(phase["period"])
        official = period_metadata.get(period)
        boundary_conflict = bool(official and (
            start < int(official["start_frame"]) or
            end > int(official["end_frame"])
        ))
        side = side_for_team(phase.get("team_in_possession_id"), metadata)
        for frame in range(max(0, start), min(n_frames, end)):
            channel["phase_team"][frame] = side
            channel["phase_index"][frame] = phase.get("index")
            channel["phase_period_conflict"][frame] = (
                boundary_conflict or
                (tracking["period"][frame] is not None and
                 tracking["period"][frame] != period)
            )
    return channel


def _dynamic_channels(events: Sequence[Mapping[str, object]], n_frames: int,
                      metadata: SkillCornerMatch) \
        -> Tuple[Dict[str, List[object]], List[Mapping[str, object]]]:
    possessions = sorted(
        events,
        key=lambda event: (int(event["period"]), int(event["frame_start"]),
                           int(event["frame_end"])),
    )
    channel = {
        "controller_team": [None] * n_frames,
        "controller_player_id": [None] * n_frames,
        "dynamic_team_possession": [None] * n_frames,
    }
    for event in possessions:
        side = side_for_team(event.get("team_id"), metadata)
        start = int(event["frame_start"])
        end = int(event["frame_end"])
        for frame in range(max(0, start), min(n_frames, end + 1)):
            channel["controller_team"][frame] = side
            channel["controller_player_id"][frame] = event.get("player_id")
            channel["dynamic_team_possession"][frame] = side

    for previous, following in zip(possessions, possessions[1:]):
        if previous["period"] != following["period"]:
            continue
        start = int(previous["frame_end"]) + 1
        end = int(following["frame_start"])
        if end <= start or end - start > int(30 * FPS):
            continue
        if (previous.get("game_interruption_after") or
                following.get("game_interruption_before")):
            continue
        side = side_for_team(previous.get("team_id"), metadata)
        for frame in range(max(0, start), min(n_frames, end)):
            channel["dynamic_team_possession"][frame] = side
    return channel, possessions


def _interruption_base(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    for suffix in ("_for", "_against"):
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value


def _interruption_perspective(value: Optional[str]) -> Optional[str]:
    if value and value.endswith("_for"):
        return "for"
    if value and value.endswith("_against"):
        return "against"
    return None


def _interruption_gaps(
        phases: Sequence[Mapping[str, object]],
        possessions: Sequence[Mapping[str, object]],
        dynamic: Mapping[str, List[object]]) -> List[Dict[str, object]]:
    """Find phase gaps and evaluate the frozen D rule in one pass."""

    phases_by_period: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    possessions_by_period: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for phase in phases:
        phases_by_period[int(phase["period"])].append(phase)
    for event in possessions:
        possessions_by_period[int(event["period"])].append(event)

    gaps: List[Dict[str, object]] = []
    for period, period_phases in sorted(phases_by_period.items()):
        period_phases.sort(
            key=lambda phase: (int(phase["frame_start"]), int(phase["frame_end"])))
        period_possessions = possessions_by_period[period]
        for previous_phase, following_phase in zip(period_phases, period_phases[1:]):
            start = int(previous_phase["frame_end"])
            end = int(following_phase["frame_start"])
            if end <= start:
                continue
            previous = max(
                (event for event in period_possessions
                 if int(event["frame_end"]) <= start),
                key=lambda event: int(event["frame_end"]), default=None,
            )
            following = min(
                (event for event in period_possessions
                 if int(event["frame_start"]) >= end),
                key=lambda event: int(event["frame_start"]), default=None,
            )
            after = previous.get("game_interruption_after") if previous else None
            before = following.get("game_interruption_before") if following else None
            after_base = _interruption_base(after)
            same_base = bool(after_base and after_base == _interruption_base(before))
            perspective_ok = False
            if same_base and previous and following:
                previous_team = previous.get("team_id")
                following_team = following.get("team_id")
                after_view = _interruption_perspective(after)
                before_view = _interruption_perspective(before)
                if previous_team is not None and following_team is not None:
                    same_team = int(previous_team) == int(following_team)
                    perspective_ok = bool(after_view and before_view and (
                        (same_team and after_view == before_view) or
                        (not same_team and after_view != before_view)
                    ))
            previous_offset = start - int(previous["frame_end"]) if previous else None
            following_offset = int(following["frame_start"]) - end if following else None
            has_control = any(
                value in ("H", "A")
                for name in ("controller_team", "dynamic_team_possession")
                for value in dynamic[name][start:end]
            )
            strict = bool(
                same_base and perspective_ok and not has_control and
                previous_offset is not None and following_offset is not None and
                0 <= previous_offset <= LOCALITY_FRAMES and
                0 <= following_offset <= LOCALITY_FRAMES
            )
            gaps.append({
                "period": period,
                "frame_start": start,
                "frame_end": end,
                "strict": strict,
                "interruption_type": after_base if same_base else None,
                "previous_offset_frames": previous_offset,
                "following_offset_frames": following_offset,
            })
    return gaps


def build_final_labels(result: Mapping[str, object]) -> Dict[str, object]:
    """Apply the fixed final policy to prepared match channels."""

    tracking = result["tracking"]
    phase = result["phase"]
    dynamic = result["dynamic"]
    n_frames = len(tracking["frame"])
    match_id = int(result["metadata"].match_id)

    gap_ids: List[Optional[int]] = [None] * n_frames
    gap_periods: List[Optional[int]] = [None] * n_frames
    interruption_types: List[Optional[str]] = [None] * n_frames
    gap_candidate = [False] * n_frames
    for gap_id, gap in enumerate(result["gaps"]):
        start = max(0, int(gap["frame_start"]))
        end = min(n_frames, int(gap["frame_end"]))
        for position in range(start, end):
            gap_candidate[position] = True
            gap_ids[position] = gap_id
            gap_periods[position] = int(gap["period"])

    accepted_gaps = [
        {
            "gap_id": gap_id,
            "period": int(gap["period"]),
            "frame_start": int(gap["frame_start"]),
            "frame_end": int(gap["frame_end"]),
            "interruption_type": gap["interruption_type"],
            "previous_offset_frames": gap["previous_offset_frames"],
            "following_offset_frames": gap["following_offset_frames"],
        }
        for gap_id, gap in enumerate(result["gaps"])
        if gap["strict"]
    ]
    strict_gap = [False] * n_frames
    for gap in accepted_gaps:
        for position in range(max(0, gap["frame_start"]), min(n_frames, gap["frame_end"])):
            strict_gap[position] = True
            interruption_types[position] = gap["interruption_type"]

    rows: List[Dict[str, object]] = []
    for position in range(n_frames):
        period = tracking["period"][position]
        phase_team = phase["phase_team"][position]
        controller_team = dynamic["controller_team"][position]
        label = "U"
        reason = "outside_phase_coverage"

        if period is None:
            reason = "period_missing"
        elif phase["phase_period_conflict"][position]:
            reason = "phase_period_conflict"
        elif (phase_team in ("H", "A") and controller_team in ("H", "A") and
              phase_team != controller_team):
            reason = "controller_phase_contradiction"
        elif phase_team in ("H", "A"):
            label = phase_team
            reason = "phase_team_possession"
        elif (strict_gap[position] and gap_periods[position] is not None and
              int(period) == int(gap_periods[position])):
            label = "D"
            reason = "strict_interruption_gap"
        elif gap_candidate[position]:
            reason = "unsupported_phase_gap"

        rows.append({
            "match_id": match_id,
            "frame": tracking["frame"][position],
            "period": period,
            "timestamp": tracking["timestamp"][position],
            "label": label,
            "label_reason": reason,
            "phase_index": phase["phase_index"][position],
            "controller_player_id": dynamic["controller_player_id"][position],
            "gap_id": gap_ids[position],
            "interruption_type": interruption_types[position],
        })

    labels = {row["label"] for row in rows}
    if not labels <= VALID_LABELS:
        raise AssertionError("final labels contain unexpected values: {}".format(labels))
    return {"rows": rows, "accepted_gaps": accepted_gaps}


def summarize_final_labels(result: Mapping[str, object], final: Mapping[str, object]) \
        -> Dict[str, object]:
    rows = final["rows"]
    counts = Counter(row["label"] for row in rows)
    reasons = Counter(row["label_reason"] for row in rows)
    metadata = result["metadata"]
    total = len(rows)
    row: Dict[str, object] = {
        "match_id": int(metadata.match_id),
        "home_team": metadata.raw["home_team"].get(
            "short_name", metadata.raw["home_team"].get("name", "")),
        "away_team": metadata.raw["away_team"].get(
            "short_name", metadata.raw["away_team"].get("name", "")),
        "frame_count": total,
        "accepted_d_gap_count": len(final["accepted_gaps"]),
    }
    for label in ("H", "A", "D", "U"):
        row["label_{}_frames".format(label.lower())] = counts[label]
    for reason in U_REASONS:
        row["u_reason_{}_frames".format(reason)] = reasons[reason]
    return row


def _atomic_write_csv(path: Path, rows: Iterable[Mapping[str, object]],
                      fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", delete=False,
            dir=str(path.parent), prefix=path.name + ".", suffix=".tmp") as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def generate_final_supervision(match_ids: Optional[Sequence[int]] = None,
                               data_root: Optional[Path] = None,
                               output_dir: Path = DEFAULT_OUTPUT_DIR,
                               progress: Optional[Callable[[str], None]] = print) \
        -> Tuple[List[Dict[str, object]], Tuple[Path, ...]]:
    """Generate per-match labels and one consolidated summary."""

    discovery = discover_matches(data_root)
    selected = ([int(value) for value in match_ids] if match_ids is not None
                else [int(value) for value in discovery["local_ids"]])
    if not selected:
        raise ValueError("no SkillCorner matches selected")
    output_dir = Path(output_dir)
    summary_rows: List[Dict[str, object]] = []
    for index, match_id in enumerate(selected, 1):
        if progress:
            progress("[{}/{}] labelling match {}".format(index, len(selected), match_id))
        paths = resolve_match_paths(match_id, data_root)
        metadata = load_match_metadata(paths)
        tracking = _tracking_columns(paths)
        phases = load_phases(paths)
        phase = _phase_channels(phases, tracking, metadata)
        dynamic, possessions = _dynamic_channels(
            load_player_possessions(paths), len(tracking["frame"]), metadata)
        result = {
            "metadata": metadata,
            "tracking": tracking,
            "phase": phase,
            "dynamic": dynamic,
            "gaps": _interruption_gaps(phases, possessions, dynamic),
        }
        final = build_final_labels(result)
        label_path = output_dir / "matches" / str(match_id) / "labels.csv"
        _atomic_write_csv(label_path, final["rows"], LABEL_COLUMNS)
        summary_rows.append(summarize_final_labels(result, final))

    summary_path = output_dir / "summary.csv"
    summary_fields = sorted({key for row in summary_rows for key in row})
    _atomic_write_csv(summary_path, summary_rows, summary_fields)
    return summary_rows, (summary_path,)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--match-ids", type=int, nargs="+")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    _, outputs = generate_final_supervision(
        match_ids=args.match_ids,
        data_root=args.data_root or DEFAULT_DATA_ROOT,
        output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
    )
    print("Outputs:")
    for path in outputs:
        print("  {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
