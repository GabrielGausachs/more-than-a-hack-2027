"""Ground-truth-free Metrica submission generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from games import GAMES, half_video_blocks

from .decoding import viterbi_decode
from .modeling.data import (ENGINEERED_SPECS, LABELS, SCHEMA_VERSION,
                            ROOT, Normalization, iter_sequence_chunks,
                            load_json, resolve_path)
from .modeling.model import DeepSetsTCN, build_model


CORE_SIZE = 512
DEFAULT_CONFIG = ROOT / "configs" / "inference.json"
MODEL_ARRAYS = (
    "player_xy_m", "player_speed_mps", "player_position_valid", "player_speed_valid",
) + tuple(dict.fromkeys(name for name, _, _ in ENGINEERED_SPECS))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model_artifact(model_dir: Path, device: torch.device) \
        -> Tuple[DeepSetsTCN, Normalization, Mapping[str, object], int]:
    checkpoint = torch.load(
        model_dir / "model.pt", map_location=device, weights_only=False)
    normalization = Normalization.load(model_dir / "normalization.npz")
    decoder_result = load_json(model_dir / "final_decoder.json")
    model = build_model(
        int(checkpoint["engineered_width"]), checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, normalization, decoder_result["decoder"], int(checkpoint["epoch"])


def _predict_probabilities(model: DeepSetsTCN,
                           arrays: Mapping[str, np.ndarray],
                           normalization: Normalization,
                           device: torch.device) -> np.ndarray:
    """Return frame-aligned H/A/D probabilities for one feature record."""

    model.eval()
    probabilities = np.zeros((len(arrays["period"]), 3), dtype=np.float32)
    halo = model.receptive_radius
    with torch.no_grad():
        for chunk in iter_sequence_chunks(
                arrays, normalization, CORE_SIZE, halo):
            players = torch.from_numpy(chunk["players"][None]).to(device)
            engineered = torch.from_numpy(chunk["engineered"][None]).to(device)
            frames = chunk["frame"]
            logits = model(players, engineered)[0, halo:halo + len(frames)]
            probabilities[frames] = torch.softmax(
                logits, dim=-1).cpu().numpy()
    return probabilities


def _load_metrica_archive(path: Path, game: str) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise IOError("Metrica feature archive not found: {}".format(path))
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}

    required = set(MODEL_ARRAYS) | {
        "schema_version", "provider", "frame_rate_hz", "frame_index",
        "timestamp_s", "period",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError("{} is missing arrays {}".format(path, missing))
    if str(arrays["schema_version"].item()) != SCHEMA_VERSION:
        raise ValueError("expected schema {}, got {!r}".format(
            SCHEMA_VERSION, arrays["schema_version"].item()))
    if str(arrays["provider"].item()) != "metrica":
        raise ValueError("expected Metrica archive")
    if "target_label" in arrays:
        raise ValueError("Metrica inference archive must be unlabelled")
    expected_frames = int(GAMES[game]["n_video_frames"])
    if len(arrays["frame_index"]) != expected_frames:
        raise ValueError("archive has {} frames, expected {}".format(
            len(arrays["frame_index"]), expected_frames))
    if not np.isclose(float(arrays["frame_rate_hz"].item()),
                      float(GAMES[game]["fps"]), rtol=0.0, atol=1e-6):
        raise ValueError("archive frame rate differs from games.py")
    return arrays


def nearest_indices(reference_timestamps: np.ndarray,
                    query_timestamps: np.ndarray) -> np.ndarray:
    """Indices of nearest references, choosing the earlier timestamp on ties."""

    reference = np.asarray(reference_timestamps, dtype=np.float64)
    query = np.asarray(query_timestamps, dtype=np.float64)
    right = np.clip(np.searchsorted(reference, query, side="left"),
                    0, len(reference) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(query - reference[left]) <= np.abs(reference[right] - query)
    return np.where(choose_left, left, right).astype(np.int32)


def _target_timestamps(source_timestamps: np.ndarray, target_rate_hz: float) -> np.ndarray:
    start, stop = float(source_timestamps[0]), float(source_timestamps[-1])
    count = int(np.floor((stop - start) * target_rate_hz + 1e-7)) + 1
    return start + np.arange(count, dtype=np.float64) / target_rate_hz


def resample_scored_halves(arrays: Mapping[str, np.ndarray], game: str,
                           target_rate_hz: float) \
        -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Nearest-neighbour 25-to-10 Hz mapping, independently per official half."""

    n_frames = len(arrays["frame_index"])
    timestamps = arrays["timestamp_s"].astype(np.float64)
    source_parts, target_parts, period_parts = [], [], []
    video_to_sample = np.full(n_frames, -1, dtype=np.int32)
    graded_mask = np.zeros(n_frames, dtype=bool)
    sample_offset = 0

    blocks = half_video_blocks(game)
    for period, first, last in blocks:
        source_frames = np.arange(first, last + 1, dtype=np.int32)
        if not np.all(arrays["period"][source_frames] == period):
            raise ValueError("archive period differs from official half {}".format(period))
        source_times = timestamps[source_frames]
        target_times = _target_timestamps(source_times, target_rate_hz)
        sampled_frames = source_frames[nearest_indices(source_times, target_times)]
        source_parts.append(sampled_frames)
        target_parts.append(target_times)
        period_parts.append(np.full(len(target_times), period, dtype=np.int8))
        video_to_sample[source_frames] = sample_offset + nearest_indices(
            target_times, source_times)
        graded_mask[source_frames] = True
        sample_offset += len(target_times)

    sample_source = np.concatenate(source_parts)
    sample_target = np.concatenate(target_parts)
    sample_period = np.concatenate(period_parts)
    record: Dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        record[name] = (value[sample_source].copy()
                        if value.ndim and value.shape[0] == n_frames else value.copy())
    record["frame_index"] = np.arange(len(sample_source), dtype=np.int32)
    record["timestamp_s"] = sample_target.astype(np.float32)
    record["period"] = sample_period
    record["frame_rate_hz"] = np.asarray(target_rate_hz, dtype=np.float32)
    return record, {
        "video_to_sample": video_to_sample,
        "graded_mask": graded_mask,
        "half_blocks": np.asarray(blocks, dtype=np.int32),
    }


def _write_submission(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("frame", "label"))
        writer.writerows((frame, str(label)) for frame, label in enumerate(labels))


def _possession_count(labels: np.ndarray, blocks: np.ndarray) -> int:
    count = 0
    for _, first, last in blocks:
        previous = "D"
        for label in labels[int(first):int(last) + 1]:
            if label in ("H", "A") and label != previous:
                count += 1
            previous = str(label)
    return count


def generate_submission(config_path: Path = DEFAULT_CONFIG,
                        model_dir: Optional[Path] = None,
                        output_dir: Optional[Path] = None,
                        submission_name: Optional[str] = None) -> Path:
    """Generate a full-timeline submission without reading ground truth."""

    config = load_json(config_path)
    game = str(config["game"])
    if game not in GAMES:
        raise ValueError("unknown game {!r}".format(game))
    model_dir = resolve_path(str(
        model_dir if model_dir is not None else config["model_dir"]))
    submission_dir = resolve_path(str(
        output_dir if output_dir is not None else config["output_dir"]))
    archive_path = resolve_path(str(config["archive"]))
    target_rate = float(config["target_rate_hz"])
    n_frames = int(GAMES[game]["n_video_frames"])

    arrays = _load_metrica_archive(archive_path, game)
    record, mapping = resample_scored_halves(arrays, game, target_rate)
    graded_mask = mapping["graded_mask"]
    blocks = mapping["half_blocks"]
    expected_graded = sum(int(last) - int(first) + 1 for _, first, last in blocks)
    if int(graded_mask.sum()) != expected_graded:
        raise AssertionError("graded frame count differs from games.py")

    device = torch.device("cpu")
    model, normalization, decoder, checkpoint_epoch = _load_model_artifact(
        model_dir, device)
    probability = _predict_probabilities(model, record, normalization, device)
    if str(decoder["method"]) != "viterbi":
        raise ValueError("the final pipeline requires Viterbi decoding")
    decoded = viterbi_decode(
        probability, record["period"], float(decoder["switch_penalty"]),
        float(decoder["probability_epsilon"]))

    labels = np.full(n_frames, "X", dtype="U1")
    labels[graded_mask] = decoded[mapping["video_to_sample"][graded_mask]]
    if not np.all(np.isin(labels[graded_mask], LABELS)):
        raise AssertionError("official half blocks must contain H/A/D")

    submission_path = submission_dir / str(
        submission_name if submission_name is not None else config["submission_name"])
    _write_submission(submission_path, labels)
    statistics = {
        "solution": submission_dir.name,
        "game": game,
        "source_frames": n_frames,
        "source_rate_hz": float(GAMES[game]["fps"]),
        "target_model_rate_hz": target_rate,
        "sampled_frames": len(record["period"]),
        "official_half_blocks": blocks.tolist(),
        "graded_frames": int(graded_mask.sum()),
        "ungraded_frames": int((~graded_mask).sum()),
        "label_counts": {
            label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")
        },
        "predicted_possession_spells": _possession_count(labels, blocks),
        "decoder": decoder,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": _sha256(model_dir / "model.pt"),
        "normalization_sha256": _sha256(model_dir / "normalization.npz"),
        "submission": str(submission_path),
    }
    (submission_dir / "prediction_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    print("Wrote {}".format(submission_path))
    return submission_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--submission-name")
    args = parser.parse_args(argv)
    generate_submission(
        args.config, args.model_dir, args.output_dir, args.submission_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
