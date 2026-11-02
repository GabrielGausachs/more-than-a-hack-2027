"""Leakage-safe temporal decoding for trained WCSF H/A/D models."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import ExperimentConfig
from .dataset import load_wcsf_data, normalize_wcsf_data
from .evaluate import normalizer_from_state, select_device, validate_checkpoint
from .evaluate_had import CLASS_NAMES, HADRun, compute_had_metrics
from .inference import predict_segment_logits
from .progress import progress
from .runs import (
    fold_checkpoint_path,
    inner_checkpoint_path,
    load_run_metadata,
)
from .splits import build_fold_split, build_refit_split, valid_subsegments
from .tcn import build_model


DEFAULT_SMOOTHING_WINDOWS = (3, 5, 9, 15, 21, 31, 41)
DEFAULT_VITERBI_PENALTIES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
CACHE_SCHEMA_VERSION = "wcsf_had_probability_cache_v1"
EXPERIMENT_SCHEMA_VERSION = "wcsf_had_temporal_decoding_v1"


@dataclass(frozen=True)
class ProbabilityRun:
    """Class probabilities and labels for one permitted contiguous interval."""

    start: int
    end: int
    half_id: int
    labels: np.ndarray
    probabilities: np.ndarray

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Decoder:
    method: str
    parameter: int | float | None = None

    def __post_init__(self) -> None:
        if self.method == "raw":
            if self.parameter is not None:
                raise ValueError("Raw decoding has no parameter")
        elif self.method == "smoothing":
            if (
                isinstance(self.parameter, bool)
                or not isinstance(self.parameter, int)
                or self.parameter < 1
                or self.parameter % 2 == 0
            ):
                raise ValueError("Smoothing requires a positive odd integer window")
        elif self.method == "viterbi":
            if self.parameter is None or not np.isfinite(self.parameter) or self.parameter < 0:
                raise ValueError("Viterbi requires a finite non-negative penalty")
        else:
            raise ValueError(f"Unknown decoder method: {self.method}")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"method": self.method}
        if self.method == "smoothing":
            result["window_frames"] = self.parameter
        elif self.method == "viterbi":
            result["transition_penalty"] = self.parameter
        return result


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    fold: int
    validation_block: int | None
    training_mode: str
    partition: str
    path: Path


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("H/A/D probabilities must have shape (T, 3)")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("H/A/D probability rows must sum to one")
    return values


def smooth_probabilities(probabilities: np.ndarray, window_frames: int) -> np.ndarray:
    """Apply a centered moving average without extending beyond sequence edges."""
    values = _validate_probabilities(probabilities)
    if window_frames < 1 or window_frames % 2 == 0:
        raise ValueError("Smoothing windows must be positive odd integers")
    radius = window_frames // 2
    cumulative = np.vstack(
        [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0)]
    )
    positions = np.arange(len(values))
    starts = np.maximum(0, positions - radius)
    ends = np.minimum(len(values), positions + radius + 1)
    smoothed = (cumulative[ends] - cumulative[starts]) / (ends - starts)[:, None]
    return smoothed.astype(np.float32)


def viterbi_decode(probabilities: np.ndarray, transition_penalty: float) -> np.ndarray:
    """Decode the maximum-score path with one symmetric state-change penalty."""
    values = _validate_probabilities(probabilities)
    if not np.isfinite(transition_penalty) or transition_penalty < 0.0:
        raise ValueError("transition_penalty must be finite and non-negative")
    emissions = np.log(np.clip(values.astype(np.float64), np.finfo(np.float64).tiny, 1.0))
    scores = emissions[0].copy()
    backpointers = np.zeros((len(values), 3), dtype=np.int8)
    change_cost = np.full((3, 3), -float(transition_penalty), dtype=np.float64)
    np.fill_diagonal(change_cost, 0.0)
    for frame in range(1, len(values)):
        candidates = scores[:, None] + change_cost
        backpointers[frame] = np.argmax(candidates, axis=0)
        scores = candidates[backpointers[frame], np.arange(3)] + emissions[frame]
    path = np.empty(len(values), dtype=np.int8)
    path[-1] = int(np.argmax(scores))
    for frame in range(len(values) - 1, 0, -1):
        path[frame - 1] = backpointers[frame, path[frame]]
    return path


def decode_probability_runs(
    runs: Sequence[ProbabilityRun], decoder: Decoder
) -> tuple[HADRun, ...]:
    """Decode every permitted interval independently."""
    decoded = []
    for run in runs:
        values = _validate_probabilities(run.probabilities)
        if run.length != len(run.labels) or run.length != len(values):
            raise ValueError("Probability run boundaries and arrays disagree")
        if not np.isin(run.labels, (0, 1, 2)).all():
            raise ValueError("Probability runs contain invalid labels")
        if decoder.method == "raw":
            predictions = values.argmax(axis=1).astype(np.int8)
        elif decoder.method == "smoothing":
            predictions = smooth_probabilities(values, int(decoder.parameter)).argmax(
                axis=1
            ).astype(np.int8)
        elif decoder.method == "viterbi":
            predictions = viterbi_decode(values, float(decoder.parameter))
        decoded.append(
            HADRun(
                run.start,
                run.end,
                run.half_id,
                run.labels.copy(),
                predictions,
            )
        )
    if not decoded:
        raise ValueError("At least one probability run is required")
    return tuple(decoded)


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_frames": metrics["n_frames"],
        "frame_metrics": metrics["frame_metrics"],
        "spell_metrics": metrics["spell_metrics"],
        "possession_summary": metrics["possession_summary"],
        "fragmentation": metrics["fragmentation"],
        "transition_metrics": metrics["transition_metrics"],
    }


def evaluate_decoder(
    block_runs: Mapping[int, Sequence[ProbabilityRun]],
    decoder: Decoder,
    source_fps: float,
) -> dict[str, Any]:
    """Evaluate one candidate equally across the three inner blocks."""
    if set(block_runs) != {1, 2, 3}:
        raise ValueError("Decoder selection requires inner blocks 1, 2 and 3")
    block_results: dict[str, Any] = {}
    for block in (1, 2, 3):
        metrics = compute_had_metrics(
            decode_probability_runs(block_runs[block], decoder), source_fps
        )
        block_results[str(block)] = _metric_summary(metrics)
    aggregate = {
        "mean_spell_f1": float(
            np.mean(
                [value["spell_metrics"]["mean_f1"] for value in block_results.values()]
            )
        ),
        "mean_frame_macro_f1": float(
            np.mean(
                [value["frame_metrics"]["macro_f1"] for value in block_results.values()]
            )
        ),
        "mean_spell_f1_at_iou_0.5": float(
            np.mean(
                [
                    value["spell_metrics"]["headline_f1"]
                    for value in block_results.values()
                ]
            )
        ),
        "mean_fragmentation_ratio": float(
            np.mean([value["fragmentation"]["ratio"] for value in block_results.values()])
        ),
        "mean_transition_f1_5_frames": float(
            np.mean(
                [
                    value["transition_metrics"]["5"]["f1"]
                    for value in block_results.values()
                ]
            )
        ),
        "mean_transition_f1_10_frames": float(
            np.mean(
                [
                    value["transition_metrics"]["10"]["f1"]
                    for value in block_results.values()
                ]
            )
        ),
        "ground_truth_possessions": int(
            sum(
                value["possession_summary"]["ground_truth_count"]
                for value in block_results.values()
            )
        ),
        "predicted_possessions": int(
            sum(
                value["possession_summary"]["predicted_count"]
                for value in block_results.values()
            )
        ),
    }
    return {"decoder": decoder.to_dict(), "aggregate": aggregate, "blocks": block_results}


def _selection_key(result: Mapping[str, Any], method_rank: int) -> tuple[float, ...]:
    aggregate = result["aggregate"]
    decoder = result["decoder"]
    parameter = decoder.get("window_frames", decoder.get("transition_penalty", 0.0))
    return (
        float(aggregate["mean_spell_f1"]),
        float(aggregate["mean_frame_macro_f1"]),
        float(aggregate["mean_spell_f1_at_iou_0.5"]),
        -float(method_rank),
        -float(parameter),
    )


def select_fold_decoders(
    block_runs: Mapping[int, Sequence[ProbabilityRun]],
    source_fps: float,
    smoothing_windows: Sequence[int] = DEFAULT_SMOOTHING_WINDOWS,
    viterbi_penalties: Sequence[float] = DEFAULT_VITERBI_PENALTIES,
) -> tuple[dict[str, Any], dict[str, Decoder]]:
    """Tune smoothing and Viterbi independently using inner validation only."""
    if not smoothing_windows or not viterbi_penalties:
        raise ValueError("Both decoder grids must be non-empty")
    if any(window < 1 or window % 2 == 0 for window in smoothing_windows):
        raise ValueError("All smoothing windows must be positive odd integers")
    if any(not np.isfinite(value) or value < 0.0 for value in viterbi_penalties):
        raise ValueError("All Viterbi penalties must be finite and non-negative")

    raw = Decoder("raw")
    smoothers = [Decoder("smoothing", int(window)) for window in smoothing_windows]
    viterbis = [Decoder("viterbi", float(value)) for value in viterbi_penalties]
    raw_result = evaluate_decoder(block_runs, raw, source_fps)
    smoothing_results = [
        evaluate_decoder(block_runs, decoder, source_fps) for decoder in smoothers
    ]
    viterbi_results = [
        evaluate_decoder(block_runs, decoder, source_fps) for decoder in viterbis
    ]
    best_smoothing_result = max(
        smoothing_results, key=lambda result: _selection_key(result, 1)
    )
    best_viterbi_result = max(
        viterbi_results, key=lambda result: _selection_key(result, 2)
    )
    finalists = [raw_result, best_smoothing_result, best_viterbi_result]
    best_overall_result = max(
        finalists,
        key=lambda result: _selection_key(
            result, {"raw": 0, "smoothing": 1, "viterbi": 2}[result["decoder"]["method"]]
        ),
    )

    def from_result(result: Mapping[str, Any]) -> Decoder:
        values = result["decoder"]
        parameter = values.get("window_frames", values.get("transition_penalty"))
        return Decoder(str(values["method"]), parameter)

    selected = {
        "raw": raw,
        "smoothing": from_result(best_smoothing_result),
        "viterbi": from_result(best_viterbi_result),
        "selected": from_result(best_overall_result),
    }
    report = {
        "selection_metric": "mean spell F1 over IoU 0.3-0.7, equally averaged across blocks",
        "tie_breaks": [
            "mean frame macro F1",
            "mean spell F1 at IoU 0.5",
            "weaker decoder method",
            "smaller parameter",
        ],
        "raw": raw_result,
        "smoothing_candidates": smoothing_results,
        "viterbi_candidates": viterbi_results,
        "best": {name: decoder.to_dict() for name, decoder in selected.items()},
    }
    return report, selected


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_specs(run_dir: Path) -> tuple[CheckpointSpec, ...]:
    specs = []
    for fold in (1, 2):
        for block in (1, 2, 3):
            specs.append(
                CheckpointSpec(
                    f"inner_f{fold}_b{block}", fold, block, "inner_cv", "validation",
                    inner_checkpoint_path(run_dir, fold, block),
                )
            )
        specs.append(
            CheckpointSpec(
                f"outer_f{fold}", fold, None, "full_half_refit", "test",
                fold_checkpoint_path(run_dir, fold),
            )
        )
    return tuple(specs)


def _cache_identity(
    run_dir: Path, metadata: Mapping[str, Any], specs: Sequence[CheckpointSpec]
) -> dict[str, Any]:
    label_path = Path(metadata["config"]["data"]["label_path"])
    feature_path = Path(metadata["config"]["data"]["feature_path"])
    paths = [
        (run_dir / "run_metadata.json").resolve(),
        label_path.resolve(),
        feature_path.resolve(),
    ]
    paths.extend(spec.path.resolve() for spec in specs)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required decoding inputs are missing: {missing}")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "run_name": metadata["run_name"],
        "class_order": list(CLASS_NAMES),
        "files": {str(path): _file_sha256(path) for path in paths},
        "entries": [spec.key for spec in specs],
    }


def _infer_checkpoint_runs(
    spec: CheckpointSpec,
    metadata: Mapping[str, Any],
    device: torch.device,
) -> tuple[ProbabilityRun, ...]:
    checkpoint = torch.load(spec.path, map_location="cpu", weights_only=False)
    config = validate_checkpoint(
        checkpoint,
        spec.fold,
        str(metadata["model_type"]),
        int(metadata["seed"]),
        spec.training_mode,
        spec.validation_block,
    )
    if config.model.classification_mode != "had":
        raise ValueError("Temporal decoding requires H/A/D checkpoints")
    raw = load_wcsf_data(
        config.data.feature_path, config.data.label_path, "had", config.model.use_ball
    )
    if spec.training_mode == "inner_cv":
        buffer_frames = config.seconds_to_frames(
            config.split.buffer_seconds, raw.source_fps
        )
        split = build_fold_split(
            raw.half_id,
            spec.fold,
            config.split.validation_fraction,
            buffer_frames,
            int(spec.validation_block),
        )
    else:
        split = build_refit_split(raw.half_id, spec.fold)
    data = normalize_wcsf_data(raw, normalizer_from_state(checkpoint["normalizer"]))
    model = build_model(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    runs = []
    for segment in split.segments[spec.partition]:
        logits = predict_segment_logits(
            model,
            data.players,
            data.player_location_valid,
            data.player_speed_valid,
            data.team_features,
            data.team_feature_valid,
            segment,
            device,
            ball=data.ball,
            ball_location_valid=data.ball_location_valid,
            ball_speed_valid=data.ball_speed_valid,
        )
        if logits.shape != (segment.length, 3):
            raise ValueError("H/A/D checkpoint produced an unexpected logit shape")
        probabilities = torch.softmax(logits, dim=-1).to(torch.float32).cpu().numpy()
        _validate_probabilities(probabilities)
        for valid_segment in valid_subsegments(segment, data.label_valid):
            local = slice(
                valid_segment.start - segment.start, valid_segment.end - segment.start
            )
            runs.append(
                ProbabilityRun(
                    valid_segment.start,
                    valid_segment.end,
                    valid_segment.half_id,
                    data.labels[valid_segment.start : valid_segment.end].copy(),
                    probabilities[local].copy(),
                )
            )
    expected = getattr(split, f"{spec.partition}_mask") & raw.label_valid
    coverage = np.zeros(raw.n_frames, dtype=np.int8)
    for run in runs:
        coverage[run.start : run.end] += 1
    if not np.array_equal(coverage, expected.astype(np.int8)):
        raise ValueError(f"{spec.key} does not exactly cover its valid labels")
    return tuple(runs)


def save_probability_cache(
    path: str | Path,
    identity: Mapping[str, Any],
    entries: Mapping[str, Sequence[ProbabilityRun]],
) -> Path:
    """Atomically save validated, checkpoint-bound probability runs."""
    path = Path(path)
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.array(json.dumps(identity, sort_keys=True))
    }
    expected = set(identity["entries"])
    if set(entries) != expected:
        raise ValueError("Probability cache entries disagree with cache identity")
    for key, runs in entries.items():
        if not runs:
            raise ValueError(f"Probability cache entry {key} is empty")
        for run in runs:
            if (
                run.length != len(run.labels)
                or run.length != len(run.probabilities)
                or not np.isin(run.labels, (0, 1, 2)).all()
            ):
                raise ValueError(f"Probability cache entry {key} has an invalid run")
        arrays[f"{key}_bounds"] = np.asarray(
            [(run.start, run.end, run.half_id) for run in runs], dtype=np.int64
        )
        arrays[f"{key}_labels"] = np.concatenate([run.labels for run in runs]).astype(
            np.int8, copy=False
        )
        arrays[f"{key}_probabilities"] = np.concatenate(
            [_validate_probabilities(run.probabilities) for run in runs]
        ).astype(np.float32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return path


def load_probability_cache(
    path: str | Path, expected_identity: Mapping[str, Any]
) -> dict[str, tuple[ProbabilityRun, ...]]:
    """Load a cache only when all source identities still agree exactly."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("Probability cache has no metadata")
        actual_identity = json.loads(str(archive["metadata_json"].item()))
        if actual_identity != expected_identity:
            raise ValueError("Probability cache does not match the current run inputs")
        result = {}
        for key in expected_identity["entries"]:
            required = {
                f"{key}_bounds", f"{key}_labels", f"{key}_probabilities"
            }
            if not required.issubset(archive.files):
                raise ValueError(f"Probability cache entry {key} is incomplete")
            bounds = archive[f"{key}_bounds"]
            labels = archive[f"{key}_labels"]
            probabilities = archive[f"{key}_probabilities"]
            if bounds.ndim != 2 or bounds.shape[1] != 3:
                raise ValueError(f"Probability cache entry {key} has invalid bounds")
            _validate_probabilities(probabilities)
            runs = []
            offset = 0
            for start, end, half_id in bounds:
                length = int(end - start)
                if length < 1 or offset + length > len(labels):
                    raise ValueError(f"Probability cache entry {key} has invalid run lengths")
                runs.append(
                    ProbabilityRun(
                        int(start), int(end), int(half_id),
                        labels[offset : offset + length].copy(),
                        probabilities[offset : offset + length].copy(),
                    )
                )
                offset += length
            if offset != len(labels) or offset != len(probabilities):
                raise ValueError(f"Probability cache entry {key} has trailing values")
            result[key] = tuple(runs)
    return result


def _validate_cached_timelines(
    entries: Mapping[str, Sequence[ProbabilityRun]],
    metadata: Mapping[str, Any],
    specs: Sequence[CheckpointSpec],
) -> None:
    """Verify cached bounds and labels against the authoritative source archives."""
    config = ExperimentConfig.from_dict(metadata["config"])
    raw = load_wcsf_data(
        config.data.feature_path, config.data.label_path, "had", config.model.use_ball
    )
    for spec in specs:
        if spec.training_mode == "inner_cv":
            buffer_frames = config.seconds_to_frames(
                config.split.buffer_seconds, raw.source_fps
            )
            split = build_fold_split(
                raw.half_id,
                spec.fold,
                config.split.validation_fraction,
                buffer_frames,
                int(spec.validation_block),
            )
        else:
            split = build_refit_split(raw.half_id, spec.fold)
        expected = tuple(
            valid_segment
            for segment in split.segments[spec.partition]
            for valid_segment in valid_subsegments(segment, raw.label_valid)
        )
        runs = tuple(entries.get(spec.key, ()))
        actual_bounds = [(run.start, run.end, run.half_id) for run in runs]
        expected_bounds = [
            (segment.start, segment.end, segment.half_id) for segment in expected
        ]
        if actual_bounds != expected_bounds:
            raise ValueError(f"Probability cache entry {spec.key} has the wrong timeline")
        for run in runs:
            if not np.array_equal(run.labels, raw.labels[run.start : run.end]):
                raise ValueError(f"Probability cache entry {spec.key} has stale labels")


def _load_or_create_cache(
    run_dir: Path,
    metadata: Mapping[str, Any],
    cache_path: Path,
    device: torch.device,
    rebuild: bool,
    show_progress: bool,
) -> tuple[dict[str, tuple[ProbabilityRun, ...]], dict[str, Any]]:
    specs = _checkpoint_specs(run_dir)
    identity = _cache_identity(run_dir, metadata, specs)
    if cache_path.is_file() and not rebuild:
        entries = load_probability_cache(cache_path, identity)
        _validate_cached_timelines(entries, metadata, specs)
        return entries, identity
    entries = {}
    iterator = progress(
        specs,
        enabled=show_progress,
        total=len(specs),
        desc="Checkpoint inference",
        unit="checkpoint",
    )
    for spec in iterator:
        entries[spec.key] = _infer_checkpoint_runs(spec, metadata, device)
    _validate_cached_timelines(entries, metadata, specs)
    save_probability_cache(cache_path, identity, entries)
    return entries, identity


def _combined_oof_report(
    outer_runs: Mapping[int, Sequence[ProbabilityRun]],
    selections: Mapping[int, Mapping[str, Decoder]],
    source_fps: float,
) -> dict[str, Any]:
    if {run.half_id for run in outer_runs.get(1, ())} != {2} or {
        run.half_id for run in outer_runs.get(2, ())
    } != {1}:
        raise ValueError("OOF decoding requires Fold 2 for H1 and Fold 1 for H2")
    approaches = {}
    for approach in ("raw", "smoothing", "viterbi", "selected"):
        fold_reports = {}
        combined: list[HADRun] = []
        for fold in (1, 2):
            decoded = decode_probability_runs(outer_runs[fold], selections[fold][approach])
            fold_reports[str(fold)] = {
                "decoder": selections[fold][approach].to_dict(),
                **_metric_summary(compute_had_metrics(decoded, source_fps)),
            }
            combined.extend(decoded)
        combined.sort(key=lambda run: run.start)
        approaches[approach] = {
            "folds": fold_reports,
            "combined": _metric_summary(compute_had_metrics(combined, source_fps)),
        }
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "oof_mapping": {"half_1": "fold_2", "half_2": "fold_1"},
        "approaches": approaches,
    }


def run_experiment(
    run_dir: str | Path,
    *,
    device_name: str = "auto",
    smoothing_windows: Sequence[int] = DEFAULT_SMOOTHING_WINDOWS,
    viterbi_penalties: Sequence[float] = DEFAULT_VITERBI_PENALTIES,
    rebuild_cache: bool = False,
    show_progress: bool = True,
) -> Path:
    run_dir = Path(run_dir).resolve()
    metadata = load_run_metadata(run_dir, "had")
    if metadata.get("status") != "completed" or set(metadata.get("folds_completed", ())) != {1, 2}:
        raise ValueError("Temporal decoding requires one completed two-fold H/A/D run")
    postprocessing_dir = run_dir / "postprocessing"
    cache_path = postprocessing_dir / "probabilities.npz"
    entries, cache_identity = _load_or_create_cache(
        run_dir,
        metadata,
        cache_path,
        select_device(device_name),
        rebuild_cache,
        show_progress,
    )
    source_fps = float(metadata["temporal"]["source_fps"])
    selection_reports = {}
    selections = {}
    for fold in (1, 2):
        blocks = {block: entries[f"inner_f{fold}_b{block}"] for block in (1, 2, 3)}
        selection_reports[str(fold)], selections[fold] = select_fold_decoders(
            blocks, source_fps, smoothing_windows, viterbi_penalties
        )
    outer_runs = {fold: entries[f"outer_f{fold}"] for fold in (1, 2)}
    oof_report = _combined_oof_report(outer_runs, selections, source_fps)

    created_at = datetime.now().astimezone()
    experiment_dir = postprocessing_dir / f"decoding_{created_at.strftime('%Y%m%d_%H%M%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    experiment_metadata = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "created_at": created_at.isoformat(),
        "source_run": str(run_dir),
        "source_run_name": metadata["run_name"],
        "model_type": metadata["model_type"],
        "seed": metadata["seed"],
        "class_order": list(CLASS_NAMES),
        "probability_cache": str(cache_path),
        "probability_cache_identity": cache_identity,
        "smoothing_windows": list(smoothing_windows),
        "viterbi_penalties": list(viterbi_penalties),
        "boundary_policy": "decode each half/partition/valid-label run independently",
        "selection_policy": "fold-specific three-block inner validation only",
    }
    for filename, values in (
        ("experiment_metadata.json", experiment_metadata),
        (
            "inner_selection.json",
            {"schema_version": EXPERIMENT_SCHEMA_VERSION, "folds": selection_reports},
        ),
        ("oof_comparison.json", oof_report),
    ):
        with (experiment_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2)
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--smoothing-windows", type=int, nargs="+", default=DEFAULT_SMOOTHING_WINDOWS
    )
    parser.add_argument(
        "--viterbi-penalties", type=float, nargs="+", default=DEFAULT_VITERBI_PENALTIES
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    output = run_experiment(
        args.run_dir,
        device_name=args.device,
        smoothing_windows=args.smoothing_windows,
        viterbi_penalties=args.viterbi_penalties,
        rebuild_cache=args.rebuild_cache,
        show_progress=not args.no_progress,
    )
    with (output / "oof_comparison.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    print("Combined OOF results:")
    for name, values in report["approaches"].items():
        metrics = values["combined"]
        print(
            f"  {name:9s} spell F1={metrics['spell_metrics']['mean_f1']:.4f} "
            f"macro F1={metrics['frame_metrics']['macro_f1']:.4f} "
            f"possessions={metrics['possession_summary']['predicted_count']}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
