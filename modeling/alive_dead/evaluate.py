"""Checkpoint-based evaluation for WCSF and Metrica ALIVE/DEAD classifiers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

from .config import ExperimentConfig
from .dataset import WCSFData, load_wcsf_data, normalize_wcsf_data
from .inference import predict_segment_logits
from .preprocessing import FeatureNormalizer
from .progress import progress
from .runs import (
    evaluation_dir,
    fold_checkpoint_path,
    inner_checkpoint_path as run_inner_checkpoint_path,
    load_run_metadata,
)
from .splits import (
    FoldSplit,
    Segment,
    build_fold_split,
    build_refit_split,
    valid_subsegments,
)
from .tcn import build_model


CLASS_NAMES = {0: "DEAD", 1: "ALIVE"}
THRESHOLD = 0.5
SEGMENT_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
SEGMENT_IOU_HEADLINE = 0.5


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


@dataclass(frozen=True)
class PredictionRun:
    """Predictions for one genuinely contiguous interval."""

    start: int
    end: int
    half_id: int
    labels: np.ndarray
    predictions: np.ndarray

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class LabeledSegment:
    run_id: int
    start: int
    end: int
    label: int


@dataclass(frozen=True)
class CheckpointEvaluation:
    fold: int
    model_type: str
    seed: int
    checkpoint_path: Path
    n_frames_total: int
    source_fps: float
    runs: tuple[PredictionRun, ...]
    report: Dict[str, object]
    label_valid: np.ndarray | None = None


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / float(denominator) if denominator else 0.0


def compute_frame_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, object]:
    """Return binary frame metrics with rows=true and columns=prediction."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.ndim != 1 or labels.shape != predictions.shape or not len(labels):
        raise ValueError("labels and predictions must be non-empty aligned 1D arrays")
    if not np.isin(labels, (0, 1)).all() or not np.isin(predictions, (0, 1)).all():
        raise ValueError("Frame metrics require binary labels and predictions")

    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (labels.astype(np.int64), predictions.astype(np.int64)), 1)
    per_class: Dict[str, object] = {}
    f1s = []
    for class_id, name in CLASS_NAMES.items():
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum() - true_positive)
        false_negative = int(confusion[class_id, :].sum() - true_positive)
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[class_id, :].sum()),
        }
        f1s.append(f1)
    return {
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def extract_segments(values: np.ndarray, run_id: int = 0) -> list[LabeledSegment]:
    """Run-length encode one contiguous binary prediction sequence."""
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isin(values, (0, 1)).all():
        raise ValueError("A segment run must be a non-empty binary 1D array")
    boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return [
        LabeledSegment(run_id, int(start), int(end), int(values[start]))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def _match_segments(
    ground_truth: Sequence[LabeledSegment],
    predictions: Sequence[LabeledSegment],
    threshold: float,
) -> tuple[int, int, int]:
    """Greedily match same-run segments by descending temporal IoU."""
    overlaps = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            if gt.run_id != pred.run_id or gt.label != pred.label:
                continue
            intersection = min(gt.end, pred.end) - max(gt.start, pred.start)
            if intersection <= 0:
                continue
            union = max(gt.end, pred.end) - min(gt.start, pred.start)
            overlaps.append((intersection / float(union), gt_index, pred_index))
    overlaps.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for iou, gt_index, pred_index in overlaps:
        if iou < threshold:
            break
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
    true_positive = len(matched_gt)
    return true_positive, len(predictions) - true_positive, len(ground_truth) - true_positive


def compute_segment_metrics(
    runs: Sequence[PredictionRun], thresholds: Iterable[float] = SEGMENT_IOUS
) -> Dict[str, object]:
    """Compute Metrica-style pooled binary segment F1 over temporal IoUs."""
    if not runs:
        raise ValueError("At least one prediction run is required")
    gt_segments = []
    pred_segments = []
    for run_id, run in enumerate(runs):
        if run.length != len(run.labels) or run.labels.shape != run.predictions.shape:
            raise ValueError("Prediction run boundaries and arrays disagree")
        gt_segments.extend(extract_segments(run.labels, run_id))
        pred_segments.extend(extract_segments(run.predictions, run_id))

    per_iou: Dict[str, object] = {}
    f1s = []
    for threshold in thresholds:
        matched, spurious, missed = _match_segments(
            gt_segments, pred_segments, threshold
        )
        precision = _safe_ratio(matched, matched + spurious)
        recall = _safe_ratio(matched, matched + missed)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        per_iou[f"{threshold:.1f}"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_matched": matched,
            "n_missed": missed,
            "n_spurious": spurious,
        }
        f1s.append(f1)
    headline_key = f"{SEGMENT_IOU_HEADLINE:.1f}"
    return {
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "headline_iou": SEGMENT_IOU_HEADLINE,
        "headline_f1": per_iou.get(headline_key, {}).get("f1"),
        "n_ground_truth": len(gt_segments),
        "n_predicted": len(pred_segments),
        "per_iou": per_iou,
    }


def score_prediction_runs(runs: Sequence[PredictionRun]) -> Dict[str, object]:
    labels = np.concatenate([run.labels for run in runs])
    predictions = np.concatenate([run.predictions for run in runs])
    return {
        "n_frames": int(len(labels)),
        "frame_metrics": compute_frame_metrics(labels, predictions),
        "segment_metrics": compute_segment_metrics(runs),
    }


def normalizer_from_state(state: Mapping[str, object]) -> FeatureNormalizer:
    """Reconstruct the exact train-fold normalizer stored in a checkpoint."""
    required = {
        "player_speed_mean",
        "player_speed_std",
        "team_speed_mean",
        "team_speed_std",
        "epsilon",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(f"Checkpoint normalizer is missing: {sorted(missing)}")
    normalizer = FeatureNormalizer(
        player_speed_mean=float(state["player_speed_mean"]),
        player_speed_std=float(state["player_speed_std"]),
        team_speed_mean=np.asarray(state["team_speed_mean"], dtype=np.float64),
        team_speed_std=np.asarray(state["team_speed_std"], dtype=np.float64),
        epsilon=float(state["epsilon"]),
    )
    if (
        normalizer.team_speed_mean.shape != (3,)
        or normalizer.team_speed_std.shape != (3,)
        or not np.isfinite(normalizer.player_speed_mean)
        or not np.isfinite(normalizer.player_speed_std)
        or not np.isfinite(normalizer.team_speed_mean).all()
        or not np.isfinite(normalizer.team_speed_std).all()
        or normalizer.player_speed_std < normalizer.epsilon
        or np.any(normalizer.team_speed_std < normalizer.epsilon)
    ):
        raise ValueError("Checkpoint normalization parameters are invalid")
    return normalizer


def validate_checkpoint(
    checkpoint: Mapping[str, object],
    expected_fold: int,
    expected_model_type: str,
    expected_seed: int,
    expected_training_mode: str | None = None,
    expected_validation_block: int | None = None,
) -> ExperimentConfig:
    required = {"model_state", "config", "fold", "normalizer", "seed", "epoch"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    if int(checkpoint["fold"]) != expected_fold or config.split.fold != expected_fold:
        raise ValueError("Requested fold does not match the checkpoint")
    if config.model.model_type != expected_model_type:
        raise ValueError("Requested model type does not match the checkpoint")
    if int(checkpoint["seed"]) != expected_seed or config.training.seed != expected_seed:
        raise ValueError("Requested seed does not match the checkpoint")
    if expected_training_mode is not None:
        if checkpoint.get("training_mode") != expected_training_mode:
            raise ValueError("Requested training mode does not match the checkpoint")
    if expected_training_mode == "inner_cv":
        if checkpoint.get("validation_block") != expected_validation_block:
            raise ValueError("Requested validation block does not match the checkpoint")
        if config.split.validation_block != expected_validation_block:
            raise ValueError("Checkpoint configuration has the wrong validation block")
    if expected_training_mode == "full_half_refit" and checkpoint.get(
        "validation_block"
    ) is not None:
        raise ValueError("A refit checkpoint cannot contain a validation block")
    # Historical checkpoints may use window strides that are no longer valid for
    # center-only training; inference does not depend on the training stride.
    config.validate(
        float(checkpoint.get("source_fps", 10.0)), require_supervised_coverage=False
    )
    normalizer_from_state(checkpoint["normalizer"])
    return config


def _predict_valid_runs(
    model: torch.nn.Module, data: WCSFData, segment: Segment, device: torch.device
) -> tuple[PredictionRun, ...]:
    """Infer with full context, then split scoring at ambiguous-label gaps."""
    probabilities = _predict_probabilities(
        model,
        data.players,
        data.player_location_valid,
        data.player_speed_valid,
        data.team_features,
        data.team_feature_valid,
        segment,
        device,
    )
    predictions = (probabilities >= THRESHOLD).astype(np.int8)
    return tuple(
        PredictionRun(
            valid_segment.start,
            valid_segment.end,
            valid_segment.half_id,
            data.labels[valid_segment.start : valid_segment.end].copy(),
            predictions[
                valid_segment.start - segment.start : valid_segment.end - segment.start
            ].copy(),
        )
        for valid_segment in valid_subsegments(segment, data.label_valid)
    )


def _predict_probabilities(
    model: torch.nn.Module,
    players: np.ndarray,
    player_location_valid: np.ndarray,
    player_speed_valid: np.ndarray,
    team_features: np.ndarray,
    team_feature_valid: np.ndarray,
    segment: Segment,
    device: torch.device,
) -> np.ndarray:
    """Infer one genuinely contiguous segment and return ALIVE probabilities."""
    logits = predict_segment_logits(
        model,
        players,
        player_location_valid,
        player_speed_valid,
        team_features,
        team_feature_valid,
        segment,
        device,
    )
    if logits.shape != (segment.length,) or not torch.isfinite(logits).all():
        raise ValueError("Model produced invalid segment logits")
    probabilities = torch.sigmoid(logits).to(torch.float32).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced non-finite probabilities")
    return probabilities


def validate_run_coverage(
    runs: Sequence[PredictionRun], split: FoldSplit, partition: str, n_frames: int,
    label_valid: np.ndarray | None = None,
) -> None:
    expected = getattr(split, f"{partition}_mask").copy()
    if label_valid is not None:
        if label_valid.shape != (n_frames,) or label_valid.dtype != np.bool_:
            raise ValueError("label_valid must be a Boolean full-timeline mask")
        expected &= label_valid
    coverage = np.zeros(n_frames, dtype=np.int8)
    for run in runs:
        if run.start < 0 or run.end > n_frames or run.start >= run.end:
            raise ValueError("Prediction run has invalid boundaries")
        coverage[run.start : run.end] += 1
    if np.any(coverage > 1) or not np.array_equal(coverage.astype(bool), expected):
        raise ValueError(f"Predictions do not exactly cover the {partition} partition")


def evaluate_checkpoint(
    path: str | Path,
    expected_fold: int,
    expected_model_type: str,
    expected_seed: int,
    partition: str,
    device_name: str = "auto",
    training_mode: str = "full_half_refit",
    validation_block: int | None = None,
    show_progress: bool = False,
) -> CheckpointEvaluation:
    """Load a best checkpoint and evaluate one fold partition."""
    if partition not in ("train", "validation", "test"):
        raise ValueError("partition must be train, validation or test")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if training_mode not in ("inner_cv", "full_half_refit"):
        raise ValueError("training_mode must be inner_cv or full_half_refit")
    if training_mode == "inner_cv" and validation_block not in (1, 2, 3):
        raise ValueError("Inner evaluation requires validation_block 1, 2 or 3")
    if training_mode == "full_half_refit" and partition == "validation":
        raise ValueError("A full-half refit has no validation partition")
    config = validate_checkpoint(
        checkpoint,
        expected_fold,
        expected_model_type,
        expected_seed,
        training_mode,
        validation_block,
    )
    if config.model.classification_mode != "binary":
        raise ValueError("The binary evaluator requires classification_mode='binary'")
    raw = load_wcsf_data(
        config.data.feature_path, config.data.label_path, "binary"
    )
    if float(checkpoint.get("source_fps", raw.source_fps)) != raw.source_fps:
        raise ValueError("Checkpoint and feature archive source frequencies disagree")
    if training_mode == "full_half_refit":
        split = build_refit_split(raw.half_id, expected_fold)
    else:
        buffer_frames = config.seconds_to_frames(config.split.buffer_seconds, raw.source_fps)
        split = build_fold_split(
            raw.half_id,
            expected_fold,
            config.split.validation_fraction,
            buffer_frames,
            int(validation_block),
        )
    data = normalize_wcsf_data(
        raw, normalizer_from_state(checkpoint["normalizer"])
    )
    device = select_device(device_name)
    model = build_model(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    segments = split.segments[partition]
    segment_iter = progress(
        segments,
        enabled=show_progress,
        total=len(segments),
        desc=f"Fold {expected_fold} {partition}",
        unit="segment",
        leave=False,
    )
    runs = tuple(
        run
        for segment in segment_iter
        for run in _predict_valid_runs(model, data, segment, device)
    )
    validate_run_coverage(
        runs, split, partition, raw.n_frames, raw.label_valid
    )
    metrics = score_prediction_runs(runs)
    report: Dict[str, object] = {
        "schema_version": "wcsf_alive_dead_evaluation_v2",
        "evaluation_type": "single_fold",
        "fold": expected_fold,
        "partition": partition,
        "model_type": expected_model_type,
        "seed": expected_seed,
        "training_mode": training_mode,
        "validation_block": validation_block,
        "checkpoint": str(path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "classification_threshold": THRESHOLD,
        "class_order": ["DEAD", "ALIVE"],
        "source_fps": raw.source_fps,
        "n_invalid_labels_in_partition": int(
            np.count_nonzero(getattr(split, f"{partition}_mask") & ~raw.label_valid)
        ),
        "runs": [
            {"start": run.start, "end": run.end, "half_id": run.half_id}
            for run in runs
        ],
        **metrics,
    }
    return CheckpointEvaluation(
        expected_fold,
        expected_model_type,
        expected_seed,
        path,
        raw.n_frames,
        raw.source_fps,
        runs,
        report,
        raw.label_valid.copy(),
    )


def combine_oof_runs(
    fold1: CheckpointEvaluation, fold2: CheckpointEvaluation
) -> tuple[PredictionRun, ...]:
    """Combine Fold 2/H1 and Fold 1/H2 held-out predictions."""
    if fold1.fold != 1 or fold2.fold != 2:
        raise ValueError("OOF combination requires Fold 1 followed by Fold 2")
    if fold1.report["partition"] != "test" or fold2.report["partition"] != "test":
        raise ValueError("OOF combination requires held-out test predictions")
    if (
        fold1.model_type != fold2.model_type
        or fold1.seed != fold2.seed
        or fold1.n_frames_total != fold2.n_frames_total
        or fold1.source_fps != fold2.source_fps
    ):
        raise ValueError("OOF checkpoints are incompatible")
    if {run.half_id for run in fold1.runs} != {2} or {run.half_id for run in fold2.runs} != {1}:
        raise ValueError("OOF source mapping must use Fold 2 for H1 and Fold 1 for H2")
    runs = tuple(sorted((*fold2.runs, *fold1.runs), key=lambda run: run.start))
    coverage = np.zeros(fold1.n_frames_total, dtype=np.int8)
    for run in runs:
        coverage[run.start : run.end] += 1
    validity = fold1.label_valid
    if validity is None:
        validity = np.ones(fold1.n_frames_total, dtype=bool)
    if fold2.label_valid is not None and not np.array_equal(validity, fold2.label_valid):
        raise ValueError("OOF checkpoints use incompatible label-validity masks")
    if not np.array_equal(coverage, validity.astype(np.int8)):
        raise ValueError("Combined OOF predictions must cover every valid frame exactly once")
    return runs


def _evaluate_oof_with_runs(
    run_dir: str | Path,
    device_name: str = "auto",
    show_progress: bool = False,
) -> tuple[Dict[str, object], tuple[PredictionRun, ...]]:
    metadata = load_run_metadata(run_dir, "alive_dead")
    model_type = str(metadata["model_type"])
    seed = int(metadata["seed"])
    fold1 = evaluate_checkpoint(
        fold_checkpoint_path(run_dir, 1),
        1,
        model_type,
        seed,
        "test",
        device_name,
        show_progress=show_progress,
    )
    fold2 = evaluate_checkpoint(
        fold_checkpoint_path(run_dir, 2),
        2,
        model_type,
        seed,
        "test",
        device_name,
        show_progress=show_progress,
    )
    runs = combine_oof_runs(fold1, fold2)
    report = {
        "schema_version": "wcsf_alive_dead_evaluation_v2",
        "evaluation_type": "combined_oof",
        "model_type": model_type,
        "seed": seed,
        "classification_threshold": THRESHOLD,
        "class_order": ["DEAD", "ALIVE"],
        "source_fps": fold1.source_fps,
        "fold_results": {"1": fold1.report, "2": fold2.report},
        "combined": {
            "runs": [
                {"start": run.start, "end": run.end, "half_id": run.half_id}
                for run in runs
            ],
            **score_prediction_runs(runs),
        },
    }
    return report, runs


def evaluate_oof(
    run_dir: str | Path,
    device_name: str = "auto",
    show_progress: bool = False,
) -> Dict[str, object]:
    """Evaluate both held-out halves and return the combined OOF report."""
    report, _ = _evaluate_oof_with_runs(
        run_dir, device_name, show_progress
    )
    return report


def _evaluate_inner_blocks_with_runs(
    fold: int,
    run_dir: str | Path,
    device_name: str = "auto",
    show_progress: bool = False,
) -> tuple[Dict[str, object], Dict[str, CheckpointEvaluation]]:
    """Evaluate all three inner validation checkpoints without touching test data."""
    metadata = load_run_metadata(run_dir, "alive_dead")
    model_type = str(metadata["model_type"])
    seed = int(metadata["seed"])
    evaluations: Dict[str, CheckpointEvaluation] = {}
    blocks = progress(
        (1, 2, 3), enabled=show_progress, total=3,
        desc=f"Fold {fold} inner validation", unit="block", leave=False,
    )
    for block in blocks:
        result = evaluate_checkpoint(
            run_inner_checkpoint_path(run_dir, fold, block),
            fold,
            model_type,
            seed,
            "validation",
            device_name,
            training_mode="inner_cv",
            validation_block=block,
            show_progress=show_progress,
        )
        evaluations[str(block)] = result
    report = {
        "schema_version": "wcsf_alive_dead_inner_cv_v1",
        "evaluation_type": "inner_validation_aggregate",
        "fold": fold,
        "model_type": model_type,
        "seed": seed,
        "blocks": {block: result.report for block, result in evaluations.items()},
        "mean_frame_macro_f1": float(
            np.mean(
                [result.report["frame_metrics"]["macro_f1"] for result in evaluations.values()]
            )
        ),
        "mean_binary_segment_f1": float(
            np.mean(
                [result.report["segment_metrics"]["mean_f1"] for result in evaluations.values()]
            )
        ),
    }
    return report, evaluations


def evaluate_inner_blocks(
    fold: int,
    run_dir: str | Path,
    device_name: str = "auto",
    show_progress: bool = False,
) -> Dict[str, object]:
    """Evaluate all inner blocks and return their aggregate report."""
    report, _ = _evaluate_inner_blocks_with_runs(
        fold, run_dir, device_name, show_progress
    )
    return report


def load_metrica_features(path: str | Path) -> Dict[str, np.ndarray]:
    """Load and fully validate the model-ready Metrica feature archive."""
    from feature_engineering.metrica_features import validate_metrica_features

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Metrica feature archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        features = {name: archive[name] for name in archive.files}
    validate_metrica_features(features)
    return features


def reconstruct_source_predictions(
    probabilities_10hz: np.ndarray,
    source_to_sample_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map 10 Hz probabilities to the source timeline using the saved reverse index."""
    probabilities_10hz = np.asarray(probabilities_10hz, dtype=np.float32)
    source_to_sample_index = np.asarray(source_to_sample_index, dtype=np.int64)
    if probabilities_10hz.ndim != 1 or not len(probabilities_10hz):
        raise ValueError("10 Hz probabilities must be a non-empty 1D array")
    if not np.isfinite(probabilities_10hz).all() or np.any(
        (probabilities_10hz < 0.0) | (probabilities_10hz > 1.0)
    ):
        raise ValueError("10 Hz probabilities must be finite and lie in [0, 1]")
    valid = source_to_sample_index >= 0
    if np.any(source_to_sample_index[valid] >= len(probabilities_10hz)):
        raise ValueError("Source alignment refers to a missing 10 Hz sample")

    probabilities_25hz = np.zeros(len(source_to_sample_index), dtype=np.float32)
    probabilities_25hz[valid] = probabilities_10hz[source_to_sample_index[valid]]
    predictions_25hz = np.zeros(len(source_to_sample_index), dtype=np.int8)
    predictions_25hz[valid] = (
        probabilities_25hz[valid] >= THRESHOLD
    ).astype(np.int8)
    return probabilities_25hz, predictions_25hz, valid


def average_fold_probabilities(
    fold1: np.ndarray, fold2: np.ndarray
) -> np.ndarray:
    """Return the fixed equal-weight transfer ensemble."""
    fold1 = np.asarray(fold1, dtype=np.float32)
    fold2 = np.asarray(fold2, dtype=np.float32)
    if fold1.ndim != 1 or fold1.shape != fold2.shape:
        raise ValueError("Fold probabilities must be aligned 1D arrays")
    if not np.isfinite(fold1).all() or not np.isfinite(fold2).all():
        raise ValueError("Fold probabilities must be finite")
    return (0.5 * fold1 + 0.5 * fold2).astype(np.float32)


def validate_transfer_configs(
    fold1: ExperimentConfig, fold2: ExperimentConfig
) -> None:
    """Reject fold checkpoints whose model architectures are not identical."""
    if fold1.model != fold2.model:
        raise ValueError("Fold checkpoints use incompatible model architectures")


def _contiguous_segments(half_id: np.ndarray) -> tuple[Segment, ...]:
    """Return one segment per Metrica half without permitting halftime context."""
    half_id = np.asarray(half_id)
    segments = []
    for half in (1, 2):
        indices = np.flatnonzero(half_id == half)
        if not len(indices) or not np.array_equal(
            indices, np.arange(indices[0], indices[-1] + 1)
        ):
            raise ValueError(f"Metrica half {half} must be one contiguous sequence")
        segments.append(Segment(int(indices[0]), int(indices[-1] + 1), "test", half))
    if sum(segment.length for segment in segments) != len(half_id):
        raise ValueError("Metrica samples must belong to exactly one half")
    return tuple(segments)


def _infer_metrica_checkpoint(
    path: str | Path,
    fold: int,
    model_type: str,
    seed: int,
    features: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, ExperimentConfig, Mapping[str, object]]:
    """Apply one frozen WCSF refit checkpoint to both Metrica halves."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = validate_checkpoint(
        checkpoint,
        fold,
        model_type,
        seed,
        expected_training_mode="full_half_refit",
    )
    source_fps = float(features["source_fps"])
    if float(checkpoint.get("source_fps", source_fps)) != source_fps:
        raise ValueError("Checkpoint and Metrica archive frequencies disagree")

    normalizer = normalizer_from_state(checkpoint["normalizer"])
    players, team_features = normalizer.transform(
        features["players"],
        features["player_location_valid"],
        features["player_speed_valid"],
        features["team_features"],
        features["team_feature_valid"],
    )
    if not np.isfinite(players).all() or not np.isfinite(team_features).all():
        raise ValueError("Normalization produced non-finite Metrica inputs")

    model = build_model(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    probabilities = np.empty(len(features["sample_index"]), dtype=np.float32)
    coverage = np.zeros(len(probabilities), dtype=np.int8)
    for segment in _contiguous_segments(features["half_id"]):
        probabilities[segment.start : segment.end] = _predict_probabilities(
            model,
            players,
            features["player_location_valid"],
            features["player_speed_valid"],
            team_features,
            features["team_feature_valid"],
            segment,
            device,
        )
        coverage[segment.start : segment.end] += 1
    if not np.all(coverage == 1) or not np.isfinite(probabilities).all():
        raise ValueError("Metrica inference did not cover every 10 Hz sample exactly once")
    return probabilities, config, checkpoint


def _public_prediction_runs(
    gt_binary: np.ndarray,
    predictions: np.ndarray,
    graded: np.ndarray,
    source_half_id: np.ndarray,
) -> tuple[PredictionRun, ...]:
    """Create gap-aware runs for binary scoring on the public ground truth."""
    if not (
        gt_binary.shape == predictions.shape == graded.shape == source_half_id.shape
    ):
        raise ValueError("Ground truth, predictions, masks and half IDs must align")
    edges = np.diff(np.r_[False, graded, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    runs = []
    for start, end in zip(starts, ends):
        halves = np.unique(source_half_id[start:end])
        if len(halves) != 1 or halves[0] not in (1, 2):
            raise ValueError("A graded run cannot cross halftime or an unscored gap")
        runs.append(
            PredictionRun(
                int(start),
                int(end),
                int(halves[0]),
                gt_binary[start:end].astype(np.int8, copy=True),
                predictions[start:end].astype(np.int8, copy=True),
            )
        )
    if not runs:
        raise ValueError("The Metrica ground truth contains no graded frames")
    return tuple(runs)


def _compact_transition_scores(scores: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in scores.items()
        if key not in {"pairs", "missed", "spurious"}
    }


def official_binary_metrics(
    gt_codes: np.ndarray, pred_binary: np.ndarray, prediction_valid: np.ndarray
) -> Dict[str, object]:
    """Reuse the official scorer for binary ALIVE/DEAD diagnostics."""
    from evaluation import possession_metrics as pm

    if not (gt_codes.shape == pred_binary.shape == prediction_valid.shape):
        raise ValueError("Official metric inputs must have equal shapes")
    pred_codes = np.full(len(gt_codes), pm.X, dtype=np.int8)
    pred_codes[prediction_valid & (pred_binary == 1)] = pm.H
    pred_codes[prediction_valid & (pred_binary == 0)] = pm.D
    cm3 = pm.scored_confusion(pm.confusion(gt_codes, pred_codes))

    collapsed_gt = gt_codes.copy()
    collapsed_gt[collapsed_gt == pm.A] = pm.H
    runs = pm.valid_runs(collapsed_gt)
    filled_predictions = pm.fill_abstentions(pred_codes, runs)
    gt_transitions = pm.transitions(collapsed_gt, runs)
    pred_transitions = pm.transitions(filled_predictions, runs)
    n_gt, n_pred, ratio = pm.fragmentation(
        collapsed_gt, filled_predictions, runs
    )
    return {
        "alive_dead_f1": float(pm.alive_dead_f1(cm3)),
        "fragmentation": {
            "n_ground_truth_segments": int(n_gt),
            "n_predicted_segments": int(n_pred),
            "ratio": float(ratio),
        },
        "transitions": {
            str(tolerance): _compact_transition_scores(
                pm.transition_scores(
                    gt_transitions, pred_transitions, tolerance, typed="to"
                )
            )
            for tolerance in pm.TAUS
        },
    }


def _load_public_ground_truth(path: str | Path, n_frames: int) -> np.ndarray:
    from evaluation import possession_metrics as pm

    frame = pm.read_label_csv(path)
    problem = pm.validate_submission(frame, n_frames)
    if problem is not None:
        raise ValueError(f"Invalid Metrica public ground truth: {problem}")
    return pm.label_codes(frame)


def _score_metrica_probabilities(
    probabilities_10hz: np.ndarray,
    features: Mapping[str, np.ndarray],
    gt_codes: np.ndarray,
) -> tuple[Dict[str, object], tuple[PredictionRun, ...], Dict[str, np.ndarray]]:
    from evaluation import possession_metrics as pm

    probabilities_25hz, predictions_25hz, prediction_valid = (
        reconstruct_source_predictions(
            probabilities_10hz, features["source_to_sample_index"]
        )
    )
    graded = gt_codes != pm.X
    if not np.all(prediction_valid[graded]):
        raise ValueError("Not every public ground-truth frame has a prediction")
    if int(graded.sum()) != 7500:
        raise ValueError(f"Expected 7,500 public frames, found {int(graded.sum()):,}")
    gt_binary = np.full(len(gt_codes), -1, dtype=np.int8)
    gt_binary[(gt_codes == pm.H) | (gt_codes == pm.A)] = 1
    gt_binary[gt_codes == pm.D] = 0
    runs = _public_prediction_runs(
        gt_binary, predictions_25hz, graded, features["source_half_id"]
    )
    metrics = score_prediction_runs(runs)
    official = official_binary_metrics(gt_codes, predictions_25hz, prediction_valid)
    alive_f1 = metrics["frame_metrics"]["per_class"]["ALIVE"]["f1"]
    if not np.isclose(official["alive_dead_f1"], alive_f1, atol=1e-12):
        raise ValueError("Official alive_dead_f1 disagrees with binary ALIVE F1")
    return (
        {**metrics, "official_binary_metrics": official},
        runs,
        {
            "alive_probability_25hz": probabilities_25hz,
            "alive_prediction_25hz": predictions_25hz,
            "prediction_valid_25hz": prediction_valid,
        },
    )


def evaluate_metrica_transfer(
    run_dir: str | Path,
    feature_path: str | Path,
    ground_truth_path: str | Path,
    device_name: str = "auto",
    show_progress: bool = False,
) -> tuple[Dict[str, object], Dict[str, np.ndarray], tuple[PredictionRun, ...]]:
    """Evaluate two frozen WCSF refit checkpoints on Metrica's public labels."""
    metadata = load_run_metadata(run_dir, "alive_dead")
    model_type = str(metadata["model_type"])
    seed = int(metadata["seed"])
    features = load_metrica_features(feature_path)
    device = select_device(device_name)
    fold_probabilities: Dict[int, np.ndarray] = {}
    configs: Dict[int, ExperimentConfig] = {}
    checkpoints: Dict[int, Mapping[str, object]] = {}
    paths: Dict[int, Path] = {}
    for fold in progress(
        (1, 2), enabled=show_progress, total=2,
        desc="Metrica checkpoints", unit="fold"
    ):
        path = fold_checkpoint_path(run_dir, fold)
        probabilities, config, checkpoint = _infer_metrica_checkpoint(
            path, fold, model_type, seed, features, device
        )
        fold_probabilities[fold] = probabilities
        configs[fold] = config
        checkpoints[fold] = checkpoint
        paths[fold] = path
    validate_transfer_configs(configs[1], configs[2])

    ensemble_probabilities = average_fold_probabilities(
        fold_probabilities[1], fold_probabilities[2]
    )
    gt_codes = _load_public_ground_truth(
        ground_truth_path, int(features["original_frame_count"])
    )
    fold_reports = {}
    for fold in (1, 2):
        metrics, _, _ = _score_metrica_probabilities(
            fold_probabilities[fold], features, gt_codes
        )
        fold_reports[str(fold)] = {
            "checkpoint": str(paths[fold]),
            "checkpoint_epoch": int(checkpoints[fold]["epoch"]),
            **metrics,
        }
    ensemble_metrics, runs, reconstructed = _score_metrica_probabilities(
        ensemble_probabilities, features, gt_codes
    )
    source_frames = np.flatnonzero(features["source_scored"])
    selected_frames = features["source_frame_index"][
        features["source_to_sample_index"][source_frames]
    ]
    full_max_alignment_frames = int(
        np.max(np.abs(source_frames - selected_frames))
    )
    public_frames = np.flatnonzero(gt_codes != 3)
    public_selected_frames = features["source_frame_index"][
        features["source_to_sample_index"][public_frames]
    ]
    max_alignment_frames = int(
        np.max(np.abs(public_frames - public_selected_frames))
    )
    if max_alignment_frames > 1:
        raise ValueError(
            "Public frames must map within one 25 Hz frame of a 10 Hz sample"
        )

    report: Dict[str, object] = {
        "schema_version": "metrica_alive_dead_transfer_evaluation_v1",
        "evaluation_type": "wcsf_to_metrica_public_transfer",
        "model_type": model_type,
        "seed": seed,
        "classification_threshold": THRESHOLD,
        "ensemble_weights": {"fold1": 0.5, "fold2": 0.5},
        "class_order": ["DEAD", "ALIVE"],
        "model_source_fps": float(features["source_fps"]),
        "evaluation_source_fps": float(features["original_source_fps"]),
        "feature_archive": str(feature_path),
        "ground_truth": str(ground_truth_path),
        "public_ground_truth_note": "H/A collapsed to ALIVE; D to DEAD; X ignored",
        "alignment": {
            "n_model_samples": int(len(features["sample_index"])),
            "n_source_frames": int(features["original_frame_count"]),
            "n_public_frames": int(sum(run.length for run in runs)),
            "public_max_source_alignment_frames": max_alignment_frames,
            "public_max_source_alignment_ms": float(
                1000.0 * max_alignment_frames / float(features["original_source_fps"])
            ),
            "full_scored_max_source_alignment_frames": full_max_alignment_frames,
            "full_scored_max_source_alignment_ms": float(
                1000.0
                * full_max_alignment_frames
                / float(features["original_source_fps"])
            ),
        },
        "fold_diagnostics": fold_reports,
        "ensemble": ensemble_metrics,
        "official_metric_scope": {
            "reported": [
                "alive_dead_f1",
                "binary transition metrics",
                "binary fragmentation",
            ],
            "not_reported": ["H/A/D macroF1", "H/A possession spellF1"],
        },
    }
    predictions = {
        "fold1_alive_probability_10hz": fold_probabilities[1].astype(np.float32),
        "fold2_alive_probability_10hz": fold_probabilities[2].astype(np.float32),
        "alive_probability_10hz": ensemble_probabilities,
        "alive_prediction_10hz": (ensemble_probabilities >= THRESHOLD).astype(np.int8),
        **reconstructed,
        "sample_index": features["sample_index"],
        "half_id": features["half_id"],
        "timestamp_ms": features["timestamp_ms"],
        "source_frame_index": features["source_frame_index"],
        "source_to_sample_index": features["source_to_sample_index"],
        "classification_threshold": np.array(THRESHOLD, dtype=np.float32),
    }
    return report, predictions, runs


def _print_metrics(title: str, report: Mapping[str, object]) -> None:
    frame = report["frame_metrics"]
    segments = report["segment_metrics"]
    print(title)
    print(f"  frames: {report['n_frames']}")
    print(f"  accuracy: {frame['accuracy']:.4f}")
    print(f"  macro F1: {frame['macro_f1']:.4f}")
    print(f"  binary segment F1 mean: {segments['mean_f1']:.4f}")
    print(f"  binary segment F1 @0.5: {segments['headline_f1']:.4f}")
    print(f"  confusion [rows=true, cols=pred]: {frame['confusion_matrix']}")


def _write_json(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="wcsf", choices=("wcsf", "metrica"))
    parser.add_argument("--fold", required=True, choices=("1", "2", "all"))
    parser.add_argument(
        "--partition", default="test", choices=("train", "validation", "test")
    )
    parser.add_argument("--inner-block", choices=("1", "2", "3", "all"))
    parser.add_argument(
        "--metrica-features",
        type=Path,
        default=Path("data/metrica/processed/metrica_player_team_features.npz"),
    )
    parser.add_argument(
        "--metrica-ground-truth",
        type=Path,
        default=Path("evaluation/gt/BAR-ATH_possession_gt_public.csv"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--save-timeline",
        action="store_true",
        help="Save ground-truth and prediction segment timelines",
    )
    args = parser.parse_args()
    metadata = load_run_metadata(args.run_dir, "alive_dead")
    model_type = str(metadata["model_type"])
    seed = int(metadata["seed"])
    output_dir = evaluation_dir(args.run_dir)
    show_progress = not args.no_progress

    if args.dataset == "metrica":
        if args.fold != "all":
            parser.error("Metrica transfer evaluation requires --fold all")
        if args.partition != "test" or args.inner_block is not None:
            parser.error("Metrica transfer evaluation does not use WCSF partitions")
        report, predictions, runs = evaluate_metrica_transfer(
            args.run_dir,
            args.metrica_features,
            args.metrica_ground_truth,
            args.device,
            show_progress,
        )
        report_path = output_dir / "metrica_alive_dead.json"
        prediction_path = output_dir / "metrica_alive_dead_predictions.npz"
        timeline_path = output_dir / "metrica_alive_dead_timeline.png"
        _write_json(report_path, report)
        np.savez_compressed(prediction_path, **predictions)
        from .plotting import save_prediction_timeline

        save_prediction_timeline(
            runs,
            float(report["evaluation_source_fps"]),
            timeline_path,
            f"Metrica public ALIVE/DEAD transfer: {model_type}",
        )
        for fold, result in report["fold_diagnostics"].items():
            _print_metrics(f"Metrica transfer: Fold {fold} checkpoint", result)
        _print_metrics("Metrica transfer: 50/50 ensemble", report["ensemble"])
        official = report["ensemble"]["official_binary_metrics"]
        print(f"  official alive/dead F1: {official['alive_dead_f1']:.4f}")
        print(
            "  binary fragmentation: "
            f"{official['fragmentation']['n_predicted_segments']} predicted / "
            f"{official['fragmentation']['n_ground_truth_segments']} ground truth"
        )
        print(f"Saved: {report_path}")
        print(f"Saved: {prediction_path}")
        print(f"Saved: {timeline_path}")
        return

    timeline_jobs = []

    if args.fold == "all":
        if args.inner_block is not None:
            parser.error("--fold all cannot be combined with --inner-block")
        if args.partition != "test":
            parser.error("--fold all only supports the held-out test partition")
        if args.save_timeline:
            report, runs = _evaluate_oof_with_runs(
                args.run_dir, args.device, show_progress
            )
            timeline_jobs.append(
                (
                    runs,
                    float(report["source_fps"]),
                    output_dir / "oof_timeline.png",
                    f"Combined OOF: {model_type}",
                )
            )
        else:
            report = evaluate_oof(
                args.run_dir, args.device, show_progress
            )
        _print_metrics("Fold 1 held-out test", report["fold_results"]["1"])
        _print_metrics("Fold 2 held-out test", report["fold_results"]["2"])
        _print_metrics("Combined out-of-fold", report["combined"])
        output_path = output_dir / "oof.json"
    else:
        fold = int(args.fold)
        if args.inner_block is not None:
            if args.partition != "validation":
                parser.error("--inner-block only supports --partition validation")
            if args.inner_block == "all":
                if args.save_timeline:
                    report, block_evaluations = _evaluate_inner_blocks_with_runs(
                        fold, args.run_dir, args.device, show_progress
                    )
                    for block, result in block_evaluations.items():
                        timeline_jobs.append(
                            (
                                result.runs,
                                result.source_fps,
                                output_dir / f"fold{fold}_inner_block{block}_timeline.png",
                                f"Fold {fold} inner block {block} validation: {model_type}",
                            )
                        )
                else:
                    report = evaluate_inner_blocks(
                        fold, args.run_dir, args.device, show_progress
                    )
                for block, block_report in report["blocks"].items():
                    _print_metrics(f"Fold {fold} inner block {block}", block_report)
                output_path = (
                    output_dir / f"fold{fold}_inner_validation.json"
                )
            else:
                block = int(args.inner_block)
                evaluation = evaluate_checkpoint(
                    run_inner_checkpoint_path(args.run_dir, fold, block),
                    fold,
                    model_type,
                    seed,
                    "validation",
                    args.device,
                    training_mode="inner_cv",
                    validation_block=block,
                    show_progress=show_progress,
                )
                report = evaluation.report
                if args.save_timeline:
                    timeline_jobs.append(
                        (
                            evaluation.runs,
                            evaluation.source_fps,
                            output_dir / f"fold{fold}_inner_block{block}_timeline.png",
                            f"Fold {fold} inner block {block} validation: {model_type}",
                        )
                    )
                _print_metrics(f"Fold {fold} inner block {block}", report)
                output_path = (
                    output_dir / f"fold{fold}_inner_block{block}_validation.json"
                )
        else:
            if args.partition == "validation":
                parser.error("Refit models have no validation partition; use --inner-block")
            evaluation = evaluate_checkpoint(
                fold_checkpoint_path(args.run_dir, fold),
                fold,
                model_type,
                seed,
                args.partition,
                args.device,
                show_progress=show_progress,
            )
            report = evaluation.report
            if args.save_timeline:
                timeline_jobs.append(
                    (
                        evaluation.runs,
                        evaluation.source_fps,
                        output_dir / f"fold{fold}_{args.partition}_timeline.png",
                        f"Fold {fold} {args.partition}: {model_type}",
                    )
                )
            _print_metrics(f"Fold {fold} refit {args.partition}", report)
            output_path = (
                output_dir / f"fold{fold}_{args.partition}.json"
            )

    _write_json(output_path, report)
    print(f"Saved: {output_path}")
    if timeline_jobs:
        from .plotting import save_prediction_timeline

        for runs, source_fps, timeline_path, title in timeline_jobs:
            saved_path = save_prediction_timeline(
                runs, source_fps, timeline_path, title
            )
            print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()
