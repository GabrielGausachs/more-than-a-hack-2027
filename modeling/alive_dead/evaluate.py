"""Checkpoint-based evaluation for WCSF ALIVE/DEAD classifiers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

from .config import ExperimentConfig
from .dataset import WCSFData, load_wcsf_data
from .preprocessing import FeatureNormalizer
from .splits import FoldSplit, Segment, build_fold_split
from .tcn import build_model
from .train import select_device


CLASS_NAMES = {0: "DEAD", 1: "ALIVE"}
THRESHOLD = 0.5
SEGMENT_IOUS = (0.50, 0.75)


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
    """Compute per-class and macro segment F1 without crossing run boundaries."""
    if not runs:
        raise ValueError("At least one prediction run is required")
    gt_segments = []
    pred_segments = []
    for run_id, run in enumerate(runs):
        if run.length != len(run.labels) or run.labels.shape != run.predictions.shape:
            raise ValueError("Prediction run boundaries and arrays disagree")
        gt_segments.extend(extract_segments(run.labels, run_id))
        pred_segments.extend(extract_segments(run.predictions, run_id))

    results: Dict[str, object] = {}
    for threshold in thresholds:
        per_class: Dict[str, object] = {}
        f1s = []
        for class_id, name in CLASS_NAMES.items():
            gt_class = [segment for segment in gt_segments if segment.label == class_id]
            pred_class = [segment for segment in pred_segments if segment.label == class_id]
            matched, spurious, missed = _match_segments(gt_class, pred_class, threshold)
            precision = _safe_ratio(matched, matched + spurious)
            recall = _safe_ratio(matched, matched + missed)
            f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
            per_class[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_ground_truth": len(gt_class),
                "n_predicted": len(pred_class),
                "n_matched": matched,
                "n_missed": missed,
                "n_spurious": spurious,
            }
            f1s.append(f1)
        results[f"{threshold:.2f}"] = {
            "macro_f1": float(np.mean(f1s)),
            "per_class": per_class,
        }
    return results


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
    checkpoint: Mapping[str, object], expected_fold: int, expected_model_type: str, expected_seed: int
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
    config.validate(float(checkpoint.get("source_fps", 10.0)))
    normalizer_from_state(checkpoint["normalizer"])
    return config


def _normalized_data(raw: WCSFData, normalizer: FeatureNormalizer) -> WCSFData:
    players, team_features = normalizer.transform(
        raw.players,
        raw.player_location_valid,
        raw.player_speed_valid,
        raw.team_features,
        raw.team_feature_valid,
    )
    return WCSFData(
        players=players,
        player_location_valid=raw.player_location_valid,
        player_speed_valid=raw.player_speed_valid,
        team_features=team_features,
        team_feature_valid=raw.team_feature_valid,
        labels=raw.labels,
        timestamp_ms=raw.timestamp_ms,
        half_id=raw.half_id,
        sample_index=raw.sample_index,
        source_fps=raw.source_fps,
    )


def _predict_segment(
    model: torch.nn.Module, data: WCSFData, segment: Segment, device: torch.device
) -> PredictionRun:
    frame_slice = slice(segment.start, segment.end)
    output_mask = torch.ones((1, segment.length), dtype=torch.bool, device=device)
    batch = {
        "players": torch.from_numpy(data.players[frame_slice]).unsqueeze(0).to(device),
        "player_location_valid": torch.from_numpy(
            data.player_location_valid[frame_slice]
        ).unsqueeze(0).to(device),
        "player_speed_valid": torch.from_numpy(
            data.player_speed_valid[frame_slice]
        ).unsqueeze(0).to(device),
        "team_features": torch.from_numpy(data.team_features[frame_slice]).unsqueeze(0).to(device),
        "team_feature_valid": torch.from_numpy(
            data.team_feature_valid[frame_slice]
        ).unsqueeze(0).to(device),
        "output_mask": output_mask,
    }
    with torch.no_grad():
        logits = model(batch).squeeze(0)
    if logits.shape != (segment.length,) or not torch.isfinite(logits).all():
        raise ValueError("Model produced invalid segment logits")
    predictions = (torch.sigmoid(logits) >= THRESHOLD).to(torch.int8).cpu().numpy()
    return PredictionRun(
        segment.start,
        segment.end,
        segment.half_id,
        data.labels[frame_slice].copy(),
        predictions,
    )


def validate_run_coverage(
    runs: Sequence[PredictionRun], split: FoldSplit, partition: str, n_frames: int
) -> None:
    expected = getattr(split, f"{partition}_mask")
    coverage = np.zeros(n_frames, dtype=np.int8)
    for run in runs:
        if run.start < 0 or run.end > n_frames or run.start >= run.end:
            raise ValueError("Prediction run has invalid boundaries")
        coverage[run.start : run.end] += 1
    if np.any(coverage > 1) or not np.array_equal(coverage.astype(bool), expected):
        raise ValueError(f"Predictions do not exactly cover the {partition} partition")


def checkpoint_path(runs_dir: str | Path, fold: int, model_type: str, seed: int) -> Path:
    return Path(runs_dir) / f"fold{fold}_{model_type}_seed{seed}" / "best_checkpoint.pt"


def evaluate_checkpoint(
    path: str | Path,
    expected_fold: int,
    expected_model_type: str,
    expected_seed: int,
    partition: str,
    device_name: str = "auto",
) -> CheckpointEvaluation:
    """Load a best checkpoint and evaluate one fold partition."""
    if partition not in ("train", "validation", "test"):
        raise ValueError("partition must be train, validation or test")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = validate_checkpoint(
        checkpoint, expected_fold, expected_model_type, expected_seed
    )
    raw = load_wcsf_data(config.data.feature_path, config.data.label_path)
    if float(checkpoint.get("source_fps", raw.source_fps)) != raw.source_fps:
        raise ValueError("Checkpoint and feature archive source frequencies disagree")
    buffer_frames = config.seconds_to_frames(config.split.buffer_seconds, raw.source_fps)
    split = build_fold_split(
        raw.half_id, expected_fold, config.split.validation_fraction, buffer_frames
    )
    data = _normalized_data(raw, normalizer_from_state(checkpoint["normalizer"]))
    device = select_device(device_name)
    model = build_model(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    runs = tuple(
        _predict_segment(model, data, segment, device)
        for segment in split.segments[partition]
    )
    validate_run_coverage(runs, split, partition, raw.n_frames)
    metrics = score_prediction_runs(runs)
    report: Dict[str, object] = {
        "schema_version": "wcsf_alive_dead_evaluation_v1",
        "evaluation_type": "single_fold",
        "fold": expected_fold,
        "partition": partition,
        "model_type": expected_model_type,
        "seed": expected_seed,
        "checkpoint": str(path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "classification_threshold": THRESHOLD,
        "class_order": ["DEAD", "ALIVE"],
        "source_fps": raw.source_fps,
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
    if not np.all(coverage == 1):
        raise ValueError("Combined OOF predictions must cover every frame exactly once")
    return runs


def evaluate_oof(
    model_type: str,
    seed: int,
    runs_dir: str | Path,
    device_name: str = "auto",
) -> Dict[str, object]:
    fold1 = evaluate_checkpoint(
        checkpoint_path(runs_dir, 1, model_type, seed),
        1,
        model_type,
        seed,
        "test",
        device_name,
    )
    fold2 = evaluate_checkpoint(
        checkpoint_path(runs_dir, 2, model_type, seed),
        2,
        model_type,
        seed,
        "test",
        device_name,
    )
    runs = combine_oof_runs(fold1, fold2)
    return {
        "schema_version": "wcsf_alive_dead_evaluation_v1",
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


def _print_metrics(title: str, report: Mapping[str, object]) -> None:
    frame = report["frame_metrics"]
    segments = report["segment_metrics"]
    print(title)
    print(f"  frames: {report['n_frames']}")
    print(f"  accuracy: {frame['accuracy']:.4f}")
    print(f"  macro F1: {frame['macro_f1']:.4f}")
    print(f"  segment macro F1 @0.50: {segments['0.50']['macro_f1']:.4f}")
    print(f"  segment macro F1 @0.75: {segments['0.75']['macro_f1']:.4f}")
    print(f"  confusion [rows=true, cols=pred]: {frame['confusion_matrix']}")


def _write_json(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=("1", "2", "all"))
    parser.add_argument(
        "--model-type", required=True, choices=("team_tcn", "deepsets_tcn")
    )
    parser.add_argument(
        "--partition", default="test", choices=("train", "validation", "test")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("data/open_data_groningen/model_runs")
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.fold == "all":
        if args.partition != "test":
            parser.error("--fold all only supports the held-out test partition")
        report = evaluate_oof(args.model_type, args.seed, args.runs_dir, args.device)
        _print_metrics("Fold 1 held-out test", report["fold_results"]["1"])
        _print_metrics("Fold 2 held-out test", report["fold_results"]["2"])
        _print_metrics("Combined out-of-fold", report["combined"])
        output_path = args.runs_dir / f"oof_{args.model_type}_seed{args.seed}.json"
    else:
        fold = int(args.fold)
        evaluation = evaluate_checkpoint(
            checkpoint_path(args.runs_dir, fold, args.model_type, args.seed),
            fold,
            args.model_type,
            args.seed,
            args.partition,
            args.device,
        )
        report = evaluation.report
        _print_metrics(f"Fold {fold} {args.partition}", report)
        output_path = evaluation.checkpoint_path.parent / f"evaluation_{args.partition}.json"

    _write_json(output_path, report)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
