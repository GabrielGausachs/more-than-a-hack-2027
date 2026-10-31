"""Checkpoint evaluation for direct WCSF Home/Away/Dead classification."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from evaluation import possession_metrics as pm

from .dataset import WCSFData, load_wcsf_data, normalize_wcsf_data
from .evaluate import normalizer_from_state, select_device, validate_checkpoint
from .inference import predict_segment_logits
from .splits import (
    FoldSplit,
    Segment,
    build_fold_split,
    build_refit_split,
    valid_subsegments,
)
from .tcn import build_model


CLASS_NAMES = ("H", "A", "D")
TRANSITION_TOLERANCES = (5, 10)


@dataclass(frozen=True)
class HADRun:
    start: int
    end: int
    half_id: int
    labels: np.ndarray
    predictions: np.ndarray

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class HADEvaluation:
    fold: int
    model_type: str
    seed: int
    checkpoint_path: Path
    n_frames_total: int
    source_fps: float
    runs: tuple[HADRun, ...]
    report: Dict[str, object]
    label_valid: np.ndarray


def _predict_valid_runs(
    model: torch.nn.Module,
    data: WCSFData,
    segment: Segment,
    device: torch.device,
) -> tuple[HADRun, ...]:
    logits = predict_segment_logits(
        model,
        data.players,
        data.player_location_valid,
        data.player_speed_valid,
        data.team_features,
        data.team_feature_valid,
        segment,
        device,
    )
    if logits.shape != (segment.length, 3) or not torch.isfinite(logits).all():
        raise ValueError("H/A/D model produced invalid segment logits")
    predictions = logits.argmax(dim=-1).to(torch.int8).cpu().numpy()

    return tuple(
        HADRun(
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


def _compact_runs(
    runs: Sequence[HADRun],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    if not runs:
        raise ValueError("At least one valid H/A/D run is required")
    labels, predictions, bounds = [], [], []
    offset = 0
    separator = 2 * max(TRANSITION_TOLERANCES) + 1
    for run_index, run in enumerate(runs):
        if run.length != len(run.labels) or run.labels.shape != run.predictions.shape:
            raise ValueError("H/A/D run boundaries and arrays disagree")
        if not np.isin(run.labels, (0, 1, 2)).all() or not np.isin(
            run.predictions, (0, 1, 2)
        ).all():
            raise ValueError("H/A/D runs contain invalid class values")
        if run_index:
            labels.append(np.full(separator, pm.X, dtype=np.int8))
            predictions.append(np.full(separator, pm.X, dtype=np.int8))
            offset += separator
        labels.append(run.labels)
        predictions.append(run.predictions)
        bounds.append((offset, offset + run.length))
        offset += run.length
    return np.concatenate(labels), np.concatenate(predictions), bounds


def compute_had_metrics(
    runs: Sequence[HADRun], source_fps: float = 10.0
) -> Dict[str, object]:
    """Score valid runs with the official competition metric definitions."""
    temporal_labels, temporal_predictions, bounds = _compact_runs(runs)
    labels = np.concatenate([temporal_labels[start:end] for start, end in bounds])
    predictions = np.concatenate(
        [temporal_predictions[start:end] for start, end in bounds]
    )
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels.astype(int), predictions.astype(int)), 1)
    per_class: Dict[str, object] = {}
    f1s = []
    for class_id, name in enumerate(CLASS_NAMES):
        precision, recall, f1 = pm.per_class_prf(confusion, class_id)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[class_id].sum()),
        }
        f1s.append(f1)

    spells = pm.spell_scores(temporal_labels, temporal_predictions, bounds)
    gt_segments, pred_segments, fragmentation = pm.fragmentation(
        temporal_labels, temporal_predictions, bounds
    )
    gt_transitions = pm.transitions(temporal_labels, bounds)
    pred_transitions = pm.transitions(temporal_predictions, bounds)
    transition_metrics = {}
    for tolerance in TRANSITION_TOLERANCES:
        values = pm.transition_scores(
            gt_transitions, pred_transitions, tolerance, typed="to"
        )
        transition_metrics[str(tolerance)] = {
            key: value
            for key, value in values.items()
            if key not in ("pairs", "missed", "spurious")
        }

    gt_spells = pm.possession_spells(temporal_labels, bounds)
    pred_spells = pm.possession_spells(temporal_predictions, bounds)
    mean_duration = lambda items: (
        float(np.mean([end - start for start, end, _ in items])) / source_fps
        if items
        else 0.0
    )
    per_iou = {f"{threshold:.1f}": values for threshold, values in spells["per_iou"].items()}
    return {
        "n_frames": int(len(labels)),
        "frame_metrics": {
            "accuracy": float(np.trace(confusion) / confusion.sum()),
            "macro_f1": float(np.mean(f1s)),
            "per_class": per_class,
            "confusion_matrix": confusion.tolist(),
        },
        "spell_metrics": {
            "mean_f1": spells["mean_f1"],
            "headline_iou": pm.SPELL_IOU_HEADLINE,
            "headline_f1": per_iou[f"{pm.SPELL_IOU_HEADLINE:.1f}"]["f1"],
            "n_ground_truth": spells["n_gt"],
            "n_predicted": spells["n_pred"],
            "per_iou": per_iou,
        },
        "possession_summary": {
            "ground_truth_count": len(gt_spells),
            "predicted_count": len(pred_spells),
            "ground_truth_mean_duration_seconds": mean_duration(gt_spells),
            "predicted_mean_duration_seconds": mean_duration(pred_spells),
        },
        "fragmentation": {
            "ground_truth_segments": gt_segments,
            "predicted_segments": pred_segments,
            "ratio": fragmentation,
        },
        "transition_metrics": transition_metrics,
    }


def _validate_coverage(
    runs: Sequence[HADRun], split: FoldSplit, partition: str, valid: np.ndarray
) -> None:
    expected = getattr(split, f"{partition}_mask") & valid
    coverage = np.zeros(len(valid), dtype=np.int8)
    for run in runs:
        coverage[run.start : run.end] += 1
    if not np.array_equal(coverage, expected.astype(np.int8)):
        raise ValueError(f"H/A/D predictions do not exactly cover valid {partition} labels")


def evaluate_had_checkpoint(
    path: str | Path,
    expected_fold: int,
    expected_model_type: str,
    expected_seed: int,
    partition: str,
    device_name: str = "auto",
    training_mode: str = "full_half_refit",
    validation_block: int | None = None,
) -> HADEvaluation:
    if partition not in ("train", "validation", "test"):
        raise ValueError("partition must be train, validation or test")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = validate_checkpoint(
        checkpoint,
        expected_fold,
        expected_model_type,
        expected_seed,
        training_mode,
        validation_block,
    )
    if config.model.classification_mode != "had":
        raise ValueError("The H/A/D evaluator requires classification_mode='had'")
    if training_mode == "full_half_refit" and partition == "validation":
        raise ValueError("A full-half refit has no validation partition")
    raw = load_wcsf_data(config.data.feature_path, config.data.label_path, "had")
    if training_mode == "full_half_refit":
        split = build_refit_split(raw.half_id, expected_fold)
    else:
        buffer_frames = config.seconds_to_frames(
            config.split.buffer_seconds, raw.source_fps
        )
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
    runs = tuple(
        run
        for segment in split.segments[partition]
        for run in _predict_valid_runs(model, data, segment, device)
    )
    _validate_coverage(runs, split, partition, raw.label_valid)
    metrics = compute_had_metrics(runs, raw.source_fps)
    report = {
        "schema_version": "wcsf_had_evaluation_v1",
        "evaluation_type": "single_fold",
        "fold": expected_fold,
        "partition": partition,
        "model_type": expected_model_type,
        "seed": expected_seed,
        "training_mode": training_mode,
        "validation_block": validation_block,
        "checkpoint": str(path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "class_order": list(CLASS_NAMES),
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
    return HADEvaluation(
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


def had_checkpoint_path(
    runs_dir: str | Path, fold: int, model_type: str, seed: int
) -> Path:
    return (
        Path(runs_dir)
        / f"fold{fold}_{model_type}_had_seed{seed}"
        / "refit"
        / "final_checkpoint.pt"
    )


def evaluate_had_oof(
    model_type: str, seed: int, runs_dir: str | Path, device_name: str = "auto"
) -> Dict[str, object]:
    fold1 = evaluate_had_checkpoint(
        had_checkpoint_path(runs_dir, 1, model_type, seed),
        1, model_type, seed, "test", device_name,
    )
    fold2 = evaluate_had_checkpoint(
        had_checkpoint_path(runs_dir, 2, model_type, seed),
        2, model_type, seed, "test", device_name,
    )
    if not np.array_equal(fold1.label_valid, fold2.label_valid):
        raise ValueError("OOF H/A/D checkpoints use different label masks")
    if {run.half_id for run in fold1.runs} != {2} or {
        run.half_id for run in fold2.runs
    } != {1}:
        raise ValueError("H/A/D OOF must use Fold 2 for H1 and Fold 1 for H2")
    runs = tuple(sorted((*fold2.runs, *fold1.runs), key=lambda run: run.start))
    coverage = np.zeros(fold1.n_frames_total, dtype=np.int8)
    for run in runs:
        coverage[run.start : run.end] += 1
    if not np.array_equal(coverage, fold1.label_valid.astype(np.int8)):
        raise ValueError("OOF H/A/D predictions must cover every valid frame once")
    return {
        "schema_version": "wcsf_had_evaluation_v1",
        "evaluation_type": "combined_oof",
        "model_type": model_type,
        "seed": seed,
        "class_order": list(CLASS_NAMES),
        "source_fps": fold1.source_fps,
        "fold_results": {"1": fold1.report, "2": fold2.report},
        "combined": {
            "runs": [
                {"start": run.start, "end": run.end, "half_id": run.half_id}
                for run in runs
            ],
            **compute_had_metrics(runs, fold1.source_fps),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=("1", "2", "all"))
    parser.add_argument("--model-type", required=True, choices=("team_tcn", "deepsets_tcn"))
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--inner-block", type=int, choices=(1, 2, 3))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=Path("data/open_data_groningen/model_runs"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.fold == "all":
        report = evaluate_had_oof(args.model_type, args.seed, args.runs_dir, args.device)
        output = args.runs_dir / f"oof_{args.model_type}_had_seed{args.seed}.json"
    else:
        fold = int(args.fold)
        training_mode = "inner_cv" if args.inner_block else "full_half_refit"
        if args.partition == "validation" and args.inner_block is None:
            parser.error("validation requires --inner-block")
        parent = args.runs_dir / f"fold{fold}_{args.model_type}_had_seed{args.seed}"
        path = (
            parent / f"inner_block{args.inner_block}" / "best_checkpoint.pt"
            if args.inner_block
            else parent / "refit" / "final_checkpoint.pt"
        )
        evaluation = evaluate_had_checkpoint(
            path, fold, args.model_type, args.seed, args.partition, args.device,
            training_mode, args.inner_block,
        )
        report = evaluation.report
        output = path.parent / f"evaluation_{args.partition}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
