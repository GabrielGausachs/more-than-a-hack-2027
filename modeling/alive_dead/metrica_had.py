"""Frozen WCSF H/A/D transfer inference and public Metrica evaluation.

The public ground truth is loaded only after inference and decoding are complete. Decoder
parameters come from WCSF inner-validation artifacts and are never selected on Metrica.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from evaluation import possession_metrics as pm

from .config import ExperimentConfig
from .evaluate import (
    load_metrica_features,
    normalizer_from_state,
    select_device,
    validate_checkpoint,
    validate_transfer_configs,
)
from .inference import predict_segment_logits
from .postprocess_had import Decoder, viterbi_decode, two_stage_viterbi_decode
from .progress import progress
from .runs import fold_checkpoint_path, load_run_metadata
from .splits import Segment
from .tcn import build_model, had_probabilities_from_logits


SCHEMA_VERSION = "metrica_had_transfer_v1"
CONFIG_SCHEMA_VERSION = "metrica_had_model_comparison_v1"
CLASS_NAMES = np.array(["H", "A", "D"])
GLOBAL_SELECTION_SCOPE = "one global decoder across six inner-validation blocks"
SUPPORTED_DECODER_SCHEMAS = {
    "wcsf_had_temporal_decoding_v2",
    "wcsf_had_temporal_decoding_v3",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    run_dir: Path
    decoder_selection: Path
    expected_transition_loss_weight: float
    expected_decoder: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_comparison_config(path: str | Path) -> tuple[tuple[ModelSpec, ...], dict[int, float]]:
    """Load the explicit two-model comparison contract."""
    path = Path(path)
    raw = _read_json(path)
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported Metrica H/A/D comparison config schema")
    weights_raw = raw.get("ensemble_weights", {})
    weights = {1: float(weights_raw.get("fold1", np.nan)), 2: float(weights_raw.get("fold2", np.nan))}
    if not all(np.isfinite(value) and value >= 0 for value in weights.values()) or not np.isclose(
        sum(weights.values()), 1.0
    ):
        raise ValueError("Fold ensemble weights must be finite, non-negative and sum to one")
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Comparison config must define at least one model")
    specs = []
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("Each model specification must be an object")
        name = str(item.get("name", ""))
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name):
            raise ValueError(f"Invalid model name: {name!r}")
        specs.append(
            ModelSpec(
                name=name,
                run_dir=Path(item["run_dir"]),
                decoder_selection=Path(item["decoder_selection"]),
                expected_transition_loss_weight=float(item["expected_transition_loss_weight"]),
                expected_decoder=dict(item["expected_decoder"]),
            )
        )
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Model names must be unique")
    return tuple(specs), weights


def decoder_from_dict(value: Mapping[str, Any]) -> Decoder:
    """Construct a validated decoder from its serialized representation."""
    method = value.get("method")
    if method == "raw":
        return Decoder("raw")
    if method == "smoothing":
        return Decoder("smoothing", int(value["window_frames"]))
    if method == "viterbi":
        return Decoder("viterbi", float(value["transition_penalty"]))
    if method == "two_stage_viterbi":
        return Decoder(
            "two_stage_viterbi",
            float(value["alive_transition_penalty"]),
            float(value["ownership_transition_penalty"]),
        )
    raise ValueError(f"Unsupported decoder method: {method!r}")


def load_global_decoder(path: str | Path, expected: Mapping[str, Any]) -> tuple[Decoder, dict[str, Any]]:
    """Load a decoder selected globally on all six WCSF inner-validation blocks."""
    path = Path(path)
    report = _read_json(path)
    if report.get("schema_version") not in SUPPORTED_DECODER_SCHEMAS:
        raise ValueError("Decoder selection must use leakage-safe schema v2 or v3")
    if report.get("selection_scope") != GLOBAL_SELECTION_SCOPE:
        raise ValueError("Decoder was not selected globally across six inner-validation blocks")
    selected = report.get("best", {}).get("selected")
    if not isinstance(selected, dict):
        raise ValueError("Decoder selection report has no best.selected decoder")
    if selected != dict(expected):
        raise ValueError(f"Selected decoder {selected} does not match expected {dict(expected)}")
    return decoder_from_dict(selected), report


def average_categorical_probabilities(
    probabilities: Mapping[int, np.ndarray], weights: Mapping[int, float]
) -> np.ndarray:
    """Return an aligned weighted mean of categorical fold probabilities."""
    if set(probabilities) != set(weights) or not probabilities:
        raise ValueError("Fold probabilities and weights must have identical non-empty keys")
    arrays = {fold: np.asarray(values, dtype=np.float32) for fold, values in probabilities.items()}
    reference_shape = next(iter(arrays.values())).shape
    if len(reference_shape) != 2 or reference_shape[1] != 3:
        raise ValueError("Fold probabilities must have shape (N, 3)")
    for values in arrays.values():
        if values.shape != reference_shape or not np.isfinite(values).all():
            raise ValueError("Fold probabilities must be finite and aligned")
        if np.any(values < 0.0) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Fold probabilities must be categorical rows")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Fold weights must sum to one")
    result = sum(float(weights[fold]) * values for fold, values in arrays.items())
    result = np.asarray(result, dtype=np.float32)
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Ensembled probability rows do not sum to one")
    return result


def _metrica_half_segments(half_id: np.ndarray) -> tuple[Segment, ...]:
    half_id = np.asarray(half_id)
    segments = []
    for half in (1, 2):
        indices = np.flatnonzero(half_id == half)
        if not len(indices) or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"Metrica half {half} must be one contiguous sequence")
        segments.append(Segment(int(indices[0]), int(indices[-1] + 1), "test", half))
    if sum(segment.length for segment in segments) != len(half_id):
        raise ValueError("Every Metrica sample must belong to exactly one half")
    return tuple(segments)


def _validate_model_metadata(metadata: Mapping[str, Any], spec: ModelSpec) -> None:
    if metadata.get("status") != "completed" or set(metadata.get("folds_completed", ())) != {1, 2}:
        raise ValueError(f"{spec.name} must be a completed two-fold run")
    if metadata.get("task") != "had" or metadata.get("class_order") != ["H", "A", "D"]:
        raise ValueError(f"{spec.name} is not an H/A/D run with class order H,A,D")
    training = metadata.get("config", {}).get("training", {})
    transition_enabled = bool(training.get("use_transition_loss", False))
    configured = (
        float(training.get("transition_loss_weight", 0.1))
        if transition_enabled
        else 0.0
    )
    if not np.isclose(configured, spec.expected_transition_loss_weight):
        raise ValueError(
            f"{spec.name} transition-loss weight {configured} does not match "
            f"{spec.expected_transition_loss_weight}"
        )


def infer_metrica_checkpoint(
    path: str | Path,
    fold: int,
    metadata: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, ExperimentConfig, Mapping[str, Any]]:
    """Apply one frozen H/A/D refit checkpoint to both complete Metrica halves."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = validate_checkpoint(
        checkpoint,
        fold,
        str(metadata["model_type"]),
        int(metadata["seed"]),
        expected_training_mode="full_half_refit",
    )
    if config.model.classification_mode != "had":
        raise ValueError("Metrica H/A/D transfer requires H/A/D checkpoints")
    expected_head = metadata.get("had_head_type") or metadata.get("config", {}).get("model", {}).get(
        "had_head_type", "direct"
    )
    if config.model.had_head_type != expected_head:
        raise ValueError("Checkpoint H/A/D head disagrees with run metadata")
    if float(checkpoint.get("source_fps", features["source_fps"])) != float(features["source_fps"]):
        raise ValueError("Checkpoint and Metrica model frequencies disagree")

    normalizer = normalizer_from_state(checkpoint["normalizer"])
    players, team_features = normalizer.transform(
        features["players"],
        features["player_location_valid"],
        features["player_speed_valid"],
        features["team_features"],
        features["team_feature_valid"],
    )
    ball = ball_location_valid = ball_speed_valid = None
    if config.model.use_ball:
        ball = normalizer.transform_ball(
            features["ball"], features["ball_location_valid"], features["ball_speed_valid"]
        )
        ball_location_valid = features["ball_location_valid"]
        ball_speed_valid = features["ball_speed_valid"]
    model = build_model(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    probabilities = np.empty((len(features["sample_index"]), 3), dtype=np.float32)
    coverage = np.zeros(len(probabilities), dtype=np.int8)
    for segment in _metrica_half_segments(features["half_id"]):
        logits = predict_segment_logits(
            model,
            players,
            features["player_location_valid"],
            features["player_speed_valid"],
            team_features,
            features["team_feature_valid"],
            segment,
            device,
            ball=ball,
            ball_location_valid=ball_location_valid,
            ball_speed_valid=ball_speed_valid,
        )
        if logits.shape != (segment.length, 3) or not torch.isfinite(logits).all():
            raise ValueError("H/A/D checkpoint produced invalid Metrica logits")
        values = had_probabilities_from_logits(logits, config.model.had_head_type)
        probabilities[segment.start : segment.end] = values.to(torch.float32).cpu().numpy()
        coverage[segment.start : segment.end] += 1
    if not np.all(coverage == 1) or not np.isfinite(probabilities).all():
        raise ValueError("Metrica inference did not cover every 10 Hz sample exactly once")
    if np.any(probabilities < 0.0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Metrica inference produced invalid categorical probabilities")
    return probabilities, config, checkpoint


def decode_probabilities_by_half(
    probabilities: np.ndarray, half_id: np.ndarray, decoder: Decoder
) -> np.ndarray:
    """Decode each half independently at the model's native 10 Hz frequency."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or len(probabilities) != len(half_id):
        raise ValueError("Probabilities and half IDs are not aligned")
    predictions = np.empty(len(probabilities), dtype=np.int8)
    for segment in _metrica_half_segments(half_id):
        values = probabilities[segment.start : segment.end]
        if decoder.method == "raw":
            decoded = values.argmax(axis=1).astype(np.int8)
        elif decoder.method == "smoothing":
            from .postprocess_had import smooth_probabilities

            decoded = smooth_probabilities(values, int(decoder.parameter)).argmax(axis=1).astype(np.int8)
        elif decoder.method == "viterbi":
            decoded = viterbi_decode(values, float(decoder.parameter))
        elif decoder.method == "two_stage_viterbi":
            decoded = two_stage_viterbi_decode(
                values, float(decoder.parameter), float(decoder.secondary_parameter)
            )
        else:  # Decoder validation should make this unreachable.
            raise ValueError(f"Unsupported decoder: {decoder.method}")
        predictions[segment.start : segment.end] = decoded
    return predictions


def reconstruct_source_labels(
    labels_10hz: np.ndarray, source_to_sample_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map H/A/D samples to the source timeline and write X outside scored halves."""
    labels_10hz = np.asarray(labels_10hz, dtype=np.int8)
    reverse = np.asarray(source_to_sample_index, dtype=np.int64)
    if labels_10hz.ndim != 1 or not np.isin(labels_10hz, (pm.H, pm.A, pm.D)).all():
        raise ValueError("10 Hz labels must be a non-empty H/A/D vector")
    valid = reverse >= 0
    if np.any(reverse[valid] >= len(labels_10hz)):
        raise ValueError("Source alignment refers to a missing 10 Hz label")
    result = np.full(len(reverse), pm.X, dtype=np.int8)
    result[valid] = labels_10hz[reverse[valid]]
    return result, valid


def reconstruct_source_probabilities(
    probabilities_10hz: np.ndarray, source_to_sample_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map categorical probabilities to the source timeline for diagnostics."""
    values = np.asarray(probabilities_10hz, dtype=np.float32)
    reverse = np.asarray(source_to_sample_index, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("10 Hz probabilities must have shape (N, 3)")
    valid = reverse >= 0
    if np.any(reverse[valid] >= len(values)):
        raise ValueError("Source alignment refers to a missing probability row")
    result = np.zeros((len(reverse), 3), dtype=np.float32)
    result[valid] = values[reverse[valid]]
    return result, valid


def _submission_frame(labels: np.ndarray) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=np.int8)
    if labels.ndim != 1 or not np.isin(labels, (pm.H, pm.A, pm.D, pm.X)).all():
        raise ValueError("Submission labels must be a one-dimensional H/A/D/X vector")
    names = np.array(["H", "A", "D", "X"])
    return pd.DataFrame(
        {"frame": np.arange(len(labels), dtype=np.int64), "label": names[labels]}
    )


def write_submission(path: str | Path, labels: np.ndarray) -> Path:
    path = Path(path)
    frame = _submission_frame(labels)
    problem = pm.validate_submission(frame, len(labels))
    if problem is not None:
        raise ValueError(f"Generated submission is invalid: {problem}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _json_safe_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    output = {key: value for key, value in result.items() if key != "transitions"}
    output["transitions"] = {
        str(tolerance): {
            key: value
            for key, value in transition.items()
            if key not in {"pairs", "missed", "spurious"}
        }
        for tolerance, transition in result["transitions"].items()
    }
    # Match the official JSON report's stringified numeric dictionary keys exactly.
    return json.loads(json.dumps(output))


def evaluate_labels(gt_codes: np.ndarray, labels: np.ndarray, bootstrap_n: int) -> dict[str, Any]:
    return _json_safe_metrics(pm.evaluate(gt_codes, labels, bootstrap_n=bootstrap_n))


def _comparison_row(name: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "submission": name,
        "spell_f1_mean": metrics["spell_f1_mean"],
        "macro_f1": metrics["macro_f1"],
        "spell_f1_at_0.5": metrics["spell_f1_at_headline"],
        "accuracy": metrics["accuracy"],
        "alive_dead_f1": metrics["alive_dead_f1"],
        "team_acc_gt_alive": metrics["team_acc_gt_alive"],
        "team_acc_both_alive": metrics["team_acc_both_alive"],
        "n_pred_spells": metrics["n_pred_spells"],
        "mean_spell_duration_s": metrics["mean_spell_duration_s"],
        "fragmentation": metrics["fragmentation"],
        "transition_f1_12": metrics["transitions"]["12"]["f1"],
        "transition_f1_25": metrics["transitions"]["25"]["f1"],
    }


def _write_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assert_official_reports_match(
    direct_reports: Mapping[str, Mapping[str, Any]], official_dir: Path
) -> None:
    keys = (
        "macro_f1",
        "spell_f1_mean",
        "spell_f1_at_headline",
        "accuracy",
        "kappa",
        "alive_dead_f1",
        "team_acc_gt_alive",
        "team_acc_both_alive",
        "n_pred_spells",
        "fragmentation",
        "confusion",
        "per_class",
        "spell_f1",
        "transitions",
    )
    for name, direct in direct_reports.items():
        official = _read_json(official_dir / "reports" / f"{name}.json")
        for key in keys:
            if official[key] != direct[key]:
                raise ValueError(f"Official scorer disagrees with direct report for {name}: {key}")


def _run_official_scorer(
    submissions_dir: Path,
    output_dir: Path,
    ground_truth: Path,
    bootstrap_n: int,
    direct_reports: Mapping[str, Mapping[str, Any]],
) -> None:
    command = [
        sys.executable,
        str(Path("evaluation/run_evaluation.py")),
        "--gt",
        str(ground_truth),
        "--results",
        str(submissions_dir),
        "--out",
        str(output_dir),
        "--bootstrap",
        str(bootstrap_n),
        "--no-detail",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scoreboard.txt").write_text(
        completed.stdout + ("\nSTDERR:\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Official evaluator failed; see {output_dir / 'scoreboard.txt'}")
    _assert_official_reports_match(direct_reports, output_dir)


def _load_ground_truth(path: Path, n_frames: int) -> np.ndarray:
    frame = pm.read_label_csv(path)
    problem = pm.validate_submission(frame, n_frames)
    if problem is not None:
        raise ValueError(f"Invalid Metrica public ground truth: {problem}")
    codes = pm.label_codes(frame)
    if int(np.count_nonzero(codes != pm.X)) != 7500:
        raise ValueError("Expected exactly 7,500 public ground-truth frames")
    return codes


def run_comparison(
    config_path: str | Path,
    feature_path: str | Path,
    ground_truth_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    bootstrap_n: int = 1000,
    show_progress: bool = True,
) -> Path:
    """Run configured frozen models, write submissions and verify official metrics."""
    config_path = Path(config_path)
    feature_path = Path(feature_path)
    ground_truth_path = Path(ground_truth_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    submissions_dir = output_dir / "submissions"
    predictions_dir = output_dir / "predictions"
    reports_dir = output_dir / "reports"
    for directory in (submissions_dir, predictions_dir, reports_dir):
        directory.mkdir()

    specs, weights = load_comparison_config(config_path)
    features = load_metrica_features(feature_path)
    n_source = int(features["original_frame_count"])
    gt_codes = _load_ground_truth(ground_truth_path, n_source)
    device = select_device(device_name)
    direct_reports: dict[str, dict[str, Any]] = {}
    comparison_rows = []
    manifest_models = []

    for spec in progress(specs, enabled=show_progress, total=len(specs), desc="Metrica models", unit="model"):
        metadata = load_run_metadata(spec.run_dir, "had")
        _validate_model_metadata(metadata, spec)
        decoder, decoder_report = load_global_decoder(spec.decoder_selection, spec.expected_decoder)
        fold_probabilities: dict[int, np.ndarray] = {}
        configs: dict[int, ExperimentConfig] = {}
        checkpoints: dict[int, Mapping[str, Any]] = {}
        checkpoint_paths = {fold: fold_checkpoint_path(spec.run_dir, fold) for fold in (1, 2)}
        for fold in (1, 2):
            values, checkpoint_config, checkpoint = infer_metrica_checkpoint(
                checkpoint_paths[fold], fold, metadata, features, device
            )
            fold_probabilities[fold] = values
            configs[fold] = checkpoint_config
            checkpoints[fold] = checkpoint
        validate_transfer_configs(configs[1], configs[2])
        ensemble = average_categorical_probabilities(fold_probabilities, weights)
        source_probabilities, probability_valid = reconstruct_source_probabilities(
            ensemble, features["source_to_sample_index"]
        )

        variants: dict[str, dict[str, Any]] = {}
        fold_diagnostics: dict[str, Any] = {}
        for fold, probabilities in fold_probabilities.items():
            fold_diagnostics[str(fold)] = {}
            for variant, selected_decoder in (("raw", Decoder("raw")), ("decoded", decoder)):
                labels_10hz = decode_probabilities_by_half(
                    probabilities, features["half_id"], selected_decoder
                )
                labels_25hz, _ = reconstruct_source_labels(
                    labels_10hz, features["source_to_sample_index"]
                )
                fold_diagnostics[str(fold)][variant] = evaluate_labels(
                    gt_codes, labels_25hz, bootstrap_n
                )

        for variant, selected_decoder in (("raw", Decoder("raw")), ("decoded", decoder)):
            labels_10hz = decode_probabilities_by_half(
                ensemble, features["half_id"], selected_decoder
            )
            labels_25hz, prediction_valid = reconstruct_source_labels(
                labels_10hz, features["source_to_sample_index"]
            )
            if not np.array_equal(prediction_valid, probability_valid):
                raise ValueError("Probability and label source coverage disagree")
            if not np.all(prediction_valid[gt_codes != pm.X]):
                raise ValueError("Not every public ground-truth frame has a prediction")
            submission_name = f"{spec.name}_{variant}"
            submission_path = write_submission(submissions_dir / f"{submission_name}.csv", labels_25hz)
            metrics = evaluate_labels(gt_codes, labels_25hz, bootstrap_n)
            direct_reports[submission_name] = metrics
            comparison_rows.append(_comparison_row(submission_name, metrics))
            variants[variant] = {
                "decoder": selected_decoder.to_dict(),
                "submission": str(submission_path),
                "labels_10hz": labels_10hz,
                "labels_25hz": labels_25hz,
                "metrics": metrics,
            }

        np.savez_compressed(
            predictions_dir / f"{spec.name}.npz",
            fold1_probabilities_10hz=fold_probabilities[1],
            fold2_probabilities_10hz=fold_probabilities[2],
            ensemble_probabilities_10hz=ensemble,
            ensemble_probabilities_25hz=source_probabilities,
            raw_labels_10hz=variants["raw"]["labels_10hz"],
            decoded_labels_10hz=variants["decoded"]["labels_10hz"],
            raw_labels_25hz=variants["raw"]["labels_25hz"],
            decoded_labels_25hz=variants["decoded"]["labels_25hz"],
            prediction_valid_25hz=probability_valid,
            sample_index=features["sample_index"],
            half_id=features["half_id"],
            source_frame_index=features["source_frame_index"],
            source_to_sample_index=features["source_to_sample_index"],
            source_half_id=features["source_half_id"],
        )
        model_report = {
            "schema_version": SCHEMA_VERSION,
            "model": spec.name,
            "source_run_name": metadata["run_name"],
            "ensemble_weights": {f"fold{fold}": weights[fold] for fold in (1, 2)},
            "selected_decoder": decoder.to_dict(),
            "public_window_warning": (
                "Diagnostics cover only 7,500 frames and 17 possessions; they must not select "
                "models, ensemble weights or decoder parameters."
            ),
            "fold_diagnostics": fold_diagnostics,
            "ensemble": {
                variant: {
                    "decoder": values["decoder"],
                    "submission": values["submission"],
                    "metrics": values["metrics"],
                }
                for variant, values in variants.items()
            },
        }
        (reports_dir / f"{spec.name}.json").write_text(
            json.dumps(model_report, indent=2), encoding="utf-8"
        )
        manifest_models.append(
            {
                "name": spec.name,
                "run_dir": str(spec.run_dir),
                "run_name": metadata["run_name"],
                "decoder_selection": str(spec.decoder_selection),
                "decoder_selection_sha256": _sha256(spec.decoder_selection),
                "decoder_schema": decoder_report["schema_version"],
                "decoder": decoder.to_dict(),
                "checkpoints": {
                    f"fold{fold}": {
                        "path": str(checkpoint_paths[fold]),
                        "sha256": _sha256(checkpoint_paths[fold]),
                        "epoch": int(checkpoints[fold]["epoch"]),
                    }
                    for fold in (1, 2)
                },
            }
        )

    comparison_rows.sort(key=lambda row: (-row["spell_f1_mean"], -row["macro_f1"], row["submission"]))
    _write_comparison_csv(reports_dir / "comparison.csv", comparison_rows)
    (reports_dir / "comparison.json").write_text(
        json.dumps(comparison_rows, indent=2), encoding="utf-8"
    )

    official_dir = output_dir / "official"
    _run_official_scorer(
        submissions_dir, official_dir, ground_truth_path, bootstrap_n, direct_reports
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "comparison_config": str(config_path),
        "comparison_config_sha256": _sha256(config_path),
        "feature_archive": str(feature_path),
        "feature_archive_sha256": _sha256(feature_path),
        "ground_truth": str(ground_truth_path),
        "ground_truth_sha256": _sha256(ground_truth_path),
        "bootstrap_draws": bootstrap_n,
        "model_source_fps": float(features["source_fps"]),
        "evaluation_source_fps": float(features["original_source_fps"]),
        "n_model_samples": int(len(features["sample_index"])),
        "n_source_frames": n_source,
        "n_public_frames": int(np.count_nonzero(gt_codes != pm.X)),
        "scored_frame_intervals": features["scored_frame_intervals"].tolist(),
        "public_window": [int(np.flatnonzero(gt_codes != pm.X)[0]), int(np.flatnonzero(gt_codes != pm.X)[-1])],
        "ensemble_weights": {f"fold{fold}": weights[fold] for fold in (1, 2)},
        "models": manifest_models,
        "official_scorer_verified": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/alive_dead/metrica_best_models.json"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/metrica/processed/metrica_player_team_ball_features.npz"),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("evaluation/gt/BAR-ATH_possession_gt_public.csv"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be non-negative")
    output_dir = args.output_dir or Path("evaluation/metrica_had_runs") / datetime.now().astimezone().strftime(
        "%Y%m%d_%H%M%S"
    )
    result = run_comparison(
        args.config,
        args.features,
        args.ground_truth,
        output_dir,
        device_name=args.device,
        bootstrap_n=args.bootstrap,
        show_progress=not args.no_progress,
    )
    with (result / "reports" / "comparison.json").open(encoding="utf-8") as handle:
        comparison = json.load(handle)
    print("Metrica public-window comparison (17 possessions; diagnostics only):")
    for row in comparison:
        print(
            f"  {row['submission']:40s} spellF1={row['spell_f1_mean']:.4f} "
            f"macroF1={row['macro_f1']:.4f} possessions={row['n_pred_spells']}"
        )
    print(f"Saved: {result}")


if __name__ == "__main__":
    main()
