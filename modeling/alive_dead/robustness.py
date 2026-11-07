"""Audit feature shift and stress-test frozen WCSF H/A/D checkpoints.

Metrica is used only as an unlabeled feature archive.  This module intentionally
does not accept or discover a Metrica ground-truth path.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from evaluation import possession_metrics as pm
from feature_engineering.robustness import (
    CorruptionSpec,
    SCENARIOS,
    audit_feature_arrays,
    compare_feature_archives,
    corrupt_feature_arrays,
    derive_corruption_profile,
    file_sha256,
    load_feature_archive,
)

from .config import ExperimentConfig
from .dataset import WCSFData, load_wcsf_data, normalize_wcsf_data
from .evaluate import normalizer_from_state, select_device, validate_checkpoint
from .evaluate_had import CLASS_NAMES, compute_had_metrics
from .inference import predict_segment_logits
from .postprocess_had import Decoder, ProbabilityRun, decode_probability_runs
from .runs import fold_checkpoint_path, load_run_metadata
from .splits import build_refit_split, valid_subsegments
from .tcn import build_model, had_probabilities_from_logits


SCHEMA_VERSION = "wcsf_had_robustness_v1"
METRIC_PATHS = {
    "spell_f1": ("spell_metrics", "mean_f1"),
    "spell_f1_iou_0_5": ("spell_metrics", "headline_f1"),
    "macro_f1": ("frame_metrics", "macro_f1"),
    "accuracy": ("frame_metrics", "accuracy"),
    "strict_team_accuracy": ("ownership", "strict_team_accuracy"),
    "both_alive_team_accuracy": ("ownership", "both_alive_team_accuracy"),
    "predicted_home_share": ("ownership", "predicted_home_share"),
    "fragmentation": ("fragmentation", "ratio"),
    "transition_f1_5": ("transition_metrics", "5", "f1"),
    "transition_f1_10": ("transition_metrics", "10", "f1"),
}


@dataclass
class FrozenFold:
    fold: int
    model: torch.nn.Module
    config: ExperimentConfig
    normalizer: Any
    checkpoint_path: Path
    checkpoint_epoch: int


@dataclass
class FrozenModel:
    name: str
    run_dir: Path
    decoder: Decoder
    decoder_report: Path
    metadata: Mapping[str, Any]
    folds: dict[int, FrozenFold]


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported robustness config schema: {config.get('schema_version')!r}")

    def reject_ground_truth(value: Any, key: str = "") -> None:
        lowered = key.lower()
        if "ground_truth" in lowered or lowered in {"gt", "metrica_labels"}:
            raise ValueError("Metrica ground truth is forbidden in robustness configuration")
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                reject_ground_truth(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                reject_ground_truth(child, key)
        elif isinstance(value, str):
            normalized = value.replace("\\", "/").lower()
            if "possession_gt" in normalized or "/evaluation/gt/" in f"/{normalized}":
                raise ValueError("Competition ground truth is forbidden in robustness configuration")

    reject_ground_truth(config)
    required = {"wcsf_features", "wcsf_labels", "metrica_features", "output_root", "models"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Robustness config is missing: {sorted(missing)}")
    factors = config.get("severity_factors", [0.5, 1.0, 1.5])
    seeds = config.get("seeds", list(range(10)))
    if not factors or any(not np.isfinite(value) or value <= 0 for value in factors):
        raise ValueError("severity_factors must contain positive finite values")
    if not seeds or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds):
        raise ValueError("seeds must contain integers")
    return config


def _decoder_from_dict(value: Mapping[str, Any]) -> Decoder:
    method = str(value.get("method"))
    if method == "raw":
        return Decoder("raw")
    if method == "smoothing":
        return Decoder(method, int(value["window_frames"]))
    if method == "viterbi":
        return Decoder(method, float(value["transition_penalty"]))
    if method == "two_stage_viterbi":
        return Decoder(
            method,
            float(value["alive_transition_penalty"]),
            float(value["ownership_transition_penalty"]),
        )
    raise ValueError(
        "Robustness models support raw, smoothing, viterbi, or two_stage_viterbi decoders"
    )


def _load_frozen_models(
    specs: Sequence[Mapping[str, Any]], device: torch.device, expected_feature_path: Path
) -> list[FrozenModel]:
    models = []
    for spec in specs:
        run_dir = Path(spec["run_dir"])
        metadata = load_run_metadata(run_dir, "had")
        decoder = _decoder_from_dict(spec["decoder"])
        decoder_report = Path(spec["decoder_report"])
        if not decoder_report.is_file():
            raise FileNotFoundError(f"Decoder report not found: {decoder_report}")
        folds: dict[int, FrozenFold] = {}
        for fold in (1, 2):
            checkpoint_path = fold_checkpoint_path(run_dir, fold)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = validate_checkpoint(
                checkpoint,
                fold,
                str(metadata["model_type"]),
                int(metadata["seed"]),
                expected_training_mode="full_half_refit",
            )
            if config.model.classification_mode != "had" or not config.model.use_ball:
                raise ValueError("Robustness requires ball-enabled H/A/D checkpoints")
            if Path(config.data.feature_path).resolve() != expected_feature_path.resolve():
                raise ValueError("Checkpoint feature archive differs from robustness configuration")
            model = build_model(config.model).to(device)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            folds[fold] = FrozenFold(
                fold,
                model,
                config,
                normalizer_from_state(checkpoint["normalizer"]),
                checkpoint_path,
                int(checkpoint["epoch"]),
            )
        models.append(
            FrozenModel(
                str(spec["name"]), run_dir, decoder, decoder_report, metadata, folds
            )
        )
    return models


def _data_with_features(raw: WCSFData, features: Mapping[str, np.ndarray]) -> WCSFData:
    return replace(
        raw,
        players=features["players"],
        player_location_valid=features["player_location_valid"],
        player_speed_valid=features["player_speed_valid"],
        team_features=features["team_features"],
        team_feature_valid=features["team_feature_valid"],
        ball=features["ball"],
        ball_location_valid=features["ball_location_valid"],
        ball_speed_valid=features["ball_speed_valid"],
    )


def _infer_oof(model: FrozenModel, raw: WCSFData, features: Mapping[str, np.ndarray], device: torch.device) -> tuple[ProbabilityRun, ...]:
    runs: list[ProbabilityRun] = []
    corrupted = _data_with_features(raw, features)
    for fold in (1, 2):
        frozen = model.folds[fold]
        data = normalize_wcsf_data(corrupted, frozen.normalizer)
        split = build_refit_split(data.half_id, fold)
        for segment in split.segments["test"]:
            logits = predict_segment_logits(
                frozen.model,
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
            if logits.shape != (segment.length, 3) or not torch.isfinite(logits).all():
                raise ValueError("Frozen H/A/D checkpoint produced invalid logits")
            probabilities = had_probabilities_from_logits(
                logits, frozen.config.model.had_head_type
            ).to(torch.float32).cpu().numpy()
            if (
                not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
            ):
                raise ValueError("Frozen H/A/D checkpoint produced invalid probabilities")
            for valid in valid_subsegments(segment, data.label_valid):
                local = slice(valid.start - segment.start, valid.end - segment.start)
                runs.append(
                    ProbabilityRun(
                        valid.start,
                        valid.end,
                        valid.half_id,
                        data.labels[valid.start : valid.end].copy(),
                        probabilities[local].copy(),
                    )
                )
    runs.sort(key=lambda run: run.start)
    coverage = np.zeros(raw.n_frames, dtype=np.int8)
    for run in runs:
        coverage[run.start : run.end] += 1
    if not np.array_equal(coverage, raw.label_valid.astype(np.int8)):
        raise ValueError("OOF probabilities do not cover each valid WCSF label exactly once")
    if {run.half_id for run in runs} != {1, 2}:
        raise ValueError("OOF robustness predictions must cover both halves")
    return tuple(runs)


def enrich_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(metrics))
    confusion = np.asarray(result["frame_metrics"]["confusion_matrix"], dtype=np.int64)
    strict, both_alive = pm.team_accuracy(confusion)
    gt_share, pred_share, abs_error, defined = pm.possession_share(confusion)
    result["ownership"] = {
        "strict_team_accuracy": strict,
        "both_alive_team_accuracy": both_alive,
        "ground_truth_home_share": gt_share,
        "predicted_home_share": pred_share,
        "absolute_home_share_error": abs_error,
        "predicted_share_defined": bool(defined),
    }
    return result


def _evaluate_probability_runs(
    runs: Sequence[ProbabilityRun], decoder: Decoder, source_fps: float
) -> dict[str, Any]:
    return enrich_metrics(compute_had_metrics(decode_probability_runs(runs, decoder), source_fps))


def _assert_reference_metrics(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    sections = (
        "frame_metrics", "spell_metrics", "possession_summary",
        "fragmentation", "transition_metrics",
    )

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(right, Mapping):
            if not isinstance(left, Mapping):
                raise AssertionError(f"{label} baseline type mismatch at {path}")
            for key, value in right.items():
                if key not in left:
                    raise AssertionError(f"{label} baseline missing {path}.{key}")
                compare(left[key], value, f"{path}.{key}")
        elif isinstance(right, list):
            if left != right:
                raise AssertionError(f"{label} baseline mismatch at {path}")
        elif isinstance(right, (int, float)) and not isinstance(right, bool):
            if not np.isclose(float(left), float(right), rtol=1e-7, atol=1e-9):
                raise AssertionError(f"{label} baseline mismatch at {path}: {left} != {right}")
        elif left != right:
            raise AssertionError(f"{label} baseline mismatch at {path}")

    for section in sections:
        compare(actual[section], expected[section], section)


def _validate_clean_baseline(model: FrozenModel, raw_metrics: Mapping[str, Any], decoded_metrics: Mapping[str, Any]) -> None:
    with (model.run_dir / "evaluation" / "oof.json").open(encoding="utf-8") as handle:
        raw_reference = json.load(handle)["combined"]
    with model.decoder_report.open(encoding="utf-8") as handle:
        decoded_reference = json.load(handle)["approaches"]["selected"]["combined"]
    _assert_reference_metrics(raw_metrics, raw_reference, f"{model.name} raw")
    _assert_reference_metrics(decoded_metrics, decoded_reference, f"{model.name} decoded")


def _get_metric(metrics: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = metrics
    for key in path:
        value = value[key]
    return float(value)


def _flatten_result(
    model_name: str,
    scenario: str,
    severity: float,
    seed: int | None,
    output: str,
    metrics: Mapping[str, Any],
    clean: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "scenario": scenario,
        "severity_factor": severity,
        "seed": "" if seed is None else seed,
        "output": output,
    }
    for name, path in METRIC_PATHS.items():
        value = _get_metric(metrics, path)
        row[name] = value
        row[f"delta_{name}"] = value - _get_metric(clean, path)
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["scenario"] == "clean":
            continue
        key = (row["model"], row["scenario"], row["severity_factor"], row["output"])
        grouped.setdefault(key, []).append(row)
    aggregated = []
    numeric = [*METRIC_PATHS, *(f"delta_{name}" for name in METRIC_PATHS)]
    for key, values in sorted(grouped.items()):
        result = {
            "model": key[0], "scenario": key[1],
            "severity_factor": key[2], "output": key[3], "n_seeds": len(values),
        }
        for name in numeric:
            samples = np.asarray([float(row[name]) for row in values])
            result[f"{name}_mean"] = float(samples.mean())
            result[f"{name}_std"] = float(samples.std(ddof=0))
            result[f"{name}_p05"] = float(np.quantile(samples, 0.05))
            result[f"{name}_p95"] = float(np.quantile(samples, 0.95))
        aggregated.append(result)
    return aggregated


def _audit_payload(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    wcsf_path = Path(config["wcsf_features"])
    metrica_path = Path(config["metrica_features"])
    wcsf = load_feature_archive(wcsf_path)
    metrica = load_feature_archive(metrica_path)
    profile = derive_corruption_profile(wcsf, metrica)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "inputs": {
            "wcsf_features": str(wcsf_path), "wcsf_sha256": file_sha256(wcsf_path),
            "metrica_features": str(metrica_path), "metrica_sha256": file_sha256(metrica_path),
        },
        "wcsf": audit_feature_arrays(wcsf),
        "metrica": audit_feature_arrays(metrica),
        "distribution_comparison": compare_feature_archives(wcsf, metrica),
        "corruption_profile": profile,
    }
    return payload, wcsf, metrica, profile


def _create_output_dir(config: Mapping[str, Any], suffix: str) -> Path:
    root = Path(config["output_root"])
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    path = root / f"{timestamp}_{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_audit(config: Mapping[str, Any]) -> Path:
    output = _create_output_dir(config, "audit")
    audit, _, _, _ = _audit_payload(config)
    _json_dump(output / "audit.json", audit)
    comparison_rows = []
    for name, values in audit["distribution_comparison"].items():
        comparison_rows.append({
            "feature": name,
            "wcsf_mean": values["wcsf"]["mean"],
            "metrica_mean": values["metrica"]["mean"],
            **values["shift"],
        })
    _write_csv(output / "distribution_comparison.csv", comparison_rows)
    return output


def run_experiment(config: Mapping[str, Any], device_name: str) -> Path:
    output = _create_output_dir(config, "run")
    audit, wcsf_features, _, profile = _audit_payload(config)
    _json_dump(output / "audit" / "audit.json", audit)
    device = select_device(device_name)
    wcsf_path = Path(config["wcsf_features"])
    label_path = Path(config["wcsf_labels"])
    raw = load_wcsf_data(wcsf_path, label_path, "had", True, "direct")
    models = _load_frozen_models(config["models"], device, wcsf_path)
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    clean_metrics: dict[tuple[str, str], dict[str, Any]] = {}

    for model in models:
        print(f"Validating clean OOF baseline: {model.name}", flush=True)
        probability_runs = _infer_oof(model, raw, wcsf_features, device)
        for output_name, decoder in (("raw", Decoder("raw")), ("decoded", model.decoder)):
            metrics = _evaluate_probability_runs(probability_runs, decoder, raw.source_fps)
            clean_metrics[(model.name, output_name)] = metrics
        _validate_clean_baseline(
            model, clean_metrics[(model.name, "raw")], clean_metrics[(model.name, "decoded")]
        )
        for output_name in ("raw", "decoded"):
            metrics = clean_metrics[(model.name, output_name)]
            records.append({
                "model": model.name, "condition": {"scenario": "clean"},
                "output": output_name, "metrics": metrics,
            })
            rows.append(_flatten_result(model.name, "clean", 0.0, None, output_name, metrics, metrics))

    save_masks = bool(config.get("save_corruption_masks", False))
    conditions = [
        (scenario, severity, seed)
        for scenario in SCENARIOS
        for severity in config.get("severity_factors", [0.5, 1.0, 1.5])
        for seed in config.get("seeds", list(range(10)))
    ]
    for condition_index, (scenario, severity, seed) in enumerate(conditions, start=1):
        print(
            f"[{condition_index}/{len(conditions)}] {scenario} "
            f"severity={severity:g} seed={seed}",
            flush=True,
        )
        spec = CorruptionSpec(scenario, float(severity), int(seed))
        corrupted, realization, masks = corrupt_feature_arrays(
            wcsf_features, spec, profile
        )
        if save_masks:
            mask_dir = output / "corruption_masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                mask_dir / f"{scenario}_s{severity:g}_seed{seed}.npz",
                **masks,
            )
        for model in models:
            probability_runs = _infer_oof(model, raw, corrupted, device)
            for output_name, decoder in (
                ("raw", Decoder("raw")), ("decoded", model.decoder)
            ):
                metrics = _evaluate_probability_runs(
                    probability_runs, decoder, raw.source_fps
                )
                clean = clean_metrics[(model.name, output_name)]
                records.append({
                    "model": model.name,
                    "condition": realization,
                    "output": output_name,
                    "metrics": metrics,
                })
                rows.append(
                    _flatten_result(
                        model.name, scenario, float(severity), int(seed),
                        output_name, metrics, clean,
                    )
                )
    aggregate = _aggregate_rows(rows)
    _json_dump(output / "reports" / "per_condition_metrics.json", records)
    _write_csv(output / "reports" / "per_condition_summary.csv", rows)
    _json_dump(output / "reports" / "aggregate.json", aggregate)
    _write_csv(output / "reports" / "aggregate.csv", aggregate)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "inputs": {
            "wcsf_features": {"path": str(wcsf_path), "sha256": file_sha256(wcsf_path)},
            "wcsf_labels": {"path": str(label_path), "sha256": file_sha256(label_path)},
            "metrica_features": {
                "path": str(config["metrica_features"]),
                "sha256": file_sha256(config["metrica_features"]),
                "usage": "unlabeled feature audit and missingness profile only",
            },
        },
        "models": [
            {
                "name": model.name,
                "run_dir": str(model.run_dir),
                "decoder": model.decoder.to_dict(),
                "decoder_report": {
                    "path": str(model.decoder_report),
                    "sha256": file_sha256(model.decoder_report),
                },
                "checkpoints": {
                    str(fold): {
                        "path": str(value.checkpoint_path),
                        "sha256": file_sha256(value.checkpoint_path),
                        "epoch": value.checkpoint_epoch,
                    }
                    for fold, value in model.folds.items()
                },
            }
            for model in models
        ],
        "corruption_profile": profile,
        "severity_factors": config.get("severity_factors", [0.5, 1.0, 1.5]),
        "seeds": config.get("seeds", list(range(10))),
        "scenarios": list(SCENARIOS),
        "oof_mapping": "fold2 -> half1; fold1 -> half2",
        "normalization": "checkpoint normalizers restored unchanged for every condition",
        "metrica_ground_truth_used": False,
        "clean_baseline_validated": True,
    }
    _json_dump(output / "manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = _load_config(args.config)
    output = run_audit(config) if args.command == "audit" else run_experiment(config, args.device)
    print(f"Robustness artifacts written to {output}")


if __name__ == "__main__":
    main()
