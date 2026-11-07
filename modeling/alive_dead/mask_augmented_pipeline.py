"""Sequential baseline, mask-augmented training, robustness, and comparison CLI."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_engineering.robustness import file_sha256

from .config import load_config
from .evaluate_had import evaluate_had_oof
from .postprocess_had import run_experiment as select_decoders
from .robustness import METRIC_PATHS, run_experiment as run_robustness
from .runs import evaluation_dir, fold_checkpoint_path, load_run_metadata
from .train import train_run


SCHEMA_VERSION = "wcsf_mask_augmented_pipeline_v1"


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def _reject_ground_truth(value: Any, key: str = "") -> None:
    if "ground_truth" in key.lower() or key.lower() in {"gt", "metrica_labels"}:
        raise ValueError("Metrica competition ground truth is forbidden in this pipeline")
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            _reject_ground_truth(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            _reject_ground_truth(child, key)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        if "possession_gt" in normalized or "/evaluation/gt/" in f"/{normalized}":
            raise ValueError("Metrica competition ground truth is forbidden in this pipeline")


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    _reject_ground_truth(config)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported mask-augmented pipeline schema")
    required = {"baseline_robustness_config", "output_root", "training_runs"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Pipeline config is missing: {sorted(missing)}")
    runs = config["training_runs"]
    if len(runs) != 2 or {item.get("family") for item in runs} != {
        "without_transition_loss", "with_transition_loss_0_1"
    }:
        raise ValueError("Exactly the no-transition and transition-loss families are required")
    if len({item.get("name") for item in runs}) != 2:
        raise ValueError("Augmented model names must be unique")
    return config


def _new_manifest(config_path: Path, config: Mapping[str, Any], device: str) -> tuple[Path, dict[str, Any]]:
    created = datetime.now().astimezone()
    output = Path(config["output_root"]) / created.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created.isoformat(),
        "status": "running",
        "device_requested": device,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "metrica_ground_truth_used": False,
        "stages": {},
    }
    _dump(output / "pipeline_manifest.json", manifest)
    return output, manifest


def _load_resume(directory: Path, config_path: Path) -> dict[str, Any]:
    with (directory / "pipeline_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Resume directory has an incompatible manifest")
    if manifest["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("Pipeline configuration changed; refusing an unsafe resume")
    manifest["status"] = "running"
    return manifest


def _stage_complete(manifest: Mapping[str, Any], name: str) -> bool:
    stage = manifest.get("stages", {}).get(name, {})
    if stage.get("status") != "completed":
        return False
    required = [Path(path) for path in stage.get("required_artifacts", ())]
    if not required or not all(path.is_file() for path in required):
        return False
    hashes = stage.get("artifact_hashes", {})
    return all(hashes.get(str(path)) == file_sha256(path) for path in required)


def _record_stage(
    directory: Path,
    manifest: dict[str, Any],
    name: str,
    status: str,
    **values: Any,
) -> None:
    if status == "completed" and "required_artifacts" in values:
        values["artifact_hashes"] = {
            str(path): file_sha256(path) for path in map(Path, values["required_artifacts"])
        }
    manifest["stages"][name] = {
        **manifest["stages"].get(name, {}),
        **values,
        "status": status,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    _dump(directory / "pipeline_manifest.json", manifest)


def _selected_decoder(decoding_dir: Path) -> dict[str, Any]:
    with (decoding_dir / "inner_selection.json").open(encoding="utf-8") as handle:
        selection = json.load(handle)
    decoder = selection.get("best", {}).get("selected")
    if not isinstance(decoder, dict) or "method" not in decoder:
        raise ValueError("Decoder selection artifact has no global selected decoder")
    return decoder


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _condition_key(row: Mapping[str, Any]) -> tuple[str, float, str, str]:
    return (
        str(row["scenario"]),
        float(row["severity_factor"]),
        str(row["seed"]),
        str(row["output"]),
    )


def build_comparison_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    augmented_rows: Sequence[Mapping[str, Any]],
    pairs: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Build paired metrics; positive robustness gain means less augmented loss."""
    baseline_by_model: dict[str, dict[tuple[str, float, str, str], Mapping[str, Any]]] = {}
    augmented_by_model: dict[str, dict[tuple[str, float, str, str], Mapping[str, Any]]] = {}
    for row in baseline_rows:
        baseline_by_model.setdefault(str(row["model"]), {})[_condition_key(row)] = row
    for row in augmented_rows:
        augmented_by_model.setdefault(str(row["model"]), {})[_condition_key(row)] = row

    result: list[dict[str, Any]] = []
    for family, augmented_name in pairs.items():
        if family not in baseline_by_model or augmented_name not in augmented_by_model:
            raise ValueError(f"Missing robustness rows for pair {family} -> {augmented_name}")
        left = baseline_by_model[family]
        right = augmented_by_model[augmented_name]
        if set(left) != set(right):
            missing_augmented = sorted(set(left) - set(right))
            missing_baseline = sorted(set(right) - set(left))
            raise ValueError(
                "Baseline and augmented robustness conditions differ: "
                f"missing augmented={missing_augmented[:3]}, "
                f"missing baseline={missing_baseline[:3]}"
            )
        for key in sorted(left):
            baseline = left[key]
            augmented = right[key]
            clean_key = ("clean", 0.0, "", key[3])
            baseline_clean = left[clean_key]
            augmented_clean = right[clean_key]
            row: dict[str, Any] = {
                "family": family,
                "baseline_model": family,
                "augmented_model": augmented_name,
                "scenario": key[0],
                "severity_factor": key[1],
                "seed": key[2],
                "output": key[3],
            }
            for metric in METRIC_PATHS:
                base_value = float(baseline[metric])
                aug_value = float(augmented[metric])
                base_clean = float(baseline_clean[metric])
                aug_clean = float(augmented_clean[metric])
                base_degradation = base_value - base_clean
                aug_degradation = aug_value - aug_clean
                row[f"baseline_{metric}"] = base_value
                row[f"augmented_{metric}"] = aug_value
                row[f"augmented_minus_baseline_{metric}"] = aug_value - base_value
                row[f"baseline_degradation_{metric}"] = base_degradation
                row[f"augmented_degradation_{metric}"] = aug_degradation
                row[f"robustness_gain_{metric}"] = aug_degradation - base_degradation
            result.append(row)
    return result


def _aggregate_comparison(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["scenario"] == "clean":
            continue
        key = (
            str(row["family"]), str(row["scenario"]),
            float(row["severity_factor"]), str(row["output"]),
        )
        grouped.setdefault(key, []).append(row)
    result = []
    fields = [
        f"{prefix}_{metric}"
        for metric in METRIC_PATHS
        for prefix in (
            "augmented_minus_baseline", "baseline_degradation",
            "augmented_degradation", "robustness_gain",
        )
    ]
    for key, values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "family": key[0], "scenario": key[1], "severity_factor": key[2],
            "output": key[3], "n_seeds": len(values),
        }
        for field in fields:
            samples = np.asarray([float(value[field]) for value in values])
            row[f"{field}_mean"] = float(samples.mean())
            row[f"{field}_std"] = float(samples.std(ddof=0))
            row[f"{field}_p05"] = float(np.quantile(samples, 0.05))
            row[f"{field}_p95"] = float(np.quantile(samples, 0.95))
        result.append(row)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refusing to write an empty comparison")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _artifact_inventory(
    models: Sequence[Mapping[str, Any]],
    robustness_dirs: Sequence[Path],
    config_paths: Sequence[Path],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "models": [], "robustness_reports": [],
        "configuration_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in config_paths
        ],
        "robustness_models": [],
    }
    for model in models:
        run_dir = Path(model["run_dir"])
        result["models"].append({
            "name": model["name"],
            "run_dir": str(run_dir),
            "checkpoints": [
                {"path": str(fold_checkpoint_path(run_dir, fold)),
                 "sha256": file_sha256(fold_checkpoint_path(run_dir, fold))}
                for fold in (1, 2)
            ],
            "decoder_artifact": {
                "path": str(model["decoder_report"]),
                "sha256": file_sha256(model["decoder_report"]),
            },
            "decoder": model["decoder"],
        })
    for directory in robustness_dirs:
        with (directory / "manifest.json").open(encoding="utf-8") as handle:
            robustness_manifest = json.load(handle)
        result["robustness_models"].extend(robustness_manifest["models"])
        for name in ("manifest.json", "reports/per_condition_summary.csv", "reports/aggregate.json"):
            path = directory / name
            result["robustness_reports"].append(
                {"path": str(path), "sha256": file_sha256(path)}
            )
    return result


def run_pipeline(config_path: Path, device: str, resume: Path | None = None) -> Path:
    config = load_pipeline_config(config_path)
    if resume is None:
        directory, manifest = _new_manifest(config_path, config, device)
    else:
        directory = resume
        manifest = _load_resume(directory, config_path)
        _dump(directory / "pipeline_manifest.json", manifest)

    try:
        stage = "baseline_robustness"
        if not _stage_complete(manifest, stage):
            _record_stage(directory, manifest, stage, "running")
            with Path(config["baseline_robustness_config"]).open(encoding="utf-8") as handle:
                baseline_config = json.load(handle)
            _reject_ground_truth(baseline_config)
            baseline_dir = run_robustness(baseline_config, device)
            required = [baseline_dir / "manifest.json", baseline_dir / "reports/per_condition_summary.csv"]
            _record_stage(directory, manifest, stage, "completed", output_dir=str(baseline_dir), required_artifacts=[str(p) for p in required])
        baseline_dir = Path(manifest["stages"][stage]["output_dir"])

        augmented_models: list[dict[str, Any]] = []
        for specification in config["training_runs"]:
            training_stage = f"train_{specification['family']}"
            if not _stage_complete(manifest, training_stage):
                _record_stage(directory, manifest, training_stage, "running")
                training_config = load_config(specification["config"])
                training_config.training.device = device
                training_config.validate()
                if not training_config.augmentation.enabled:
                    raise ValueError("Pipeline training configs must enable mask augmentation")
                run_dir, _ = train_run(training_config, (1, 2), show_progress=True)
                required = [fold_checkpoint_path(run_dir, 1), fold_checkpoint_path(run_dir, 2)]
                _record_stage(
                    directory, manifest, training_stage, "completed",
                    run_dir=str(run_dir),
                    required_artifacts=[str(path) for path in required],
                )
            run_dir = Path(manifest["stages"][training_stage]["run_dir"])

            evaluation_stage = f"evaluate_{specification['family']}"
            if not _stage_complete(manifest, evaluation_stage):
                _record_stage(directory, manifest, evaluation_stage, "running")
                oof_path = evaluation_dir(run_dir) / "oof.json"
                _dump(oof_path, evaluate_had_oof(run_dir, device, show_progress=True))
                decoding_dir = select_decoders(run_dir, device_name=device, show_progress=True)
                decoder = _selected_decoder(decoding_dir)
                decoder_report = decoding_dir / "oof_comparison.json"
                required = [
                    fold_checkpoint_path(run_dir, 1), fold_checkpoint_path(run_dir, 2),
                    oof_path, decoding_dir / "inner_selection.json", decoder_report,
                ]
                _record_stage(
                    directory, manifest, evaluation_stage, "completed",
                    run_dir=str(run_dir), decoder=decoder,
                    decoder_report=str(decoder_report),
                    required_artifacts=[str(path) for path in required],
                )
            values = manifest["stages"][evaluation_stage]
            run_metadata = load_run_metadata(values["run_dir"], "had")
            if run_metadata.get("status") != "completed":
                raise ValueError(f"Augmented run is not complete: {values['run_dir']}")
            augmented_models.append({
                "family": specification["family"], "name": specification["name"],
                "run_dir": values["run_dir"], "decoder": values["decoder"],
                "decoder_report": values["decoder_report"],
            })

        stage = "augmented_robustness"
        if not _stage_complete(manifest, stage):
            _record_stage(directory, manifest, stage, "running")
            with Path(config["baseline_robustness_config"]).open(encoding="utf-8") as handle:
                augmented_config = json.load(handle)
            augmented_config["models"] = [
                {key: model[key] for key in ("name", "run_dir", "decoder", "decoder_report")}
                for model in augmented_models
            ]
            effective_config_path = directory / "configs/augmented_robustness.json"
            _dump(effective_config_path, augmented_config)
            augmented_dir = run_robustness(augmented_config, device)
            required = [
                effective_config_path, augmented_dir / "manifest.json",
                augmented_dir / "reports/per_condition_summary.csv",
            ]
            _record_stage(
                directory, manifest, stage, "completed",
                output_dir=str(augmented_dir), config_path=str(effective_config_path),
                required_artifacts=[str(p) for p in required],
            )
        augmented_dir = Path(manifest["stages"][stage]["output_dir"])

        stage = "comparison"
        if not _stage_complete(manifest, stage):
            _record_stage(directory, manifest, stage, "running")
            rows = build_comparison_rows(
                _read_rows(baseline_dir / "reports/per_condition_summary.csv"),
                _read_rows(augmented_dir / "reports/per_condition_summary.csv"),
                {model["family"]: model["name"] for model in augmented_models},
            )
            aggregate = _aggregate_comparison(rows)
            comparison_csv = directory / "reports/comparison.csv"
            aggregate_csv = directory / "reports/comparison_aggregate.csv"
            comparison_json = directory / "reports/comparison.json"
            _write_csv(comparison_csv, rows)
            _write_csv(aggregate_csv, aggregate)
            _dump(comparison_json, {"rows": rows, "aggregate": aggregate})
            required = [comparison_csv, aggregate_csv, comparison_json]
            _record_stage(directory, manifest, stage, "completed", required_artifacts=[str(p) for p in required])

        manifest["artifact_inventory"] = _artifact_inventory(
            augmented_models,
            (baseline_dir, augmented_dir),
            (
                config_path,
                Path(config["baseline_robustness_config"]),
                *(Path(item["config"]) for item in config["training_runs"]),
                Path(manifest["stages"]["augmented_robustness"]["config_path"]),
            ),
        )
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        _dump(directory / "pipeline_manifest.json", manifest)
        return directory
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        _dump(directory / "pipeline_manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    output = run_pipeline(args.config, args.device, args.resume)
    print(f"Mask-augmented pipeline completed: {output}")


if __name__ == "__main__":
    main()
