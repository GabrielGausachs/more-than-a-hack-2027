"""Timestamped experiment-run naming, metadata and artifact paths."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import ExperimentConfig

RUN_SCHEMA_VERSION = "wcsf_temporal_run_v1"


def task_name(config: ExperimentConfig) -> str:
    return "had" if config.model.classification_mode == "had" else "alive_dead"


def _duration_token(seconds: float) -> str:
    return f"{seconds:g}".replace(".", "p")


def run_name(config: ExperimentConfig, created_at: datetime) -> str:
    """Return the stable human-readable identity for one experiment."""
    return (
        f"{task_name(config)}_rf{config.model.receptive_field_frames}"
        f"_seq{_duration_token(config.data.sequence_seconds)}"
        f"_{created_at.strftime('%Y%m%d_%H%M%S')}"
    )


def _write_metadata(run_dir: Path, metadata: Mapping[str, Any]) -> None:
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def build_run_metadata(
    config: ExperimentConfig,
    created_at: datetime,
    folds: Iterable[int],
    *,
    status: str = "running",
) -> dict[str, Any]:
    fps = 10.0
    sequence_frames = config.seconds_to_frames(config.data.sequence_seconds, fps)
    stride_frames = config.seconds_to_frames(config.data.train_stride_seconds, fps)
    buffer_frames = config.seconds_to_frames(config.split.buffer_seconds, fps)
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_name": run_name(config, created_at),
        "created_at": created_at.isoformat(),
        "status": status,
        "task": task_name(config),
        "model_type": config.model.model_type,
        "seed": config.training.seed,
        "folds_requested": list(folds),
        "folds_completed": [],
        "class_order": ["H", "A", "D"]
        if config.model.classification_mode == "had"
        else ["DEAD", "ALIVE"],
        "temporal": {
            "source_fps": fps,
            "sequence_seconds": config.data.sequence_seconds,
            "sequence_frames": sequence_frames,
            "train_stride_seconds": config.data.train_stride_seconds,
            "train_stride_frames": stride_frames,
            "receptive_field_frames": config.model.receptive_field_frames,
            "context_radius_frames": config.model.context_radius_frames,
            "buffer_seconds": config.split.buffer_seconds,
            "buffer_frames": buffer_frames,
        },
        "methodology": {
            "outer_folds": "H1 train/refit -> H2 test; H2 train/refit -> H1 test",
            "inner_validation": "three purged contiguous 15% validation blocks",
            "training_supervision": "receptive-field-aware center-only window loss",
            "normalization": "fit on training frames only; saved with each checkpoint",
            "validation_inference": "one prediction per frame from contiguous segments",
            "refit": "complete outer training half for median inner best epoch",
        },
        "config": config.to_dict(),
    }


def create_run_directory(
    config: ExperimentConfig,
    folds: Iterable[int],
    created_at: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create a unique run directory and its initial manifest."""
    created_at = created_at or datetime.now().astimezone()
    folds = tuple(folds)
    run_dir = Path(config.training.output_dir) / run_name(config, created_at)
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = build_run_metadata(config, created_at, folds)
    _write_metadata(run_dir, metadata)
    return run_dir, metadata


def load_run_metadata(run_dir: str | Path, expected_task: str | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    path = run_dir / "run_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {path}")
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("Unsupported run metadata schema")
    if metadata.get("run_name") != run_dir.name:
        raise ValueError("Run directory name disagrees with run metadata")
    if expected_task is not None and metadata.get("task") != expected_task:
        raise ValueError(
            f"Run task is {metadata.get('task')!r}; expected {expected_task!r}"
        )
    if metadata.get("model_type") not in ("team_tcn", "deepsets_tcn"):
        raise ValueError("Run metadata contains an invalid model type")
    return metadata


def update_run_status(
    run_dir: str | Path,
    metadata: dict[str, Any],
    *,
    status: str,
    completed_fold: int | None = None,
) -> None:
    if completed_fold is not None and completed_fold not in metadata["folds_completed"]:
        metadata["folds_completed"].append(completed_fold)
        metadata["folds_completed"].sort()
    metadata["status"] = status
    metadata["updated_at"] = datetime.now().astimezone().isoformat()
    _write_metadata(Path(run_dir), metadata)


def fold_dir(run_dir: str | Path, fold: int) -> Path:
    if fold not in (1, 2):
        raise ValueError("fold must be 1 or 2")
    return Path(run_dir) / f"fold{fold}"


def fold_checkpoint_path(run_dir: str | Path, fold: int) -> Path:
    return fold_dir(run_dir, fold) / "refit" / "final_checkpoint.pt"


def inner_checkpoint_path(run_dir: str | Path, fold: int, block: int) -> Path:
    if block not in (1, 2, 3):
        raise ValueError("block must be 1, 2 or 3")
    return fold_dir(run_dir, fold) / f"inner_block{block}" / "best_checkpoint.pt"


def evaluation_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "evaluation"
    path.mkdir(parents=True, exist_ok=True)
    return path
