"""Purged three-block cross-validation and full-half refit training."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import ExperimentConfig, load_config
from .dataset import LoaderBundle, WCSFData, build_dataloaders
from .evaluate import evaluate_checkpoint, select_device
from .inference import predict_segment_logits
from .progress import progress, progress_write
from .runs import create_run_directory, fold_dir, update_run_status
from .splits import Segment
from .tcn import build_model


@dataclass(frozen=True)
class EpochStats:
    loss: float
    effective_frames: float


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def masked_bce_components(
    logits: torch.Tensor,
    labels: torch.Tensor,
    supervision_mask: torch.Tensor,
    loss_weight: torch.Tensor,
    label_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted BCE numerator and denominator for valid outputs."""
    if logits.shape != labels.shape or logits.shape != supervision_mask.shape:
        raise ValueError("logits, labels and supervision_mask must have identical shapes")
    if label_valid.shape != logits.shape:
        raise ValueError("label_valid must have the same shape as logits")
    effective_weight = (
        supervision_mask.to(logits.dtype)
        * label_valid.to(logits.dtype)
        * loss_weight.to(logits.dtype)
    )
    denominator = effective_weight.sum()
    if denominator.item() <= 0:
        raise ValueError("A batch must contain at least one valid output")
    safe_labels = torch.where(label_valid, labels, torch.zeros_like(labels))
    element_loss = F.binary_cross_entropy_with_logits(
        logits, safe_labels, reduction="none"
    )
    return (element_loss * effective_weight).sum(), denominator


def masked_cross_entropy_components(
    logits: torch.Tensor,
    labels: torch.Tensor,
    supervision_mask: torch.Tensor,
    loss_weight: torch.Tensor,
    label_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted H/A/D cross-entropy over supervised outputs only."""
    if logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError("H/A/D logits must have shape (B, T, 3)")
    expected = logits.shape[:2]
    if any(
        value.shape != expected
        for value in (labels, supervision_mask, loss_weight, label_valid)
    ):
        raise ValueError("H/A/D targets and masks must have shape (B, T)")
    effective_weight = (
        supervision_mask.to(logits.dtype)
        * label_valid.to(logits.dtype)
        * loss_weight.to(logits.dtype)
    )
    denominator = effective_weight.sum()
    if denominator.item() <= 0:
        raise ValueError("A batch must contain at least one valid output")
    element_loss = F.cross_entropy(
        logits.transpose(1, 2), labels.long(), reduction="none", ignore_index=-1
    )
    return (element_loss * effective_weight).sum(), denominator


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float | None = None,
    classification_mode: str = "binary",
    show_progress: bool = False,
    description: str = "Training",
) -> EpochStats:
    """Train for one epoch using center-supervised temporal windows."""
    model.train()
    total_numerator = 0.0
    total_denominator = 0.0
    batches = progress(
        loader,
        enabled=show_progress,
        total=len(loader),
        desc=description,
        unit="batch",
        leave=False,
    )
    for cpu_batch in batches:
        batch = move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss_function = (
            masked_bce_components
            if classification_mode == "binary"
            else masked_cross_entropy_components
        )
        numerator, denominator = loss_function(
            logits,
            batch["labels"],
            batch["supervision_mask"],
            batch["loss_weight"],
            batch["label_valid"],
        )
        loss = numerator / denominator
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        total_numerator += float(numerator.detach().cpu())
        total_denominator += float(denominator.detach().cpu())
        if hasattr(batches, "set_postfix"):
            batches.set_postfix(loss=f"{total_numerator / total_denominator:.4f}")
    if total_denominator <= 0:
        raise ValueError("Epoch produced no valid outputs")
    return EpochStats(total_numerator / total_denominator, total_denominator)


def validate_full_segments(
    model: nn.Module,
    data: WCSFData,
    segments: tuple[Segment, ...],
    device: torch.device,
    classification_mode: str,
    chunk_frames: int | None = None,
    show_progress: bool = False,
    description: str = "Validation",
) -> EpochStats:
    """Calculate mean validation loss from one prediction per valid frame."""
    total_numerator = 0.0
    total_frames = 0
    segment_iter = progress(
        segments,
        enabled=show_progress,
        total=len(segments),
        desc=description,
        unit="segment",
        leave=False,
    )
    for segment in segment_iter:
        logits = predict_segment_logits(
            model,
            data.players,
            data.player_location_valid,
            data.player_speed_valid,
            data.team_features,
            data.team_feature_valid,
            segment,
            device,
            chunk_frames,
        )
        frame_slice = slice(segment.start, segment.end)
        valid = torch.from_numpy(data.label_valid[frame_slice]).bool()
        labels = torch.from_numpy(data.labels[frame_slice])
        if not valid.any():
            continue
        if classification_mode == "binary":
            losses = F.binary_cross_entropy_with_logits(
                logits[valid], labels[valid].to(logits.dtype), reduction="sum"
            )
        elif classification_mode == "had":
            losses = F.cross_entropy(logits[valid], labels[valid].long(), reduction="sum")
        else:
            raise ValueError("classification_mode must be 'binary' or 'had'")
        total_numerator += float(losses)
        total_frames += int(valid.sum())
    if total_frames == 0:
        raise ValueError("Validation contains no valid labelled frames")
    return EpochStats(total_numerator / total_frames, float(total_frames))


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    config: ExperimentConfig,
    loaders: LoaderBundle,
    epoch: int,
    best_validation_loss: float | None,
    training_mode: str,
    validation_block: int | None,
    inner_best_epochs: list[int] | None = None,
) -> Dict[str, object]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": config.to_dict(),
        "fold": config.split.fold,
        "normalizer": loaders.normalizer.state_dict(),
        "seed": config.training.seed,
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "source_fps": loaders.data.source_fps,
        "receptive_field_frames": config.model.receptive_field_frames,
        "training_mode": training_mode,
        "validation_block": validation_block,
        "inner_best_epochs": inner_best_epochs,
    }


def train_experiment(
    config: ExperimentConfig,
    run_dir: str | Path,
    show_progress: bool = False,
) -> Dict[str, object]:
    """Train one inner validation block; held-out test data is never iterated."""
    set_reproducible_seed(config.training.seed)
    loaders = build_dataloaders(config)
    device = select_device(config.training.device)
    model = build_model(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
    )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    history = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    epochs = progress(
        range(1, config.training.epochs + 1),
        enabled=show_progress,
        total=config.training.epochs,
        desc=f"Fold {config.split.fold} block {config.split.validation_block}",
        unit="epoch",
        leave=False,
    )
    for epoch in epochs:
        train_stats = train_epoch(
            model,
            loaders.train,
            device,
            optimizer,
            config.training.gradient_clip_norm,
            config.model.classification_mode,
            show_progress,
            f"F{config.split.fold} B{config.split.validation_block} epoch {epoch}",
        )
        validation_stats = validate_full_segments(
            model,
            loaders.data,
            loaders.split.segments["validation"],
            device,
            config.model.classification_mode,
            show_progress=show_progress,
            description=f"F{config.split.fold} B{config.split.validation_block} validation",
        )
        scheduler.step(validation_stats.loss)
        record = {
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "validation_loss": validation_stats.loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if hasattr(epochs, "set_postfix"):
            epochs.set_postfix(
                train=f"{train_stats.loss:.4f}",
                val=f"{validation_stats.loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2g}",
            )
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        if validation_stats.loss < best_validation_loss:
            best_validation_loss = validation_stats.loss
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    config,
                    loaders,
                    epoch,
                    best_validation_loss,
                    "inner_cv",
                    config.split.validation_block,
                ),
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.early_stopping_patience:
                break

    return {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(history),
        "best_epoch": min(history, key=lambda item: item["validation_loss"])["epoch"],
    }


def train_full_half_refit(
    config: ExperimentConfig,
    epochs: int,
    run_dir: str | Path,
    inner_best_epochs: list[int],
    show_progress: bool = False,
) -> Dict[str, object]:
    """Refit from scratch on the complete training half for a fixed epoch count."""
    if epochs < 1:
        raise ValueError("Refit epochs must be positive")
    set_reproducible_seed(config.training.seed)
    loaders = build_dataloaders(config, full_half_refit=True)
    if loaders.split.validation_mask.any() or loaders.split.buffer_mask.any():
        raise ValueError("A full-half refit cannot contain validation or buffer frames")
    device = select_device(config.training.device)
    model = build_model(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "final_checkpoint.pt"
    history = []
    epoch_iter = progress(
        range(1, epochs + 1),
        enabled=show_progress,
        total=epochs,
        desc=f"Fold {config.split.fold} refit",
        unit="epoch",
        leave=False,
    )
    for epoch in epoch_iter:
        train_stats = train_epoch(
            model,
            loaders.train,
            device,
            optimizer,
            config.training.gradient_clip_norm,
            config.model.classification_mode,
            show_progress,
            f"F{config.split.fold} refit epoch {epoch}",
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(train=f"{train_stats.loss:.4f}")
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            None,
            config,
            loaders,
            epochs,
            None,
            "full_half_refit",
            None,
            inner_best_epochs,
        ),
        checkpoint_path,
    )
    return {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "epochs_completed": epochs,
    }


def run_cross_validation(
    config: ExperimentConfig, parent: str | Path, show_progress: bool = False
) -> Dict[str, object]:
    """Run three inner blocks, summarize validation, then refit the complete half."""
    config.validate()
    parent = Path(parent)
    block_results = []
    best_epochs = []
    blocks = progress(
        (1, 2, 3),
        enabled=show_progress,
        total=3,
        desc=f"Fold {config.split.fold} inner CV",
        unit="block",
        leave=False,
    )
    for block in blocks:
        inner_config = copy.deepcopy(config)
        inner_config.split.validation_block = block
        block_dir = parent / f"inner_block{block}"
        training_result = train_experiment(inner_config, block_dir, show_progress)
        if config.model.classification_mode == "binary":
            evaluation = evaluate_checkpoint(
                training_result["checkpoint_path"],
                inner_config.split.fold,
                inner_config.model.model_type,
                inner_config.training.seed,
                "validation",
                inner_config.training.device,
                training_mode="inner_cv",
                validation_block=block,
                show_progress=show_progress,
            )
        else:
            from .evaluate_had import evaluate_had_checkpoint

            evaluation = evaluate_had_checkpoint(
                training_result["checkpoint_path"],
                inner_config.split.fold,
                inner_config.model.model_type,
                inner_config.training.seed,
                "validation",
                inner_config.training.device,
                training_mode="inner_cv",
                validation_block=block,
                show_progress=show_progress,
            )
        evaluation_path = block_dir / "evaluation_validation.json"
        with evaluation_path.open("w", encoding="utf-8") as handle:
            json.dump(evaluation.report, handle, indent=2)
        best_epoch = int(training_result["best_epoch"])
        best_epochs.append(best_epoch)
        block_result = {
            "validation_block": block,
            "validation_interval": evaluation.report["runs"][0],
            "best_epoch": best_epoch,
            "best_validation_loss": training_result["best_validation_loss"],
            "frame_macro_f1": evaluation.report["frame_metrics"]["macro_f1"],
            "checkpoint_path": training_result["checkpoint_path"],
        }
        if config.model.classification_mode == "binary":
            block_result["best_validation_bce"] = training_result["best_validation_loss"]
            block_result["binary_segment_mean_f1"] = evaluation.report[
                "segment_metrics"
            ]["mean_f1"]
        else:
            block_result["best_validation_ce"] = training_result["best_validation_loss"]
            block_result["possession_spell_mean_f1"] = evaluation.report[
                "spell_metrics"
            ]["mean_f1"]
        block_results.append(block_result)

    refit_epochs = int(np.median(best_epochs))
    refit_config = copy.deepcopy(config)
    refit_result = train_full_half_refit(
        refit_config, refit_epochs, parent / "refit", best_epochs, show_progress
    )
    summary = {
        "schema_version": "wcsf_temporal_cv_v2",
        "classification_mode": config.model.classification_mode,
        "fold": config.split.fold,
        "model_type": config.model.model_type,
        "seed": config.training.seed,
        "blocks": block_results,
        "mean_best_validation_loss": float(
            np.mean([item["best_validation_loss"] for item in block_results])
        ),
        "mean_frame_macro_f1": float(
            np.mean([item["frame_macro_f1"] for item in block_results])
        ),
        "inner_best_epochs": best_epochs,
        "refit_epochs": refit_epochs,
        "refit_checkpoint_path": refit_result["checkpoint_path"],
    }
    if config.model.classification_mode == "binary":
        summary["mean_best_validation_bce"] = summary["mean_best_validation_loss"]
        summary["mean_binary_segment_f1"] = float(
            np.mean([item["binary_segment_mean_f1"] for item in block_results])
        )
    else:
        summary["mean_best_validation_ce"] = summary["mean_best_validation_loss"]
        summary["mean_possession_spell_f1"] = float(
            np.mean([item["possession_spell_mean_f1"] for item in block_results])
        )
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / "cv_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def train_run(
    config: ExperimentConfig, folds: tuple[int, ...], show_progress: bool = True
) -> tuple[Path, dict[str, object]]:
    """Create one timestamped run and train every requested outer fold."""
    run_dir, metadata = create_run_directory(config, folds)
    summaries: dict[str, object] = {}
    try:
        progress_write(f"Run directory: {run_dir}", enabled=show_progress)
        fold_iter = progress(
            folds,
            enabled=show_progress,
            total=len(folds),
            desc="Outer folds",
            unit="fold",
        )
        for fold in fold_iter:
            fold_config = copy.deepcopy(config)
            fold_config.split.fold = fold
            summaries[str(fold)] = run_cross_validation(
                fold_config, fold_dir(run_dir, fold), show_progress
            )
            update_run_status(
                run_dir, metadata, status="running", completed_fold=fold
            )
        update_run_status(run_dir, metadata, status="completed")
    except Exception:
        update_run_status(run_dir, metadata, status="failed")
        raise
    return run_dir, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fold", choices=("1", "2", "all"), default="all")
    parser.add_argument("--model-type", choices=("team_tcn", "deepsets_tcn"))
    parser.add_argument("--classification-mode", choices=("binary", "had"))
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model_type is not None:
        config.model.model_type = args.model_type
    if args.classification_mode is not None:
        config.model.classification_mode = args.classification_mode
        if (
            args.classification_mode == "had"
            and Path(config.data.label_path).name
            == "wcsf_alive_dead_labels_strict.npz"
        ):
            config.data.label_path = (
                "data/open_data_groningen/processed/wcsf_had_labels.npz"
            )
    config.validate()
    folds = (1, 2) if args.fold == "all" else (int(args.fold),)
    run_dir, summaries = train_run(config, folds, not args.no_progress)
    print(f"Completed run: {run_dir}")
    for fold, summary in summaries.items():
        print(
            f"  Fold {fold}: refit epochs={summary['refit_epochs']}, "
            f"mean validation loss={summary['mean_best_validation_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
