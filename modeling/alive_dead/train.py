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

from .config import ExperimentConfig, ModelConfig, load_config
from .dataset import LoaderBundle, WCSFData, build_dataloaders
from .evaluate import evaluate_checkpoint, select_device
from .inference import predict_segment_logits
from .progress import progress, progress_write
from .runs import create_run_directory, fold_dir, update_run_status
from .splits import Segment, valid_subsegments
from .tcn import build_model, had_probabilities_from_logits


@dataclass(frozen=True)
class EpochStats:
    loss: float
    effective_frames: float
    classification_loss: float | None = None
    transition_loss: float | None = None
    transition_pairs: float | None = None
    spell_f1: float | None = None
    macro_f1: float | None = None
    alive_loss: float | None = None
    possession_loss: float | None = None
    alive_frames: float | None = None
    possession_frames: float | None = None


def _stats_fields(prefix: str, stats: EpochStats) -> Dict[str, float]:
    values = {f"{prefix}_loss": stats.loss}
    if stats.classification_loss is not None:
        values[f"{prefix}_classification_loss"] = stats.classification_loss
    if stats.transition_loss is not None:
        values.update(
            {
                f"{prefix}_transition_loss": stats.transition_loss,
                f"{prefix}_transition_pairs": stats.transition_pairs,
            }
        )
    if stats.spell_f1 is not None:
        values.update(
            {
                f"{prefix}_spell_f1": stats.spell_f1,
                f"{prefix}_macro_f1": stats.macro_f1,
            }
        )
    if stats.alive_loss is not None:
        values.update(
            {
                f"{prefix}_alive_bce": stats.alive_loss,
                f"{prefix}_possession_ce": stats.possession_loss,
                f"{prefix}_alive_frames": stats.alive_frames,
                f"{prefix}_possession_frames": stats.possession_frames,
            }
        )
    return values


def checkpoint_selection_key(
    stats: EpochStats, classification_mode: str
) -> tuple[float, ...]:
    """Return the deterministic checkpoint ordering for one validation epoch."""
    if classification_mode == "binary":
        return (-stats.loss,)
    if classification_mode != "had":
        raise ValueError("classification_mode must be 'binary' or 'had'")
    if stats.spell_f1 is None or stats.macro_f1 is None:
        raise ValueError("H/A/D checkpoint selection requires spell and macro F1")
    classification_loss = (
        stats.classification_loss
        if stats.classification_loss is not None
        else stats.loss
    )
    return (stats.spell_f1, stats.macro_f1, -classification_loss)


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
    """Return weighted categorical cross-entropy over supervised outputs only."""
    if logits.ndim != 3 or logits.shape[-1] < 2:
        raise ValueError("Categorical logits must have shape (B, T, C), C >= 2")
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


def hierarchical_loss_components(
    logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return independently normalized ALIVE and conditional-possession components."""
    if logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError("Hierarchical outputs must have shape (B, T, 3)")
    required = {
        "alive_labels",
        "alive_label_valid",
        "possession_labels",
        "possession_label_valid",
        "supervision_mask",
        "loss_weight",
    }
    missing = required.difference(batch)
    if missing:
        raise ValueError(f"Hierarchical batch is missing values: {sorted(missing)}")
    alive_numerator, alive_denominator = masked_bce_components(
        logits[..., 0],
        batch["alive_labels"],
        batch["supervision_mask"],
        batch["loss_weight"],
        batch["alive_label_valid"],
    )
    possession_weight = (
        batch["supervision_mask"].to(logits.dtype)
        * batch["possession_label_valid"].to(logits.dtype)
        * batch["loss_weight"].to(logits.dtype)
    )
    possession_denominator = possession_weight.sum()
    if possession_denominator.item() > 0:
        possession_loss = F.cross_entropy(
            logits[..., 1:].transpose(1, 2),
            batch["possession_labels"].long(),
            reduction="none",
            ignore_index=-1,
        )
        possession_numerator = (possession_loss * possession_weight).sum()
    else:
        possession_numerator = logits[..., 1:].sum() * 0.0
    return (
        alive_numerator,
        alive_denominator,
        possession_numerator,
        possession_denominator,
    )


def transition_loss_components(
    logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    had_head_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted soft transition BCE for fully contextualized adjacent pairs."""
    required = {
        "labels",
        "label_valid",
        "transition_supervision_mask",
        "transition_loss_weight",
    }
    missing = required.difference(batch)
    if missing:
        raise ValueError(f"Transition-loss batch is missing values: {sorted(missing)}")
    probabilities = had_probabilities_from_logits(logits, had_head_type)
    predicted = 1.0 - (probabilities[:, :-1] * probabilities[:, 1:]).sum(dim=-1)
    valid = (
        batch["transition_supervision_mask"][:, 1:]
        & batch["label_valid"][:, :-1]
        & batch["label_valid"][:, 1:]
    )
    weight = valid.to(logits.dtype) * batch["transition_loss_weight"][:, 1:].to(
        logits.dtype
    )
    denominator = weight.sum()
    if denominator.item() <= 0:
        return logits.sum() * 0.0, denominator
    target = (batch["labels"][:, :-1] != batch["labels"][:, 1:]).to(logits.dtype)
    epsilon = torch.finfo(predicted.dtype).eps
    element_loss = F.binary_cross_entropy(
        predicted.clamp(epsilon, 1.0 - epsilon), target, reduction="none"
    )
    return (element_loss * weight).sum(), denominator


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    gradient_clip_norm: float | None = None,
    show_progress: bool = False,
    description: str = "Training",
    use_transition_loss: bool = False,
    transition_loss_weight: float = 0.1,
) -> EpochStats:
    """Train for one epoch using center-supervised temporal windows."""
    model.train()
    total_numerator = 0.0
    total_denominator = 0.0
    alive_numerator_total = 0.0
    alive_denominator_total = 0.0
    possession_numerator_total = 0.0
    possession_denominator_total = 0.0
    transition_numerator_total = 0.0
    transition_denominator_total = 0.0
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
        if (
            model_config.classification_mode == "had"
            and model_config.had_head_type == "hierarchical"
        ):
            alive_num, alive_den, possession_num, possession_den = (
                hierarchical_loss_components(logits, batch)
            )
            loss = alive_num / alive_den
            if possession_den.item() > 0:
                loss = loss + model_config.possession_loss_weight * (
                    possession_num / possession_den
                )
            alive_numerator_total += float(alive_num.detach().cpu())
            alive_denominator_total += float(alive_den.detach().cpu())
            possession_numerator_total += float(possession_num.detach().cpu())
            possession_denominator_total += float(possession_den.detach().cpu())
        else:
            loss_function = (
                masked_bce_components
                if model_config.classification_mode == "binary"
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
            total_numerator += float(numerator.detach().cpu())
            total_denominator += float(denominator.detach().cpu())
        if use_transition_loss:
            transition_num, transition_den = transition_loss_components(
                logits, batch, model_config.had_head_type
            )
            if transition_den.item() > 0:
                loss = loss + transition_loss_weight * (
                    transition_num / transition_den
                )
                transition_numerator_total += float(transition_num.detach().cpu())
                transition_denominator_total += float(transition_den.detach().cpu())
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if hasattr(batches, "set_postfix"):
            if alive_denominator_total > 0:
                running = alive_numerator_total / alive_denominator_total
                if possession_denominator_total > 0:
                    running += model_config.possession_loss_weight * (
                        possession_numerator_total / possession_denominator_total
                    )
            else:
                running = total_numerator / total_denominator
            if transition_denominator_total > 0:
                running += transition_loss_weight * (
                    transition_numerator_total / transition_denominator_total
                )
            batches.set_postfix(loss=f"{running:.4f}")
    if use_transition_loss and transition_denominator_total <= 0:
        raise ValueError("Epoch contains no valid adjacent transition pairs")
    if alive_denominator_total > 0:
        if possession_denominator_total <= 0:
            raise ValueError("Epoch contains no valid possession targets")
        alive_loss = alive_numerator_total / alive_denominator_total
        possession_loss = possession_numerator_total / possession_denominator_total
        classification_loss = (
            alive_loss + model_config.possession_loss_weight * possession_loss
        )
        transition_loss = (
            transition_numerator_total / transition_denominator_total
            if transition_denominator_total > 0
            else None
        )
        return EpochStats(
            classification_loss
            + (transition_loss_weight * transition_loss if transition_loss is not None else 0.0),
            alive_denominator_total,
            classification_loss,
            transition_loss,
            transition_denominator_total if transition_loss is not None else None,
            None,
            None,
            alive_loss,
            possession_loss,
            alive_denominator_total,
            possession_denominator_total,
        )
    if total_denominator <= 0:
        raise ValueError("Epoch produced no valid outputs")
    classification_loss = total_numerator / total_denominator
    transition_loss = (
        transition_numerator_total / transition_denominator_total
        if transition_denominator_total > 0
        else None
    )
    return EpochStats(
        classification_loss
        + (transition_loss_weight * transition_loss if transition_loss is not None else 0.0),
        total_denominator,
        classification_loss,
        transition_loss,
        transition_denominator_total if transition_loss is not None else None,
    )


def validate_full_segments(
    model: nn.Module,
    data: WCSFData,
    segments: tuple[Segment, ...],
    device: torch.device,
    classification_mode: str,
    had_head_type: str = "direct",
    possession_loss_weight: float = 1.0,
    chunk_frames: int | None = None,
    show_progress: bool = False,
    description: str = "Validation",
    use_transition_loss: bool = False,
    transition_loss_weight: float = 0.1,
) -> EpochStats:
    """Calculate mean validation loss from one prediction per valid frame."""
    total_numerator = 0.0
    total_frames = 0
    alive_numerator = 0.0
    alive_frames = 0
    possession_numerator = 0.0
    possession_frames = 0
    transition_numerator = 0.0
    transition_pairs = 0
    had_runs = []
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
            ball=data.ball,
            ball_location_valid=data.ball_location_valid,
            ball_speed_valid=data.ball_speed_valid,
        )
        frame_slice = slice(segment.start, segment.end)
        valid = torch.from_numpy(data.label_valid[frame_slice]).bool()
        labels = torch.from_numpy(data.labels[frame_slice])
        if classification_mode == "had":
            from .evaluate_had import HADRun

            probabilities = had_probabilities_from_logits(logits, had_head_type)
            predictions = probabilities.argmax(dim=-1).to(torch.int8).cpu().numpy()
            for valid_segment in valid_subsegments(segment, data.label_valid):
                local_start = valid_segment.start - segment.start
                local_end = valid_segment.end - segment.start
                had_runs.append(
                    HADRun(
                        valid_segment.start,
                        valid_segment.end,
                        valid_segment.half_id,
                        data.labels[valid_segment.start : valid_segment.end].copy(),
                        predictions[local_start:local_end].copy(),
                    )
                )
            if use_transition_loss and segment.length > 1:
                pair_valid = valid[:-1] & valid[1:]
                if pair_valid.any():
                    predicted_transition = 1.0 - (
                        probabilities[:-1] * probabilities[1:]
                    ).sum(dim=-1)
                    target_transition = (labels[:-1] != labels[1:]).to(
                        predicted_transition.dtype
                    )
                    epsilon = torch.finfo(predicted_transition.dtype).eps
                    transition_numerator += float(
                        F.binary_cross_entropy(
                            predicted_transition[pair_valid].clamp(
                                epsilon, 1.0 - epsilon
                            ),
                            target_transition[pair_valid],
                            reduction="sum",
                        )
                    )
                    transition_pairs += int(pair_valid.sum())
        if classification_mode == "had" and had_head_type == "hierarchical":
            if data.alive_labels is None:
                raise ValueError("Hierarchical validation targets are unavailable")
            alive_valid = torch.from_numpy(data.alive_label_valid[frame_slice]).bool()
            possession_valid = torch.from_numpy(
                data.possession_label_valid[frame_slice]
            ).bool()
            alive_targets = torch.from_numpy(data.alive_labels[frame_slice])
            possession_targets = torch.from_numpy(data.possession_labels[frame_slice])
            if alive_valid.any():
                alive_numerator += float(
                    F.binary_cross_entropy_with_logits(
                        logits[alive_valid, 0],
                        alive_targets[alive_valid].to(logits.dtype),
                        reduction="sum",
                    )
                )
                alive_frames += int(alive_valid.sum())
            if possession_valid.any():
                possession_numerator += float(
                    F.cross_entropy(
                        logits[possession_valid, 1:],
                        possession_targets[possession_valid].long(),
                        reduction="sum",
                    )
                )
                possession_frames += int(possession_valid.sum())
            continue
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
    if classification_mode == "had" and had_head_type == "hierarchical":
        if alive_frames == 0 or possession_frames == 0:
            raise ValueError("Validation lacks hierarchical supervision targets")
        alive_loss = alive_numerator / alive_frames
        possession_loss = possession_numerator / possession_frames
        classification_loss = alive_loss + possession_loss_weight * possession_loss
        transition_loss = (
            transition_numerator / transition_pairs if transition_pairs else None
        )
        from .evaluate_had import compute_had_metrics

        metrics = compute_had_metrics(had_runs, data.source_fps)
        return EpochStats(
            classification_loss
            + (transition_loss_weight * transition_loss if transition_loss is not None else 0.0),
            float(alive_frames),
            classification_loss,
            transition_loss,
            float(transition_pairs) if transition_loss is not None else None,
            metrics["spell_metrics"]["mean_f1"],
            metrics["frame_metrics"]["macro_f1"],
            alive_loss,
            possession_loss,
            float(alive_frames),
            float(possession_frames),
        )
    if total_frames == 0:
        raise ValueError("Validation contains no valid labelled frames")
    classification_loss = total_numerator / total_frames
    if classification_mode == "had":
        from .evaluate_had import compute_had_metrics

        metrics = compute_had_metrics(had_runs, data.source_fps)
        transition_loss = (
            transition_numerator / transition_pairs if transition_pairs else None
        )
        return EpochStats(
            classification_loss
            + (transition_loss_weight * transition_loss if transition_loss is not None else 0.0),
            float(total_frames),
            classification_loss,
            transition_loss,
            float(transition_pairs) if transition_loss is not None else None,
            metrics["spell_metrics"]["mean_f1"],
            metrics["frame_metrics"]["macro_f1"],
        )
    return EpochStats(
        classification_loss, float(total_frames), classification_loss=classification_loss
    )


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
    best_validation_spell_f1: float | None = None,
    best_validation_macro_f1: float | None = None,
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
        "checkpoint_selection_metric": (
            "raw_validation_spell_f1"
            if config.model.classification_mode == "had" and training_mode == "inner_cv"
            else "validation_loss"
        ),
        "best_validation_spell_f1": best_validation_spell_f1,
        "best_validation_macro_f1": best_validation_macro_f1,
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
    best_validation_spell_f1 = float("-inf")
    best_validation_macro_f1 = float("-inf")
    best_selection_key: tuple[float, ...] | None = None
    best_record: Dict[str, float] | None = None
    best_epoch = 0
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
            config.model,
            config.training.gradient_clip_norm,
            show_progress,
            f"F{config.split.fold} B{config.split.validation_block} epoch {epoch}",
            config.training.use_transition_loss,
            config.training.transition_loss_weight,
        )
        validation_stats = validate_full_segments(
            model,
            loaders.data,
            loaders.split.segments["validation"],
            device,
            config.model.classification_mode,
            config.model.had_head_type,
            config.model.possession_loss_weight,
            show_progress=show_progress,
            description=f"F{config.split.fold} B{config.split.validation_block} validation",
            use_transition_loss=config.training.use_transition_loss,
            transition_loss_weight=config.training.transition_loss_weight,
        )
        scheduler_metric = (
            validation_stats.classification_loss
            if validation_stats.classification_loss is not None
            else validation_stats.loss
        )
        scheduler.step(scheduler_metric)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **_stats_fields("train", train_stats),
            **_stats_fields("validation", validation_stats),
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

        if config.model.classification_mode == "had":
            selection_key = checkpoint_selection_key(validation_stats, "had")
            better_checkpoint = (
                best_selection_key is None or selection_key > best_selection_key
            )
            strict_primary_improvement = (
                validation_stats.spell_f1 > best_validation_spell_f1
            )
        else:
            selection_key = checkpoint_selection_key(validation_stats, "binary")
            better_checkpoint = (
                best_selection_key is None or selection_key > best_selection_key
            )
            strict_primary_improvement = better_checkpoint

        if better_checkpoint:
            best_selection_key = selection_key
            best_validation_loss = validation_stats.loss
            best_validation_spell_f1 = (
                validation_stats.spell_f1
                if validation_stats.spell_f1 is not None
                else best_validation_spell_f1
            )
            best_validation_macro_f1 = (
                validation_stats.macro_f1
                if validation_stats.macro_f1 is not None
                else best_validation_macro_f1
            )
            best_record = record.copy()
            best_epoch = epoch
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
                    best_validation_spell_f1=(
                        best_validation_spell_f1
                        if config.model.classification_mode == "had"
                        else None
                    ),
                    best_validation_macro_f1=(
                        best_validation_macro_f1
                        if config.model.classification_mode == "had"
                        else None
                    ),
                ),
                checkpoint_path,
            )
        if strict_primary_improvement:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.early_stopping_patience:
                break

    if best_record is None or best_epoch < 1:
        raise ValueError("Training did not select a checkpoint")
    return {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "best_validation_loss": best_validation_loss,
        "best_validation_stats": best_record,
        "best_validation_spell_f1": (
            best_validation_spell_f1
            if config.model.classification_mode == "had"
            else None
        ),
        "best_validation_macro_f1": (
            best_validation_macro_f1
            if config.model.classification_mode == "had"
            else None
        ),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
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
            config.model,
            config.training.gradient_clip_norm,
            show_progress,
            f"F{config.split.fold} refit epoch {epoch}",
            config.training.use_transition_loss,
            config.training.transition_loss_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **_stats_fields("train", train_stats),
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
            "checkpoint_selection_metric": (
                "raw_validation_spell_f1"
                if config.model.classification_mode == "had"
                else "validation_loss"
            ),
        }
        if config.model.classification_mode == "binary":
            block_result["best_validation_bce"] = training_result["best_validation_loss"]
            block_result["binary_segment_mean_f1"] = evaluation.report[
                "segment_metrics"
            ]["mean_f1"]
        else:
            best_stats = training_result["best_validation_stats"]
            if config.model.had_head_type == "hierarchical":
                block_result.update(
                    {
                        "best_validation_alive_bce": best_stats[
                            "validation_alive_bce"
                        ],
                        "best_validation_possession_ce": best_stats[
                            "validation_possession_ce"
                        ],
                        "possession_loss_weight": config.model.possession_loss_weight,
                    }
                )
            else:
                block_result["best_validation_ce"] = best_stats[
                    "validation_classification_loss"
                ]
            if config.training.use_transition_loss:
                block_result.update(
                    {
                        "best_validation_transition_loss": best_stats[
                            "validation_transition_loss"
                        ],
                        "transition_loss_weight": config.training.transition_loss_weight,
                    }
                )
            block_result["selected_validation_spell_f1"] = training_result[
                "best_validation_spell_f1"
            ]
            block_result["selected_validation_macro_f1"] = training_result[
                "best_validation_macro_f1"
            ]
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
        "schema_version": "wcsf_temporal_cv_v3",
        "classification_mode": config.model.classification_mode,
        "had_head_type": config.model.had_head_type,
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
        "checkpoint_selection_metric": (
            "raw_validation_spell_f1"
            if config.model.classification_mode == "had"
            else "validation_loss"
        ),
    }
    if config.model.classification_mode == "binary":
        summary["mean_best_validation_bce"] = summary["mean_best_validation_loss"]
        summary["mean_binary_segment_f1"] = float(
            np.mean([item["binary_segment_mean_f1"] for item in block_results])
        )
    else:
        summary["mean_selected_validation_spell_f1"] = float(
            np.mean([item["selected_validation_spell_f1"] for item in block_results])
        )
        summary["mean_selected_validation_macro_f1"] = float(
            np.mean([item["selected_validation_macro_f1"] for item in block_results])
        )
        if config.model.had_head_type == "hierarchical":
            summary.update(
                {
                    "mean_best_validation_alive_bce": float(
                        np.mean(
                            [
                                item["best_validation_alive_bce"]
                                for item in block_results
                            ]
                        )
                    ),
                    "mean_best_validation_possession_ce": float(
                        np.mean(
                            [
                                item["best_validation_possession_ce"]
                                for item in block_results
                            ]
                        )
                    ),
                    "possession_loss_weight": config.model.possession_loss_weight,
                }
            )
        else:
            summary["mean_best_validation_ce"] = summary["mean_best_validation_loss"]
        summary["mean_possession_spell_f1"] = float(
            np.mean([item["possession_spell_mean_f1"] for item in block_results])
        )
        summary["use_transition_loss"] = config.training.use_transition_loss
        summary["transition_loss_weight"] = config.training.transition_loss_weight
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
    parser.add_argument("--had-head-type", choices=("direct", "hierarchical"))
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
    if args.had_head_type is not None:
        config.model.had_head_type = args.had_head_type
    config.validate()
    folds = (1, 2) if args.fold == "all" else (int(args.fold),)
    run_dir, summaries = train_run(config, folds, not args.no_progress)
    print(f"Completed run: {run_dir}")
    for fold, summary in summaries.items():
        if summary["classification_mode"] == "had":
            metric = (
                "mean selected spell F1="
                f"{summary['mean_selected_validation_spell_f1']:.4f}"
            )
        else:
            metric = f"mean validation loss={summary['mean_best_validation_loss']:.4f}"
        print(f"  Fold {fold}: refit epochs={summary['refit_epochs']}, {metric}")


if __name__ == "__main__":
    main()
