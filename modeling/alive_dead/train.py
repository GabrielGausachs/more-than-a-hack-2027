"""Future training entry point for WCSF ALIVE/DEAD models.

This module defines training infrastructure but does not execute unless called explicitly.
"""

from __future__ import annotations

import argparse
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
from .dataset import LoaderBundle, build_dataloaders
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


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def masked_bce_components(
    logits: torch.Tensor,
    labels: torch.Tensor,
    output_mask: torch.Tensor,
    loss_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted BCE numerator and denominator for valid outputs."""
    if logits.shape != labels.shape or logits.shape != output_mask.shape:
        raise ValueError("logits, labels and output_mask must have identical shapes")
    effective_weight = output_mask.to(logits.dtype) * loss_weight.to(logits.dtype)
    denominator = effective_weight.sum()
    if denominator.item() <= 0:
        raise ValueError("A batch must contain at least one valid output")
    element_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (element_loss * effective_weight).sum(), denominator


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> EpochStats:
    """Run one training or validation epoch; no test loader is accepted implicitly."""
    training = optimizer is not None
    model.train(training)
    total_numerator = 0.0
    total_denominator = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            numerator, denominator = masked_bce_components(
                logits,
                batch["labels"],
                batch["output_mask"],
                batch["loss_weight"],
            )
            loss = numerator / denominator
            if training:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            total_numerator += float(numerator.detach().cpu())
            total_denominator += float(denominator.detach().cpu())
    if total_denominator <= 0:
        raise ValueError("Epoch produced no valid outputs")
    return EpochStats(total_numerator / total_denominator, total_denominator)


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    config: ExperimentConfig,
    loaders: LoaderBundle,
    epoch: int,
    best_validation_loss: float,
) -> Dict[str, object]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config.to_dict(),
        "fold": config.split.fold,
        "normalizer": loaders.normalizer.state_dict(),
        "seed": config.training.seed,
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "source_fps": loaders.data.source_fps,
        "receptive_field_frames": config.model.receptive_field_frames,
    }


def train_experiment(config: ExperimentConfig) -> Dict[str, object]:
    """Train one configured fold/model; held-out test data is never iterated here."""
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

    run_name = (
        f"fold{config.split.fold}_{config.model.model_type}_seed{config.training.seed}"
    )
    run_dir = Path(config.training.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    history = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.training.epochs + 1):
        train_stats = run_epoch(
            model,
            loaders.train,
            device,
            optimizer,
            config.training.gradient_clip_norm,
        )
        validation_stats = run_epoch(model, loaders.validation, device)
        scheduler.step(validation_stats.loss)
        record = {
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "validation_loss": validation_stats.loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fold", type=int, choices=(1, 2))
    parser.add_argument("--model-type", choices=("team_tcn", "deepsets_tcn"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.fold is not None:
        config.split.fold = args.fold
    if args.model_type is not None:
        config.model.model_type = args.model_type
    config.validate()
    result = train_experiment(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
