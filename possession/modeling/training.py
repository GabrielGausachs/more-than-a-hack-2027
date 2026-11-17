"""Train the final 20-match Deep Sets plus TCN possession model."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .data import (IGNORE_INDEX, ROOT, engineered_matrix, fit_normalization,
                   iter_sequence_chunks, load_archive, load_json,
                   possession_duration_weights, resolve_path)
from .model import build_model


DEFAULT_CONFIG = ROOT / "configs" / "training.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _batches(chunks: List[Dict[str, np.ndarray]], batch_size: int):
    for start in range(0, len(chunks), batch_size):
        group = chunks[start:start + batch_size]
        yield (
            torch.from_numpy(np.stack([item["players"] for item in group])),
            torch.from_numpy(np.stack([item["engineered"] for item in group])),
            torch.from_numpy(np.stack([item["target"] for item in group])),
            torch.from_numpy(np.stack([item["loss_weight"] for item in group])),
        )


def train(config_path: Path = DEFAULT_CONFIG) -> Path:
    config = load_json(config_path)
    archive_root = resolve_path(str(config["archive_root"]))
    output_dir = resolve_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    match_ids = [int(value) for value in config["match_ids"]]
    epochs = int(config["epochs"])
    seed = int(config["seed"])
    device = _device(str(config["device"]))
    training = config["training"]
    loss_config = config["loss"]
    decoder = config["decoder"]

    set_seed(seed)
    normalization = fit_normalization(match_ids, archive_root)
    normalization.save(output_dir / "normalization.npz")

    sample = load_archive(match_ids[0], archive_root)
    engineered_width = engineered_matrix(sample).shape[1]
    model_config = config["model"]
    model = build_model(engineered_width, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]))
    core_size = int(training["core_size"])
    batch_size = int(training["batch_size"])
    halo = model.receptive_radius

    print("Final model training | matches={} | epochs={} | "
          "loss=duration-weighted | device={}".format(
              len(match_ids), epochs, device.type))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        match_order = list(match_ids)
        random.Random(seed + epoch).shuffle(match_order)
        loss_total = loss_denominator = 0.0
        labelled_total = 0

        for match_id in match_order:
            arrays = load_archive(match_id, archive_root)
            frame_weights = possession_duration_weights(
                arrays["target_label"], arrays["period"],
                float(arrays["frame_rate_hz"].item()),
                float(loss_config["reference_duration_s"]),
                float(loss_config["minimum_weight"]),
                float(loss_config["maximum_weight"]))
            chunks = list(iter_sequence_chunks(
                arrays, normalization, core_size, halo, frame_weights))
            random.Random(seed + epoch * 1009 + match_id).shuffle(chunks)

            for players, engineered, targets, weights in _batches(chunks, batch_size):
                denominator = float(weights.sum().item())
                if denominator <= 0:
                    continue
                labelled_total += int((targets != IGNORE_INDEX).sum())
                players = players.to(device)
                engineered = engineered.to(device)
                targets = targets.to(device)
                weights = weights.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(players, engineered)[:, halo:halo + core_size]
                frame_loss = F.cross_entropy(
                    logits.reshape(-1, 3), targets.reshape(-1),
                    ignore_index=IGNORE_INDEX, reduction="none")
                loss = (frame_loss * weights.reshape(-1)).sum() / weights.sum()
                loss.backward()
                optimizer.step()
                loss_total += float(loss.detach().cpu()) * denominator
                loss_denominator += denominator

        train_loss = loss_total / max(loss_denominator, 1.0)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "labelled_training_frames": labelled_total})
        print("Epoch {:02d}/{:02d} | loss={:.6f}".format(
            epoch, epochs, train_loss))

    checkpoint = {
        "model_state": model.state_dict(),
        "epoch": epochs,
        "model_config": dict(model_config),
        "engineered_width": engineered_width,
        "metrics": {},
        "loss_config": dict(loss_config),
    }
    torch.save(checkpoint, output_dir / "model.pt")
    with (output_dir / "history.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    final_decoder = {
        "decoder": dict(decoder),
        "checkpoint_epoch": epochs,
        "validation_match_ids": [],
        "metrics": {},
    }
    (output_dir / "final_decoder.json").write_text(
        json.dumps(final_decoder, indent=2) + "\n", encoding="utf-8")
    print("Saved {}".format(output_dir / "model.pt"))
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    train(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
