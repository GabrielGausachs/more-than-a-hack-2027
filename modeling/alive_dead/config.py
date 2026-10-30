"""Central configuration for WCSF ALIVE/DEAD experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


def receptive_field_frames(
    kernel_size: int, dilations: Tuple[int, ...], convolutions_per_block: int
) -> int:
    """Return the receptive field of a stack of stride-one temporal convolutions."""
    return 1 + convolutions_per_block * (kernel_size - 1) * sum(dilations)


@dataclass
class DataConfig:
    feature_path: str = "data/open_data_groningen/processed/wcsf_player_team_features.npz"
    label_path: str = "data/open_data_groningen/processed/wcsf_alive_dead_labels.npz"
    sequence_seconds: float = 10.0
    train_stride_seconds: float = 5.0
    eval_stride_seconds: float = 10.0
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class SplitConfig:
    fold: int = 1
    validation_fraction: float = 0.15
    buffer_seconds: float = 10.0


@dataclass
class ModelConfig:
    model_type: str = "team_tcn"
    temporal_channels: int = 64
    player_hidden_dim: int = 32
    player_embedding_dim: int = 32
    kernel_size: int = 3
    dilations: Tuple[int, ...] = (1, 2, 4, 8)
    convolutions_per_block: int = 2
    dropout: float = 0.15

    @property
    def receptive_field_frames(self) -> int:
        return receptive_field_frames(
            self.kernel_size, self.dilations, self.convolutions_per_block
        )


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float | None = 1.0
    early_stopping_patience: int = 8
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    device: str = "auto"
    output_dir: str = "data/open_data_groningen/model_runs"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self, source_fps: float = 10.0) -> None:
        if self.split.fold not in (1, 2):
            raise ValueError("fold must be 1 or 2")
        if not 0.0 < self.split.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between 0 and 1")
        if self.model.model_type not in ("team_tcn", "deepsets_tcn"):
            raise ValueError("model_type must be 'team_tcn' or 'deepsets_tcn'")
        if self.model.kernel_size < 1 or self.model.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.model.convolutions_per_block < 1 or not self.model.dilations:
            raise ValueError("The TCN requires positive convolutions and at least one dilation")
        if any(dilation < 1 for dilation in self.model.dilations):
            raise ValueError("All TCN dilations must be positive")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        sequence_frames = self.seconds_to_frames(self.data.sequence_seconds, source_fps)
        train_stride = self.seconds_to_frames(self.data.train_stride_seconds, source_fps)
        eval_stride = self.seconds_to_frames(self.data.eval_stride_seconds, source_fps)
        buffer_frames = self.seconds_to_frames(self.split.buffer_seconds, source_fps)
        if train_stride > sequence_frames or eval_stride > sequence_frames:
            raise ValueError("Window strides cannot exceed the sequence length")
        receptive_field = self.model.receptive_field_frames
        if receptive_field > sequence_frames:
            raise ValueError(
                f"TCN receptive field ({receptive_field}) exceeds sequence length "
                f"({sequence_frames})"
            )
        if receptive_field > buffer_frames:
            raise ValueError(
                f"TCN receptive field ({receptive_field}) exceeds validation buffer "
                f"({buffer_frames})"
            )

    @staticmethod
    def seconds_to_frames(seconds: float, source_fps: float) -> int:
        frames = int(round(seconds * source_fps))
        if frames < 1:
            raise ValueError("Durations must convert to at least one frame")
        return frames

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ExperimentConfig":
        model_values = dict(values.get("model", {}))
        if "dilations" in model_values:
            model_values["dilations"] = tuple(model_values["dilations"])
        return cls(
            data=DataConfig(**values.get("data", {})),
            split=SplitConfig(**values.get("split", {})),
            model=ModelConfig(**model_values),
            training=TrainingConfig(**values.get("training", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load a JSON experiment configuration or return the documented defaults."""
    return ExperimentConfig.from_json(path) if path is not None else ExperimentConfig()
