"""Central configuration for WCSF binary and H/A/D temporal experiments."""

from __future__ import annotations

import json
import math
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
    label_path: str = "data/open_data_groningen/processed/wcsf_alive_dead_labels_strict.npz"
    sequence_seconds: float = 10.0
    train_stride_seconds: float = 4.0
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class SplitConfig:
    fold: int = 1
    validation_fraction: float = 0.15
    buffer_seconds: float = 10.0
    validation_block: int = 2


@dataclass
class ModelConfig:
    model_type: str = "team_tcn"
    classification_mode: str = "binary"
    had_head_type: str = "direct"
    possession_loss_weight: float = 1.0
    use_ball: bool = False
    player_encoder_type: str = "legacy_mean"
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

    @property
    def context_radius_frames(self) -> int:
        """Return the symmetric context required on either side of an output."""
        return (self.receptive_field_frames - 1) // 2

    @property
    def output_classes(self) -> int:
        return 1 if self.classification_mode == "binary" else 3


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

    def validate(
        self, source_fps: float = 10.0, require_supervised_coverage: bool = True
    ) -> None:
        if self.split.fold not in (1, 2):
            raise ValueError("fold must be 1 or 2")
        if not 0.0 < self.split.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between 0 and 1")
        if self.split.validation_block not in (1, 2, 3):
            raise ValueError("validation_block must be 1, 2 or 3")
        if self.model.model_type not in ("team_tcn", "deepsets_tcn"):
            raise ValueError("model_type must be 'team_tcn' or 'deepsets_tcn'")
        if self.model.classification_mode not in ("binary", "had"):
            raise ValueError("classification_mode must be 'binary' or 'had'")
        if self.model.had_head_type not in ("direct", "hierarchical"):
            raise ValueError("had_head_type must be 'direct' or 'hierarchical'")
        if (
            self.model.classification_mode != "had"
            and self.model.had_head_type != "direct"
        ):
            raise ValueError("hierarchical heads require classification_mode='had'")
        if not math.isfinite(self.model.possession_loss_weight) or (
            self.model.possession_loss_weight <= 0.0
        ):
            raise ValueError("possession_loss_weight must be positive")
        if (
            self.model.had_head_type != "hierarchical"
            and self.model.possession_loss_weight != 1.0
        ):
            raise ValueError(
                "possession_loss_weight is configurable only for hierarchical H/A/D"
            )
        if not isinstance(self.model.use_ball, bool):
            raise ValueError("use_ball must be Boolean")
        if self.model.player_encoder_type not in ("legacy_mean", "ball_conditioned"):
            raise ValueError(
                "player_encoder_type must be 'legacy_mean' or 'ball_conditioned'"
            )
        if self.model.player_encoder_type == "ball_conditioned":
            if self.model.model_type != "deepsets_tcn":
                raise ValueError("ball_conditioned encoder requires model_type='deepsets_tcn'")
            if not self.model.use_ball:
                raise ValueError("ball_conditioned encoder requires use_ball=true")
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
        buffer_frames = self.seconds_to_frames(self.split.buffer_seconds, source_fps)
        if train_stride > sequence_frames:
            raise ValueError("Training stride cannot exceed the sequence length")
        receptive_field = self.model.receptive_field_frames
        if receptive_field % 2 == 0:
            raise ValueError("The noncausal TCN receptive field must be odd")
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
        central_width = sequence_frames - 2 * self.model.context_radius_frames
        if require_supervised_coverage and train_stride > central_width:
            raise ValueError(
                f"Training stride ({train_stride}) exceeds the fully contextualized "
                f"central width ({central_width})"
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
        data_values = dict(values.get("data", {}))
        data_values.pop("eval_stride_seconds", None)
        model_values = dict(values.get("model", {}))
        if "dilations" in model_values:
            model_values["dilations"] = tuple(model_values["dilations"])
        return cls(
            data=DataConfig(**data_values),
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
