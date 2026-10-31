"""Noncausal residual TCN and selectable ALIVE/DEAD model frontends."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .config import ModelConfig, receptive_field_frames
from .player_encoder import DeepSetPlayerEncoder


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels independently at each timestamp."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(inputs.transpose(1, 2)).transpose(1, 2)


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        convolutions: int = 2,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Noncausal same-length convolutions require an odd kernel")
        padding = dilation * (kernel_size - 1) // 2
        layers = []
        current_channels = in_channels
        for _ in range(convolutions):
            layers.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv1d(
                            current_channels,
                            out_channels,
                            kernel_size,
                            padding=padding,
                            dilation=dilation,
                        ),
                        "norm": ChannelLayerNorm(out_channels),
                        "dropout": nn.Dropout(dropout),
                    }
                )
            )
            current_channels = out_channels
        self.layers = nn.ModuleList(layers)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.GELU()

    @staticmethod
    def _apply_mask(values: torch.Tensor, time_mask: torch.Tensor | None) -> torch.Tensor:
        if time_mask is None:
            return values
        return values * time_mask.unsqueeze(1).to(values.dtype)

    def forward(
        self, inputs: torch.Tensor, time_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = inputs
        for layer in self.layers:
            output = layer["conv"](output)
            output = layer["norm"](output)
            output = self.activation(output)
            output = layer["dropout"](output)
            output = self._apply_mask(output, time_mask)
        residual = self._apply_mask(self.residual(inputs), time_mask)
        return self._apply_mask(self.activation(output + residual), time_mask)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        dropout: float,
        convolutions_per_block: int,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.convolutions_per_block = convolutions_per_block
        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    channels,
                    channels,
                    kernel_size,
                    dilation,
                    dropout,
                    convolutions_per_block,
                )
                for dilation in dilations
            ]
        )

    @property
    def receptive_field_frames(self) -> int:
        return receptive_field_frames(
            self.kernel_size, self.dilations, self.convolutions_per_block
        )

    def forward(
        self, inputs: torch.Tensor, time_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = inputs
        for block in self.blocks:
            output = block(output, time_mask)
        return output


def _team_representation(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    features = batch["team_features"]
    valid = batch["team_feature_valid"].to(features.dtype)
    return torch.cat((features.flatten(-2), valid.flatten(-2)), dim=-1)


class BaseAliveDeadTCN(nn.Module):
    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, config.temporal_channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.temporal = TemporalConvNet(
            config.temporal_channels,
            config.kernel_size,
            config.dilations,
            config.dropout,
            config.convolutions_per_block,
        )
        self.output_classes = config.output_classes
        self.classifier = nn.Linear(config.temporal_channels, self.output_classes)

    def temporal_forward(
        self, representation: torch.Tensor, output_mask: torch.Tensor
    ) -> torch.Tensor:
        projected = self.input_projection(representation)
        projected = projected * output_mask.unsqueeze(-1).to(projected.dtype)
        temporal = self.temporal(projected.transpose(1, 2), output_mask)
        logits = self.classifier(temporal.transpose(1, 2))
        return logits.squeeze(-1) if self.output_classes == 1 else logits


class TeamStatisticsTCN(BaseAliveDeadTCN):
    """Team statistics and validity masks followed by the shared TCN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(input_dim=24, config=config)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.temporal_forward(_team_representation(batch), batch["output_mask"])


class DeepSetsTCN(BaseAliveDeadTCN):
    """Deep Sets player embeddings fused with team statistics before the TCN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(input_dim=2 * config.player_embedding_dim + 24, config=config)
        self.player_encoder = DeepSetPlayerEncoder(
            config.player_hidden_dim, config.player_embedding_dim
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        player_embeddings = self.player_encoder(
            batch["players"],
            batch["player_location_valid"],
            batch["player_speed_valid"],
        )
        representation = torch.cat(
            (player_embeddings.flatten(-2), _team_representation(batch)), dim=-1
        )
        return self.temporal_forward(representation, batch["output_mask"])


def build_model(config: ModelConfig) -> nn.Module:
    if config.model_type == "team_tcn":
        return TeamStatisticsTCN(config)
    if config.model_type == "deepsets_tcn":
        return DeepSetsTCN(config)
    raise ValueError(f"Unknown model_type: {config.model_type}")
