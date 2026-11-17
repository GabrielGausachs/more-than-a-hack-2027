"""Small Deep Sets plus noncausal residual TCN possession model."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation,
                      padding=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation,
                      padding=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class DeepSetsTCN(nn.Module):
    def __init__(self, engineered_width: int, hidden_channels: int = 64,
                 player_hidden: int = 32, player_embedding: int = 64,
                 dilations: Sequence[int] = (1, 2, 4, 8, 16),
                 dropout: float = 0.1):
        super().__init__()
        self.engineered_width = int(engineered_width)
        self.dilations = tuple(int(value) for value in dilations)
        self.player_mlp = nn.Sequential(
            nn.Linear(5, player_hidden), nn.ReLU(),
            nn.Linear(player_hidden, player_embedding), nn.ReLU(),
        )
        self.frame_projection = nn.Sequential(
            nn.Linear(2 * player_embedding + engineered_width, hidden_channels),
            nn.ReLU(),
        )
        self.tcn = nn.Sequential(*[
            ResidualTCNBlock(hidden_channels, dilation, dropout)
            for dilation in self.dilations
        ])
        self.classifier = nn.Linear(hidden_channels, 3)

    @property
    def receptive_field_frames(self) -> int:
        return 1 + 4 * sum(self.dilations)

    @property
    def receptive_radius(self) -> int:
        return (self.receptive_field_frames - 1) // 2

    def forward(self, players: torch.Tensor, engineered: torch.Tensor) -> torch.Tensor:
        """Return [batch,time,3] logits.

        ``players`` is [B,T,2,24,5]. Position validity is channel 3 and
        controls pooling; speed validity remains an input channel.
        """

        embedding = self.player_mlp(players)
        position_valid = players[..., 3:4]
        pooled = (embedding * position_valid).sum(dim=3) / position_valid.sum(dim=3).clamp_min(1.0)
        pooled = pooled.reshape(pooled.shape[0], pooled.shape[1], -1)
        frame = self.frame_projection(torch.cat((pooled, engineered), dim=-1))
        temporal = self.tcn(frame.transpose(1, 2)).transpose(1, 2)
        return self.classifier(temporal)


def build_model(engineered_width: int, config: Mapping[str, object]) -> DeepSetsTCN:
    return DeepSetsTCN(
        engineered_width=engineered_width,
        hidden_channels=int(config.get("hidden_channels", 64)),
        player_hidden=int(config.get("player_hidden", 32)),
        player_embedding=int(config.get("player_embedding", 64)),
        dilations=tuple(config.get("dilations", (1, 2, 4, 8, 16))),
        dropout=float(config.get("dropout", 0.1)),
    )

