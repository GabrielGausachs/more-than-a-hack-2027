"""Permutation-invariant player-set encoder."""

from __future__ import annotations

import torch
from torch import nn


class DeepSetPlayerEncoder(nn.Module):
    """Encode each team with a shared player MLP and masked mean pooling."""

    def __init__(self, hidden_dim: int = 32, embedding_dim: int = 32) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.player_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
        )

    def forward(
        self,
        players: torch.Tensor,
        location_valid: torch.Tensor,
        speed_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return `(B, T, 2, embedding_dim)` Home/Away embeddings."""
        if players.ndim != 5 or players.shape[-3:] != (2, 24, 3):
            raise ValueError("players must have shape (B, T, 2, 24, 3)")
        expected_mask_shape = players.shape[:-1]
        if location_valid.shape != expected_mask_shape or speed_valid.shape != expected_mask_shape:
            raise ValueError("Player validity masks do not match players")

        dtype = players.dtype
        encoded_input = torch.cat(
            (
                players,
                location_valid.unsqueeze(-1).to(dtype),
                speed_valid.unsqueeze(-1).to(dtype),
            ),
            dim=-1,
        )
        encoded = self.player_mlp(encoded_input)
        present = location_valid | speed_valid
        encoded = encoded * present.unsqueeze(-1).to(dtype)
        count = present.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)
        return encoded.sum(dim=-2) / count
