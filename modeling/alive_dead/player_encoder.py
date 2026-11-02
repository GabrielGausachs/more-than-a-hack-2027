"""Permutation-invariant player-set encoder."""

from __future__ import annotations

import torch
from torch import nn


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PITCH_DIAGONAL_M = (PITCH_LENGTH_M**2 + PITCH_WIDTH_M**2) ** 0.5
NEARBY_RADII_M = (2.0, 3.0, 5.0)


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


def player_ball_geometry(
    players: torch.Tensor,
    player_location_valid: torch.Tensor,
    ball: torch.Tensor,
    ball_location_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return masked `(dx, dy, normalized_distance)`, metric distance and validity."""
    if players.ndim != 5 or players.shape[-3:] != (2, 24, 3):
        raise ValueError("players must have shape (B, T, 2, 24, 3)")
    if player_location_valid.shape != players.shape[:-1]:
        raise ValueError("player_location_valid does not match players")
    if ball.shape != (*players.shape[:2], 3):
        raise ValueError("ball must have shape (B, T, 3)")
    if ball_location_valid.shape != players.shape[:2]:
        raise ValueError("ball_location_valid must have shape (B, T)")

    valid = player_location_valid & ball_location_valid[..., None, None]
    delta = players[..., :2] - ball[..., None, None, :2]
    distance_m = torch.sqrt(
        (delta[..., 0] * PITCH_LENGTH_M).square()
        + (delta[..., 1] * PITCH_WIDTH_M).square()
    )
    relative = torch.cat(
        (delta, (distance_m / PITCH_DIAGONAL_M).unsqueeze(-1)), dim=-1
    )
    relative = relative * valid.unsqueeze(-1).to(relative.dtype)
    distance_m = distance_m * valid.to(distance_m.dtype)
    return relative, distance_m, valid


class BallConditionedDeepSetPlayerEncoder(nn.Module):
    """Encode unordered player sets after exposing each player to the ball geometry."""

    summary_dim = 8
    cross_team_dim = 2

    def __init__(self, hidden_dim: int = 32, embedding_dim: int = 32) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.team_output_dim = 3 * embedding_dim + self.summary_dim
        self.output_dim = 2 * self.team_output_dim + self.cross_team_dim
        self.player_mlp = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
        )
        self.attention_score = nn.Linear(embedding_dim, 1)

    @staticmethod
    def _masked_max(encoded: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        masked = encoded.masked_fill(
            ~present.unsqueeze(-1), torch.finfo(encoded.dtype).min
        )
        result = masked.max(dim=-2).values
        return torch.where(present.any(dim=-1, keepdim=True), result, 0.0)

    def _masked_attention(
        self, encoded: torch.Tensor, present: torch.Tensor
    ) -> torch.Tensor:
        scores = self.attention_score(encoded).squeeze(-1)
        scores = scores.masked_fill(~present, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(present, weights, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return (encoded * weights.unsqueeze(-1)).sum(dim=-2)

    @staticmethod
    def _geometry_summary(
        distance_m: torch.Tensor,
        geometry_valid: torch.Tensor,
        player_location_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = distance_m.dtype
        valid_count = geometry_valid.sum(dim=-1)
        sortable = distance_m.masked_fill(~geometry_valid, torch.inf)
        nearest_values = sortable.topk(k=2, dim=-1, largest=False).values
        nearest_valid = valid_count >= 1
        second_valid = valid_count >= 2
        nearest = torch.where(nearest_valid, nearest_values[..., 0], 0.0)
        second = torch.where(second_valid, nearest_values[..., 1], 0.0)
        nearest = nearest / PITCH_DIAGONAL_M
        second = second / PITCH_DIAGONAL_M
        nearby = [
            ((distance_m <= radius) & geometry_valid).sum(dim=-1).to(dtype)
            for radius in NEARBY_RADII_M
        ]
        tracked = player_location_valid.sum(dim=-1).to(dtype)
        summary = torch.stack(
            (
                nearest,
                second,
                *nearby,
                tracked,
                nearest_valid.to(dtype),
                second_valid.to(dtype),
            ),
            dim=-1,
        )
        return summary, nearest, nearest_valid

    def forward(
        self,
        players: torch.Tensor,
        location_valid: torch.Tensor,
        speed_valid: torch.Tensor,
        ball: torch.Tensor,
        ball_location_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return one `(B, T, output_dim)` ball-conditioned set representation."""
        if location_valid.shape != players.shape[:-1] or speed_valid.shape != players.shape[:-1]:
            raise ValueError("Player validity masks do not match players")
        relative, distance_m, geometry_valid = player_ball_geometry(
            players, location_valid, ball, ball_location_valid
        )
        dtype = players.dtype
        encoded_input = torch.cat(
            (
                players,
                location_valid.unsqueeze(-1).to(dtype),
                speed_valid.unsqueeze(-1).to(dtype),
                relative,
                geometry_valid.unsqueeze(-1).to(dtype),
            ),
            dim=-1,
        )
        encoded = self.player_mlp(encoded_input)
        present = location_valid | speed_valid
        encoded = encoded * present.unsqueeze(-1).to(dtype)
        count = present.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)
        mean = encoded.sum(dim=-2) / count
        maximum = self._masked_max(encoded, present)
        attention = self._masked_attention(encoded, present)
        summary, nearest, nearest_valid = self._geometry_summary(
            distance_m, geometry_valid, location_valid
        )
        teams = torch.cat((mean, maximum, attention, summary), dim=-1)

        margin_valid = nearest_valid[..., 0] & nearest_valid[..., 1]
        margin = torch.where(
            margin_valid, nearest[..., 0] - nearest[..., 1], 0.0
        )
        cross_team = torch.stack((margin, margin_valid.to(dtype)), dim=-1)
        return torch.cat((teams.flatten(-2), cross_team), dim=-1)
