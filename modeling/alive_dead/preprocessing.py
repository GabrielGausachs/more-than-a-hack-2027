"""Train-fold-only normalization for WCSF tracking features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class FeatureNormalizer:
    player_speed_mean: float
    player_speed_std: float
    team_speed_mean: np.ndarray
    team_speed_std: np.ndarray
    epsilon: float = 1e-6
    ball_speed_mean: float | None = None
    ball_speed_std: float | None = None

    @classmethod
    def fit(
        cls,
        players: np.ndarray,
        player_speed_valid: np.ndarray,
        team_features: np.ndarray,
        team_feature_valid: np.ndarray,
        train_mask: np.ndarray,
        epsilon: float = 1e-6,
        ball: np.ndarray | None = None,
        ball_speed_valid: np.ndarray | None = None,
    ) -> "FeatureNormalizer":
        """Fit shared-team speed statistics using training frames only."""
        if train_mask.dtype != np.bool_ or train_mask.shape != (len(players),):
            raise ValueError("train_mask must be Boolean with one value per frame")
        player_mask = player_speed_valid & train_mask[:, None, None]
        valid_player_speeds = players[..., 2][player_mask].astype(np.float64)
        if not len(valid_player_speeds):
            raise ValueError("Training partition contains no valid player speeds")
        player_mean = float(valid_player_speeds.mean())
        player_std = float(valid_player_speeds.std(ddof=0))
        if player_std < epsilon:
            raise ValueError("Training player-speed standard deviation is too small")

        team_means = np.zeros(3, dtype=np.float64)
        team_stds = np.zeros(3, dtype=np.float64)
        for feature_index in range(3):
            valid = team_feature_valid[..., feature_index] & train_mask[:, None]
            values = team_features[..., feature_index][valid].astype(np.float64)
            if not len(values):
                raise ValueError(
                    f"Training partition contains no valid team feature {feature_index}"
                )
            team_means[feature_index] = values.mean()
            team_stds[feature_index] = values.std(ddof=0)
            if team_stds[feature_index] < epsilon:
                raise ValueError(
                    f"Training team-feature standard deviation is too small at index {feature_index}"
                )
        ball_mean = None
        ball_std = None
        if (ball is None) != (ball_speed_valid is None):
            raise ValueError("ball and ball_speed_valid must be provided together")
        if ball is not None:
            if ball.shape != (len(players), 3) or ball_speed_valid.shape != (len(players),):
                raise ValueError("Ball arrays have unexpected shapes")
            values = ball[:, 2][ball_speed_valid & train_mask].astype(np.float64)
            if not len(values):
                raise ValueError("Training partition contains no valid ball speeds")
            ball_mean = float(values.mean())
            ball_std = float(values.std(ddof=0))
            if ball_std < epsilon:
                raise ValueError("Training ball-speed standard deviation is too small")
        return cls(
            player_mean, player_std, team_means, team_stds, epsilon,
            ball_mean, ball_std,
        )

    def transform(
        self,
        players: np.ndarray,
        player_location_valid: np.ndarray,
        player_speed_valid: np.ndarray,
        team_features: np.ndarray,
        team_feature_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply normalization while preserving masks and zero-filled invalid entries."""
        normalized_players = players.copy()
        normalized_players[..., :2][~player_location_valid] = 0.0
        speed_output = np.zeros_like(normalized_players[..., 2])
        speed_output[player_speed_valid] = (
            players[..., 2][player_speed_valid] - self.player_speed_mean
        ) / self.player_speed_std
        normalized_players[..., 2] = speed_output

        normalized_team = team_features.copy()
        for feature_index in range(3):
            valid = team_feature_valid[..., feature_index]
            output = np.zeros_like(normalized_team[..., feature_index])
            output[valid] = (
                team_features[..., feature_index][valid]
                - self.team_speed_mean[feature_index]
            ) / self.team_speed_std[feature_index]
            normalized_team[..., feature_index] = output
        normalized_team[~team_feature_valid] = 0.0

        if not np.isfinite(normalized_players).all() or not np.isfinite(normalized_team).all():
            raise ValueError("Normalization produced non-finite values")
        return normalized_players.astype(np.float32), normalized_team.astype(np.float32)

    def transform_ball(
        self,
        ball: np.ndarray,
        ball_location_valid: np.ndarray,
        ball_speed_valid: np.ndarray,
    ) -> np.ndarray:
        """Normalize valid ball speed and preserve zero-filled invalid measurements."""
        if self.ball_speed_mean is None or self.ball_speed_std is None:
            raise ValueError("Normalizer does not contain ball-speed statistics")
        if ball.shape != (len(ball), 3):
            raise ValueError("ball must have shape (N, 3)")
        if (
            ball_location_valid.shape != (len(ball),)
            or ball_speed_valid.shape != (len(ball),)
        ):
            raise ValueError("Ball validity masks must have shape (N,)")
        output = ball.copy()
        output[:, :2][~ball_location_valid] = 0.0
        speed = np.zeros(len(ball), dtype=np.float32)
        speed[ball_speed_valid] = (
            ball[:, 2][ball_speed_valid] - self.ball_speed_mean
        ) / self.ball_speed_std
        output[:, 2] = speed
        if not np.isfinite(output).all():
            raise ValueError("Ball normalization produced non-finite values")
        return output.astype(np.float32)

    def state_dict(self) -> Dict[str, object]:
        state = {
            "player_speed_mean": self.player_speed_mean,
            "player_speed_std": self.player_speed_std,
            "team_speed_mean": self.team_speed_mean.tolist(),
            "team_speed_std": self.team_speed_std.tolist(),
            "epsilon": self.epsilon,
        }
        if self.ball_speed_mean is not None and self.ball_speed_std is not None:
            state.update(
                {
                    "ball_speed_mean": self.ball_speed_mean,
                    "ball_speed_std": self.ball_speed_std,
                }
            )
        return state
