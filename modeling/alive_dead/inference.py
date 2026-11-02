"""Shared full-segment and receptive-field-aware chunked inference."""

from __future__ import annotations

import numpy as np
import torch

from .splits import Segment


def predict_segment_logits(
    model: torch.nn.Module,
    players: np.ndarray,
    player_location_valid: np.ndarray,
    player_speed_valid: np.ndarray,
    team_features: np.ndarray,
    team_feature_valid: np.ndarray,
    segment: Segment,
    device: torch.device,
    chunk_frames: int | None = None,
    ball: np.ndarray | None = None,
    ball_location_valid: np.ndarray | None = None,
    ball_speed_valid: np.ndarray | None = None,
) -> torch.Tensor:
    """Return one logit per frame without crossing the supplied segment."""
    if chunk_frames is not None and chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    temporal = getattr(model, "temporal", None)
    if temporal is None or not hasattr(temporal, "receptive_field_frames"):
        raise ValueError("Model does not expose its temporal receptive field")
    receptive_field = int(temporal.receptive_field_frames)
    if receptive_field % 2 == 0:
        raise ValueError("Chunked noncausal inference requires an odd receptive field")
    radius = (receptive_field - 1) // 2
    ball_arrays = (ball, ball_location_valid, ball_speed_valid)
    if any(value is not None for value in ball_arrays) and not all(
        value is not None for value in ball_arrays
    ):
        raise ValueError("Ball values and both validity masks must be provided together")
    if getattr(model, "use_ball", False) and ball is None:
        raise ValueError("Ball-enabled model requires ball arrays for inference")
    output_chunk = segment.length if chunk_frames is None else chunk_frames
    outputs = []
    model.eval()
    with torch.no_grad():
        for local_start in range(0, segment.length, output_chunk):
            local_end = min(segment.length, local_start + output_chunk)
            input_start = max(0, local_start - radius)
            input_end = min(segment.length, local_end + radius)
            frame_slice = slice(segment.start + input_start, segment.start + input_end)
            length = input_end - input_start
            batch = {
                "players": torch.from_numpy(players[frame_slice]).unsqueeze(0).to(device),
                "player_location_valid": torch.from_numpy(
                    player_location_valid[frame_slice]
                ).unsqueeze(0).to(device),
                "player_speed_valid": torch.from_numpy(
                    player_speed_valid[frame_slice]
                ).unsqueeze(0).to(device),
                "team_features": torch.from_numpy(team_features[frame_slice]).unsqueeze(0).to(device),
                "team_feature_valid": torch.from_numpy(
                    team_feature_valid[frame_slice]
                ).unsqueeze(0).to(device),
                "output_mask": torch.ones((1, length), dtype=torch.bool, device=device),
            }
            if ball is not None:
                batch.update(
                    {
                        "ball": torch.from_numpy(ball[frame_slice]).unsqueeze(0).to(device),
                        "ball_location_valid": torch.from_numpy(
                            ball_location_valid[frame_slice]
                        ).unsqueeze(0).to(device),
                        "ball_speed_valid": torch.from_numpy(
                            ball_speed_valid[frame_slice]
                        ).unsqueeze(0).to(device),
                    }
                )
            logits = model(batch).squeeze(0)
            keep_start = local_start - input_start
            keep_end = keep_start + local_end - local_start
            outputs.append(logits[keep_start:keep_end].detach().cpu())
    result = torch.cat(outputs, dim=0)
    if result.shape[0] != segment.length or not torch.isfinite(result).all():
        raise ValueError("Model produced invalid segment logits")
    return result
