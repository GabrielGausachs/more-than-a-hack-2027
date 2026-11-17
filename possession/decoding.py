"""Period-aware Viterbi decoding for H/A/D probabilities."""

from __future__ import annotations

import numpy as np

from .modeling.data import LABELS, LABEL_TO_INDEX, period_slices


LABEL_ARRAY = np.asarray(LABELS, dtype="U1")


def viterbi_decode(probability: np.ndarray, period: np.ndarray,
                   switch_penalty: float = 1.5,
                   epsilon: float = 1e-8) -> np.ndarray:
    """Decode P1 and P2 independently with a symmetric switch cost."""

    if probability.shape != (len(period), 3):
        raise ValueError("probabilities must have shape [frames,3]")
    transition = np.full((3, 3), -float(switch_penalty), dtype=np.float64)
    np.fill_diagonal(transition, 0.0)
    codes = np.full(len(period), LABEL_TO_INDEX["D"], dtype=np.int8)

    for _, start, stop in period_slices(period):
        emission = np.log(np.maximum(probability[start:stop], float(epsilon)))
        back = np.zeros((len(emission), 3), dtype=np.int8)
        score = emission[0].copy()
        for frame in range(1, len(emission)):
            candidate = score[:, None] + transition
            back[frame] = candidate.argmax(axis=0)
            score = emission[frame] + candidate.max(axis=0)
        state = int(score.argmax())
        decoded = np.empty(len(emission), dtype=np.int8)
        decoded[-1] = state
        for frame in range(len(emission) - 1, 0, -1):
            state = int(back[frame, state])
            decoded[frame - 1] = state
        codes[start:stop] = decoded
    return LABEL_ARRAY[codes]
