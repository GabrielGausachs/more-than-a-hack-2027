"""Purged temporal inner folds and full-half refit splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    partition: str
    half_id: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    validation_block: int | None
    train_mask: np.ndarray
    validation_mask: np.ndarray
    buffer_mask: np.ndarray
    test_mask: np.ndarray
    segments: Dict[str, Tuple[Segment, ...]]

    def assert_valid(self) -> None:
        masks = (
            self.train_mask,
            self.validation_mask,
            self.buffer_mask,
            self.test_mask,
        )
        if len({mask.shape for mask in masks}) != 1:
            raise ValueError("All partition masks must have the same shape")
        membership = sum(mask.astype(np.int8) for mask in masks)
        if not np.all(membership == 1):
            raise ValueError("Every frame must belong to exactly one partition")
        names = ("train", "validation", "buffer", "test")
        for name, mask in zip(names, masks):
            reconstructed = np.zeros_like(mask)
            for segment in self.segments[name]:
                if segment.start < 0 or segment.end > len(mask) or segment.start >= segment.end:
                    raise ValueError(f"Invalid {name} segment: {segment}")
                reconstructed[segment.start : segment.end] = True
            if not np.array_equal(mask, reconstructed):
                raise ValueError(f"{name} segments do not reproduce the {name} mask")


def _half_segment(half_id: np.ndarray, half: int, partition: str) -> Segment:
    indices = np.flatnonzero(half_id == half)
    if not len(indices) or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        raise ValueError(f"Half {half} must be one non-empty contiguous block")
    return Segment(int(indices[0]), int(indices[-1] + 1), partition, half)


def valid_subsegments(segment: Segment, valid: np.ndarray) -> Tuple[Segment, ...]:
    """Split one segment at invalid-label gaps without changing its half."""
    if valid.ndim != 1 or valid.dtype != np.bool_ or segment.end > len(valid):
        raise ValueError("valid must be a full-timeline Boolean mask")
    local = valid[segment.start : segment.end]
    edges = np.diff(np.r_[False, local, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return tuple(
        Segment(
            segment.start + int(start),
            segment.start + int(end),
            segment.partition,
            segment.half_id,
        )
        for start, end in zip(starts, ends)
    )


def build_fold_split(
    half_id: np.ndarray,
    fold: int,
    validation_fraction: float = 0.15,
    buffer_frames: int = 100,
    validation_block: int = 2,
) -> FoldSplit:
    """Build one of three purged inner-validation splits."""
    if fold not in (1, 2):
        raise ValueError("fold must be 1 or 2")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if buffer_frames < 0:
        raise ValueError("buffer_frames cannot be negative")
    if validation_block not in (1, 2, 3):
        raise ValueError("validation_block must be 1, 2 or 3")
    if set(np.unique(half_id).tolist()) != {1, 2}:
        raise ValueError("half_id must contain exactly halves 1 and 2")

    train_half = fold
    test_half = 2 if fold == 1 else 1
    train_block = _half_segment(half_id, train_half, "training_half")
    test_segment = _half_segment(half_id, test_half, "test")
    half_length = train_block.length
    validation_length = int(round(half_length * validation_fraction))
    validation_start_fractions = (0.175, 0.425, 0.675)
    validation_local_start = int(
        round(half_length * validation_start_fractions[validation_block - 1])
    )
    validation_start = train_block.start + validation_local_start
    validation_end = validation_start + validation_length
    left_buffer_start = validation_start - buffer_frames
    right_buffer_end = validation_end + buffer_frames
    if left_buffer_start <= train_block.start or right_buffer_end >= train_block.end:
        raise ValueError("Validation block and buffers leave no training data on both sides")

    segments = {
        "train": (
            Segment(train_block.start, left_buffer_start, "train", train_half),
            Segment(right_buffer_end, train_block.end, "train", train_half),
        ),
        "validation": (
            Segment(validation_start, validation_end, "validation", train_half),
        ),
        "buffer": (
            Segment(left_buffer_start, validation_start, "buffer", train_half),
            Segment(validation_end, right_buffer_end, "buffer", train_half),
        ),
        "test": (test_segment,),
    }
    masks = {}
    for name, partition_segments in segments.items():
        mask = np.zeros(len(half_id), dtype=bool)
        for segment in partition_segments:
            mask[segment.start : segment.end] = True
        masks[name] = mask

    split = FoldSplit(
        fold=fold,
        validation_block=validation_block,
        train_mask=masks["train"],
        validation_mask=masks["validation"],
        buffer_mask=masks["buffer"],
        test_mask=masks["test"],
        segments=segments,
    )
    split.assert_valid()
    return split


def build_refit_split(half_id: np.ndarray, fold: int) -> FoldSplit:
    """Use the complete selected half for training and the opposite half for testing."""
    if fold not in (1, 2):
        raise ValueError("fold must be 1 or 2")
    if set(np.unique(half_id).tolist()) != {1, 2}:
        raise ValueError("half_id must contain exactly halves 1 and 2")
    train_segment = _half_segment(half_id, fold, "train")
    test_half = 2 if fold == 1 else 1
    test_segment = _half_segment(half_id, test_half, "test")
    segments = {
        "train": (train_segment,),
        "validation": (),
        "buffer": (),
        "test": (test_segment,),
    }
    masks = {}
    for name, partition_segments in segments.items():
        mask = np.zeros(len(half_id), dtype=bool)
        for segment in partition_segments:
            mask[segment.start : segment.end] = True
        masks[name] = mask
    split = FoldSplit(
        fold=fold,
        validation_block=None,
        train_mask=masks["train"],
        validation_mask=masks["validation"],
        buffer_mask=masks["buffer"],
        test_mask=masks["test"],
        segments=segments,
    )
    split.assert_valid()
    return split
