"""Diagnostic timelines for ALIVE/DEAD predictions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from .evaluate import PredictionRun


CLASS_COLORS = {0: "#b8b8b8", 1: "#2ca25f"}


def _group_runs_by_half(
    runs: Sequence[PredictionRun],
) -> tuple[tuple[int, tuple[PredictionRun, ...]], ...]:
    """Validate and chronologically group prediction runs by half."""
    if not runs:
        raise ValueError("At least one prediction run is required")

    grouped: dict[int, list[PredictionRun]] = {}
    for run in runs:
        if int(run.half_id) not in (1, 2):
            raise ValueError("Prediction run half_id must be 1 or 2")
        labels = np.asarray(run.labels)
        predictions = np.asarray(run.predictions)
        if run.start < 0 or run.end <= run.start or run.length != len(labels):
            raise ValueError("Prediction run has invalid boundaries")
        if labels.ndim != 1 or predictions.shape != labels.shape:
            raise ValueError("Prediction labels and predictions must be aligned 1D arrays")
        if not np.isin(labels, (0, 1)).all() or not np.isin(predictions, (0, 1)).all():
            raise ValueError("Prediction timelines require binary labels and predictions")
        grouped.setdefault(int(run.half_id), []).append(run)

    result = []
    for half_id in sorted(grouped):
        half_runs = tuple(sorted(grouped[half_id], key=lambda run: run.start))
        if any(left.end > right.start for left, right in zip(half_runs, half_runs[1:])):
            raise ValueError(f"Prediction runs overlap in half {half_id}")
        result.append((half_id, half_runs))
    return tuple(result)


def save_prediction_timeline(
    runs: Sequence[PredictionRun],
    source_fps: float,
    output_path: str | Path,
    title: str,
) -> Path:
    """Save aligned ground-truth and prediction segment bands for each half."""
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be a positive finite value")
    grouped = _group_runs_by_half(runs)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Timeline plotting requires Matplotlib; install requirements-starter.txt"
        ) from exc

    from .evaluate import extract_segments

    figure, axes = plt.subplots(
        len(grouped),
        1,
        figsize=(16, 2.2 * len(grouped) + 1.0),
        squeeze=False,
    )
    for axis, (half_id, half_runs) in zip(axes[:, 0], grouped):
        origin = half_runs[0].start
        for run in half_runs:
            for values, y_range in ((run.labels, (1.1, 0.7)), (run.predictions, (0.1, 0.7))):
                for segment in extract_segments(values):
                    start_minutes = (
                        run.start - origin + segment.start
                    ) / source_fps / 60.0
                    width_minutes = (segment.end - segment.start) / source_fps / 60.0
                    axis.broken_barh(
                        [(start_minutes, width_minutes)],
                        y_range,
                        facecolors=CLASS_COLORS[segment.label],
                        edgecolors="white",
                        linewidth=0.25,
                    )

        end_minutes = (max(run.end for run in half_runs) - origin) / source_fps / 60.0
        axis.set_xlim(0.0, end_minutes)
        axis.set_ylim(0.0, 2.0)
        axis.set_yticks((1.45, 0.45), labels=("Ground truth", "Prediction"))
        axis.set_xlabel("Elapsed time within half (minutes)")
        axis.set_title("First half" if half_id == 1 else "Second half")
        axis.grid(axis="x", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)

    figure.suptitle(title)
    figure.legend(
        handles=(
            Patch(facecolor=CLASS_COLORS[1], label="ALIVE"),
            Patch(facecolor=CLASS_COLORS[0], label="DEAD"),
        ),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.965),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path
